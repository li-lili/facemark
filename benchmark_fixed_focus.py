"""Independent focus experiment. No camera unless --run-camera is supplied.

Only native CameraControl_Focus is writable (manual/auto); OpenCV setters only
negotiate width/height/FPS/FOURCC. Original focus state is restored in finally.
No serial/controller imports. Production detection and baselines are read-only.
"""
import argparse
import ast
import ctypes as C
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import benchmark_front_camera as bench
import compare_camera_quality as quality
import query_front_camera_modes as ds
from query_front_camera_focus import inspect_focus

FOCUS = 450
FIELDS = ("L_dX", "L_dY", "L_dist", "R_dX", "R_dY", "R_dist")
MAIN_SEQUENCE = (("auto", "YUY2"), ("auto", "MJPG"),
                 ("manual", "YUY2"), ("manual", "MJPG"),
                 ("manual", "MJPG"), ("manual", "YUY2"),
                 ("auto", "MJPG"), ("auto", "YUY2"))
MODE_PROPERTIES = ("FRAME_WIDTH", "FRAME_HEIGHT", "FPS", "FOURCC")


class FocusAccess:
    """Retain a property interface in a live STA; no graph is run here."""
    def __init__(self):
        self.ptr = None
        self.original = None
        self.ole = None
        self.device = None

    def open(self):
        self.ole = C.OleDLL("ole32")
        self.ole.CoInitializeEx.argtypes = [ds.PTR, ds.ULONG]
        self.ole.CoInitializeEx.restype = ds.LONG
        ds.checked(self.ole.CoInitializeEx(None, 2), "Focus COM initialize")
        try:
            def bind(source, ole):
                capabilities = inspect_focus(source, ole)
                if not capabilities.get("supports_auto") or not capabilities.get("supports_manual"):
                    raise RuntimeError("Both auto/manual focus capabilities are required.")
                if not capabilities["minimum"] <= FOCUS <= capabilities["maximum"]:
                    raise RuntimeError("Candidate focus outside capability range.")
                self.ptr = ds.query_interface(source, "c6e13370-30ac-11d0-a18c-00a0c9118956")
                if not self.ptr:
                    raise RuntimeError("Cannot retain camera focus interface.")
                return capabilities
            result = ds.query_camera_one(filter_inspector=bind)
            self.device = result["selected_device"]
            self.original = self.read()
            return result["filter_inspection"]
        except BaseException:
            self.close()
            raise

    def read(self):
        value, flags = ds.LONG(), ds.LONG()
        ds.checked(ds.method(self.ptr, 5, ds.LONG, [ds.LONG, ds.PTR, ds.PTR],
                             6, C.byref(value), C.byref(flags)), "Get focus")
        return {"value": value.value, "flags": flags.value,
                "auto": bool(flags.value & 1), "manual": bool(flags.value & 2)}

    def write_focus(self, value, flags):
        # IAMCameraControl.Set, property 6 ONLY. No other native setters exist.
        if flags not in (1, 2):
            raise ValueError("Only Auto(1) or Manual(2) allowed.")
        ds.checked(ds.method(self.ptr, 4, ds.LONG, [ds.LONG, ds.LONG, ds.LONG],
                             6, int(value), flags), "Set focus only")

    def configure(self, mode):
        self.write_focus(FOCUS if mode == "manual" else self.read()["value"],
                         2 if mode == "manual" else 1)
        return self.verify(mode)

    def verify(self, mode):
        current = self.read()
        if mode == "manual":
            if current["auto"] or not current["manual"] or current["value"] != FOCUS:
                raise RuntimeError(f"Focus lock failed: {current}")
        elif not current["auto"]:
            raise RuntimeError(f"Autofocus reference not automatic: {current}")
        return current

    def restore(self):
        if self.original is None:
            return {"attempted": False, "verified": False}
        log = []
        flags = 1 if self.original["auto"] else 2
        for attempt in range(1, 4):
            try:
                self.write_focus(self.original["value"], flags)
                time.sleep(.15)
                current = self.read()
                ok = current["auto"] if flags == 1 else (
                    current["manual"] and not current["auto"]
                    and current["value"] == self.original["value"])
                log.append({"attempt": attempt, "readback": current, "verified": bool(ok)})
                if ok:
                    return {"attempted": True, "verified": True, "original": self.original,
                            "readback": current, "attempts": log}
            except Exception as exc:
                log.append({"attempt": attempt, "error": str(exc)})
        return {"attempted": True, "verified": False, "original": self.original, "attempts": log}

    def close(self):
        if self.ptr:
            ds.release(self.ptr)
            self.ptr = None
        if self.ole is not None:
            self.ole.CoUninitialize()
            self.ole = None


def reduce60(signals):
    """Exact production sorting/trimming/arithmetic (stable sort, default zeros)."""
    if len(signals) != 60:
        raise ValueError("Exactly 60 non-None guide signals are required.")
    offsets = [{k: signal.get(k, 0) for k in FIELDS} for signal in signals]
    order = sorted(range(60), key=lambda i: offsets[i]["L_dist"]+offsets[i]["R_dist"])
    kept = order[10:-10]
    avg = {k: sum(offsets[i][k] for i in kept)/len(kept) for k in FIELDS}
    for side in ("L", "R"):
        avg[f"{side}_adj_X"] = -avg[f"{side}_dX"]
        avg[f"{side}_adj_Y"] = -avg[f"{side}_dY"]
    avg["total_dist"] = avg["L_dist"]+avg["R_dist"]
    return {"average": avg, "kept_zero_based": kept,
            "discard_low": order[:10], "discard_high": order[-10:],
            "left_pass": avg["L_dist"] <= 1.0, "right_pass": avg["R_dist"] <= 1.0,
            "pass": avg["L_dist"] <= 1.0 and avg["R_dist"] <= 1.0}


def intent(avg, locks=(False, False)):
    """Pure axis intentions; not a command, not a simulated movement."""
    output, next_locks = {}, []
    for side, locked in zip(("L", "R"), locks):
        dx, dy, distance = (avg[f"{side}_{k}"] for k in ("dX", "dY", "dist"))
        locked = locked or abs(dx) <= 1.0
        next_locks.append(locked)
        x = -dx if distance > 1.0 and not locked else 0.0
        y = -dy if distance > 1.0 and abs(dy) > 1.0 else 0.0
        output[side] = {"x_sign": (x > 0)-(x < 0), "y_sign": (y > 0)-(y < 0)}
    return {"intent": output, "horizontal_locks": next_locks}


def collect(block, data, rois, folder, access, precheck=False):
    import cv2
    import numpy as np
    visual, detector = bench.load_visual_pipeline(data)
    for name in ("lm_smoother", "ell_smooth_L", "ell_smooth_R"):
        setattr(detector, name, quality.Observer(getattr(detector, name)))
    cap = None
    pictures = []
    rows = block["rows"] = []
    native = block["focus_samples"] = []
    block["batches"] = []
    try:
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise RuntimeError("Camera 1 failed to open; no fallback index.")
        initial = block["opened_settings"] = bench.camera_settings(cap, cv2)
        block["mode_requests"] = []
        for name, value in (("FRAME_WIDTH", 1920), ("FRAME_HEIGHT", 1080),
                            ("FPS", 5 if block["format"] == "YUY2" else 30),
                            ("FOURCC", cv2.VideoWriter_fourcc(*block["format"]))):
            assert name in MODE_PROPERTIES
            accepted = bool(cap.set(getattr(cv2, "CAP_PROP_"+name), value))
            current = bench.camera_settings(cap, cv2)
            block["mode_requests"].append(
                {"property": name, "value": value, "accepted": accepted, "readback": current})
            if not accepted:
                raise RuntimeError(f"Mode request failed: {name}")
            for key in ("exposure", "auto_exposure", "gain", "auto_wb"):
                if current[key] != initial[key]:
                    raise RuntimeError(f"Non-focus control changed during mode negotiation: {key}")
        block["mode_before_focus"] = quality.verify_mode(cap, cv2, block["format"])
        block["focus_after_mode_before_setting"] = access.read()
        block["focus_after_setting"] = access.configure(block["focus_mode"])
        # Cross-check the capturing OpenCV filter sees the same native device setting.
        cv_focus = bench.camera_settings(cap, cv2)
        if block["focus_mode"] == "manual":
            if cv_focus["focus"] != FOCUS or int(cv_focus["autofocus"]) & 1:
                raise RuntimeError(f"Capturing filter disagrees with native focus lock: {cv_focus}")
        settle = time.perf_counter()
        while time.perf_counter()-settle < 6:
            if not cap.read()[0]:
                raise RuntimeError("Settle read failed.")
        access.verify(block["focus_mode"])
        for _ in range(20):
            ok, camera_frame = cap.read()
            if not ok:
                raise RuntimeError("Model warmup read failed.")
            detector.detect(cv2.rotate(camera_frame, cv2.ROTATE_180))
        block["before_measurement"] = quality.verify_mode(cap, cv2, block["format"])
        started = time.perf_counter()
        attempts = read_failures = guide_count = consecutive_missing = 0
        last_poll = last_progress = -100.
        batch_signals, batch_row_indices = [], []
        batch_start = started
        target = 12 if precheck else 300
        while (len(rows) if precheck else guide_count) < target:
            elapsed = time.perf_counter()-started
            if elapsed > 150:
                raise TimeoutError("Block exceeded 150s; native calls may still block individually.")
            if elapsed-last_poll >= 1:
                settings = bench.camera_settings(cap, cv2)
                focus = access.verify(block["focus_mode"])
                native.append({"seconds": elapsed, "native": focus, "opencv": settings})
                if block["focus_mode"] == "manual" and settings["focus"] != FOCUS:
                    raise RuntimeError("OpenCV focus changed while native lock was requested.")
                last_poll = elapsed
            t0 = time.perf_counter()
            ok, camera_frame = cap.read()
            t1 = time.perf_counter()
            attempts += 1
            if not ok:
                read_failures += 1
                time.sleep(.001)
                continue
            if camera_frame.shape != (1080, 1920, 3):
                raise RuntimeError("Unexpected decoded resolution/shape.")
            frame = cv2.rotate(camera_frame, cv2.ROTATE_180)
            for name in ("lm_smoother", "ell_smooth_L", "ell_smooth_R"):
                getattr(detector, name).clear()
            result = detector.detect(frame)
            signal = detector.get_guide_signal()
            row = quality.observation(frame, result, detector, visual, rois, cv2, np)
            row.update({"frame": len(rows)+1, "t_seconds": t1-started,
                        "read_ms": 1000*(t1-t0), "guide": signal,
                        "target_offset": result.get("target_offset"),
                        "raw_offset": result.get("offset")})
            rows.append(row)
            if signal is None:
                consecutive_missing += 1
                if consecutive_missing > 10:
                    raise RuntimeError("Production sampling stops after >10 consecutive missing guides.")
                time.sleep(.001)
            else:
                consecutive_missing = 0
                guide_count += 1
                batch_signals.append(signal)
                batch_row_indices.append(len(rows)-1)
                if len(batch_signals) == 60:
                    b = reduce60(batch_signals)
                    b["row_indices"] = batch_row_indices
                    b["elapsed_seconds"] = time.perf_counter()-batch_start
                    b["both_eyes_complete"] = all(rows[i]["left"]["valid"] and
                                                 rows[i]["right"]["valid"] for i in batch_row_indices)
                    b["intent_fresh_locks"] = intent(b["average"])
                    block["batches"].append(b)
                    print(f"BLOCK {block['number']} BATCH {len(block['batches'])}: "
                          f"Ldist={b['average']['L_dist']:.3f}, "
                          f"Rdist={b['average']['R_dist']:.3f}, pass={b['pass']}", flush=True)
                    batch_signals, batch_row_indices = [], []
                    batch_start = time.perf_counter()
            if len(rows) in ((1, 6, 12) if precheck else (1, 60, 120, 180, 240, 300)):
                pictures.append((len(rows), camera_frame.copy(), frame.copy()))
            if elapsed-last_progress >= 10:
                print(f"BLOCK {block['number']} {block['focus_mode']} {block['format']}: "
                      f"{len(rows)} processed, {guide_count} guides, {elapsed:.1f}s", flush=True)
                last_progress = elapsed
        block.update({"elapsed_seconds": time.perf_counter()-started, "attempts": attempts,
                      "read_failures": read_failures, "guide_count": guide_count,
                      "final_focus": access.verify(block["focus_mode"]),
                      "final_settings": quality.verify_mode(cap, cv2, block["format"]),
                      "status": "complete"})
        block["processing_fps"] = len(rows)/block["elapsed_seconds"]
        ids = sorted({int(i) for r in rows for i in r["mp_raw"]})
        block["statistics"] = quality.series_summary(rows, ids, np)
    finally:
        if cap is not None:
            cap.release()
        detector.close()
        block["images"] = []
        for number, camera, oriented in pictures:
            stem = f"block{block['number']}_{block['focus_mode']}_{block['format']}_{number:04d}"
            entry = {"frame": number}
            for kind, img in (("camera", camera), ("oriented", oriented)):
                filename = f"{stem}_{kind}.png"
                if not cv2.imwrite(str(folder / filename), img):
                    raise IOError(filename)
                entry[kind+"_png"] = filename
            block["images"].append(entry)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-camera", action="store_true")
    parser.add_argument("--phase", choices=("precheck", "experiment"), default="precheck")
    parser.add_argument("--precheck-dir", type=Path)
    args = parser.parse_args()
    data = bench.preflight()
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    if not args.run_camera:
        print("Preflight OK. No COM/camera/model initialization or settings.")
        return 0
    if args.phase == "experiment":
        if args.precheck_dir is None:
            raise RuntimeError("Reviewed precheck directory is required.")
        checked = json.loads((args.precheck_dir/"data.json").read_text(encoding="utf-8"))
        review = json.loads((args.precheck_dir/"visual_review.json").read_text(encoding="utf-8"))
        if not (checked["status"] == "complete" and checked["focus_restore"]["verified"]
                and review["accepted"] and review["focus"] == FOCUS):
            raise RuntimeError("Candidate focus did not pass the precheck.")
    folder = bench.ROOT/"benchmark_results"/(
        f"focus450_{args.phase}_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    folder.mkdir()
    before = bench.protected_hashes()
    report = {"phase": args.phase, "status": "running", "camera_index": 1, "focus": FOCUS,
              "started_at": datetime.now().astimezone().isoformat(),
              "blocks": [], "fixed_eye_rois": {},
              "baseline_snapshot": {"common": data["common"],
                                    "active_template": data["active_template"],
                                    "head_adapter": data["templates"][data["active_template"]].get("head_adapter")},
              "source_hashes": {n: hashlib.sha256((bench.ROOT/n).read_bytes()).hexdigest()
                                for n in ("ellseg_scorer.py","eye_auto_tuner.py","eye_constants.py",
                                          "face_point.json","servo_motion_rules.yaml")}}
    access = FocusAccess()
    try:
        report["capabilities"] = access.open()
        report["device"] = access.device
        report["original_focus"] = access.original
        quality.save_json(folder/"data.json", report)  # Persist recovery information BEFORE setters.
        sequence = (("auto", "YUY2"), ("manual", "YUY2"), ("manual", "MJPG")) \
            if args.phase == "precheck" else MAIN_SEQUENCE
        for number, (mode, codec) in enumerate(sequence, 1):
            block = {"number": number, "focus_mode": mode, "format": codec, "status": "running"}
            report["blocks"].append(block)
            print(f"START {args.phase} block {number}: {mode} {codec}, camera 1 only", flush=True)
            collect(block, data, report["fixed_eye_rois"], folder, access,
                    precheck=args.phase == "precheck")
            quality.save_json(folder/"data.json", report)
        report["status"] = "complete"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], flush=True)
    finally:
        try:
            report["focus_restore"] = access.restore()
        except BaseException as exc:
            report["focus_restore"] = {"verified": False, "error": repr(exc)}
        finally:
            access.close()
        after = bench.protected_hashes()
        report["protected_files_changed"] = [
            k for k in sorted(before.keys() | after.keys()) if before.get(k) != after.get(k)]
        report["forbidden_modules_loaded"] = [
            n for n in ("serial","Motor","Communication","eye_auto_tuner","qt_calibration_panel")
            if n in sys.modules]
        report["finished_at"] = datetime.now().astimezone().isoformat()
        quality.save_json(folder/"data.json", report)
        print("RESTORE", json.dumps(report["focus_restore"]), flush=True)
        print("REPORT", folder/"data.json", flush=True)
    return 0 if report["status"] == "complete" and report["focus_restore"]["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
