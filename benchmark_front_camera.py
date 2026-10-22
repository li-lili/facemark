"""Isolated camera-1 benchmark; no motors, baseline writes, or GUI.

Run with the project's Python:
    python -B benchmark_front_camera.py --run-camera

Without --run-camera this only checks local inputs. Camera mode requests match
EllSegDetector, but exposure/focus/gain/white-balance setters are never called.
Only benchmark_results/ is written. Existing detection methods are reused;
the detector's camera-opening/baseline-migration constructor is bypassed.
"""

import argparse
import hashlib
import importlib.abc
import importlib.metadata
import json
import math
from pathlib import Path
import statistics
import sys
import time
from datetime import datetime

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
FRAMES = 60
ROUNDS = 3
CAMERA = 1
ROUND_TIMEOUT = 90.0  # Checked between reads; a native driver call may block.


class NoMotorImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {
            "serial", "Motor", "Communication", "eye_auto_tuner",
            "qt_calibration_panel", "run_interface",
        }:
            raise ImportError(f"Hardware-control import forbidden: {fullname}")
        return None


sys.meta_path.insert(0, NoMotorImports())


def preflight():
    data = json.loads((ROOT / "face_point.json").read_text(encoding="utf-8"))
    config = data.get("common", {}).get("camera_config") or data.get("camera_config", {})
    if int(config.get("front_camera_index", -1)) != CAMERA:
        raise RuntimeError("Configured front camera is not 1; refusing to open a camera.")
    for name in ("EllSeg/weights/all.git_ok", "models/face_landmarker.task",
                 "EllSeg/modelSummary.py", "ellseg_scorer.py"):
        if not (ROOT / name).is_file():
            raise FileNotFoundError(f"Missing local input (no downloads allowed): {name}")
    if data.get("common", {}).get("eye_lr_basis") != "marked":
        raise RuntimeError("Baseline would require L/R migration; refusing to migrate.")
    return data


def protected_hashes():
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in ROOT.iterdir()
        if p.is_file() and p.suffix in {".py", ".yaml", ".yml", ".json", ".md", ".html", ".svg"}
    }


def summary(values):
    if not values:
        return None
    ordered = sorted(values)
    return {
        "count": len(values),
        "sum_ms": sum(values),
        "mean_ms": statistics.mean(values),
        "p50_ms": statistics.median(values),
        "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def camera_settings(cap, cv2):
    names = {
        "width": "FRAME_WIDTH", "height": "FRAME_HEIGHT", "fps_property": "FPS",
        "fourcc_code": "FOURCC", "buffer_size": "BUFFERSIZE",
        "exposure": "EXPOSURE", "auto_exposure": "AUTO_EXPOSURE", "gain": "GAIN",
        "focus": "FOCUS", "autofocus": "AUTOFOCUS", "auto_wb": "AUTO_WB",
        "white_balance": "WB_TEMPERATURE", "brightness": "BRIGHTNESS",
    }
    result = {"backend": cap.getBackendName()}
    for name, prop in names.items():
        value = float(cap.get(getattr(cv2, "CAP_PROP_" + prop)))
        result[name] = value if math.isfinite(value) else None
    code = int(result["fourcc_code"] or 0)
    result["fourcc"] = "".join(chr((code >> (8 * i)) & 255) for i in range(4))
    return result


def timed_round(cap, cv2, detector=None, number=0, frames=FRAMES):
    mode = "raw" if detector is None else "full"
    samples = []
    failures = 0
    attempts = 0
    last_frame = None
    start = time.perf_counter()
    while len(samples) < frames:
        if time.perf_counter() - start > ROUND_TIMEOUT:
            raise TimeoutError(f"{mode} round exceeded {ROUND_TIMEOUT}s")
        t0 = time.perf_counter()
        ok, frame = cap.read()
        t1 = time.perf_counter()
        attempts += 1
        if not ok or frame is None or frame.size == 0:
            failures += 1
            time.sleep(0.001)
            continue
        row = {"read_ms": (t1 - t0) * 1000,
               "shape": list(frame.shape), "read_completed_s": t1 - start}
        if detector is not None:
            rotated = cv2.rotate(frame, cv2.ROTATE_180)
            t2 = time.perf_counter()
            detector.landmarker.last_ms = 0.0
            result = detector.detect(rotated)
            t3 = time.perf_counter()
            timing = result.get("_timing")
            row.update({
                "rotation_ms": (t2 - t1) * 1000,
                "detect_wall_ms": (t3 - t2) * 1000,
                "mp_call_ms": detector.landmarker.last_ms,
                "mp_bundle_ms": timing[1] if timing else None,
                "ellseg_both_ms": timing[2] if timing else None,
                "post_timing_ms": (t3 - t2) * 1000 - timing[0] if timing else None,
                "face_valid": result.get("nose") is not None,
                "left_iris_valid": result.get("left_iris") is not None,
                "right_iris_valid": result.get("right_iris") is not None,
            })
            last_frame = rotated
        samples.append(row)
    elapsed = time.perf_counter() - start
    fields = ("read_ms", "rotation_ms", "detect_wall_ms", "mp_call_ms",
              "mp_bundle_ms", "ellseg_both_ms", "post_timing_ms")
    record = {
        "mode": mode, "round": number, "frames": len(samples),
        "attempts": attempts, "read_failures": failures,
        "elapsed_seconds": elapsed, "fps": len(samples) / elapsed,
        "timing": {k: summary([s[k] for s in samples if s.get(k) is not None])
                   for k in fields},
        "samples": samples,
    }
    if detector is not None:
        record["valid_faces"] = sum(s["face_valid"] for s in samples)
        record["valid_both_eyes"] = sum(
            s["left_iris_valid"] and s["right_iris_valid"] for s in samples)
    return record, last_frame


def load_visual_pipeline(data):
    # No EllSegDetector.__init__, load_baseline(), or migration helpers are called.
    import ellseg_scorer as visual

    class TimedLandmarker:
        def __init__(self):
            self.inner = visual._init_mediapipe()
            self.last_ms = 0.0

        def detect_for_video(self, *args):
            start = time.perf_counter()
            result = self.inner.detect_for_video(*args)
            self.last_ms = (time.perf_counter() - start) * 1000
            return result

        def close(self):
            self.inner.close()

    class BenchmarkDetector(visual.EllSegDetector):
        def __init__(self):
            self.camera_index = CAMERA
            self.flip_horizontal = False
            self.flip_vertical = False
            self.rotate_180 = True
            self.enable_mp = True
            self.enable_ellseg = True
            self.landmarker = TimedLandmarker()
            self.lm_smoother = visual.LMASmoother()
            self.ell_smooth_L = visual.EllipseSmoother()
            self.ell_smooth_R = visual.EllipseSmoother()
            self.landmark_snapshots = visual.LandmarkSnapshots()
            self.baseline = data["common"]
            self.head_adapter_baseline = (
                data["templates"][data["active_template"]].get("head_adapter") or None)
            self._timestamp = 0
            self.last_result = None
            self._last_lms_smooth = None

        def close(self):
            self.landmarker.close()

        def __del__(self):
            # Cleanup is explicit; do not invoke the production GUI destructor.
            pass

    return visual, BenchmarkDetector()


def verify_cuda(visual, detector, frame):
    torch = visual.torch
    result = {
        "torch": torch.__version__, "built_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "selected_device": str(visual._DEVICE),
        "parameter_devices": sorted({str(p.device) for p in visual.ellseg_model.parameters()}),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "forward_inputs": [], "forward_outputs": [],
    }

    def tensors(value):
        if torch.is_tensor(value):
            return [{"device": str(value.device), "shape": list(value.shape)}]
        if isinstance(value, (tuple, list)):
            return [item for child in value for item in tensors(child)]
        return []

    def before(module, args):
        result["forward_inputs"].append(tensors(args))

    def after(module, args, output):
        result["forward_outputs"].append(tensors(output))

    hooks = [visual.ellseg_model.enc.register_forward_pre_hook(before),
             visual.ellseg_model.enc.register_forward_hook(after)]
    activities = [torch.profiler.ProfilerActivity.CPU]
    if visual._DEVICE.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    try:
        if visual._DEVICE.type == "cuda":
            torch.cuda.synchronize()
        # A real final camera image, outside all timed rounds.
        with torch.profiler.profile(activities=activities) as profiler:
            detection = detector.detect(frame)
            if visual._DEVICE.type == "cuda":
                torch.cuda.synchronize()
        events = [e for e in profiler.events() if "CUDA" in str(e.device_type)]
        result["cuda_event_count"] = len(events)
        result["cuda_event_names"] = sorted({e.name for e in events})[:20]
        result["diagnostic_both_eyes_valid"] = all(
            detection.get(k) is not None for k in ("left_iris", "right_iris"))
        result["actual_cuda_forward_verified"] = (
            len(result["forward_inputs"]) == 2
            and all(t["device"].startswith("cuda")
                    for call in result["forward_inputs"] for t in call)
            and len(events) > 0)
    finally:
        for hook in hooks:
            hook.remove()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-camera", action="store_true",
                        help="Explicitly authorize camera 1 acquisition and application mode requests.")
    args = parser.parse_args()
    data = preflight()
    if not args.run_camera:
        print("Preflight OK: configured front camera=1; local models present. Camera NOT opened.")
        return 0

    before_hashes = protected_hashes()
    output_dir = ROOT / "benchmark_results"
    output_dir.mkdir(exist_ok=True)
    path = output_dir / ("front_camera_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json")
    report = {
        "started_at": datetime.now().astimezone().isoformat(), "camera_index": CAMERA,
        "python": sys.executable, "frames_per_round": FRAMES, "rounds_per_group": ROUNDS,
        "notes": [
            "No motors, serial, GUI, side camera, or baseline/config writes.",
            "Only FOURCC/FPS/width/height/buffer setters; no exposure/gain/focus/WB setters.",
            "All 60 successfully read frames count, including failed visual detections.",
            "Model loading, warmup, and CUDA profiler excluded from measured rounds.",
            "MP+EllSeg reuses existing detect(), including in-memory landmark publication.",
            "Camera released after this standalone measurement; no calibration is run.",
        ],
        "rounds": [], "status": "running",
    }

    def save():
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
                        encoding="utf-8")

    cap = None
    detector = None
    try:
        import cv2
        report["packages"] = {
            name: importlib.metadata.version(name)
            for name in ("opencv-python", "mediapipe", "torch", "numpy")}
        print("Opening ONLY camera 1 with DirectShow.", flush=True)
        cap = cv2.VideoCapture(CAMERA, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise RuntimeError("Camera 1 cannot be opened; no fallback to another index.")
        report["opened_settings"] = camera_settings(cap, cv2)
        requested = [
            ("FOURCC", cv2.VideoWriter_fourcc(*"MJPG")), ("FPS", 120),
            ("FRAME_WIDTH", 1920), ("FRAME_HEIGHT", 1080), ("BUFFERSIZE", 1),
        ]
        report["mode_requests"] = []
        for name, value in requested:
            prop = getattr(cv2, "CAP_PROP_" + name)
            accepted = bool(cap.set(prop, value))
            report["mode_requests"].append({
                "property": name, "requested": value, "set_return": accepted,
                "immediate_readback": float(cap.get(prop)),
            })
        report["configured_settings"] = camera_settings(cap, cv2)
        old, new = report["opened_settings"], report["configured_settings"]
        report["control_readback_changes_after_mode_request"] = {
            k: [old[k], new[k]] for k in (
                "exposure", "auto_exposure", "gain", "focus",
                "autofocus", "auto_wb", "white_balance") if old[k] != new[k]}
        print("MODE " + json.dumps(new, ensure_ascii=False), flush=True)
        save()
        if any(old[k] != new[k] for k in ("auto_exposure", "autofocus", "auto_wb")):
            raise RuntimeError("Driver changed an automatic-control mode; stopped without further setters.")
        if new["auto_exposure"] == 0 and old["exposure"] != new["exposure"]:
            raise RuntimeError("Driver changed manual exposure; stopped without further setters.")

        warm, _ = timed_round(cap, cv2, frames=30)
        report["raw_warmup_seconds"] = warm["elapsed_seconds"]
        for number in range(1, ROUNDS + 1):
            record, _ = timed_round(cap, cv2, number=number)
            report["rounds"].append(record)
            report[f"after_raw_{number}"] = camera_settings(cap, cv2)
            save()
            print(f"RAW {number}: {record['elapsed_seconds']:.3f}s / 60, "
                  f"{record['fps']:.2f} FPS, failures={record['read_failures']}", flush=True)

        print("Loading local visual models; this time is excluded.", flush=True)
        start = time.perf_counter()
        visual, detector = load_visual_pipeline(data)
        report["model_load_seconds"] = time.perf_counter() - start
        # Discard queued startup images, then warm both actual visual models.
        timed_round(cap, cv2, frames=30)
        warm, _ = timed_round(cap, cv2, detector, frames=10)
        report["visual_warmup_seconds"] = warm["elapsed_seconds"]
        for number in range(1, ROUNDS + 1):
            record, last_frame = timed_round(cap, cv2, detector, number=number)
            report["rounds"].append(record)
            report[f"after_full_{number}"] = camera_settings(cap, cv2)
            save()
            print(f"FULL {number}: {record['elapsed_seconds']:.3f}s / 60, "
                  f"{record['fps']:.2f} FPS, both eyes={record['valid_both_eyes']}/60, "
                  f"read={record['timing']['read_ms']['mean_ms']:.2f}ms, "
                  f"MP={record['timing']['mp_call_ms']['mean_ms']:.2f}ms, "
                  f"Ell={record['timing']['ellseg_both_ms']}", flush=True)

        print("Checking actual EllSeg CUDA execution outside timed rounds.", flush=True)
        report["cuda"] = verify_cuda(visual, detector, last_frame)
        report["final_settings"] = camera_settings(cap, cv2)
        report["status"] = "complete"
        print("CUDA " + json.dumps(report["cuda"], ensure_ascii=False), flush=True)
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        import traceback
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], flush=True)
    finally:
        if detector is not None:
            detector.close()
        if cap is not None:
            cap.release()
        after_hashes = protected_hashes()
        report["protected_files_changed"] = [
            name for name in sorted(before_hashes.keys() | after_hashes.keys())
            if before_hashes.get(name) != after_hashes.get(name)]
        report["forbidden_modules_loaded"] = [
            name for name in ("serial", "Motor", "Communication", "eye_auto_tuner",
                              "qt_calibration_panel") if name in sys.modules]
        report["finished_at"] = datetime.now().astimezone().isoformat()
        save()
        print(f"REPORT {path}", flush=True)
        print(f"Protected files changed: {report['protected_files_changed']}", flush=True)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
