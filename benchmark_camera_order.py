"""Camera 1 only: compare DirectShow mode-setting orders without visual models.

Only FOURCC, FPS, FRAME_WIDTH and FRAME_HEIGHT are written. No image-control
setters, GUI, motor imports, or production/configuration writes. Each trial
reopens the camera, warms 30 frames, then times 60 successful reads.
"""

import argparse
from datetime import datetime
import json
import statistics
import sys
import time
import traceback

import benchmark_front_camera as bench


ORDERS = {
    "A": ("FOURCC", "FPS", "FRAME_WIDTH", "FRAME_HEIGHT"),
    "B": ("FRAME_WIDTH", "FRAME_HEIGHT", "FPS", "FOURCC"),
    "C": ("FRAME_WIDTH", "FRAME_HEIGHT", "FOURCC", "FPS"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-camera", action="store_true")
    args = parser.parse_args()
    bench.preflight()
    modes_path = bench.ROOT / "benchmark_results" / "camera1_supported_modes.json"
    modes = json.loads(modes_path.read_text(encoding="utf-8"))
    supported = [m for m in modes["modes"]
                 if m["width"] == 1920 and m["height"] == 1080
                 and m["fourcc"] == "MJPG"
                 and 29.9 <= m["max_fps"] <= 30.1]
    if modes["selected_device"]["index"] != 1 or not supported:
        raise RuntimeError("Camera 1 1080p MJPG 30 FPS capability not confirmed.")
    if not args.run_camera:
        print("Preflight OK. Camera NOT opened. Orders:", ORDERS)
        print("Target: camera 1, DirectShow, 1920x1080, MJPG, 30 FPS.")
        return 0

    import cv2
    before = bench.protected_hashes()
    report = {
        "started_at": datetime.now().astimezone().isoformat(),
        "camera_index": 1, "backend": "DSHOW", "opencv": cv2.__version__,
        "device_from_readonly_query": modes["selected_device"],
        "requested_fps": 30, "frames_per_trial": 60, "trials_per_order": 3,
        "warmup_frames_per_trial": 30,
        "notes": [
            "Every trial opens ONLY camera 1 afresh, then releases it.",
            "No exposure/gain/focus/white-balance or other image-control setters.",
            "No inference, serial, motor, production/configuration writes.",
            "Reported FPS is compared with measured FPS; setter True is not sufficient.",
            "Open/set/warmup time is excluded from the 60-frame timing.",
        ],
        "trials": [], "status": "running",
    }
    destination = bench.ROOT / "benchmark_results" / (
        "camera1_order_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json")
    values = {"FOURCC": cv2.VideoWriter_fourcc(*"MJPG"), "FPS": 30,
              "FRAME_WIDTH": 1920, "FRAME_HEIGHT": 1080}
    cap = None

    def save():
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                          allow_nan=False), encoding="utf-8")

    try:
        # Repeat A/B/C cycles so each order is measured across the same session.
        for repetition in range(1, 4):
            for label, order in ORDERS.items():
                trial = {"order": label, "repetition": repetition, "steps": []}
                report["trials"].append(trial)
                print(f"OPEN {label}{repetition}: ONLY camera 1", flush=True)
                opening = time.perf_counter()
                cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
                trial["open_seconds"] = time.perf_counter() - opening
                if not cap.isOpened():
                    raise RuntimeError("Camera 1 failed to open; no other index attempted.")
                initial = trial["initial_settings"] = bench.camera_settings(cap, cv2)
                for name in order:
                    started = time.perf_counter()
                    accepted = bool(cap.set(getattr(cv2, "CAP_PROP_" + name), values[name]))
                    setter_seconds = time.perf_counter() - started
                    settings = bench.camera_settings(cap, cv2)
                    trial["steps"].append({
                        "property": name, "requested": values[name],
                        "set_return": accepted, "setter_seconds": setter_seconds,
                        "settings_after": settings,
                    })
                    print(f"  {name}: return={accepted}, {settings['fourcc']}, "
                          f"{settings['width']}x{settings['height']}, "
                          f"reported_fps={settings['fps_property']}", flush=True)
                    for control in ("exposure", "auto_exposure", "gain", "autofocus", "auto_wb"):
                        if settings[control] != initial[control]:
                            raise RuntimeError(
                                f"Driver changed {control} during mode setup; stopped "
                                "without attempting image-control writes.")
                trial["configured_settings"] = bench.camera_settings(cap, cv2)
                save()
                warm, _ = bench.timed_round(cap, cv2, frames=30)
                trial["warmup_seconds"] = warm["elapsed_seconds"]
                trial["before_measurement"] = bench.camera_settings(cap, cv2)
                record, _ = bench.timed_round(cap, cv2, number=repetition, frames=60)
                trial["measurement"] = record
                trial["after_measurement"] = bench.camera_settings(cap, cv2)
                trial["actual_frame_shapes"] = sorted({
                    tuple(row["shape"]) for row in record["samples"]})
                cap.release()
                cap = None
                save()
                print(f"RESULT {label}{repetition}: {record['elapsed_seconds']:.6f}s / 60, "
                      f"{record['fps']:.3f} FPS, "
                      f"fourcc={trial['after_measurement']['fourcc']}, "
                      f"failures={record['read_failures']}", flush=True)
        report["summary"] = {}
        for label in ORDERS:
            trials = [t for t in report["trials"] if t["order"] == label]
            seconds = [t["measurement"]["elapsed_seconds"] for t in trials]
            report["summary"][label] = {
                "seconds_per_60": seconds, "mean_seconds": statistics.mean(seconds),
                "fastest_seconds": min(seconds), "aggregate_fps": 180 / sum(seconds),
                "final_fourcc": [t["after_measurement"]["fourcc"] for t in trials],
            }
        report["status"] = "complete"
        print("SUMMARY " + json.dumps(report["summary"]), flush=True)
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], flush=True)
    finally:
        if cap is not None:
            cap.release()
        after = bench.protected_hashes()
        report["protected_files_changed"] = [
            k for k in sorted(before.keys() | after.keys()) if before.get(k) != after.get(k)]
        report["forbidden_modules_loaded"] = [
            name for name in ("serial", "Motor", "Communication", "eye_auto_tuner",
                              "qt_calibration_panel", "ellseg_scorer", "torch", "mediapipe")
            if name in sys.modules]
        report["finished_at"] = datetime.now().astimezone().isoformat()
        save()
        print("REPORT", destination, flush=True)
        print("Protected files changed:", report["protected_files_changed"], flush=True)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
