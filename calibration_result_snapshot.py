"""Read-only export of an existing detection. No camera capture or controller imports."""

import copy
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from reference_head_overlay import COORDINATE_SYSTEM


def _json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Unsupported snapshot value: {type(value).__name__}")


def _write_new(path, content):
    with path.open("xb") as stream:
        stream.write(content)


def save_calibration_result(*, root, detector, template_name, camera_info,
                            servo_cache, head_id, qt_log, detection_metadata,
                            prior_yaml=None):
    """Called in the idle detector thread; never run a detector or move a servo.

    The result is one new observation, NOT the original closing 60-frame batch.
    Complete exports have result.json; incomplete directories remain for diagnosis.
    """
    import cv2

    root = Path(root).resolve()
    if (not head_id or len(head_id) > 80
            or any(not (c.isascii() and (c.isalnum() or c in "_-")) for c in head_id)):
        raise ValueError("Head ID must contain 1-80 ASCII letters, digits, '_' or '-'.")
    det = detector.last_result
    if det is None or det.get("frame") is None:
        raise ValueError("No current detection frame; no snapshot saved.")
    if not detector.enable_mp or not detector.enable_ellseg:
        raise ValueError("Enable existing MP/EllSeg visual detection before saving.")
    if detection_metadata.get("status") != "detected":
        raise ValueError("No fresh detected front-frame metadata.")
    # Detach from display and detector state before encoding or filesystem work.
    det = copy.deepcopy(det)
    frame = det.pop("frame")
    if (frame.shape[1], frame.shape[0]) != (camera_info.get("width"), camera_info.get("height")):
        raise ValueError("Camera metadata and detection frame dimensions disagree.")
    baseline = copy.deepcopy(detector.baseline)
    adapter = copy.deepcopy(detector.head_adapter_baseline)
    eyes = {}
    target_offsets = det.get("target_offset", det.get("offset", {}))
    for side, label in (("left", "L"), ("right", "R")):
        center = det.get(f"{side}_iris")
        target = detector._eyeball_target_point(side)
        error = target_offsets.get(side)
        if center is None or target is None or error is None:
            raise ValueError(f"Missing {label} center, baseline or target error.")
        values = list(center) + list(target) + list(error)
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError(f"Nonfinite {label} detection values.")
        # Reject stale results computed using a different in-memory baseline.
        expected = (center[0] - target[0], center[1] - target[1])
        if any(abs(float(a) - float(b)) > 1e-8 for a, b in zip(expected, error[:2])):
            raise ValueError("Detection and loaded target disagree; wait for the next frame.")
        ellipse = det.get(f"{side}_iris_ellipse")
        eyes[label] = {
            "integer_center": list(center),
            "smoothed_float_center": list(ellipse[0]) if ellipse is not None else None,
            "raw_fit_float_center": None,
            "raw_fit_unavailable_reason": "Existing detector does not retain pre-EMA fit.",
            "effective_target": list(target),
            "target_error": {"dx": error[0], "dy": error[1], "distance": error[2]},
            "common_baseline_error": copy.deepcopy(det.get("offset", {}).get(side)),
        }

    disk_bytes = (root / "face_point.json").read_bytes()
    disk_data = json.loads(disk_bytes.decode("utf-8-sig"))
    prior_path = Path(prior_yaml).resolve() if prior_yaml else None
    prior_bytes = prior_path.read_bytes() if prior_path else None
    now = datetime.now().astimezone()
    directory = root / "calibration_results" / f"{head_id}_{now:%Y%m%d_%H%M%S_%f}"
    directory.mkdir(parents=True, exist_ok=False)
    payload = {
        "schema_version": 1, "head_id": head_id, "saved_at": now.isoformat(),
        "observation_kind": "current_single_frame_after_adjustment_not_original_final_batch",
        "detection_metadata": copy.deepcopy(detection_metadata),
        "coordinate_system": COORDINATE_SYSTEM,
        "template_name_loaded": template_name,
        "baseline_loaded": baseline, "head_adapter_loaded": adapter,
        "face_point_disk": {"file": "face_point.json",
                            "sha256": hashlib.sha256(disk_bytes).hexdigest(),
                            "active_template": disk_data.get("active_template"),
                            "note": "Disk copy; loaded baseline/adapter recorded separately."},
        "camera": copy.deepcopy(camera_info), "eyes": eyes,
        "single_frame_qualified": det.get("qualified"),
        "last_calibration_60": None,
        "last_calibration_60_unavailable_reason": "Existing code does not retain the historical batch.",
        "calibration_iterations": None,
        "servo_cache": {"values_by_channel": copy.deepcopy(servo_cache),
                        "source": "current_process_software_cache" if servo_cache is not None else "unavailable",
                        "note": "Configuration-initialized / software command values; not encoder feedback."},
        "prior_servo_export": None,
        "files": {"raw_frame": "frame.png", "detection_image": "detection.png",
                  "qt_log": "qt_log.txt"},
        "image_note": "Same-frame export overlay: yellow centers, cyan effective targets; not desktop HUD capture.",
    }
    if prior_bytes is not None:
        payload["prior_servo_export"] = {
            "file": "pre_restart_servo_commands.yaml", "source_path": str(prior_path),
            "sha256": hashlib.sha256(prior_bytes).hexdigest(),
            "source": "user_selected_pre_restart_export_from_software_command_cache",
            "note": "Historical export; not current cache or encoder feedback; does not restore angles.",
        }
        _write_new(directory / "pre_restart_servo_commands.yaml", prior_bytes)
    overlay = frame.copy()
    for eye in eyes.values():
        cv2.circle(overlay, tuple(int(round(v)) for v in eye["effective_target"]), 8, (255, 255, 0), 1)
    for eye in eyes.values():
        cv2.circle(overlay, tuple(int(v) for v in eye["integer_center"]), 4, (0, 255, 255), -1)
    for name, img in (("frame.png", frame), ("detection.png", overlay)):
        ok, encoded = cv2.imencode(".png", img)
        if not ok:
            raise RuntimeError(f"PNG encoding failed: {name}; incomplete directory: {directory}")
        _write_new(directory / name, encoded.tobytes())
    _write_new(directory / "face_point.json", disk_bytes)
    _write_new(directory / "qt_log.txt", qt_log.encode("utf-8"))
    # The completion record is written last; never overwrite any existing file.
    _write_new(directory / "result.json", json.dumps(
        payload, ensure_ascii=False, indent=2, allow_nan=False,
        default=_json_default).encode("utf-8"))
    return directory
