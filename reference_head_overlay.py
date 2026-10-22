"""Display-only reference coordinates. No detector, target or motor dependencies."""
import json
import math
from pathlib import Path

COORDINATE_SYSTEM = "full detector input frame; x right, y down; L screen-right"
CAMERA_KEYS = ("front_camera_index", "width", "height", "fourcc",
               "rotate_180", "flip_horizontal", "flip_vertical")


def compatibility_error(reference, camera):
    for key in CAMERA_KEYS:
        if key not in reference["camera"] or key not in camera:
            return f"missing {key}"
        if reference["camera"][key] != camera[key]:
            return f"{key} mismatch"
    if reference.get("coordinate_system") != COORDINATE_SYSTEM:
        return "coordinate_system mismatch"
    return None


def load_reference(path, camera):
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("Unsupported reference result schema.")
    if not isinstance(data.get("camera"), dict):
        raise ValueError("Reference lacks camera metadata.")
    reference = {"camera": data["camera"], "coordinate_system": data.get("coordinate_system"),
                 "head_id": str(data.get("head_id", "unknown")), "centers": {},
                 "source_path": str(Path(path).resolve())}
    error = compatibility_error(reference, camera)
    if error:
        raise ValueError(f"Reference overlay blocked: {error}. Re-save with complete matching metadata.")
    for side in ("L", "R"):
        # Only the saved yellow point, never effective_target or baseline.
        point = data.get("eyes", {}).get(side, {}).get("integer_center")
        if (not isinstance(point, list) or len(point) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) or int(v) != v for v in point)):
            raise ValueError(f"Invalid saved {side} integer center.")
        if not (0 <= point[0] < camera["width"] and 0 <= point[1] < camera["height"]):
            raise ValueError(f"Saved {side} center outside frame.")
        reference["centers"][side] = tuple(int(v) for v in point)
    return reference


def draw_reference(image, detection, state, camera):
    """Draw onto a DISPLAY COPY only; returns three persistent right-HUD lines."""
    import cv2
    reference, enabled = state
    lines = ["RefHead: none OFF", "Ref L: --", "Ref R: --"]
    if reference is None:
        return lines
    lines[0] = f"RefHead {reference['head_id'][:20]}: {'ON' if enabled else 'OFF'}"
    if not enabled:
        return lines
    error = compatibility_error(reference, camera)
    if error or (image.shape[1], image.shape[0]) != (camera["width"], camera["height"]):
        return ["RefHead: BLOCKED", f"Ref: {error or 'frame size mismatch'}", "Ref L/R: --"]
    for row, (side, key) in enumerate((("L", "left_iris"), ("R", "right_iris")), 1):
        point = reference["centers"][side]
        cv2.circle(image, point, 10, (0, 255, 0), 1, cv2.LINE_AA)
        current = detection.get(key)
        if current is not None:
            dx, dy = current[0] - point[0], current[1] - point[1]
            lines[row] = f"Ref {side}: dX={dx:+.1f} dY={dy:+.1f} d={math.hypot(dx, dy):.2f}px"
        # Repaint exactly the existing current point if the green ring crossed it.
        # Preserve FaceDraw OFF: caller tells us whether current dots are visible.
        if current is not None and camera.get("current_dots_visible", True):
            cv2.circle(image, tuple(current), 4, (0, 255, 255), -1)
    return lines
