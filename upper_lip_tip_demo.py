"""
Upper/lower lip front-point demo.

Usage:
    python upper_lip_tip_demo.py
    python upper_lip_tip_demo.py 1

Controls:
    Drag left mouse button  Set mouth ROI
    R                       Clear ROI
    H                       Horizontal flip
    V                       Vertical flip
    S                       Save current frame
    D                       Toggle face direction
    Q / Esc                 Quit

Trackbars:
    Score Pct   Higher means fewer lip-color pixels
    A Delta     Red-channel margin above local skin median in Lab space
    Min Sat     Minimum HSV saturation
    Split %     Upper lip searches above this ROI height percentage; lower lip searches below it
    Min Area    Ignore small noisy connected components
"""

from __future__ import annotations

import datetime as _dt
import sys
from dataclasses import dataclass

import cv2
import numpy as np


CAMERA_INDEX = 1
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
FACE_DIRECTION = "right"  # "right": front point is max x, "left": min x


@dataclass
class RoiState:
    start: tuple[int, int] | None = None
    current: tuple[int, int] | None = None
    rect: tuple[int, int, int, int] | None = None
    dragging: bool = False


def clamp_rect(rect: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = rect
    x1, x2 = sorted((max(0, min(width - 1, x1)), max(0, min(width - 1, x2))))
    y1, y2 = sorted((max(0, min(height - 1, y1)), max(0, min(height - 1, y2))))
    if x2 - x1 < 12 or y2 - y1 < 12:
        return None
    return x1, y1, x2, y2


def make_mouse_callback(state: RoiState):
    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state.start = (x, y)
            state.current = (x, y)
            state.dragging = True
        elif event == cv2.EVENT_MOUSEMOVE and state.dragging:
            state.current = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and state.dragging:
            state.current = (x, y)
            state.dragging = False
            if state.start is not None:
                state.rect = (*state.start, x, y)

    return on_mouse


def largest_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            keep[labels == label] = 255
    return keep


def front_point_from_mask(
    mask: np.ndarray,
    face_direction: str,
) -> tuple[int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    if face_direction == "left":
        front_x = int(xs.min())
        candidate_ys = ys[xs == front_x]
    else:
        front_x = int(xs.max())
        candidate_ys = ys[xs == front_x]
    return front_x, int(np.median(candidate_ys))


def find_lip_tips(
    frame: np.ndarray,
    rect: tuple[int, int, int, int],
    score_pct: int,
    a_delta: int,
    min_sat: int,
    split_pct: int,
    min_area: int,
    face_direction: str,
) -> tuple[tuple[int, int] | None, tuple[int, int] | None, np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = rect
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        empty = np.zeros((1, 1), dtype=np.uint8)
        return None, None, empty, cv2.cvtColor(empty, cv2.COLOR_GRAY2BGR)

    blur = cv2.GaussianBlur(roi, (5, 5), 0)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    a = lab[:, :, 1].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)

    a_med = float(np.median(a))
    s_med = float(np.median(s))
    v_med = float(np.median(v))

    # Lips are usually redder, a little more saturated, and often slightly darker than nearby skin.
    score = (a - a_med) * 1.8 + (s - s_med) * 0.35 - (v - v_med) * 0.12
    threshold = np.percentile(score, max(50, min(99, score_pct)))
    mask = ((score >= threshold) & (a >= a_med + a_delta) & (s >= min_sat)).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = largest_components(mask, min_area)

    h, w = mask.shape[:2]
    split_y = int(h * max(15, min(95, split_pct)) / 100)
    upper_mask = np.zeros_like(mask)
    lower_mask = np.zeros_like(mask)
    upper_mask[:split_y, :] = mask[:split_y, :]
    lower_mask[split_y:, :] = mask[split_y:, :]

    upper_local = front_point_from_mask(upper_mask, face_direction)
    lower_local = front_point_from_mask(lower_mask, face_direction)

    upper_tip = (x1 + upper_local[0], y1 + upper_local[1]) if upper_local else None
    lower_tip = (x1 + lower_local[0], y1 + lower_local[1]) if lower_local else None

    debug = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.line(debug, (0, split_y), (w - 1, split_y), (0, 255, 255), 1)
    if upper_local:
        cv2.circle(debug, upper_local, 5, (0, 0, 255), -1)
    if lower_local:
        cv2.circle(debug, lower_local, 5, (255, 0, 0), -1)
    return upper_tip, lower_tip, mask, debug


def draw_text(img: np.ndarray, text: str, pos: tuple[int, int], color=(0, 255, 255)) -> None:
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def main() -> int:
    camera_index = CAMERA_INDEX
    if len(sys.argv) > 1:
        try:
            camera_index = int(sys.argv[1])
        except ValueError:
            pass

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"ERROR: failed to open camera {camera_index}")
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    win = "Upper Lip Tip Demo"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    cv2.createTrackbar("Score Pct", win, 76, 99, lambda _v: None)
    cv2.createTrackbar("A Delta", win, 3, 40, lambda _v: None)
    cv2.createTrackbar("Min Sat", win, 18, 120, lambda _v: None)
    cv2.createTrackbar("Split %", win, 58, 100, lambda _v: None)
    cv2.createTrackbar("Min Area", win, 80, 2000, lambda _v: None)

    roi_state = RoiState()
    cv2.setMouseCallback(win, make_mouse_callback(roi_state))

    flipped_h = False
    flipped_v = False
    face_direction = FACE_DIRECTION
    last_upper_tip: tuple[int, int] | None = None
    last_lower_tip: tuple[int, int] | None = None
    print(f"Opened camera {camera_index}. Drag a mouth ROI, then tune trackbars if needed.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("ERROR: failed to read frame")
            break

        if flipped_h and flipped_v:
            frame = cv2.flip(frame, -1)
        elif flipped_h:
            frame = cv2.flip(frame, 1)
        elif flipped_v:
            frame = cv2.flip(frame, 0)

        h, w = frame.shape[:2]
        display = frame.copy()
        active_rect = roi_state.rect
        if roi_state.dragging and roi_state.start and roi_state.current:
            active_rect = (*roi_state.start, *roi_state.current)

        rect = clamp_rect(active_rect, w, h) if active_rect else None
        debug_panel = np.zeros((180, 240, 3), dtype=np.uint8)

        if rect:
            score_pct = cv2.getTrackbarPos("Score Pct", win)
            a_delta = cv2.getTrackbarPos("A Delta", win)
            min_sat = cv2.getTrackbarPos("Min Sat", win)
            split_pct = cv2.getTrackbarPos("Split %", win)
            min_area = max(1, cv2.getTrackbarPos("Min Area", win))

            upper_tip, lower_tip, _mask, debug = find_lip_tips(
                frame,
                rect,
                score_pct=score_pct,
                a_delta=a_delta,
                min_sat=min_sat,
                split_pct=split_pct,
                min_area=min_area,
                face_direction=face_direction,
            )
            last_upper_tip = upper_tip
            last_lower_tip = lower_tip

            x1, y1, x2, y2 = rect
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 80, 255), 2)
            if upper_tip:
                cv2.circle(display, upper_tip, 7, (0, 0, 255), -1)
                cv2.circle(display, upper_tip, 12, (255, 255, 255), 2)
            if lower_tip:
                cv2.circle(display, lower_tip, 7, (255, 0, 0), -1)
                cv2.circle(display, lower_tip, 12, (255, 255, 255), 2)

            if upper_tip or lower_tip:
                draw_text(display, f"upper: {upper_tip}  lower: {lower_tip}", (x1, max(24, y1 - 10)), (0, 0, 255))
                print(f"\rupper_lip_tip={upper_tip} lower_lip_tip={lower_tip} roi={rect}      ", end="")
            else:
                draw_text(display, "lip tips: not found", (x1, max(24, y1 - 10)), (0, 0, 255))

            debug_panel = cv2.resize(debug, (240, 180), interpolation=cv2.INTER_NEAREST)

        draw_text(
            display,
            f"Camera {camera_index}  drag ROI  Dir={face_direction}  D dir  R clear  H/V flip  S save  Q quit",
            (10, 24),
        )
        if last_upper_tip or last_lower_tip:
            draw_text(display, f"Upper: {last_upper_tip}  Lower: {last_lower_tip}", (10, 48), (0, 255, 0))
        else:
            draw_text(display, "Lip tips: none", (10, 48), (0, 255, 0))

        panel_h, panel_w = debug_panel.shape[:2]
        px = max(0, w - panel_w - 12)
        py = 64
        display[py : py + panel_h, px : px + panel_w] = debug_panel
        cv2.rectangle(display, (px, py), (px + panel_w, py + panel_h), (255, 255, 255), 1)
        draw_text(display, "lip mask / split line", (px, py - 8), (255, 255, 255))

        cv2.imshow(win, display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            roi_state.rect = None
            last_upper_tip = None
            last_lower_tip = None
            print("\nROI cleared")
        elif key == ord("h"):
            flipped_h = not flipped_h
            print(f"\nHorizontal flip: {'ON' if flipped_h else 'OFF'}")
        elif key == ord("v"):
            flipped_v = not flipped_v
            print(f"\nVertical flip: {'ON' if flipped_v else 'OFF'}")
        elif key == ord("d"):
            face_direction = "left" if face_direction == "right" else "right"
            print(f"\nFace direction: {face_direction}")
        elif key == ord("s"):
            ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"upper_lip_demo_cam{camera_index}_{ts}.png"
            cv2.imwrite(name, display)
            print(f"\nSaved {name}")

    print()
    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
