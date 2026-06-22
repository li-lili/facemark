"""Color based side-view lip front-point detector.

The caller provides a fixed mouth ROI. The detector returns the most forward
upper/lower lip color points inside that ROI.
"""

from __future__ import annotations

import cv2
import numpy as np


def largest_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == label] = 255
    return keep


def front_point_from_mask(mask: np.ndarray, face_direction: str) -> tuple[int, int] | None:
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


def find_lip_front_points(
    frame: np.ndarray,
    rect: tuple[int, int, int, int],
    score_pct: int = 76,
    a_delta: int = 3,
    min_sat: int = 18,
    split_pct: int = 58,
    min_area: int = 80,
    face_direction: str = "right",
) -> dict:
    x1, y1, x2, y2 = rect
    h_frame, w_frame = frame.shape[:2]
    x1 = max(0, min(w_frame - 1, int(x1)))
    x2 = max(0, min(w_frame, int(x2)))
    y1 = max(0, min(h_frame - 1, int(y1)))
    y2 = max(0, min(h_frame, int(y2)))
    if x2 <= x1 + 8 or y2 <= y1 + 8:
        return {"upper": None, "lower": None, "mask": None}

    roi = frame[y1:y2, x1:x2]
    blur = cv2.GaussianBlur(roi, (5, 5), 0)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    a = lab[:, :, 1].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)

    a_med = float(np.median(a))
    s_med = float(np.median(s))
    v_med = float(np.median(v))

    score = (a - a_med) * 1.8 + (s - s_med) * 0.35 - (v - v_med) * 0.12
    threshold = np.percentile(score, max(50, min(99, score_pct)))
    mask = ((score >= threshold) & (a >= a_med + a_delta) & (s >= min_sat)).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = largest_components(mask, max(1, int(min_area)))

    h, _w = mask.shape[:2]
    split_y = int(h * max(15, min(95, split_pct)) / 100)
    upper_mask = np.zeros_like(mask)
    lower_mask = np.zeros_like(mask)
    upper_mask[:split_y, :] = mask[:split_y, :]
    lower_mask[split_y:, :] = mask[split_y:, :]

    upper_local = front_point_from_mask(upper_mask, face_direction)
    lower_local = front_point_from_mask(lower_mask, face_direction)

    upper = (x1 + upper_local[0], y1 + upper_local[1]) if upper_local else None
    lower = (x1 + lower_local[0], y1 + lower_local[1]) if lower_local else None
    return {"upper": upper, "lower": lower, "mask": mask}
