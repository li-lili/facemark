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


def center_square_rect(frame: np.ndarray, size: int = 250) -> tuple[int, int, int, int]:
    """返回以画面中心为中心的正方形，不依赖手工 ROI。"""
    h, w = frame.shape[:2]
    side = max(16, min(int(size), h, w))
    x1 = max(0, (w - side) // 2)
    y1 = max(0, (h - side) // 2)
    return x1, y1, x1 + side, y1 + side


def _largest_profile_component(mask: np.ndarray, face_direction: str) -> np.ndarray:
    """选择最像从画面后侧进入的脸部连通域。"""
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return np.zeros_like(mask)
    h, w = mask.shape[:2]
    band = max(3, w // 8)
    best_label = None
    best_score = -1.0
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < max(30, h * w // 40):
            continue
        component = labels == label
        if face_direction == "left":
            back_overlap = int(np.count_nonzero(component[:, w - band:]))
        else:
            back_overlap = int(np.count_nonzero(component[:, :band]))
        score = float(area + back_overlap * 3)
        if score > best_score:
            best_label = label
            best_score = score
    if best_label is None:
        return np.zeros_like(mask)
    return (labels == best_label).astype(np.uint8) * 255


def _profile_face_mask(
    roi: np.ndarray,
    face_direction: str,
    background_delta: float,
) -> np.ndarray:
    """以朝向前方的边缘窄条估计背景，提取头部轮廓。"""
    blur = cv2.GaussianBlur(roi, (5, 5), 0)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    strip_width = max(5, w // 10)
    if face_direction == "left":
        background_strip = lab[:, :strip_width]
        face_strip = lab[:, w - strip_width:]
    else:
        background_strip = lab[:, w - strip_width:]
        face_strip = lab[:, :strip_width]
    background = np.median(background_strip.reshape(-1, 3), axis=0)
    face = np.median(face_strip.reshape(-1, 3), axis=0)
    background_distance = np.linalg.norm(
        lab - background.reshape(1, 1, 3), axis=2)
    face_distance = np.linalg.norm(lab - face.reshape(1, 1, 3), axis=2)
    mask = (
        (background_distance >= float(background_delta) * 0.55)
        & (face_distance + 2.0 < background_distance)
    ).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    return _largest_profile_component(mask, face_direction)


def _profile_lip_mask(
    roi: np.ndarray,
    face_mask: np.ndarray,
    min_area: int,
    nose_y: int | None = None,
) -> np.ndarray:
    """从脸部连通域内提取偏红/偏粉的唇部像素。"""
    blur = cv2.GaussianBlur(roi, (5, 5), 0)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    a = lab[:, :, 1].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)
    valid = face_mask > 0
    if np.count_nonzero(valid) < 30:
        return np.zeros_like(face_mask)
    a_med = float(np.median(a[valid]))
    s_med = float(np.median(s[valid]))
    v_med = float(np.median(v[valid]))
    score = (a - a_med) * 1.8 + (s - s_med) * 0.35 - (v - v_med) * 0.12
    threshold = float(np.percentile(score[valid], 78))
    mask = (
        valid
        & (score >= threshold)
        & (a >= a_med + 2.0)
        & (s >= 15.0)
    ).astype(np.uint8) * 255
    h = mask.shape[0]
    vertical_band = np.zeros_like(mask)
    if nose_y is None:
        band_top = int(h * 0.28)
        band_bottom = int(h * 0.80)
    else:
        # 嘴唇只在鼻尖明显下方搜索，避免鼻孔和鼻翼进入候选区域。
        band_top = max(
            int(h * 0.30),
            int(nose_y) + max(8, int(h * 0.10)),
        )
        # 鼻尖先由轮廓取得；嘴唇只能出现在鼻尖明显下方，排除鼻孔/鼻翼偏红区域。
        band_bottom = min(int(h * 0.82), int(nose_y) + int(h * 0.42))
    if band_bottom <= band_top + 5:
        return np.zeros_like(face_mask)
    vertical_band[band_top:band_bottom, :] = 255
    mask = cv2.bitwise_and(mask, vertical_band)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    return largest_components(mask, max(1, int(min_area)))


def _robust_front_point(mask: np.ndarray, face_direction: str) -> tuple[int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    if face_direction == "left":
        edge = float(np.percentile(xs, 3))
        selected = xs <= edge + 1.5
    else:
        edge = float(np.percentile(xs, 97))
        selected = xs >= edge - 1.5
    if not np.any(selected):
        return None
    return int(round(float(np.median(xs[selected])))), int(round(float(np.median(ys[selected]))))


def _split_profile_lips(mask: np.ndarray, face_direction: str):
    ys, _xs = np.where(mask > 0)
    if len(ys) < 8 or int(ys.max()) - int(ys.min()) < 5:
        return None, None
    centers = np.array([np.percentile(ys, 25), np.percentile(ys, 75)], dtype=np.float32)
    for _ in range(8):
        groups = np.argmin(np.abs(ys[:, None] - centers[None, :]), axis=1)
        updated = centers.copy()
        for index in (0, 1):
            if np.any(groups == index):
                updated[index] = float(np.mean(ys[groups == index]))
        if np.allclose(updated, centers):
            break
        centers = updated
    split_y = int(round(float(np.mean(np.sort(centers)))))
    upper_mask = np.zeros_like(mask)
    lower_mask = np.zeros_like(mask)
    upper_mask[:split_y + 1] = mask[:split_y + 1]
    lower_mask[split_y + 1:] = mask[split_y + 1:]
    upper = _robust_front_point(upper_mask, face_direction)
    lower = _robust_front_point(lower_mask, face_direction)
    if upper is not None and lower is not None and lower[1] <= upper[1] + 2:
        return None, None
    return upper, lower


def _profile_boundary(face_mask: np.ndarray, face_direction: str) -> dict[int, float]:
    boundary = {}
    for y in range(face_mask.shape[0]):
        xs = np.flatnonzero(face_mask[y] > 0)
        if len(xs) < 4:
            continue
        boundary[y] = float(xs.min() if face_direction == "left" else xs.max())
    if not boundary:
        return boundary
    smoothed = {}
    for y in boundary:
        nearby = [boundary[row] for row in range(y - 2, y + 3) if row in boundary]
        smoothed[y] = float(np.median(nearby))
    return smoothed


def _contrast_profile_boundary(
    roi: np.ndarray,
    face_direction: str,
    x_range: tuple[int, int],
    y_range: tuple[int, int],
    min_contrast: float,
    all_tracks: bool = False,
    color_edges: bool = False,
) -> dict[int, float] | list[dict[int, float]]:
    """Find the background-to-skin silhouette edge inside a local search box."""
    gray = cv2.cvtColor(cv2.GaussianBlur(roi, (5, 5), 0), cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32)
    color = cv2.GaussianBlur(roi, (5, 5), 0).astype(np.float32) if color_edges else None
    h, w = gray.shape[:2]
    gap = max(2, w // 160)
    x_min = max(gap, min(int(value) for value in x_range))
    x_max = min(w - gap - 1, max(int(value) for value in x_range))
    y_min = max(0, min(int(value) for value in y_range))
    y_max = min(h - 1, max(int(value) for value in y_range))
    if x_max <= x_min or y_max < y_min:
        return [] if all_tracks else {}

    boundary = {}
    tracks = []
    xs = np.arange(x_min, x_max + 1, dtype=np.int32)
    for y in range(y_min, y_max + 1):
        if face_direction == "left":
            contrast = gray[y, xs + gap] - gray[y, xs - gap]
        else:
            contrast = gray[y, xs - gap] - gray[y, xs + gap]
        if color is not None:
            contrast = np.max(np.abs(color[y, xs+gap] - color[y, xs-gap]), axis=1)
        # Keep the first strong local peak from the background side. A stronger
        # nostril/highlight edge farther inside the face is not the silhouette.
        peaks = np.flatnonzero(
            (contrast >= float(min_contrast))
            & (contrast >= np.r_[np.inf, contrast[:-1]])
            & (contrast >= np.r_[contrast[1:], np.inf]))
        if len(peaks):
            if all_tracks:
                # A flat gradient maximum is one edge, not competing tracks.
                groups = np.split(peaks, np.flatnonzero(np.diff(peaks) > 1) + 1)
                peaks = np.array([int(round(float(np.mean(group)))) for group in groups])
                # Keep separate continuous edges: the support's edge must not
                # hide a valid nose farther inside the manually selected ROI.
                available = [i for i, track in enumerate(tracks)
                             if y - max(track) <= 2]
                for peak in peaks:
                    px = float(xs[peak])
                    matches = [(abs(tracks[i][max(tracks[i])] - px), i)
                               for i in available]
                    distance, track_index = min(matches, default=(float("inf"), -1))
                    if distance <= 4:
                        tracks[track_index][y] = px
                        available.remove(track_index)
                    else:
                        tracks.append({y: px})
            index = peaks[0] if face_direction == "left" else peaks[-1]
            boundary[y] = float(xs[index])

    if all_tracks:
        return [track for track in tracks if len(track) >= 15]
    if not boundary:
        return {}
    smoothed = {}
    for y in boundary:
        nearby = [boundary[row] for row in range(y - 2, y + 3) if row in boundary]
        smoothed[y] = float(np.median(nearby))
    return smoothed


def _nose_from_boundary(
    boundary: dict[int, float],
    crop_height: int,
    face_direction: str,
    nose_x_range: tuple[int, int] | None = None,
    nose_y_range: tuple[int, int] | None = None,
) -> tuple[int, int] | None:
    # 鼻尖是轮廓上半部最朝前的位置，不再依赖可能被鼻孔误判的“上唇点”。
    if nose_y_range is None:
        y_min = max(2, int(crop_height * 0.08))
        y_max = min(crop_height - 3, int(crop_height * 0.52))
    else:
        y_min = max(0, min(int(value) for value in nose_y_range))
        y_max = min(crop_height - 1, max(int(value) for value in nose_y_range))
    if nose_x_range is None:
        x_min = float("-inf")
        x_max = float("inf")
    else:
        x_min = float(min(nose_x_range))
        x_max = float(max(nose_x_range))
    candidates = [
        (y, x) for y, x in boundary.items()
        if y_min <= y <= y_max and x_min <= x <= x_max
    ]
    if not candidates:
        return None
    # Split at missing rows or abrupt edge changes, so separate objects cannot
    # be averaged into a fictitious nose. Require a rounded, interior extremum.
    runs = []
    for point in sorted(candidates):
        if (not runs or point[0] - runs[-1][-1][0] > 2
                or abs(point[1] - runs[-1][-1][1]) > 4):
            runs.append([])
        runs[-1].append(point)
    tips = []
    sign = 1.0 if face_direction == "left" else -1.0
    for run in runs:
        if len(run) < 15:
            continue
        rows, values = np.asarray(run, dtype=np.float64).T
        forward = sign * values
        index = int(np.argmin(forward))
        near = np.flatnonzero(forward <= forward[index] + 1.5)
        center = float(np.median(rows[near]))
        local = np.abs(rows - center) <= 10
        if np.count_nonzero(local) < 15:
            continue
        offsets = rows[local] - center
        coeff = np.polyfit(offsets, forward[local], 2)
        if coeff[0] <= 0:
            continue
        vertex = -coeff[1] / (2 * coeff[0])
        if not (offsets.min() + 3 < vertex < offsets.max() - 3):
            continue
        if max(abs(np.polyval(coeff, offsets) - forward[local])) > 2.5:
            continue
        if min(forward[0], forward[-1]) - forward[index] < 2:
            continue
        y = center + vertex
        x = sign * float(np.polyval(coeff, vertex))
        if x_min + 2 < x < x_max - 2 and y_min + 2 < y < y_max - 2:
            tips.append((sign * x, int(round(x)), int(round(y))))
    if not tips:
        return None
    _, x, y = min(tips)
    return x, y


def _chin_from_boundary(
    boundary: dict[int, float],
    lower_lip: tuple[int, int],
    crop_height: int,
) -> tuple[int, int] | None:
    y_min = max(lower_lip[1] + 7, int(crop_height * 0.68))
    rows = sorted(y for y in boundary if y >= y_min)
    if len(rows) < 3:
        return None
    target_y = int(round(float(np.percentile(rows, 90))))
    nearby = [boundary[y] for y in rows if abs(y - target_y) <= 2]
    if not nearby:
        return None
    return int(round(float(np.median(nearby)))), target_y


def measure_side_profile_geometry(
    points: dict,
    face_direction: str = "right",
    nose_upper_target_slope: float = 1.50,
    lip_chin_target_slope: float = 2.80,
    slope_tolerance: float = 0.12,
    lower_line_tolerance_px: float = 2.0,
    chin_line_tolerance_px: float = 3.0,
) -> dict:
    """计算鼻尖—上唇斜率及上唇—下唇—下巴目标线偏差。"""
    required = ("nose", "upper_lip", "lower_lip", "chin")
    if any(points.get(name) is None for name in required):
        return {"valid": False, "reason": "missing profile point", "points": points}
    nose = np.asarray(points["nose"], dtype=np.float64)
    upper = np.asarray(points["upper_lip"], dtype=np.float64)
    lower = np.asarray(points["lower_lip"], dtype=np.float64)
    chin = np.asarray(points["chin"], dtype=np.float64)
    if not (nose[1] < upper[1] < lower[1] < chin[1]):
        return {"valid": False, "reason": "invalid vertical point order", "points": points}

    forward_sign = -1.0 if face_direction == "left" else 1.0
    nose_retreat = forward_sign * (nose[0] - upper[0])
    chin_retreat = forward_sign * (upper[0] - chin[0])
    if nose_retreat <= 1.0 or chin_retreat <= 1.0:
        return {"valid": False, "reason": "profile direction mismatch", "points": points}

    nose_upper_slope = float((upper[1] - nose[1]) / nose_retreat)
    lip_chin_slope = float((chin[1] - upper[1]) / chin_retreat)

    def target_x(y_value: float) -> float:
        retreat = (float(y_value) - upper[1]) / float(lip_chin_target_slope)
        return float(upper[0] - forward_sign * retreat)

    lower_target_x = target_x(lower[1])
    chin_target_x = target_x(chin[1])
    lower_line_error = float(forward_sign * (lower[0] - lower_target_x))
    chin_line_error = float(forward_sign * (chin[0] - chin_target_x))
    nose_upper_error = float(nose_upper_slope - nose_upper_target_slope)
    lip_chin_slope_error = float(lip_chin_slope - lip_chin_target_slope)
    phase1_ok = abs(nose_upper_error) <= slope_tolerance
    lower_ok = abs(lower_line_error) <= lower_line_tolerance_px
    chin_ok = abs(chin_line_error) <= chin_line_tolerance_px
    slope_ok = abs(lip_chin_slope_error) <= slope_tolerance
    return {
        "valid": True,
        "reason": "",
        "points": {name: tuple(float(v) for v in points[name]) for name in required},
        "face_direction": face_direction,
        "nose_upper_slope": nose_upper_slope,
        "nose_upper_target_slope": float(nose_upper_target_slope),
        "nose_upper_slope_error": nose_upper_error,
        "lip_chin_slope": lip_chin_slope,
        "lip_chin_target_slope": float(lip_chin_target_slope),
        "lip_chin_slope_error": lip_chin_slope_error,
        "lower_target_x": lower_target_x,
        "chin_target_x": chin_target_x,
        "lower_line_error_px": lower_line_error,
        "chin_line_error_px": chin_line_error,
        "phase1_qualified": phase1_ok,
        "lower_qualified": lower_ok,
        "chin_qualified": chin_ok,
        "slope_qualified": slope_ok,
        "qualified": bool(phase1_ok and lower_ok and chin_ok and slope_ok),
        "slope_tolerance": float(slope_tolerance),
        "lower_line_tolerance_px": float(lower_line_tolerance_px),
        "chin_line_tolerance_px": float(chin_line_tolerance_px),
    }


class StableFeaturePoint:
    """Coarse display latch; missing samples never become valid measurements."""
    def __init__(self, deadband=6.0, confirm=5, cluster=None):
        self.deadband, self.confirm = deadband, confirm
        self.cluster = deadband if cluster is None else cluster
        self.strict_cluster = cluster is not None
        self.point = None
        self.pending = []
        self.missing = 0

    def update(self, point):
        if point is None:
            self.missing += 1
            self.pending.clear()
            if self.missing >= self.confirm:
                self.point = None
            return None
        self.missing = 0
        candidate = np.asarray(point, dtype=float)
        if self.point is not None and np.linalg.norm(candidate-self.point) <= self.deadband:
            self.pending.clear()
            return tuple(self.point)
        if self.pending and (
                any(np.linalg.norm(candidate-p) > self.cluster for p in self.pending)
                if self.strict_cluster else
                np.linalg.norm(candidate-np.median(self.pending, axis=0)) > self.deadband):
            self.pending.clear()
        self.pending.append(candidate)
        if len(self.pending) >= self.confirm:
            self.point = np.round(np.median(self.pending, axis=0))
            self.pending.clear()
        return tuple(self.point) if self.point is not None else None


def _manual_profile_tips(frame, rect, direction, contrast, paired=False, chin_turn=False, single_lip=False, lower_lip=False):
    """Find rounded forward extrema; a lip ROI must contain two distinct lobes."""
    if rect is None or len(rect) != 4:
        return []
    x1, y1, x2, y2 = map(int, rect)
    h, w = frame.shape[:2]
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        return []
    crop = frame[y1:y2, x1:x2]
    tracks = _contrast_profile_boundary(crop, direction, (0, x2-x1-1),
                                       (0, y2-y1-1), contrast, all_tracks=True,
                                       color_edges=lower_lip)
    sign = 1 if direction == "left" else -1
    candidates = []
    for track in tracks:
        # Median along the contour removes isolated edge noise before choosing
        # an extremum. Chin curvature is broad, unlike a small lip lobe.
        smooth = {y: float(np.median([track[r] for r in range(y-2, y+3) if r in track]))
                  for y in track}
        if lower_lip:
            # A broad lower-lip front need not turn back inside a 20-row window.
            # Assess the full continuous track, then average its forward band.
            ys = sorted(smooth)
            xs = np.array([sign*smooth[y] for y in ys])
            if min(xs[0], xs[-1])-min(xs) < 2:
                continue
            near = np.flatnonzero(xs <= min(xs)+1.5)
            ty = int(round(float(np.median(np.array(ys)[near]))))
            tx = sign*int(round(float(np.median(xs[near]))))
            if 2 < tx < x2-x1-3 and 2 < ty < y2-y1-3:
                candidates.append((x1+tx, y1+ty))
            continue
        if chin_turn:
            # Lower transition: tangent changes from vertical front to sloping
            # underside. Use the 45-degree tangent, not the foremost pixel.
            front_y = min(smooth, key=lambda y: sign*smooth[y])
            turns = []
            for y in sorted(smooth):
                if y <= front_y or y-10 not in smooth or y+10 not in smooth:
                    continue
                before = sign*(smooth[y]-smooth[y-10])/10
                after = sign*(smooth[y+10]-smooth[y])/10
                if before < 1 <= after:
                    turns.append((abs((before+after)/2-1), y))
            if turns:
                _, y = min(turns)
                candidates.append((x1+int(round(smooth[y])), y1+y))
            continue
        radius = 10 if paired or single_lip else 20
        for row in smooth:
            local = {y: x for y, x in smooth.items() if abs(y-row) <= radius}
            if len(local) < 15:
                continue
            ys = sorted(local)
            xs = np.array([sign * local[y] for y in ys])
            if min(xs[0], xs[-1]) - min(xs) < 2:
                continue
            near = np.flatnonzero(xs <= min(xs) + 1.5)
            ty = int(round(float(np.median(np.array(ys)[near]))))
            tx = int(round(float(np.median(xs[near])))) * sign
            if not (2 < tx < x2-x1-3 and 2 < ty < y2-y1-3):
                continue
            point = (x1+tx, y1+ty)
            candidates.append(point)
    candidates.sort(key=lambda p: sign*p[0])
    # Previously the first scanned window won and suppressed later, better
    # candidates at the same height (including the actual outer silhouette).
    merged = []
    for point in candidates:
        if all(abs(point[1]-old[1]) > 10 for old in merged):
            nearby = [p for p in candidates if abs(p[1]-point[1]) <= 10
                      and abs(p[0]-point[0]) <= 1.5]
            merged.append(tuple(int(round(float(v))) for v in np.median(nearby, axis=0)))
    candidates = merged
    if not paired:
        return candidates[:1]
    # Use the two strongest forward lobes separated vertically. Do not invent
    # a lower lip when only one lobe is visible (e.g. closed/occluded mouth).
    if len(candidates) < 2:
        return []
    return sorted(candidates[:2], key=lambda p: p[1])


def find_side_profile_points(
    frame: np.ndarray,
    crop_size: int = 250,
    face_direction: str = "right",
    background_delta: float = 14.0,
    lip_min_area: int = 6,
    nose_x_range: tuple[int, int] | None = None,
    nose_y_range: tuple[int, int] | None = None,
    nose_edge_contrast_min: float = 12.0,
    nose_roi: list[int] | None = None,
    require_nose_roi: bool = False,
    mouth_roi: list[int] | None = None,
    upper_lip_roi: list[int] | None = None,
    lower_lip_roi: list[int] | None = None,
    separate_lip_rois: bool = False,
    chin_roi: list[int] | None = None,
    require_feature_rois: bool = False,
) -> dict:
    """在画面中心正方形内检测鼻尖、上唇、下唇和下巴轮廓端点。"""
    rect = center_square_rect(frame, crop_size)
    x1, y1, x2, y2 = rect
    roi = frame[y1:y2, x1:x2]
    face_mask = _profile_face_mask(roi, face_direction, background_delta)
    boundary = _profile_boundary(face_mask, face_direction)
    nose_boundary = boundary
    if nose_x_range is not None and nose_y_range is not None:
        nose_boundary = _contrast_profile_boundary(
            roi,
            face_direction,
            nose_x_range,
            nose_y_range,
            nose_edge_contrast_min,
        )
    nose_local = _nose_from_boundary(
        nose_boundary,
        roi.shape[0],
        face_direction,
        nose_x_range=nose_x_range,
        nose_y_range=nose_y_range,
    )
    manual_rect = None
    if require_nose_roi or nose_roi is not None:
        nose_local = None
        if nose_roi is not None and len(nose_roi) == 4:
            nx1, ny1, nx2, ny2 = map(int, nose_roi)
            h, w = frame.shape[:2]
            if 0 <= nx1 < nx2 <= w and 0 <= ny1 < ny2 <= h:
                manual_rect = (nx1, ny1, nx2, ny2)
                nose_crop = frame[ny1:ny2, nx1:nx2]
                xr, yr = (0, nx2 - nx1 - 1), (0, ny2 - ny1 - 1)
                tracks = _contrast_profile_boundary(nose_crop, face_direction, xr, yr,
                                                   nose_edge_contrast_min, all_tracks=True)
                tips = [_nose_from_boundary(track, ny2 - ny1, face_direction, xr, yr)
                        for track in tracks]
                tips = [point for point in tips if point is not None]
                tip = (min(tips, key=lambda p: p[0]) if face_direction == "left"
                       else max(tips, key=lambda p: p[0])) if tips else None
                if tip is not None:
                    nose_local = (nx1 + tip[0] - x1, ny1 + tip[1] - y1)
    lip_mask = _profile_lip_mask(
        roi,
        face_mask,
        lip_min_area,
        nose_y=(nose_local[1] if nose_local is not None else None),
    )
    upper_local, lower_local = _split_profile_lips(lip_mask, face_direction)
    local_points = {
        "upper_lip": upper_local,
        "lower_lip": lower_local,
        "nose": nose_local,
        "chin": None,
    }
    if upper_local is not None and lower_local is not None:
        local_points["chin"] = _chin_from_boundary(
            boundary, lower_local, roi.shape[0])

    points = {
        name: ((x1 + point[0], y1 + point[1]) if point is not None else None)
        for name, point in local_points.items()
    }
    if separate_lip_rois:
        for name, box in (("upper_lip", upper_lip_roi), ("lower_lip", lower_lip_roi)):
            tips = _manual_profile_tips(frame, box, face_direction,
                                        nose_edge_contrast_min, single_lip=True,
                                        lower_lip=(name == "lower_lip"))
            points[name] = tips[0] if tips else None
    elif require_feature_rois or mouth_roi is not None:
        lips = _manual_profile_tips(frame, mouth_roi, face_direction,
                                    nose_edge_contrast_min, paired=True)
        points["upper_lip"] = lips[0] if lips else None
        points["lower_lip"] = lips[1] if lips else None
    if require_feature_rois or chin_roi is not None:
        chin = _manual_profile_tips(frame, chin_roi, face_direction, nose_edge_contrast_min,
                                    chin_turn=True)
        points["chin"] = chin[0] if chin else None
    valid = all(points[name] is not None for name in ("nose", "upper_lip", "lower_lip", "chin"))
    return {
        "valid": valid,
        "reason": "" if valid else ("Select nose ROI" if require_nose_roi and manual_rect is None
                                     else "nose/lips/chin not all found"),
        "rect": rect,
        "points": points,
        "face_mask": face_mask,
        "lip_mask": lip_mask,
        "face_direction": face_direction,
        "mouth_search_rect": mouth_roi,
        "upper_search_rect": upper_lip_roi,
        "lower_search_rect": lower_lip_roi,
        "chin_search_rect": chin_roi,
        "nose_search_rect": manual_rect if require_nose_roi or nose_roi is not None else (
            (
                x1 + min(nose_x_range),
                y1 + min(nose_y_range),
                x1 + max(nose_x_range),
                y1 + max(nose_y_range),
            )
            if nose_x_range is not None and nose_y_range is not None else None
        ),
    }


def draw_side_profile_geometry(
    frame: np.ndarray,
    signal: dict,
    show_face_overlays: bool = True,
    draw_lines: bool = True,
    draw_expected_dashed: bool = False,
) -> None:
    """只绘制贴在人脸上的轮廓几何；文字 HUD 由调用方单独绘制。"""
    if not show_face_overlays or not signal:
        return
    rect = signal.get("rect")
    if rect is not None:
        cv2.rectangle(frame, tuple(rect[:2]), tuple(rect[2:]), (180, 180, 180), 1)
    nose_search_rect = signal.get("nose_search_rect")
    if nose_search_rect is not None:
        cv2.rectangle(
            frame,
            tuple(int(value) for value in nose_search_rect[:2]),
            tuple(int(value) for value in nose_search_rect[2:]),
            (0, 255, 255), 2, cv2.LINE_AA,
        )
    points = signal.get("points") or {}
    for key, color in (("upper_search_rect", (0, 165, 255)),
                       ("lower_search_rect", (0, 255, 255)),
                       ("mouth_search_rect", (0, 165, 255)),
                       ("chin_search_rect", (255, 200, 0))):
        box = signal.get(key)
        if box is not None:
            cv2.rectangle(frame, tuple(map(int, box[:2])), tuple(map(int, box[2:])), color, 1)
    colors = {
        "nose": (0, 0, 255),
        "upper_lip": (0, 165, 255),
        "lower_lip": (0, 255, 255),
        "chin": (255, 180, 0),
    }
    for name, color in colors.items():
        point = points.get(name)
        if point is not None:
            p = tuple(int(round(value)) for value in point)
            cv2.circle(frame, p, 4, color, -1, cv2.LINE_AA)

    if not signal.get("valid"):
        return
    nose = tuple(int(round(value)) for value in points["nose"])
    upper = tuple(int(round(value)) for value in points["upper_lip"])
    lower = tuple(int(round(value)) for value in points["lower_lip"])
    chin = tuple(int(round(value)) for value in points["chin"])

    if draw_expected_dashed:
        forward_sign = -1.0 if signal.get("face_direction") == "left" else 1.0
        nose_slope = max(1e-6, float(signal["nose_upper_target_slope"]))
        nose_upper_retreat = (upper[1] - nose[1]) / nose_slope
        expected_upper = (
            int(round(nose[0] - forward_sign * nose_upper_retreat)),
            upper[1],
        )
        target_chin = (int(round(signal["chin_target_x"])), chin[1])
        _draw_dashed_line(frame, nose, expected_upper, (0, 0, 255), 2)
        _draw_dashed_line(frame, upper, target_chin, (0, 255, 255), 2)

    if not draw_lines:
        return
    # 红线：鼻尖到上唇；黄线：以上唇为起点的目标斜率线。
    cv2.line(frame, nose, upper, (0, 0, 255), 2, cv2.LINE_AA)
    target_chin = (int(round(signal["chin_target_x"])), chin[1])
    cv2.line(frame, upper, target_chin, (0, 255, 255), 2, cv2.LINE_AA)
    actual_color = (0, 255, 0) if signal.get("qualified") else (0, 140, 255)
    cv2.polylines(
        frame,
        [np.asarray([upper, lower, chin], dtype=np.int32)],
        False,
        actual_color,
        1,
        cv2.LINE_AA,
    )


def _draw_dashed_line(
    frame: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 1,
    dash_length: float = 9.0,
    gap_length: float = 6.0,
) -> None:
    start_v = np.asarray(start, dtype=np.float64)
    end_v = np.asarray(end, dtype=np.float64)
    delta = end_v - start_v
    distance = float(np.linalg.norm(delta))
    if distance < 1.0:
        return
    direction = delta / distance
    step = max(1.0, dash_length + gap_length)
    offset = 0.0
    while offset < distance:
        segment_end = min(distance, offset + dash_length)
        p1 = tuple(int(round(v)) for v in start_v + direction * offset)
        p2 = tuple(int(round(v)) for v in start_v + direction * segment_end)
        cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)
        offset += step
