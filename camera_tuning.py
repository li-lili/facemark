"""Camera calibration metrics and checked DirectShow property operations. No motors."""
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np

SAMPLE_SECONDS = 8.0
SETTLE_SECONDS = 2.0
MIN_SAMPLES = 30
MIN_VALID_RATE = 0.90
MIN_PROCESS_FPS = 3.0
MIN_SHARPNESS_RATIO = 0.60
MAX_CLIPPED_RATIO = 0.10
MIN_GRAY = 20.0
MAX_GRAY = 235.0
PROPERTIES = {
    'fps': cv2.CAP_PROP_FPS,
    'exposure': cv2.CAP_PROP_EXPOSURE,
    'gain': cv2.CAP_PROP_GAIN,
    'focus': cv2.CAP_PROP_FOCUS,
    'auto_exposure': cv2.CAP_PROP_AUTO_EXPOSURE,
    'autofocus': cv2.CAP_PROP_AUTOFOCUS,
    'auto_wb': cv2.CAP_PROP_AUTO_WB,
    'wb': cv2.CAP_PROP_WB_TEMPERATURE,
}


def snapshot(cap):
    return {name: float(cap.get(prop)) for name, prop in PROPERTIES.items()}


def apply_checked(cap, requested):
    """Reject ignored/clamped settings. Readback is driver evidence, not sensor proof."""
    errors = []
    # FPS first: a mode change may reset the remaining controls.
    for name in ('fps', 'auto_exposure', 'autofocus', 'auto_wb',
                 'exposure', 'gain', 'focus', 'wb'):
        if name not in requested:
            continue
        value = float(requested[name])
        if not math.isfinite(value):
            errors.append(f'{name}: non-finite value')
            continue
        accepted = cap.set(PROPERTIES[name], value)
        actual = float(cap.get(PROPERTIES[name]))
        if not accepted or not math.isclose(actual, value, abs_tol=0.01, rel_tol=0.001):
            errors.append(f'{name}: requested={value}, readback={actual}, set={accepted}')
    return snapshot(cap), errors


def restore(cap, original):
    """Restore manual values first, automatic modes last; return failures to the UI."""
    failures = []
    automatic = {
        'exposure': original['auto_exposure'] != 0,
        'gain': original['auto_exposure'] != 0,
        'focus': original['autofocus'] != 0,
        'wb': original['auto_wb'] != 0,
    }
    for name in ('fps', 'exposure', 'gain', 'focus', 'wb',
                 'auto_exposure', 'autofocus', 'auto_wb'):
        if automatic.get(name, False):
            continue  # Automatic values may legitimately drift; restore their mode.
        value = original[name]
        mode = {'exposure': 'auto_exposure', 'focus': 'autofocus', 'wb': 'auto_wb'}.get(name)
        if mode and float(cap.get(PROPERTIES[mode])) != 0:
            if not cap.set(PROPERTIES[mode], 0):
                failures.append(mode)
        if math.isfinite(value) and not math.isclose(float(cap.get(PROPERTIES[name])), value,
                                                   abs_tol=0.01, rel_tol=0.001):
            if not cap.set(PROPERTIES[name], value):
                failures.append(name)
    actual = snapshot(cap)
    for name, value in original.items():
        if automatic.get(name, False):
            continue
        if not math.isclose(actual[name], value, abs_tol=0.01, rel_tol=0.001):
            failures.append(name)
    return sorted(set(failures))


def measurement(det, frame):
    """Read existing detector outputs unchanged; never substitute stale/missing eyes."""
    centers, sharpness, brightness, clipped = [], [], [], []
    h, w = frame.shape[:2]
    for side in ('left', 'right'):
        center = det.get(f'{side}_iris')
        ellipse = det.get(f'{side}_iris_ellipse')
        if center is None or ellipse is None:
            return None
        x, y = map(float, center)
        axes = np.asarray(ellipse[1], dtype=float)
        if not np.isfinite([x, y, *axes]).all() or not (0 <= x < w and 0 <= y < h):
            return None
        radius = float(max(axes)) / 2
        if radius < 2 or radius > min(h, w) / 4:
            return None
        r = max(8, int(radius * 1.5))
        crop = frame[max(0, int(y)-r):min(h, int(y)+r),
                     max(0, int(x)-r):min(w, int(x)+r)]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        centers.extend([x, y])
        sharpness.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        brightness.append(float(gray.mean()))
        clipped.append(float(np.mean((gray <= 2) | (gray >= 253))))
    raw_offsets = {side: list(det.get('offset', {}).get(side, ()))
                   for side in ('left', 'right')}
    return dict(centers=centers, sharpness=min(sharpness), brightness=brightness,
                clipped=max(clipped), raw_offsets=raw_offsets)


def summarize(samples, attempts, elapsed, reference_sharpness=None):
    valid = [s for s in samples if s is not None]
    rate = len(valid) / max(1, attempts)
    result = dict(valid=len(valid), attempts=attempts, valid_rate=rate,
                  process_fps=attempts / max(elapsed, 1e-6), eligible=False,
                  jitter_px=None, reasons=[])
    reasons = result['reasons']
    if len(valid) < MIN_SAMPLES:
        reasons.append('有效帧不足')
    if rate < MIN_VALID_RATE:
        reasons.append('双眼检测率不足')
    if result['process_fps'] < MIN_PROCESS_FPS:
        reasons.append('处理帧率不足')
    hashes = {s.get('frame_digest') for s in valid if s.get('frame_digest') is not None}
    if len(valid) >= 2 and len(hashes) == 1:
        reasons.append('全部有效帧内容完全相同，疑似冻结画面')
    if len(valid) >= 2:
        std = np.std([s['centers'] for s in valid], axis=0, ddof=1)
        result['center_std_xy'] = std.tolist()
        result['jitter_px'] = float(max(np.hypot(*std[:2]), np.hypot(*std[2:])))
        result['sharpness'] = float(np.median([s['sharpness'] for s in valid]))
        result['clipped'] = float(np.mean([s['clipped'] for s in valid]))
        brightness = np.mean([s['brightness'] for s in valid], axis=0)
        result['brightness'] = brightness.tolist()
        if (brightness < MIN_GRAY).any() or (brightness > MAX_GRAY).any():
            reasons.append('眼部过暗或过亮')
        if result['clipped'] > MAX_CLIPPED_RATIO:
            reasons.append('眼部黑白截断比例过高')
        if reference_sharpness is not None and result['sharpness'] < reference_sharpness * MIN_SHARPNESS_RATIO:
            reasons.append('清晰度低于初始参考')
    result['eligible'] = not reasons
    return result


def save_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    try:
        with temp.open('w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def draw_camera_hud(img, settings, sample_seconds=SAMPLE_SECONDS):
    """Shared permanent right-column HUD; no change to detection values."""
    x = max(340, img.shape[1] - 430)
    rows = [
        'Cam ' + ' '.join(f'{k}={settings.get(k, float("nan")):g}' for k in ('fps', 'exposure', 'gain')),
        'Cam ' + ' '.join(f'{k}={settings.get(k, float("nan")):g}' for k in ('focus', 'autofocus', 'auto_exposure')),
        f'Cam WB={settings.get("wb", float("nan")):g} auto={settings.get("auto_wb", float("nan")):g}',
        f'CamTest {sample_seconds:g}s settle{SETTLE_SECONDS:g}s N>={MIN_SAMPLES} valid>={MIN_VALID_RATE:.0%}',
        f'CamTest proc>={MIN_PROCESS_FPS:g} sharp>={MIN_SHARPNESS_RATIO:g}x clip<={MAX_CLIPPED_RATIO:.0%}',
        f'CamTest gray {MIN_GRAY:g}..{MAX_GRAY:g}; rank worst-eye SD(px)',
    ]
    for i, text in enumerate(rows):
        cv2.putText(img, text, (x, 926 + i * 16), 0, 0.34, (200, 220, 255), 1)
