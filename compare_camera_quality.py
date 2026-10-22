"""Independent camera-1 quality experiment. Default is strictly NO CAMERA.

Run only after user confirmation:
    python -B -u compare_camera_quality.py --run-camera

Uses the existing detect() without changing its math, timestamps, or EMA.
Observers copy smoother inputs/outputs and delegate unchanged. No CUDA test.
"""

import argparse
import ast
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import benchmark_front_camera as bench

SEQUENCE = ("YUY2", "MJPG", "MJPG", "YUY2")
MIN_VALID = 100                 # BOTH eyes, per block (>=200 per format).
MIN_FRAMES = 120                # Failures remain in the denominator.
MIN_SECONDS = 24.0             # Comparable observation duration, not just count.
MAX_SECONDS = 90.0             # Checked BETWEEN native calls.
MAX_FRAMES = 3000
SAVE_INDICES = (1, 30, 60, 90, 120)  # Prespecified, not chosen for sharpness.
ALLOWED_PROPERTIES = ("FRAME_WIDTH", "FRAME_HEIGHT", "FPS", "FOURCC")
CONTROL_KEYS = ("exposure", "auto_exposure", "gain", "focus", "autofocus",
                "auto_wb", "white_balance", "brightness")


class Observer:
    """Observe values without changing the wrapped algorithm's return value."""

    def __init__(self, inner):
        self.inner = inner
        self.clear()

    def clear(self):
        self.raw = self.output = None

    def update(self, value):
        self.raw = copy.deepcopy(value)
        result = self.inner.update(value)
        self.output = copy.deepcopy(result)
        return result

    def reset(self):
        self.clear()
        self.inner.reset()


def static_check():
    """Parse sources; no cv2/visual import, model initialization or COM access."""
    paths = [Path(__file__), bench.ROOT / "benchmark_front_camera.py",
             bench.ROOT / "ellseg_scorer.py", bench.ROOT / "landmark_snapshot.py"]
    for path in paths:
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    setters = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
               and isinstance(n.func, ast.Attribute) and n.func.attr == "set"]
    if len(setters) != 1:
        raise RuntimeError("Unexpected property-setter call count; review required.")
    data = bench.preflight()
    modes = json.loads((bench.ROOT / "benchmark_results" /
                       "camera1_supported_modes.json").read_text(encoding="utf-8"))
    for codec, fps in (("YUY2", 5), ("MJPG", 30)):
        if not any(m["width"] == 1920 and m["height"] == 1080
                   and m["fourcc"] == codec and abs(m["max_fps"] - fps) < .01
                   for m in modes["modes"]):
            raise RuntimeError(f"Previously queried mode unavailable: {codec} {fps}")
    return data, modes


def save_json(path, value):
    def numeric_value(obj):
        if type(obj).__module__.split(".")[0] == "numpy" and hasattr(obj, "tolist"):
            return obj.tolist()
        raise TypeError(f"Unsupported JSON value: {type(obj).__name__}")

    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False, default=numeric_value), encoding="utf-8")


def stats(values, np):
    if not values:
        return {"count": 0}
    a = np.asarray(values, dtype=float)
    return {"count": len(a), "mean": a.mean(axis=0).tolist(),
            "std_sample": a.std(axis=0, ddof=1).tolist() if len(a) > 1 else None,
            "min": a.min(axis=0).tolist(), "max": a.max(axis=0).tolist(),
            "peak_to_peak": np.ptp(a, axis=0).tolist()}


def series_summary(rows, eye_ids, np):
    output = {"processed_frames": len(rows),
              "mp_success_rate": sum(r["mp_valid"] for r in rows) / len(rows)}
    for side in ("left", "right"):
        valid = [r for r in rows if r[side]["valid"]]
        output[side] = {"success_count": len(valid),
                        "success_rate": len(valid) / len(rows)}
        for field in ("raw_center", "smooth_center", "integer_center", "relative_to_corners"):
            output[side][field] = stats(
                [r[side][field] for r in valid if r[side].get(field) is not None], np)
        output[side]["quality"] = {
            key: stats([r["quality"][side][key] for r in rows
                        if side in r["quality"]], np)
            for key in ("brightness", "laplacian_variance", "tenengrad")}
    output["mp_raw_points"] = {
        str(idx): stats([r["mp_raw"][str(idx)] for r in rows
                        if str(idx) in r["mp_raw"]], np) for idx in eye_ids}
    output["mp_smoothed_points"] = {
        str(idx): stats([r["mp_smooth"][str(idx)] for r in rows
                        if str(idx) in r["mp_smooth"]], np) for idx in eye_ids}
    return output


def subset_5hz(rows):
    """First observed frame in each 200-ms bin; retain failures too."""
    seen, selected = set(), []
    for row in rows:
        bucket = int(row["t_seconds"] / .2)
        if bucket not in seen:
            selected.append(row)
            seen.add(bucket)
    return selected


def image_quality(image, cv2, np):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    return {"brightness": float(gray.mean()),
            "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            "tenengrad": float(np.mean(gx * gx + gy * gy))}


def observation(frame, result, detector, visual, rois, cv2, np):
    lm = detector.lm_smoother
    ids = sorted(set(visual.LEFT_EYE_IDX + visual.RIGHT_EYE_IDX +
                     list(range(468, 478)) + [1]))
    row = {"mp_valid": lm.raw is not None, "mp_raw": {}, "mp_smooth": {},
           "quality": {}}
    if lm.raw is not None:
        row["mp_raw"] = {str(i): lm.raw[i].tolist() for i in ids if i < len(lm.raw)}
        row["mp_smooth"] = {
            str(i): lm.output[i].tolist() for i in ids if i < len(lm.output)}
    for side, indices, observer, corners in (
        ("left", visual.RIGHT_EYE_IDX, detector.ell_smooth_L, (362, 263)),
        ("right", visual.LEFT_EYE_IDX, detector.ell_smooth_R, (33, 133)),
    ):
        ellipse = result.get(f"{side}_iris_ellipse")
        valid = ellipse is not None and observer.raw is not None
        if valid:
            valid = bool(np.isfinite(np.asarray(ellipse[0])).all()
                         and np.isfinite(np.asarray(observer.raw[0])).all())
        row[side] = {"valid": valid}
        if valid:
            crop, (x, y) = visual.crop_eye(frame, lm.output, indices)
            center = np.asarray(observer.raw[0], dtype=float) + [x, y]
            row[side].update({
                "raw_center": center.tolist(),
                "smooth_center": list(ellipse[0]),
                "integer_center": list(result[f"{side}_iris"]),
                "raw_ellipse_global": [center.tolist(), list(observer.raw[1]),
                                       float(observer.raw[2])],
                "smooth_ellipse_global": [list(ellipse[0]), list(ellipse[1]), float(ellipse[2])],
                "relative_to_corners": (center - lm.raw[list(corners)].mean(axis=0)).tolist(),
            })
            if side not in rois:
                rois[side] = [int(x), int(y), int(crop.shape[1]), int(crop.shape[0])]
        if side in rois:
            x, y, w, h = rois[side]
            row["quality"][side] = image_quality(frame[y:y+h, x:x+w], cv2, np)
    return row


def verify_mode(cap, cv2, codec):
    settings = bench.camera_settings(cap, cv2)
    if (settings["fourcc"], settings["width"], settings["height"]) != (codec, 1920, 1080):
        raise RuntimeError(f"Unexpected negotiated mode: {settings}")
    return settings


def run_block(block, data, rois, folder, cv2, np):
    codec = block["format"]
    visual, detector = bench.load_visual_pipeline(data)
    # Only these observer objects change in this isolated process.
    detector.lm_smoother = Observer(detector.lm_smoother)
    detector.ell_smooth_L = Observer(detector.ell_smooth_L)
    detector.ell_smooth_R = Observer(detector.ell_smooth_R)
    cap = None
    pictures = []
    try:
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise RuntimeError("Cannot open camera 1; no other index attempted.")
        initial = block["initial_settings"] = bench.camera_settings(cap, cv2)
        requests = (("FRAME_WIDTH", 1920), ("FRAME_HEIGHT", 1080),
                    ("FPS", 5 if codec == "YUY2" else 30),
                    ("FOURCC", cv2.VideoWriter_fourcc(*codec)))
        block["mode_requests"] = []
        for name, value in requests:
            if name not in ALLOWED_PROPERTIES:
                raise RuntimeError("Forbidden camera setter.")
            accepted = bool(cap.set(getattr(cv2, "CAP_PROP_" + name), value))
            current = bench.camera_settings(cap, cv2)
            block["mode_requests"].append(
                {"property": name, "requested": value, "set_return": accepted,
                 "readback": current})
            if not accepted:
                raise RuntimeError(f"Rejected mode request: {name}")
            # Automatic focus position may move; do not disable or override it.
            for key in ("exposure", "auto_exposure", "gain", "autofocus", "auto_wb"):
                if current[key] != initial[key]:
                    raise RuntimeError(f"Driver changed {key} while negotiating mode.")
        block["negotiated_settings"] = verify_mode(cap, cv2, codec)
        # Six wall-clock seconds settling, independent of selected frame rate.
        settle_start = time.perf_counter()
        while time.perf_counter() - settle_start < 6:
            ok, _ = cap.read()
            if not ok:
                raise RuntimeError("Camera warmup read failed.")
        raw, _ = bench.timed_round(cap, cv2, frames=60)
        block["pure_read_60"] = raw
        # Fresh MediaPipe/EMA state per block; identical number of model warmup frames.
        for _ in range(20):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Visual warmup read failed.")
            detector.detect(cv2.rotate(frame, cv2.ROTATE_180))
        block["settings_before_measurement"] = verify_mode(cap, cv2, codec)
        rows = block["rows"] = []
        controls = block["control_samples"] = []
        attempts = failures = both_valid = 0
        started = time.perf_counter()
        last_control = last_progress = -100.0
        while True:
            elapsed = time.perf_counter() - started
            if len(rows) >= MIN_FRAMES and both_valid >= MIN_VALID and elapsed >= MIN_SECONDS:
                break
            if elapsed >= MAX_SECONDS or len(rows) >= MAX_FRAMES:
                raise RuntimeError(f"Insufficient valid data: {both_valid} both-eye frames.")
            if elapsed - last_control >= 1:
                controls.append({"t_seconds": elapsed, "settings": bench.camera_settings(cap, cv2)})
                last_control = elapsed
            read_start = time.perf_counter()
            ok, camera_frame = cap.read()
            read_end = time.perf_counter()
            attempts += 1
            if not ok or camera_frame is None:
                failures += 1
                continue
            if camera_frame.shape != (1080, 1920, 3):
                raise RuntimeError(f"Unexpected frame shape: {camera_frame.shape}")
            frame = cv2.rotate(camera_frame, cv2.ROTATE_180)
            for obs in (detector.lm_smoother, detector.ell_smooth_L, detector.ell_smooth_R):
                obs.clear()  # Never reuse a previous frame's success on a failed detection.
            detect_start = time.perf_counter()
            result = detector.detect(frame)
            detect_end = time.perf_counter()
            row = observation(frame, result, detector, visual, rois, cv2, np)
            row.update({"frame": len(rows)+1, "t_seconds": read_end-started,
                        "read_ms": 1000*(read_end-read_start),
                        "detect_ms": 1000*(detect_end-detect_start)})
            rows.append(row)
            both_valid += int(row["left"]["valid"] and row["right"]["valid"])
            if len(rows) in SAVE_INDICES:
                pictures.append((len(rows), camera_frame.copy(), frame.copy()))
            if elapsed - last_progress >= 10:
                print(f"BLOCK {block['number']} {codec}: frames={len(rows)}, "
                      f"both_valid={both_valid}, elapsed={elapsed:.1f}s", flush=True)
                last_progress = elapsed
        block.update({"elapsed_seconds": time.perf_counter()-started,
                      "read_attempts": attempts, "read_failures": failures,
                      "both_valid_frames": both_valid})
        block["processing_fps"] = len(rows) / block["elapsed_seconds"]
        block["final_settings"] = verify_mode(cap, cv2, codec)
        block["status"] = "complete"
    finally:
        if cap is not None:
            cap.release()
        detector.close()
        # Image compression/disk writes occur after measurement and camera release.
        block["images"] = []
        for number, camera_frame, frame in pictures:
            stem = f"block{block['number']}_{codec}_frame{number:04d}"
            camera_name, oriented_name = stem+"_camera.png", stem+"_oriented.png"
            for name, img in ((camera_name, camera_frame), (oriented_name, frame)):
                if not cv2.imwrite(str(folder / name), img):
                    raise IOError(f"Cannot save {name}")
            block["images"].append({"frame": number, "camera_png": camera_name,
                                    "oriented_png": oriented_name})


def analyze(report, np):
    eye_ids = sorted({int(i) for b in report["blocks"] for r in b.get("rows", [])
                      for i in r["mp_raw"]})
    for block in report["blocks"]:
        if block.get("rows"):
            block["statistics"] = series_summary(block["rows"], eye_ids, np)
            block["statistics_5hz"] = series_summary(subset_5hz(block["rows"]), eye_ids, np)
        snapshots = [block[k] for k in ("initial_settings", "negotiated_settings",
                                       "settings_before_measurement", "final_settings")
                     if k in block]
        snapshots += [entry["settings"] for entry in block.get("control_samples", [])]
        block["control_readback_values"] = {
            key: sorted({s[key] for s in snapshots if s.get(key) is not None})
            for key in CONTROL_KEYS}
    aggregate = report["statistics_by_format"] = {}
    for codec in ("YUY2", "MJPG"):
        rows = [r for b in report["blocks"] if b["format"] == codec for r in b.get("rows", [])]
        if rows:
            aggregate[codec] = series_summary(rows, eye_ids, np)
    # Use equal block weighting rather than let faster/longer blocks dominate.
    contrasts = report["mean_shift_MJPG_minus_YUY2"] = {}
    for side in ("left", "right"):
        contrasts[side] = {}
        for field in ("raw_center", "smooth_center", "integer_center", "relative_to_corners"):
            pairs = []
            for yi, mi in ((0, 1), (3, 2)):
                if max(yi, mi) >= len(report["blocks"]):
                    continue
                y = report["blocks"][yi].get("statistics", {}).get(side, {}).get(field, {})
                m = report["blocks"][mi].get("statistics", {}).get(side, {}).get(field, {})
                if y.get("count", 0) and m.get("count", 0):
                    pairs.append((np.asarray(m["mean"])-np.asarray(y["mean"])).tolist())
            if pairs:
                mean = np.mean(pairs, axis=0)
                contrasts[side][field] = {
                    "pair_deltas_xy": pairs, "equal_pair_mean_delta_xy": mean.tolist(),
                    "norm_px": float(np.linalg.norm(mean)),
                    "at_least_1px": bool(np.linalg.norm(mean) >= 1),
                    "either_pair_at_least_1px": any(np.linalg.norm(p) >= 1 for p in pairs)}
    report["mp_raw_mean_shift_by_point"] = {}
    for idx in eye_ids:
        pairs = []
        for yi, mi in ((0, 1), (3, 2)):
            if max(yi, mi) >= len(report["blocks"]):
                continue
            y = report["blocks"][yi].get("statistics", {}).get("mp_raw_points", {}).get(str(idx), {})
            m = report["blocks"][mi].get("statistics", {}).get("mp_raw_points", {}).get(str(idx), {})
            if y.get("count", 0) and m.get("count", 0):
                pairs.append((np.asarray(m["mean"])-np.asarray(y["mean"])).tolist())
        if pairs:
            mean = np.mean(pairs, axis=0)
            report["mp_raw_mean_shift_by_point"][str(idx)] = {
                "pair_deltas_xy": pairs, "equal_pair_mean_delta_xy": mean.tolist(),
                "norm_px": float(np.linalg.norm(mean))}
    report["interpretation"] = (
        "Repeatability and between-mode differences only, not absolute accuracy. "
        "1 px is a review flag, not a validated motor tolerance. Differences cannot "
        "be attributed solely to compression. Inspect each block and paired direction, "
        "controls, brightness, fixed-ROI images, and MP drift. No independent-frame "
        "confidence intervals: successive video frames and EMA outputs are correlated.")


def make_report(report, folder, cv2, np):
    lines = ["# YUY2 / MJPG 独立对比", "",
             f"状态：{report['status']}。设备仅 camera 1，1920×1080，DirectShow。",
             "原图为 OpenCV 解码/颜色转换后的 BGR，PNG 保存不再引入 JPEG 压缩；"
             "不是传感器 RAW，也不能据此称 MJPG 无损。",
             "所有统计坐标位于旋转 180° 的检测输入图；L=机器左眼（画面右），R=机器右眼。",
             "", "|段|模式|纯读60帧/秒|纯读FPS|含检测与记录FPS|有效双眼/处理帧|",
             "|---|---|---:|---:|---:|---|"]
    for b in report["blocks"]:
        raw = b.get("pure_read_60", {})
        lines.append(f"|{b['number']}|{b['format']}|{raw.get('elapsed_seconds',0):.3f}|"
                     f"{raw.get('fps',0):.2f}|{b.get('processing_fps',0):.2f}|"
                     f"{b.get('both_valid_frames',0)}/{len(b.get('rows',[]))}|")
    lines += ["", "## 虹膜稳定性（像素）", "",
              "以下为全部有效帧的平滑前椭圆中心；不剔除离群值。"
              "完整 JSON 同时提供每段、约5Hz子集、平滑后浮点和整数输出统计。",
              "", "|模式|眼|检测成功率|均值 x,y|样本标准差 x,y|峰峰值 x,y|",
              "|---|---|---:|---|---|---|"]
    for codec, s in report.get("statistics_by_format", {}).items():
        for side in ("left", "right"):
            a = s[side]["raw_center"]
            lines.append(f"|{codec}|{side}|{s[side]['success_rate']:.2%}|"
                         f"{a.get('mean')}|{a.get('std_sample')}|{a.get('peak_to_peak')}|")
    lines += ["", "## 每段稳定性（用于检查时间漂移）", "",
              "|段|模式|眼|成功率|原始中心均值 x,y|原始中心标准差 x,y|",
              "|---|---|---|---:|---|---|"]
    for b in report["blocks"]:
        for side in ("left", "right"):
            s = b.get("statistics", {}).get(side)
            if s:
                a = s["raw_center"]
                lines.append(f"|{b['number']}|{b['format']}|{side}|{s['success_rate']:.2%}|"
                             f"{a.get('mean')}|{a.get('std_sample')}|")
    lines += ["", "## MediaPipe 与图像质量辅助指标", "",
              "|模式|MP检测成功率|眼|亮度均值|Laplacian方差均值|Tenengrad均值|",
              "|---|---:|---|---:|---:|---:|"]
    for codec, s in report.get("statistics_by_format", {}).items():
        for side in ("left", "right"):
            q = s[side]["quality"]
            lines.append(f"|{codec}|{s['mp_success_rate']:.2%}|{side}|"
                         f"{q['brightness'].get('mean')}|"
                         f"{q['laplacian_variance'].get('mean')}|"
                         f"{q['tenengrad'].get('mean')}|")
    mp_shift = report.get("mp_raw_mean_shift_by_point", {})
    if mp_shift:
        worst_id = max(mp_shift, key=lambda idx: mp_shift[idx]["norm_px"])
        lines += ["", f"所记录点（眼部点及鼻点1）中最大等权均值偏移："
                  f"点 {worst_id}，{mp_shift[worst_id]['norm_px']:.4f} px。"
                  "各点两次差值及原始/平滑坐标统计见 JSON。"]
    lines += ["", "控制参数仅回读；-1 表示接口未提供有效信息，不等于自动模式已关闭。", ""]
    for b in report["blocks"]:
        lines += [f"第{b['number']}段 {b['format']}：",
                  "```json", json.dumps(b.get("control_readback_values", {}),
                                       ensure_ascii=False), "```"]
    lines += ["", "## 模式间均值差（MJPG − YUY2）", "",
              "两次相邻对比等权平均；绝对坐标和眼角相对坐标分别报告。≥1 px 仅为关注标记。",
              "```json", json.dumps(report.get("mean_shift_MJPG_minus_YUY2", {}),
                                   ensure_ascii=False, indent=2), "```",
              "", "## 清晰度图片", "",
              "下列拼图固定选各段第60帧（不按清晰度或检测成功筛选），"
              "同一眼使用全程相同像素 ROI，原尺寸、不插值。完整帧另存 PNG。"]
    for side, (x, y, w, h) in report["fixed_eye_rois"].items():
        panels = []
        for block in report["blocks"]:
            entry = next((e for e in block.get("images", []) if e["frame"] == 60), None)
            if entry is None:
                continue
            frame = cv2.imread(str(folder / entry["oriented_png"]))
            patch = frame[y:y+h, x:x+w]
            panel = cv2.copyMakeBorder(patch, 32, 0, 0, 0, cv2.BORDER_CONSTANT)
            cv2.putText(panel, f"{block['number']} {block['format']} {side}",
                        (4, 22), cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1)
            panels.append(panel)
        if panels:
            name = f"comparison_{side}_1to1.png"
            if not cv2.imwrite(str(folder / name), np.hstack(panels)):
                raise IOError(name)
            lines += [f"![{side} 1:1]({name})", ""]
    for block in report["blocks"]:
        for entry in block.get("images", []):
            lines.append(f"- 第{block['number']}段，第{entry['frame']}帧："
                         f"[未旋转原画面]({entry['camera_png']}) / "
                         f"[旋转后无HUD]({entry['oriented_png']})")
    lines += ["", "## 解读限制与建议", "",
              "这是重复性与模式间差异测试，不是绝对定位精度测试。"
              "自动对焦、光照、驱动内部处理及采样时长可能造成差异；"
              "已有 EMA 按帧更新，两种帧率的时间平滑效果不同。"
              "约5Hz子集只减少采样密度差异，不能消除原流程的时间历史差异。",
              "Laplacian 方差、Tenengrad 与亮度为固定眼 ROI 的辅助指标（见 JSON），"
              "噪声或压缩边缘也可能使数值升高，不能单独判定画质好坏。",
              "MediaPipe 原始/平滑后眼部点的均值、标准差、极值与峰峰值见 JSON；"
              "眼角相对坐标仅用于区分整体漂移，不回写算法。",
              "**是否正式切换 MJPG，需人工检查原图/边缘伪影、各段稳定性、"
              "往返均值差是否重复以及控制参数变化后再判断；脚本不自动宣告无损或验收通过。**",
              "", f"受保护文件变化：{report.get('protected_files_changed', [])}",
              f"禁止模块加载：{report.get('forbidden_modules_loaded', [])}"]
    (folder / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-camera", action="store_true")
    args = parser.parse_args()
    data, modes = static_check()
    if not args.run_camera:
        print("STATIC/PREFLIGHT OK. No camera, COM, visual models, or CUDA test started.")
        print("Await confirmation: camera 1; YUY2/MJPG/MJPG/YUY2; "
              "each >=100 both-eye valid frames, >=120 processed frames, >=24 seconds.")
        return 0
    import cv2
    import numpy as np
    before = bench.protected_hashes()
    folder = bench.ROOT / "benchmark_results" / (
        "quality_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    folder.mkdir()
    report = {
        "started_at": datetime.now().astimezone().isoformat(), "status": "running",
        "previous_device_query": modes["selected_device"], "camera_index": 1,
        "blocks": [], "fixed_eye_rois": {},
        "settings": {"min_valid_both_per_block": MIN_VALID, "min_frames": MIN_FRAMES,
                     "min_seconds": MIN_SECONDS, "max_seconds": MAX_SECONDS,
                     "max_frames": MAX_FRAMES, "image_frame_indices": SAVE_INDICES},
        "source_sha256": {name: hashlib.sha256((bench.ROOT/name).read_bytes()).hexdigest()
                          for name in ("ellseg_scorer.py", "benchmark_front_camera.py",
                                       "compare_camera_quality.py")},
    }
    try:
        for number, codec in enumerate(SEQUENCE, 1):
            block = {"number": number, "format": codec, "status": "running"}
            report["blocks"].append(block)
            print(f"Starting block {number}: {codec}, ONLY camera 1.", flush=True)
            run_block(block, data, report["fixed_eye_rois"], folder, cv2, np)
            save_json(folder / "data.json", report)
        report["status"] = "complete"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], flush=True)
    finally:
        after = bench.protected_hashes()
        report["protected_files_changed"] = [
            k for k in sorted(before.keys() | after.keys()) if before.get(k) != after.get(k)]
        report["forbidden_modules_loaded"] = [
            n for n in ("serial", "Motor", "Communication", "eye_auto_tuner",
                        "qt_calibration_panel", "run_interface") if n in sys.modules]
        report["finished_at"] = datetime.now().astimezone().isoformat()
        save_json(folder / "data.json", report)
        analyze(report, np)
        save_json(folder / "data.json", report)
        make_report(report, folder, cv2, np)
        print("REPORT", folder / "report.md", flush=True)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
