"""
Manual Eye Calibrator — 拍照标定中性脸眼睛参数
================================================
流程：
  1. 打开摄像头，实时显示 MediaPipe 眼部关键点 + 几何评分
  2. 人工通过其他方式调好舵机，观察评分变化
  3. 满意后按 [S] 保存当前帧的眼睛关键点参数

按键：
  [S/Space]  保存当前眼睛参数
  [M]        切换模板匹配模式
  [Q/Esc]    退出

依赖：mediapipe, opencv-python, numpy
"""

import os
import json
import time
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import RunningMode

from expression_match import (
    ensure_model, FaceEMA,
    L_EYE, R_EYE, L_IRIS, R_IRIS,
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, FLIP_HORIZONTAL,
)
from eye_scorer import calc_geometry_eye_score


# 画面裁剪：像素值，按需自行修改
CROP_LEFT_PX   = 650   # 左侧裁掉像素
CROP_RIGHT_PX  = 650   # 右侧裁掉像素
CROP_TOP_PX    = 60    # 上方裁掉像素
CROP_BOTTOM_PX = 300    # 下方裁掉像素
# CROP_LEFT_PX   = 0   # 左侧裁掉像素
# CROP_RIGHT_PX  = 0   # 右侧裁掉像素
# CROP_TOP_PX    = 0    # 上方裁掉像素
# CROP_BOTTOM_PX = 0    # 下方裁掉像素
# ============================================================
# 配置
# ============================================================
EYE_CHANNELS = [8, 9, 10, 11, 12, 13]
EYE_SHORT = ["A8", "A9", "A10", "A11", "A12", "A13"]

SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "neutral_eye_config.json")

# Colors (BGR)
C_GREEN = (0, 255, 0)
C_YELLOW = (0, 215, 255)
C_RED = (0, 0, 255)
C_CYAN = (255, 255, 0)
C_WHITE = (255, 255, 255)
C_GRAY = (128, 128, 128)
C_DARK = (40, 40, 40)
C_ORANGE = (0, 165, 255)
C_MAGENTA = (255, 0, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# 模板匹配：每个特征的容差范围（超出此范围得 0 分）
MATCH_TOLERANCE = {
    "openness": 0.08,    # 眼高/眼宽 的可接受偏差
    "h_position": 0.10,  # 水平比例可接受偏差 (0~1 范围, 0.5=中)
    "v_position": 0.10,  # 垂直比例可接受偏差 (0~1 范围, 0.5=中)
}

# ===================== 【新增：居中裁剪宽度】 =====================
CROP_WIDTH_RATIO = 0.65  # 只保留中间 65% 画面，裁掉两边
# =================================================================


def load_saved_template():
    if not os.path.exists(SAVE_PATH):
        return None
    try:
        with open(SAVE_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg.get("template")
    except Exception:
        return None


def calc_template_match(current, saved):
    if not current or not saved:
        return 0.0, {}

    keys = ["L_openness", "L_h_position", "L_v_position",
            "R_openness", "R_h_position", "R_v_position"]
    base_map = {}
    for k in keys:
        if k.endswith("_openness"):
            base_map[k] = "openness"
        elif k.endswith("_h_position"):
            base_map[k] = "h_position"
        elif k.endswith("_v_position"):
            base_map[k] = "v_position"

    scores = {}
    for k in keys:
        if k in current and k in saved:
            diff = abs(current[k] - saved[k])
            tol = MATCH_TOLERANCE.get(base_map.get(k, "openness"), 0.1)
            scores[k] = max(0.0, 1.0 - diff / tol) * 100.0
        else:
            scores[k] = 0.0

    overall = sum(scores.values()) / len(scores) if scores else 0.0
    return overall, scores


def draw_eye_points(frame, current_kps, w, h):
    if current_kps is None:
        return
    lms_px = (current_kps * np.array([w, h])).astype(np.int32)

    for idx in L_EYE:
        px, py = int(lms_px[idx, 0]), int(lms_px[idx, 1])
        cv2.circle(frame, (px, py), 2, (0, 255, 255), -1, cv2.LINE_AA)
    for idx in R_EYE:
        px, py = int(lms_px[idx, 0]), int(lms_px[idx, 1])
        cv2.circle(frame, (px, py), 2, (0, 255, 255), -1, cv2.LINE_AA)

    for idx in L_IRIS:
        px, py = int(lms_px[idx, 0]), int(lms_px[idx, 1])
        cv2.circle(frame, (px, py), 3, (0, 200, 0), -1, cv2.LINE_AA)
    for idx in R_IRIS:
        px, py = int(lms_px[idx, 0]), int(lms_px[idx, 1])
        cv2.circle(frame, (px, py), 3, (0, 200, 0), -1, cv2.LINE_AA)

    # 虹膜中心
    for center_idx in [468, 473]:
        px = int(current_kps[center_idx, 0] * w)
        py = int(current_kps[center_idx, 1] * h)
        cv2.circle(frame, (px, py), 4, (0, 0, 255), -1, cv2.LINE_AA)

    # ===== 标注运算用到的 10 个关键点 =====
    CORE_POINTS = {
        # 左眼
        33:  ("Lo", C_MAGENTA),   # L outer
        133: ("Li", C_MAGENTA),   # L inner
        159: ("Lt", C_ORANGE),    # L top
        145: ("Lb", C_ORANGE),    # L bottom
        468: ("Lp", C_WHITE),     # L pupil
        # 右眼
        263: ("Ro", C_MAGENTA),   # R outer
        362: ("Ri", C_MAGENTA),   # R inner
        386: ("Rt", C_ORANGE),    # R top
        374: ("Rb", C_ORANGE),    # R bottom
        473: ("Rp", C_WHITE),     # R pupil
    }
    for idx, (label, color) in CORE_POINTS.items():
        px = int(current_kps[idx, 0] * w)
        py = int(current_kps[idx, 1] * h)
        cv2.circle(frame, (px, py), 6, color, 2, cv2.LINE_AA)
        cv2.putText(frame, label, (px + 8, py + 4), FONT, 0.4, color, 1, cv2.LINE_AA)


def draw_overlay(frame, eye_score, detail, template, match_mode, match_score, match_detail, saved_template, w, h, save_flash):
    x0, y = 10, 30

    if eye_score > 0:
        sc = C_GREEN if eye_score >= 80 else C_YELLOW if eye_score >= 50 else C_RED
        cv2.putText(frame, f"Score: {eye_score:.1f}%", (x0, y), FONT, 0.9, sc, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, f"Score: --", (x0, y), FONT, 0.9, C_GRAY, 2, cv2.LINE_AA)
    y += 28

    if template:
        labels = [
            (f"L 开合度: {template['L_openness']:.3f}", C_CYAN),
            (f"L 水平:   {template['L_h_position']:.3f}  (0.5=中)", C_CYAN),
            (f"L 垂直:   {template['L_v_position']:.3f}  (0.5=中)", C_CYAN),
            (f"R 开合度: {template['R_openness']:.3f}", C_CYAN),
            (f"R 水平:   {template['R_h_position']:.3f}  (0.5=中)", C_CYAN),
            (f"R 垂直:   {template['R_v_position']:.3f}  (0.5=中)", C_CYAN),
        ]
        for text, color in labels:
            cv2.putText(frame, text, (x0, y), FONT, 0.45, color, 1, cv2.LINE_AA)
            y += 20
    y += 6

    if match_mode:
        if saved_template is None:
            cv2.putText(frame, "MATCH: No saved template", (x0, y), FONT, 0.55, C_RED, 1, cv2.LINE_AA)
            y += 22
        else:
            mc = C_GREEN if match_score >= 80 else C_YELLOW if match_score >= 50 else C_RED
            cv2.putText(frame, f"MATCH: {match_score:.1f}%", (x0, y), FONT, 0.75, mc, 2, cv2.LINE_AA)
            y += 26
            sub_names = [
                ("L_openness", "L_Opn"),
                ("L_h_position", "L_H"),
                ("L_v_position", "L_V"),
                ("R_openness", "R_Opn"),
                ("R_h_position", "R_H"),
                ("R_v_position", "R_V"),
            ]
            for key, short in sub_names:
                v = match_detail.get(key, 0)
                c = C_GREEN if v >= 70 else C_YELLOW if v >= 40 else C_RED
                cv2.putText(frame, f"  {short}: {v:.0f}%", (x0, y), FONT, 0.4, c, 1, cv2.LINE_AA)
                y += 16
    y += 6

    if detail:
        sub_labels = [
            ("L_Openness", "L_Opn"),
            ("R_Openness", "R_Opn"),
            ("L_Centered", "L_Cen"),
            ("R_Centered", "R_Cen"),
        ]
        for full, short in sub_labels:
            v = detail.get(full, 0)
            c = C_GREEN if v >= 70 else C_YELLOW if v >= 40 else C_RED
            cv2.putText(frame, f"  {short}: {v:.0f}%", (x0, y), FONT, 0.45, c, 1, cv2.LINE_AA)
            y += 18

    match_hint = "[M] Match:ON" if match_mode else "[M] Match:OFF"
    cv2.putText(frame, f"[S/Space] Save  {match_hint}  [Q/Esc] Quit", (x0, h - 15), FONT, 0.4, C_ORANGE if match_mode else C_GRAY, 1, cv2.LINE_AA)

    if save_flash > 0:
        cv2.putText(frame, "SAVED!", (w // 2 - 70, h // 2), FONT, 1.8, C_GREEN, 4, cv2.LINE_AA)


def extract_eye_template(kps):
    def _eye_features(outer, inner, top, bottom, iris_ctr):
        eye_width = outer[0] - inner[0]
        if abs(eye_width) < 1e-8:
            return {"openness": 0.0, "h_position": 0.5, "v_position": 0.5}

        eye_height = bottom[1] - top[1]
        openness = abs(eye_height / eye_width)
        h_position = (iris_ctr[0] - inner[0]) / eye_width

        if abs(eye_height) > 1e-8:
            v_position = (iris_ctr[1] - top[1]) / eye_height
        else:
            v_position = 0.5

        return {
            "openness": float(openness),
            "h_position": float(h_position),
            "v_position": float(v_position),
        }

    l = _eye_features(kps[33], kps[133], kps[159], kps[145], kps[468])
    r = _eye_features(kps[263], kps[362], kps[386], kps[374], kps[473])

    return {
        "L_openness": l["openness"],
        "L_h_position": l["h_position"],
        "L_v_position": l["v_position"],
        "R_openness": r["openness"],
        "R_h_position": r["h_position"],
        "R_v_position": r["v_position"],
    }


# 保存时多帧采样的数量
SAVE_SAMPLE_COUNT = 10


def save_config(current_kps, eye_score, detail, sample_buf=None):
    config = {
        "description": "中性脸眼睛几何模板",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "eye_score": float(eye_score),
    }

    if sample_buf is not None and len(sample_buf) > 0:
        # 多帧取均值后再提取模板，消除 EMA 帧间抖动
        avg_kps = np.mean(sample_buf, axis=0)
        template = extract_eye_template(avg_kps)
        config["sample_count"] = len(sample_buf)
    elif current_kps is not None:
        template = extract_eye_template(current_kps)
    else:
        template = None

    if template is not None:
        config["template"] = template

    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] 中性脸眼睛模板已保存 -> {SAVE_PATH}")
    print(f"[INFO] 得分: {eye_score:.1f}%")


def main():
    model_path = ensure_model()
    options = vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开摄像头 {CAMERA_INDEX}")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera: {actual_w}x{actual_h}")

    ema = FaceEMA(alpha=0.5, max_dis=3)
    timestamp = 0
    save_flash = 0
    match_mode = False
    saved_template = load_saved_template()
    # 采样缓冲区：保存时取最近 N 帧的 kps 做均值
    kps_sample_buf = []

    print("\n" + "="*45)
    print("  Manual Eye Calibrator (已居中裁剪)")
    print("="*45)
    print("  [S/Space]  保存当前眼睛参数")
    print("  [M]        切换模板匹配模式")
    print("  [Q/Esc]    退出")
    print("="*45)

    try:
        while True:
            ok, frame = cap.read()
            if not ok: continue
            if FLIP_HORIZONTAL: frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]

            # ====== 替换为精准像素裁剪代码 =====
            orig_h, orig_w = frame.shape[:2]
            # 边界防越界保护
            x1 = max(0, CROP_LEFT_PX)
            x2 = min(orig_w, orig_w - CROP_RIGHT_PX)
            y1 = max(0, CROP_TOP_PX)
            y2 = min(orig_h, orig_h - CROP_BOTTOM_PX)

            # 实际裁剪
            frame = frame[y1:y2, x1:x2]
            # 更新裁剪后画面尺寸
            h, w = frame.shape[:2]
            # ==========================================================================

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp += 33
            result = landmarker.detect_for_video(mp_img, timestamp)

            face_lms_list = result.face_landmarks or []
            if face_lms_list:
                ema.update(face_lms_list[0])
            else:
                ema.skip()

            eye_score = 0.0
            detail = {}
            template = {}
            match_score = 0.0
            match_detail = {}
            current_kps = None

            if ema.ok:
                current_kps = ema.norm_coords()
                if current_kps is not None:
                    eye_score, detail = calc_geometry_eye_score(current_kps)
                    template = extract_eye_template(current_kps)
                    if match_mode and saved_template:
                        match_score, match_detail = calc_template_match(template, saved_template)
                    draw_eye_points(frame, current_kps, w, h)
                    # 持续收集最近 SAVE_SAMPLE_COUNT 帧
                    kps_sample_buf.append(current_kps.copy())
                    if len(kps_sample_buf) > SAVE_SAMPLE_COUNT:
                        kps_sample_buf.pop(0)

            if save_flash > 0: save_flash -= 1

            draw_overlay(frame, eye_score, detail, template, match_mode, match_score, match_detail, saved_template, w, h, save_flash)
            cv2.imshow("Eye Calibrator (Cropped)", frame)

            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
            elif key in (ord('s'), ord('S'), ord(' ')):
                if current_kps is not None:
                    samples = kps_sample_buf if len(kps_sample_buf) > 0 else None
                    save_config(current_kps, eye_score, detail, sample_buf=samples)
                    saved_template = load_saved_template()
                    kps_sample_buf.clear()
                    save_flash = 20
                else:
                    print("[WARN] 未检测到人脸，无法保存")
            elif key in (ord('m'), ord('M')):
                match_mode = not match_mode
                saved_template = load_saved_template()
                state = "ON" if match_mode else "OFF"
                print(f"[INFO] 模板匹配模式: {state}")

    finally:
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] 已退出")


if __name__ == "__main__":
    main()