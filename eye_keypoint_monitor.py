"""
Eye Keypoint Monitor — 抖动检测 + 模板记录 + 实时误差
======================================================
[R]   Record: 取最近 10 帧均值记录为参考模板
[M]   Toggle: 切换误差显示模式
[Q/Esc] Quit

依赖：mediapipe, opencv-python, numpy
"""

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import RunningMode

from expression_match import (
    ensure_model, FaceEMA,
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, FLIP_HORIZONTAL,
)

# 10 个核心关键点
CORE_INDICES = {
    33:  "L_outer",
    133: "L_inner",
    159: "L_top",
    145: "L_bottom",
    468: "L_iris",
    263: "R_outer",
    362: "R_inner",
    386: "R_top",
    374: "R_bottom",
    473: "R_iris",
}

INDEX_LIST = list(CORE_INDICES.keys())
NAME_LIST = list(CORE_INDICES.values())

# 滑动窗口大小 (帧数)
WINDOW_SIZE = 30
# 采样帧数 (记录模板时取最近 N 帧均值)
SAMPLE_COUNT = 10

# 画面裁剪：像素值，按需自行修改
CROP_LEFT_PX   = 650   # 左侧裁掉像素
CROP_RIGHT_PX  = 650   # 右侧裁掉像素
CROP_TOP_PX    = 60    # 上方裁掉像素
CROP_BOTTOM_PX = 300   # 下方裁掉像素

# 抖动等级阈值 (std)
JITTER_GOOD = 0.002
JITTER_OK   = 0.005

# Colors (BGR)
C_GREEN  = (0, 255, 0)
C_YELLOW = (0, 215, 255)
C_RED    = (0, 0, 255)
C_CYAN   = (255, 255, 0)
C_WHITE  = (255, 255, 255)
C_GRAY   = (128, 128, 128)
C_ORANGE = (0, 165, 255)
C_MAGENTA = (255, 0, 255)
FONT     = cv2.FONT_HERSHEY_SIMPLEX


def jitter_color(v):
    if v < JITTER_GOOD:
        return C_GREEN
    elif v < JITTER_OK:
        return C_YELLOW
    return C_RED


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
        print(f"[ERROR] Cannot open camera {CAMERA_INDEX}")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    ema = FaceEMA(alpha=0.5, max_dis=3)
    timestamp = 0

    # 滑动窗口: (N, 10, 2)
    buf = np.zeros((0, len(INDEX_LIST), 2), dtype=np.float64)
    # 采样缓冲区: 最近 N 帧完整 kps
    kps_buf = []
    # 记录的参考: 10 个点的 (10, 2) 均值坐标
    ref_kps = None
    # 误差显示模式
    show_error = True

    print("\n" + "=" * 55)
    print("  Eye Keypoint Monitor")
    print("  [R] Record template  [M] Toggle error  [Q] Quit")
    print("=" * 55)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            if FLIP_HORIZONTAL:
                frame = cv2.flip(frame, 1)

            # ---- 裁剪 ----
            fh, fw = frame.shape[:2]
            y1 = min(CROP_TOP_PX, fh)
            y2 = max(fh - CROP_BOTTOM_PX, y1)
            x1 = min(CROP_LEFT_PX, fw)
            x2 = max(fw - CROP_RIGHT_PX, x1)
            frame = frame[y1:y2, x1:x2]

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp += 33
            result = landmarker.detect_for_video(mp_img, timestamp)

            face_lms = result.face_landmarks or []
            if face_lms:
                ema.update(face_lms[0])
            else:
                ema.skip()

            if ema.ok:
                raw = ema._lms  # MediaPipe 原始坐标 (0~1)
                kps = ema.norm_coords()
                if kps is not None:
                    # ---- 抖动统计 ----
                    cur = kps[INDEX_LIST, :2].reshape(1, len(INDEX_LIST), 2)
                    buf = np.concatenate([buf, cur], axis=0)
                    if len(buf) > WINDOW_SIZE:
                        buf = buf[-WINDOW_SIZE:]

                    if len(buf) >= 5:
                        stds = buf.std(axis=0)
                        jitter = np.sqrt((stds ** 2).sum(axis=1))
                    else:
                        stds = np.zeros((len(INDEX_LIST), 2))
                        jitter = np.zeros(len(INDEX_LIST))

                    # ---- 采样缓冲区持续积累 ----
                    kps_buf.append(kps[INDEX_LIST, :2].copy())  # 只存 10 个点
                    if len(kps_buf) > SAMPLE_COUNT:
                        kps_buf.pop(0)

                    # ---- 画点 ----
                    for i, (idx, name) in enumerate(CORE_INDICES.items()):
                        px = int(kps[idx, 0] * w)
                        py = int(kps[idx, 1] * h)
                        c = jitter_color(jitter[i])
                        cv2.circle(frame, (px, py), 4, c, -1, cv2.LINE_AA)

                    # ---- HUD ----
                    ty = 22

                    # 左栏: 原始坐标 (MediaPipe 0~1)
                    cv2.putText(frame, "Raw Coords (MP)", (10, ty),
                                FONT, 0.45, C_CYAN, 1, cv2.LINE_AA)
                    ty += 18
                    cv2.putText(frame, f"{'Key':<9} {'X':>8} {'Y':>8}", (10, ty),
                                FONT, 0.28, C_GRAY, 1, cv2.LINE_AA)
                    ty += 14
                    for i, (idx, name) in enumerate(CORE_INDICES.items()):
                        rx = raw[idx, 0]
                        ry_v = raw[idx, 1]
                        cv2.putText(frame, f"{name:<9} {rx:>8.4f} {ry_v:>8.4f}", (10, ty),
                                    FONT, 0.28, C_WHITE, 1, cv2.LINE_AA)
                        ty += 14
                    ty += 6

                    # 中栏: 抖动
                    cv2.putText(frame, "Jitter", (10, ty),
                                FONT, 0.4, C_WHITE, 1, cv2.LINE_AA)
                    ty += 16
                    for i, name in enumerate(NAME_LIST):
                        j = jitter[i]
                        c = jitter_color(j)
                        cv2.putText(frame, f"{name:<9} {j:.4f}", (10, ty),
                                    FONT, 0.28, c, 1, cv2.LINE_AA)
                        ty += 14
                    avg_j = np.mean(jitter) if len(jitter) else 0
                    max_j = np.max(jitter) if len(jitter) else 0
                    c = jitter_color(max_j)
                    cv2.putText(frame, f"Avg:{avg_j:.4f} Max:{max_j:.4f}", (10, ty),
                                FONT, 0.32, c, 1, cv2.LINE_AA)
                    ty += 16

                    # 右栏: 10 个点位偏差
                    if show_error and ref_kps is not None:
                        rx = 300
                        ry = 22
                        cv2.putText(frame, "Error (cur - ref)", (rx, ry),
                                    FONT, 0.45, C_MAGENTA, 1, cv2.LINE_AA)
                        ry += 22

                        # 表头
                        cv2.putText(frame, f"{'Key':<9} {'dX':>8} {'dY':>8} {'dist':>8}", (rx, ry),
                                    FONT, 0.3, C_GRAY, 1, cv2.LINE_AA)
                        ry += 16
                        cv2.line(frame, (rx, ry), (rx + 300, ry), C_GRAY, 1, cv2.LINE_AA)
                        ry += 4

                        max_dist = 0.0
                        for i, name in enumerate(NAME_LIST):
                            dx = kps[INDEX_LIST[i], 0] - ref_kps[i, 0]
                            dy = kps[INDEX_LIST[i], 1] - ref_kps[i, 1]
                            dist = np.sqrt(dx * dx + dy * dy)
                            max_dist = max(max_dist, dist)
                            # 着色: <0.005绿, <0.015黄, >=0.015红
                            if dist < 0.005:
                                dc = C_GREEN
                            elif dist < 0.015:
                                dc = C_YELLOW
                            else:
                                dc = C_RED
                            cv2.putText(frame, f"{name:<9} {dx:>+8.4f} {dy:>+8.4f} {dist:>8.4f}",
                                        (rx, ry), FONT, 0.3, dc, 1, cv2.LINE_AA)
                            ry += 16

                        ry += 4
                        if max_dist < 0.005:
                            level, lc = "MATCHED", C_GREEN
                        elif max_dist < 0.015:
                            level, lc = "CLOSE", C_YELLOW
                        else:
                            level, lc = "OFFSET", C_RED
                        cv2.putText(frame, f"Max: {max_dist:.4f}  {level}", (rx, ry),
                                    FONT, 0.4, lc, 1, cv2.LINE_AA)

                    elif show_error and ref_kps is None:
                        cv2.putText(frame, "Press [R] to record ref", (320, 22),
                                    FONT, 0.45, C_GRAY, 1, cv2.LINE_AA)

                    # 底部提示
                    status = "REF:SET" if ref_kps is not None else "REF:NONE"
                    err_s = "ERR:ON" if show_error else "ERR:OFF"
                    cv2.putText(frame, f"[R]Record  [M]{err_s}  {status}  [Q]Quit",
                                (10, h - 12), FONT, 0.35, C_ORANGE, 1, cv2.LINE_AA)

            cv2.imshow("Eye Monitor", frame)

            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
            elif key in (ord('r'), ord('R')):
                if len(kps_buf) >= 3:
                    ref_kps = np.mean(kps_buf, axis=0)  # (10, 2)
                    print(f"\n[RECORD] Ref saved from {len(kps_buf)} frames:")
                    for i, name in enumerate(NAME_LIST):
                        print(f"  {name:<9} X={ref_kps[i,0]:+.5f}  Y={ref_kps[i,1]:+.5f}")
                else:
                    print("[WARN] Need at least 3 frames, wait a moment...")
            elif key in (ord('m'), ord('M')):
                show_error = not show_error
                print(f"[INFO] Error display: {'ON' if show_error else 'OFF'}")

    finally:
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Exited")


if __name__ == "__main__":
    main()
