"""
Expression Match Score — 表情区域匹配度检测
=============================================
流程：
  1. 按 [S/空格] 捕获当前表情为"标准脸"(standard_kps)
  2. 每帧实时计算当前表情与标准脸的区域差异
  3. 显示各区域匹配度百分比 + 不匹配区域高亮

按键：
  [S/Space] 保存当前表情为标准脸
  [C]       清除标准脸
  [q/Esc]   退出

依赖：mediapipe>=0.10.35, opencv-python, numpy
"""

import os
import urllib.request
import numpy as np
import json
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import RunningMode

# ============================================================
# Config
# ============================================================
CAMERA_INDEX = 0
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
FLIP_HORIZONTAL = True
EMA_ALPHA = 0.35
MAX_DISAPPEAR = 5
NUM_FACES = 1
SIDEBAR_WIDTH = 300

# 不匹配阈值（归一化坐标下的平均欧氏距离）
WARN_THRESHOLD = 0.015
ALERT_THRESHOLD = 0.030

# 区域权重（用于总分计算）
REGION_WEIGHTS = {
    "L Eye":  0.15,
    "R Eye":  0.15,
    "L Iris": 0.075,
    "R Iris": 0.075,
    "L Brow": 0.10,
    "R Brow": 0.10,
    "Mouth":  0.30,
    "Face":   0.05,
}

# Colors (BGR)
C_GREEN  = (0, 255, 0)
C_YELLOW = (0, 215, 255)
C_RED    = (0, 0, 255)
C_WHITE  = (255, 255, 255)
C_CYAN   = (255, 255, 0)
C_GRAY   = (128, 128, 128)
FONT     = cv2.FONT_HERSHEY_SIMPLEX

# ============================================================
# MediaPipe 区域关键点索引 (478-point face mesh)
# ============================================================
L_EYE  = [33, 246, 161, 160, 159, 158, 157, 173,
           133, 155, 154, 153, 145, 144, 163, 7]
R_EYE  = [362, 398, 384, 385, 386, 387, 388, 466,
           263, 249, 390, 373, 374, 380, 381, 382]
L_IRIS = [469, 470, 471, 472]
R_IRIS = [474, 475, 476, 477]
L_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
R_BROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
MOUTH  = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
           291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
           78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
           308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
             361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
             176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
             162, 21, 54, 103, 67, 109]

REGIONS = {
    "L Eye":  L_EYE,
    "R Eye":  R_EYE,
    "L Iris": L_IRIS,
    "R Iris": R_IRIS,
    "L Brow": L_BROW,
    "R Brow": R_BROW,
    "Mouth":  MOUTH,
    "Face":   FACE_OVAL,
}

# 归一化基准点：鼻尖(1)、左眼外角(33)、右眼外角(263)
NORM_PTS = [1, 33, 263]

# ============================================================
# Model
# ============================================================
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/"
    "face_landmarker.task"
)


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)
        print("[INFO] Downloading face_landmarker.task ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


# ============================================================
# Standard face persistence
# ============================================================
STANDARD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "standard_face.json")


def save_standard(kps):
    """Save standard keypoints to JSON."""
    data = {"keypoints": kps.tolist(), "norm_pts": NORM_PTS}
    with open(STANDARD_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f)
    print(f"[INFO] Standard face saved → {STANDARD_PATH}")


def load_standard():
    """Load standard keypoints from JSON. Returns None if not found."""
    if not os.path.exists(STANDARD_PATH):
        return None
    try:
        with open(STANDARD_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        kps = np.array(data["keypoints"], dtype=np.float64)
        print(f"[INFO] Standard face loaded ← {STANDARD_PATH} ({kps.shape[0]} keypoints)")
        return kps
    except Exception as e:
        print(f"[WARN] Failed to load standard face: {e}")
        return None


def delete_standard():
    """Delete saved standard face file."""
    if os.path.exists(STANDARD_PATH):
        os.remove(STANDARD_PATH)
        print(f"[INFO] Standard face file deleted: {STANDARD_PATH}")


# ============================================================
# EMA Smoother
# ============================================================
class FaceEMA:
    def __init__(self, alpha=EMA_ALPHA, max_dis=MAX_DISAPPEAR):
        self.alpha = alpha
        self.max_dis = max_dis
        self._lms = None
        self._miss = 0

    @property
    def ok(self):
        return self._lms is not None and self._miss <= self.max_dis

    def update(self, landmarks):
        cur = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float64)
        if self._lms is None:
            self._lms = cur
        else:
            self._lms = self.alpha * cur + (1 - self.alpha) * self._lms
        self._miss = 0

    def skip(self):
        self._miss += 1

    def norm_coords(self):
        """Return normalized (centered + scaled) landmark array."""
        if not self.ok:
            return None
        kps = self._lms.copy()
        ref = kps[NORM_PTS]
        center = ref[0]
        scale = np.linalg.norm(ref[1] - ref[2]) + 1e-8
        kps = (kps - center) / scale
        return kps

    def pixel_coords(self, w, h):
        """Return pixel landmark array."""
        if not self.ok:
            return None
        return (self._lms * np.array([w, h], dtype=np.float64)).astype(np.int32)


# ============================================================
# IoU Tracker
# ============================================================
def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(b[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0
    aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    bb = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / (aa + bb - inter + 1e-8)


def update_trackers(trackers, face_lms_list, w, h):
    nd, nt = len(face_lms_list), len(trackers)
    if nd == 0 and nt == 0:
        return []
    if nd == 0:
        for t in trackers:
            t.skip()
        trackers[:] = [t for t in trackers if t.ok]
        return list(trackers)

    dets = []
    for lms in face_lms_list:
        xs = [lm.x for lm in lms]
        ys = [lm.y for lm in lms]
        dets.append((min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)))

    iou_mat = np.zeros((nd, nt), dtype=np.float64)
    for i, (bx, by, bw, bh) in enumerate(dets):
        r1 = (bx * w, by * h, (bx + bw) * w, (by + bh) * h)
        for j, t in enumerate(trackers):
            if not t.ok:
                continue
            kps = t._lms
            r2 = (float(kps[:, 0].min()) * w, float(kps[:, 1].min()) * h,
                  float(kps[:, 0].max()) * w, float(kps[:, 1].max()) * h)
            iou_mat[i, j] = _iou(r1, r2)

    md, mt = set(), set()
    pairs = sorted(
        [(iou_mat[i, j], i, j) for i in range(nd) for j in range(nt)
         if iou_mat[i, j] > 0.35], reverse=True)
    for _, i, j in pairs:
        if i in md or j in mt:
            continue
        trackers[j].update(face_lms_list[i])
        md.add(i)
        mt.add(j)

    for i in range(nd):
        if i not in md:
            t = FaceEMA()
            t.update(face_lms_list[i])
            trackers.append(t)

    for j in range(nt):
        if j not in mt:
            trackers[j].skip()

    trackers[:] = [t for t in trackers if t.ok]
    return list(trackers)


# ============================================================
# 核心：区域差异计算
# ============================================================
def calc_region_diff(standard_kps, current_kps, region_indices):
    """计算指定区域关键点的平均欧氏距离（归一化坐标）。"""
    std_pts = standard_kps[list(region_indices)]
    cur_pts = current_kps[list(region_indices)]
    diffs = np.linalg.norm(std_pts - cur_pts, axis=1)
    return float(np.mean(diffs))


def calc_all_regions(standard_kps, current_kps):
    """计算所有区域的差异。"""
    return {name: calc_region_diff(standard_kps, current_kps, indices)
            for name, indices in REGIONS.items()}


def diff_to_score(diff, warn=WARN_THRESHOLD, alert=ALERT_THRESHOLD):
    """将差异值转换为 0~100 匹配度。"""
    if diff <= warn:
        return 100.0 * (1.0 - diff / warn * 0.2)
    elif diff <= alert:
        t = (diff - warn) / (alert - warn)
        return 80.0 * (1.0 - t) + 20.0
    else:
        return max(0.0, 20.0 * (1.0 - (diff - alert) / alert))


def calc_total_score(region_diffs):
    """加权总分。"""
    total = 0.0
    for name, weight in REGION_WEIGHTS.items():
        score = diff_to_score(region_diffs.get(name, 0))
        total += weight * score
    return total


def diff_color(diff):
    if diff <= WARN_THRESHOLD:
        return C_GREEN
    elif diff <= ALERT_THRESHOLD:
        return C_YELLOW
    return C_RED


def score_color(score):
    if score >= 80:
        return C_GREEN
    elif score >= 50:
        return C_YELLOW
    return C_RED


# ============================================================
# Drawing
# ============================================================
def draw_region_highlights(image, standard_kps, current_kps, lms_px):
    """高亮不匹配区域的关键点（像素坐标）。"""
    for name, indices in REGIONS.items():
        diff = calc_region_diff(standard_kps, current_kps, indices)
        if diff <= WARN_THRESHOLD:
            continue
        color = diff_color(diff)
        for idx in indices:
            px, py = int(lms_px[idx, 0]), int(lms_px[idx, 1])
            r = 3 if diff > ALERT_THRESHOLD else 2
            cv2.circle(image, (px, py), r, color, -1, cv2.LINE_AA)


def draw_sidebar(canvas, region_diffs, total_score, has_standard, offset_x):
    """右侧信息面板。"""
    px = offset_x + 8
    y = 6

    # 标题
    cv2.rectangle(canvas, (offset_x, 0),
                  (offset_x + SIDEBAR_WIDTH, 30), (30, 30, 30), -1)
    cv2.putText(canvas, "Expression Match", (px, y + 20),
                FONT, 0.55, C_CYAN, 1, cv2.LINE_AA)
    y += 36

    if has_standard:
        # 总分
        sc = score_color(total_score)
        cv2.putText(canvas, f"Total: {total_score:.0f}%", (px, y + 16),
                    FONT, 0.7, sc, 2, cv2.LINE_AA)
        y += 30

        # 各区域条形图
        cv2.rectangle(canvas, (offset_x, y - 2),
                      (offset_x + SIDEBAR_WIDTH, y + len(REGIONS) * 24 + 4),
                      (25, 25, 25), -1)
        bar_w = 100
        for name, diff in region_diffs.items():
            score = diff_to_score(diff)
            sc = score_color(score)

            cv2.putText(canvas, f"{name}", (px, y + 14),
                        FONT, 0.38, C_WHITE, 1, cv2.LINE_AA)

            bx = px + 75
            by = y + 3
            bh = 12
            cv2.rectangle(canvas, (bx, by), (bx + bar_w, by + bh),
                          (60, 60, 60), -1)
            fill_w = int(bar_w * score / 100)
            if fill_w > 0:
                cv2.rectangle(canvas, (bx, by), (bx + fill_w, by + bh), sc, -1)
            cv2.putText(canvas, f"{score:.0f}%", (bx + bar_w + 4, by + 10),
                        FONT, 0.32, sc, 1, cv2.LINE_AA)
            y += 24

        # 提示
        y += 8
        cv2.putText(canvas, "[S] Re-capture standard", (px, y + 12),
                    FONT, 0.33, C_GRAY, 1, cv2.LINE_AA)
        y += 16
        cv2.putText(canvas, "[C] Clear standard", (px, y + 12),
                    FONT, 0.33, C_GRAY, 1, cv2.LINE_AA)
    else:
        cv2.putText(canvas, "No standard face", (px, y + 16),
                    FONT, 0.5, C_RED, 1, cv2.LINE_AA)
        y += 30
        cv2.putText(canvas, "Press [S] to capture", (px, y + 14),
                    FONT, 0.4, C_GRAY, 1, cv2.LINE_AA)
        y += 20
        cv2.putText(canvas, "standard neutral face", (px, y + 14),
                    FONT, 0.4, C_GRAY, 1, cv2.LINE_AA)


# ============================================================
# Main
# ============================================================
def main():
    ensure_model()

    options = vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_faces=NUM_FACES,
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
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera: {actual_w}x{actual_h}")

    print()
    print("=" * 50)
    print("  Expression Match Score")
    print("=" * 50)
    print("  [S/Space] Capture standard face")
    print("  [C]       Clear standard")
    print("  [q/Esc]   Quit")
    print("=" * 50)

    trackers = []
    timestamp = 0
    standard_kps = load_standard()  # 启动时自动加载已保存的标准脸

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            if FLIP_HORIZONTAL:
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp += 33
            result = landmarker.detect_for_video(mp_img, timestamp)

            face_lms_list = result.face_landmarks or []
            active = update_trackers(trackers, face_lms_list, w, h)

            current_kps = None
            lms_px = None
            if active:
                current_kps = active[0].norm_coords()
                lms_px = active[0].pixel_coords(w, h)

            region_diffs = {}
            total_score = 0.0
            if standard_kps is not None and current_kps is not None and lms_px is not None:
                region_diffs = calc_all_regions(standard_kps, current_kps)
                total_score = calc_total_score(region_diffs)
                draw_region_highlights(frame, standard_kps, current_kps, lms_px)

            # 左上角总分
            if standard_kps is not None and current_kps is not None:
                sc = score_color(total_score)
                cv2.putText(frame, f"{total_score:.0f}%", (10, 36),
                            FONT, 1.0, sc, 2, cv2.LINE_AA)

            # 状态提示
            status = "STANDARD SET" if standard_kps is not None else "NO STANDARD"
            status_color = C_GREEN if standard_kps is not None else C_RED
            cv2.putText(frame, status, (10, 62),
                        FONT, 0.5, status_color, 1, cv2.LINE_AA)
            cv2.putText(frame, f"Faces: {len(active)}", (w - 120, 28),
                        FONT, 0.55, C_WHITE, 1, cv2.LINE_AA)

            # 侧边栏
            canvas = cv2.copyMakeBorder(
                frame, 0, 0, 0, SIDEBAR_WIDTH,
                borderType=cv2.BORDER_CONSTANT, value=(20, 20, 20))
            draw_sidebar(canvas, region_diffs, total_score,
                         standard_kps is not None, w)

            cv2.imshow("Expression Match", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key in (ord("s"), ord("S"), ord(" ")):
                if current_kps is not None:
                    standard_kps = current_kps.copy()
                    save_standard(standard_kps)
                else:
                    print("[WARN] No face detected - cannot capture standard")
            elif key in (ord("c"), ord("C")):
                standard_kps = None
                delete_standard()

    finally:
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Exited")


if __name__ == "__main__":
    main()
