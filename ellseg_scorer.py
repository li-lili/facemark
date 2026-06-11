"""
EllSeg Scorer — 基线标定 + 虹膜偏移检测 → 舵机引导
======================================================
核心逻辑：
  1. 标定基线：保存鼻子+左右虹膜中心像素位置 → face_points.json
  2. 偏移检测：实时对比当前帧与基线 (TOLERANCE=2px)
  3. 通过判定：规定帧数内，≥50% 帧合格(qualified)则通过
  4. 舵机引导：偏移方向反馈给舵机，指导眼球调整
"""

import os, sys, subprocess, urllib.request, cv2, numpy as np, json, torch, threading, time

# ============================================================
# 配置
# ============================================================
DIR = os.path.dirname(os.path.abspath(__file__))
ELLSEG_REPO = os.path.join(DIR, "EllSeg")
WEIGHTS_PATH = os.path.join(ELLSEG_REPO, "weights", "all.git_ok")
FACE_POINTS_FILE = os.path.join(DIR, "face_points.json")
EYELID_BASELINE_FILE = os.path.join(DIR, "eyelid_baseline.json")
EYEBROW_BASELINE_FILE = os.path.join(DIR, "eyebrow_baseline.json")
MOUTH_BASELINE_FILE = os.path.join(DIR, "mouth_baseline.json")
LOWER_LIP_BASELINE_FILE = os.path.join(DIR, "lower_lip_baseline.json")
UPPER_LIP_BASELINE_FILE = os.path.join(DIR, "upper_lip_baseline.json")
MOUTH_CORNERS_BASELINE_FILE = os.path.join(DIR, "mouth_corners_baseline.json")

from eye_constants import EYEBALL_OFFSET_TOLERANCE
TOLERANCE = EYEBALL_OFFSET_TOLERANCE        # 合格阈值 (像素)
EMA_LM = 0.35         # MediaPipe 关键点 EMA
EMA_ELL = 0.3         # EllSeg 椭圆 EMA
EYE_PAD_X = 0.5       # 眼部裁剪水平扩充
EYE_PAD_Y = 0.8       # 眼部裁剪垂直扩充

LEFT_EYE_IDX = [33, 246, 161, 160, 159, 158, 157, 173,
                133, 155, 154, 153, 145, 144, 163, 7]
RIGHT_EYE_IDX = [362, 398, 384, 385, 386, 387, 388, 466,
                 263, 249, 390, 373, 374, 380, 381, 382]
NOSE_IDX = 1

# ============================================================
# 自动安装依赖
# ============================================================
if not os.path.exists(ELLSEG_REPO):
    print("[EllSeg] Cloning...")
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/RSKothari/EllSeg.git", ELLSEG_REPO], check=True)
if not os.path.exists(WEIGHTS_PATH):
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    print("[EllSeg] Downloading weights...")
    urllib.request.urlretrieve(
        "https://github.com/RSKothari/EllSeg/raw/refs/heads/master/weights/all.git_ok",
        WEIGHTS_PATH)
sys.path.insert(0, ELLSEG_REPO)

MP_DIR = os.path.join(DIR, "models")
MP_MODEL_PATH = os.path.join(MP_DIR, "face_landmarker.task")
if not os.path.exists(MP_MODEL_PATH):
    os.makedirs(MP_DIR, exist_ok=True)
    print("[MediaPipe] Downloading model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
        MP_MODEL_PATH)

from modelSummary import model_dict
from utils import get_predictions
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import RunningMode

# 全局单例模型
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[EllSeg] Using device: {_DEVICE}")
_net = torch.load(WEIGHTS_PATH, map_location=_DEVICE, weights_only=False)
ellseg_model = model_dict["ritnet_v3"]
ellseg_model.load_state_dict(_net["state_dict"], strict=True)
ellseg_model.to(_DEVICE)
ellseg_model.eval()
print("[EllSeg] Model loaded")


# ============================================================
# EMA 平滑器
# ============================================================
class LMASmoother:
    def __init__(self, alpha=EMA_LM):
        self.alpha = alpha
        self._v = None

    def update(self, lms):
        if self._v is None:
            self._v = lms.copy()
        else:
            self._v = self.alpha * lms + (1 - self.alpha) * self._v
        return self._v.copy()

    def reset(self):
        self._v = None


class EllipseSmoother:
    """EMA 平滑椭圆参数 (含角度环绕)"""
    def __init__(self, alpha=EMA_ELL):
        self.alpha = alpha
        self._v = None

    def update(self, ellipse):
        if ellipse is None:
            return None
        (cx, cy), (aw, ah), ang = ellipse
        cur = np.array([cx, cy, aw, ah, ang], dtype=np.float64)
        if self._v is None:
            self._v = cur
        else:
            prev_ang = self._v[4]
            diff = ang - prev_ang
            if diff > 90:   ang -= 180
            elif diff < -90: ang += 180
            cur[4] = ang
            a = self.alpha
            self._v = a * cur + (1 - a) * self._v
            self._v[4] = self._v[4] % 360
        v = self._v
        return ((v[0], v[1]), (v[2], v[3]), v[4])

    def reset(self):
        self._v = None


# ============================================================
# 工具函数
# ============================================================
def _init_mediapipe():
    opts = vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MP_MODEL_PATH),
        running_mode=RunningMode.VIDEO, num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return vision.FaceLandmarker.create_from_options(opts)


def crop_eye(frame, lms_px, eye_idx):
    pts = lms_px[eye_idx]
    x1, y1 = pts.min(axis=0).astype(int)
    x2, y2 = pts.max(axis=0).astype(int)
    w, h = x2 - x1, y2 - y1
    pad_x = int(w * EYE_PAD_X)
    pad_y = int(h * EYE_PAD_Y)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(frame.shape[1], x2 + pad_x)
    y2 = min(frame.shape[0], y2 + pad_y)
    return frame[y1:y2, x1:x2], (x1, y1)


def run_ellseg(eye_gray):
    h, w = eye_gray.shape
    scale = 320 / w
    new_h = int(h * scale)
    resized = cv2.resize(eye_gray, (320, new_h), interpolation=cv2.INTER_LANCZOS4)

    pad = 0
    if new_h < 240:
        pad = 240 - new_h
        resized = np.pad(resized, ((pad // 2, pad - pad // 2), (0, 0)))
    elif new_h > 240:
        resized = resized[(new_h - 240) // 2:(new_h - 240) // 2 + 240, :]

    img = (resized.astype(np.float64) - resized.mean()) / (resized.std() + 1e-8)
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(_DEVICE)

    with torch.no_grad():
        x4, x3, x2, x1, x = ellseg_model.enc(tensor)
        seg = get_predictions(ellseg_model.dec(x4, x3, x2, x1, x))

    seg_map = seg.squeeze().cpu().numpy()
    result = {}
    for name, cls_id in [("iris", 1), ("pupil", 2)]:
        mask = (seg_map == cls_id).astype(np.uint8)
        pts = cv2.findNonZero(mask)
        if pts is None or len(pts) < 5:
            result[name] = None
            continue
        (cx, cy), (aw, ah), ang = cv2.fitEllipse(pts)
        cy -= pad // 2
        cx /= scale; cy /= scale; aw /= scale; ah /= scale
        result[name] = ((cx, cy), (aw, ah), ang)
    return result


# ============================================================
# 基线管理
# ============================================================
def save_baseline(nose_pos, left_iris_pos, right_iris_pos, filepath=None):
    if filepath is None:
        filepath = FACE_POINTS_FILE
    data = {
        "nose": list(nose_pos),
        "iris_left": list(left_iris_pos),
        "iris_right": list(right_iris_pos),
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[Baseline] Saved: nose={nose_pos} L={left_iris_pos} R={right_iris_pos}")


def load_baseline(filepath=None):
    if filepath is None:
        filepath = FACE_POINTS_FILE
    if not os.path.isfile(filepath):
        return None
    try:
        with open(filepath, encoding="utf-8") as f:
            d = json.load(f)
        return {
            "nose": tuple(d["nose"]),
            "iris_left": tuple(d["iris_left"]),
            "iris_right": tuple(d["iris_right"]),
        }
    except Exception as e:
        print(f"[Baseline] Load failed: {e}")
        return None


# ============================================================
# 眼皮基线管理
# ============================================================
def save_eyelid_baseline(left_ear, right_ear, filepath=None):
    if filepath is None:
        filepath = EYELID_BASELINE_FILE
    data = {
        "left_ear": float(left_ear),
        "right_ear": float(right_ear),
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[Eyelid Baseline] Saved: L_EAR={left_ear:.4f} R_EAR={right_ear:.4f}")


def load_eyelid_baseline(filepath=None):
    if filepath is None:
        filepath = EYELID_BASELINE_FILE
    if not os.path.isfile(filepath):
        return None
    try:
        with open(filepath, encoding="utf-8") as f:
            d = json.load(f)
        return {
            "left_ear": float(d["left_ear"]),
            "right_ear": float(d["right_ear"]),
        }
    except Exception as e:
        print(f"[Eyelid Baseline] Load failed: {e}")
        return None


# ============================================================
# 眉毛基线管理
# ============================================================
def save_eyebrow_baseline(left_ebhr, right_ebhr, filepath=None):
    if filepath is None:
        filepath = EYEBROW_BASELINE_FILE
    data = {"left_ebhr": float(left_ebhr), "right_ebhr": float(right_ebhr)}
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[Eyebrow Baseline] Saved: L_EBHR={left_ebhr:.4f} R_EBHR={right_ebhr:.4f}")


def load_eyebrow_baseline(filepath=None):
    if filepath is None:
        filepath = EYEBROW_BASELINE_FILE
    if not os.path.isfile(filepath):
        return None
    try:
        with open(filepath, encoding="utf-8") as f:
            d = json.load(f)
        return {"left_ebhr": float(d["left_ebhr"]), "right_ebhr": float(d["right_ebhr"])}
    except Exception as e:
        print(f"[Eyebrow Baseline] Load failed: {e}")
        return None


# ============================================================
# 嘴部基线管理
# ============================================================
def save_mouth_baseline(mar, filepath=None):
    if filepath is None:
        filepath = MOUTH_BASELINE_FILE
    data = {"mar": float(mar)}
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[Mouth Baseline] Saved: MAR={mar:.4f}")


def load_mouth_baseline(filepath=None):
    if filepath is None:
        filepath = MOUTH_BASELINE_FILE
    if not os.path.isfile(filepath):
        return None
    try:
        with open(filepath, encoding="utf-8") as f:
            d = json.load(f)
        return {"mar": float(d["mar"])}
    except Exception as e:
        print(f"[Mouth Baseline] Load failed: {e}")
        return None


# ============================================================
# 下唇基线管理
# ============================================================
def save_lower_lip_baseline(llr, filepath=None):
    if filepath is None:
        filepath = LOWER_LIP_BASELINE_FILE
    data = {"llr": float(llr)}
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[LowerLip Baseline] Saved: LLR={llr:.4f}")

def load_lower_lip_baseline(filepath=None):
    if filepath is None:
        filepath = LOWER_LIP_BASELINE_FILE
    if not os.path.isfile(filepath):
        return None
    try:
        with open(filepath, encoding="utf-8") as f:
            d = json.load(f)
        # 兼容旧格式 {llt, llp}
        if "llr" in d:
            return {"llr": float(d["llr"])}
        else:
            return None
    except Exception as e:
        print(f"[LowerLip Baseline] Load failed: {e}")
        return None


# ============================================================
# 上唇基线管理
# ============================================================
def save_upper_lip_baseline(ulr, filepath=None):
    if filepath is None:
        filepath = UPPER_LIP_BASELINE_FILE
    data = {"ulr": float(ulr)}
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[UpperLip Baseline] Saved: ULR={ulr:.4f}")

def load_upper_lip_baseline(filepath=None):
    if filepath is None:
        filepath = UPPER_LIP_BASELINE_FILE
    if not os.path.isfile(filepath):
        return None
    try:
        with open(filepath, encoding="utf-8") as f:
            d = json.load(f)
        return {"ulr": float(d["ulr"])}
    except Exception as e:
        print(f"[UpperLip Baseline] Load failed: {e}")
        return None


# ============================================================
# 嘴角基线管理 — A22-A25
# ============================================================
def save_mouth_corners_baseline(nose_pos, left_corner_pos, right_corner_pos, filepath=None):
    if filepath is None:
        filepath = MOUTH_CORNERS_BASELINE_FILE
    data = {
        "nose": list(nose_pos),
        "corner_left": list(left_corner_pos),
        "corner_right": list(right_corner_pos),
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[MouthCorners Baseline] Saved: nose={nose_pos} L_corner={left_corner_pos} R_corner={right_corner_pos}")


def load_mouth_corners_baseline(filepath=None):
    if filepath is None:
        filepath = MOUTH_CORNERS_BASELINE_FILE
    if not os.path.isfile(filepath):
        return None
    try:
        with open(filepath, encoding="utf-8") as f:
            d = json.load(f)
        return {
            "nose": tuple(d["nose"]),
            "corner_left": tuple(d["corner_left"]),
            "corner_right": tuple(d["corner_right"]),
        }
    except Exception as e:
        print(f"[MouthCorners Baseline] Load failed: {e}")
        return None


# ============================================================
# 主类
# ============================================================
class EllSegDetector:
    """
    核心：MediaPipe + EllSeg 检测虹膜 → 对比基线 → 输出偏移和舵机引导

    公开接口：
      capture()           → (ok, frame)
      detect(frame)       → dict {left_iris, right_iris, nose, score, qualified, offset, ...}
      capture_and_score() → (score, region_scores)   兼容 optimizer
      get_guide_signal()  → dict 或 None             舵机引导信号
      save_current_baseline(det)  手动保存当前帧为基线
    """

    def __init__(self, camera_index=0, width=1920, height=1080,
                 flip_horizontal=True, stabilize_frames=8):
        self.width = width
        self.height = height
        self.flip_horizontal = flip_horizontal
        self.stabilize_frames = stabilize_frames

        # 检测模块开关
        self.enable_mp = False        # MediaPipe 人脸关键点检测
        self.enable_ellseg = False    # EllSeg 虹膜椭圆检测
        self.show_all_landmarks = False  # 显示所有 MediaPipe 关键点

        print("[Detector] Initializing MediaPipe...")
        self.landmarker = _init_mediapipe()

        print(f"[Detector] Opening camera {camera_index}...")
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_index}")

        # 打印摄像头当前参数
        #self._print_camera_capabilities()

        # 强制 MJPG 格式以支持高帧率 (YUY2 默认不支持 120fps)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FPS, 120)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # 避免缓冲区堆积旧帧

        # EMA
        self.lm_smoother = LMASmoother(alpha=EMA_LM)
        self.ell_smooth_L = EllipseSmoother(alpha=EMA_ELL)
        self.ell_smooth_R = EllipseSmoother(alpha=EMA_ELL)

        # 基线
        self.baseline = load_baseline()
        if self.baseline:
            print(f"[Baseline] Loaded from {FACE_POINTS_FILE}")
            print(f"  nose   = {self.baseline['nose']}")
            print(f"  L_iris = {self.baseline['iris_left']}")
            print(f"  R_iris = {self.baseline['iris_right']}")
        else:
            print("[Baseline] No baseline found.")

        # 眼皮基线
        self.eyelid_baseline = load_eyelid_baseline()
        if self.eyelid_baseline:
            print(f"[Eyelid Baseline] Loaded from {EYELID_BASELINE_FILE}")
            print(f"  L_EAR = {self.eyelid_baseline['left_ear']:.4f}")
            print(f"  R_EAR = {self.eyelid_baseline['right_ear']:.4f}")
        else:
            print("[Eyelid Baseline] No eyelid baseline found.")

        # 眉毛基线
        self.eyebrow_baseline = load_eyebrow_baseline()
        if self.eyebrow_baseline:
            print(f"[Eyebrow Baseline] Loaded from {EYEBROW_BASELINE_FILE}")
            print(f"  L_EBHR = {self.eyebrow_baseline['left_ebhr']:.4f}")
            print(f"  R_EBHR = {self.eyebrow_baseline['right_ebhr']:.4f}")
        else:
            print("[Eyebrow Baseline] No eyebrow baseline found.")

        # 嘴部基线
        self.mouth_baseline = load_mouth_baseline()
        if self.mouth_baseline:
            print(f"[Mouth Baseline] Loaded from {MOUTH_BASELINE_FILE}")
            print(f"  MAR = {self.mouth_baseline['mar']:.4f}")
        else:
            print("[Mouth Baseline] No mouth baseline found.")

        # 下唇基线
        self.lower_lip_baseline = load_lower_lip_baseline()
        if self.lower_lip_baseline:
            print(f"[LowerLip Baseline] Loaded from {LOWER_LIP_BASELINE_FILE}")
            print(f"  LLR = {self.lower_lip_baseline['llr']:.4f}")
        else:
            print("[LowerLip Baseline] No lower lip baseline found.")

        # 上唇基线
        self.upper_lip_baseline = load_upper_lip_baseline()
        if self.upper_lip_baseline:
            print(f"[UpperLip Baseline] Loaded from {UPPER_LIP_BASELINE_FILE}")
            print(f"  ULR = {self.upper_lip_baseline['ulr']:.4f}")
        else:
            print("[UpperLip Baseline] No upper lip baseline found.")

        # 嘴角基线
        self.mouth_corners_baseline = load_mouth_corners_baseline()
        if self.mouth_corners_baseline:
            print(f"[MouthCorners Baseline] Loaded from {MOUTH_CORNERS_BASELINE_FILE}")
            print(f"  nose   = {self.mouth_corners_baseline['nose']}")
            print(f"  L_corner = {self.mouth_corners_baseline['corner_left']}")
            print(f"  R_corner = {self.mouth_corners_baseline['corner_right']}")
        else:
            print("[MouthCorners Baseline] No mouth corners baseline found.")

        self._timestamp = 0
        self.last_result = None

        # 显示线程 (独立线程刷新 cv2.imshow，不阻塞主逻辑)
        self._disp_extra_texts = []       # 额外HUD文字 [(text, pos, color), ...]
        self._stop_event = threading.Event()
        self._display_thread = None
        self._user_key = None             # 停止按键 (显示线程写入)
        self._last_key = None              # 最近一次任意按键
        self._capture_fps = 0.0             # 主循环捕获帧率 (display线程读取)
        self._cap_fps_counter = 0
        self._cap_fps_time = time.time()

    def _measure_fps(self, cap, warmup=5, measure=20):
        """实际抓帧测量 FPS"""
        for _ in range(warmup):
            cap.read()
        t0 = time.perf_counter()
        for _ in range(measure):
            ok, _ = cap.read()
            if not ok:
                return 0.0
        elapsed = time.perf_counter() - t0
        return round(measure / elapsed, 1) if elapsed > 0 else 0.0

    def _print_camera_capabilities(self):
        """打印摄像头支持的分辨率和实测帧率"""
        cap = self.cap
        print("\n[Camera] 查询支持的分辨率/帧率 (实测)...")

        # 常见分辨率测试列表 (从高到低)
        test_sizes = [
            (3840, 2160), (2560, 1440), (1920, 1080), (1600, 900),
            (1280, 720), (960, 540), (800, 600), (640, 480),
        ]
        supported = []
        for tw, th in test_sizes:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, tw)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, th)
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if actual_w == tw and actual_h == th:
                fps = self._measure_fps(cap, warmup=3, measure=10)
                supported.append((tw, th, fps))

        if supported:
            print(f"  支持的分辨率 (实测FPS):")
            for sw, sh, sfps in supported:
                marker = " ★ 当前" if (sw == self.width and sh == self.height) else ""
                print(f"    {sw}x{sh}  @ {sfps:.1f} fps{marker}")
        else:
            print("  [WARN] 无法查询具体支持情况")

        # 恢复目标分辨率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    # ==================================================================
    # 显示线程 — 独立线程持续刷新检测画面，主逻辑只管推数据
    # ==================================================================

    def _display_loop(self):
        """显示线程主循环: 持续读取 last_result 并刷新窗口"""
        while not self._stop_event.is_set():
            det = self.last_result
            if det and det.get("frame") is not None:
                img = det["frame"].copy()

                # 关键点圆
                if det.get("nose"):
                    cv2.circle(img, det["nose"], 5, (0, 255, 0), -1)
                if det.get("left_iris"):
                    cv2.circle(img, det["left_iris"], 4, (0, 255, 255), -1)
                if det.get("right_iris"):
                    cv2.circle(img, det["right_iris"], 4, (0, 255, 255), -1)

                # PASS/FAIL + 偏移 HUD
                q = det.get("qualified", False)
                cv2.putText(img, f"{'PASS' if q else 'FAIL'}", (10, 22), 0, 0.55,
                           (0, 255, 0) if q else (0, 0, 255), 1)
                y = 44
                off = det.get("offset", {})
                for label, o in [("L", off.get("left")), ("R", off.get("right"))]:
                    if o:
                        dx, dy, d = o
                        cv2.putText(img,
                                   f"{label}:dX={dx:+.1f} dY={dy:+.1f} d={d:.1f}px",
                                   (10, y), 0, 0.35,
                                   (0, 255, 0) if d <= TOLERANCE else (0, 0, 255), 1)
                    y += 18

                # mp0(人中) 实时坐标 — 画面左侧
                if hasattr(self, '_last_lms_smooth') and self._last_lms_smooth is not None:
                    mp0 = self._last_lms_smooth[0]
                    cv2.putText(img, f"mp0:({mp0[0]:.0f},{mp0[1]:.0f})", (10, 400), 0, 0.35, (255, 200, 0), 1)

                # FPS 显示 (来自主采集线程的实际捕获速率)
                cap_fps = self._capture_fps
                cv2.putText(img, f"Cap FPS: {cap_fps:.1f}", (img.shape[1] - 170, 22),
                           0, 0.45, (200, 200, 200), 1)

                # EAR 眼皮高宽比 (如果有基线)
                if self.eyelid_baseline and hasattr(self, '_last_lms_smooth') and self._last_lms_smooth is not None:
                    signal = self.get_eyelid_signal()
                    if signal:
                        bl = self.eyelid_baseline
                        l_ear = signal["left_ear"]
                        r_ear = signal["right_ear"]
                        l_ok = signal["left_qualified"]
                        r_ok = signal["right_qualified"]
                        y_ear = 80
                        cv2.putText(img, f"L_EAR={l_ear:.3f} (bl={bl['left_ear']:.3f})",
                                   (10, y_ear), 0, 0.35,
                                   (0, 255, 0) if l_ok else (0, 0, 255), 1)
                        cv2.putText(img, f"R_EAR={r_ear:.3f} (bl={bl['right_ear']:.3f})",
                                   (10, y_ear + 16), 0, 0.35,
                                   (0, 255, 0) if r_ok else (0, 0, 255), 1)

                # EBHR 眉毛高度比 (如果有基线)
                if self.eyebrow_baseline and hasattr(self, '_last_lms_smooth') and self._last_lms_smooth is not None:
                    signal = self.get_eyebrow_signal()
                    if signal:
                        bl = self.eyebrow_baseline
                        l_ebhr = signal["left_ebhr"]
                        r_ebhr = signal["right_ebhr"]
                        l_ok = signal["left_qualified"]
                        r_ok = signal["right_qualified"]
                        y_brow = 115
                        cv2.putText(img, f"L_EBHR={l_ebhr:.3f} (bl={bl['left_ebhr']:.3f})",
                                   (10, y_brow), 0, 0.35,
                                   (0, 255, 0) if l_ok else (0, 0, 255), 1)
                        cv2.putText(img, f"R_EBHR={r_ebhr:.3f} (bl={bl['right_ebhr']:.3f})",
                                   (10, y_brow + 16), 0, 0.35,
                                   (0, 255, 0) if r_ok else (0, 0, 255), 1)

                # MAR 嘴部高宽比 (如果有基线)
                if self.mouth_baseline and hasattr(self, '_last_lms_smooth') and self._last_lms_smooth is not None:
                    signal = self.get_mouth_signal()
                    if signal:
                        bl = self.mouth_baseline
                        mar = signal["mar"]
                        ok = signal["qualified"]
                        y_mouth = 150
                        cv2.putText(img, f"MAR={mar:.3f} (bl={bl['mar']:.3f})",
                                   (10, y_mouth), 0, 0.35,
                                   (0, 255, 0) if ok else (0, 0, 255), 1)

                # LLR 下唇比例 (如果有基线)
                if self.lower_lip_baseline and hasattr(self, '_last_lms_smooth') and self._last_lms_smooth is not None:
                    signal = self.get_lower_lip_signal()
                    if signal:
                        bl = self.lower_lip_baseline
                        ok = signal["qualified"]
                        y_ll = 168
                        cv2.putText(img, f"LLR={signal['llr']:.3f} (bl={bl['llr']:.3f})",
                                   (10, y_ll), 0, 0.35,
                                   (0, 255, 0) if ok else (0, 0, 255), 1)

                # ULR 上唇比例 (如果有基线)
                if self.upper_lip_baseline and hasattr(self, '_last_lms_smooth') and self._last_lms_smooth is not None:
                    signal = self.get_upper_lip_signal()
                    if signal:
                        bl = self.upper_lip_baseline
                        ok = signal["qualified"]
                        y_ul = 202
                        cv2.putText(img, f"ULR={signal['ulr']:.3f} (bl={bl['ulr']:.3f})",
                                   (10, y_ul), 0, 0.35,
                                   (0, 255, 0) if ok else (0, 0, 255), 1)

                # 嘴角偏移 HUD (如果有基线)
                if self.mouth_corners_baseline and hasattr(self, '_last_lms_smooth') and self._last_lms_smooth is not None:
                    signal = self.get_mouth_corner_signal()
                    if signal:
                        y_mc = 220
                        cv2.putText(img, "MouthCorner:",
                                   (10, y_mc), 0, 0.35, (255, 200, 0), 1)
                        y_mc += 16
                        for label, key in [("L", "left"), ("R", "right")]:
                            d = signal.get(key, {})
                            dx, dy, dist = d.get("dx", 0), d.get("dy", 0), d.get("dist", 0)
                            ok = d.get("qualified", False)
                            cv2.putText(img,
                                       f"  {label}:dX={dx:+.1f} dY={dy:+.1f} d={dist:.1f}px",
                                       (10, y_mc), 0, 0.35,
                                       (0, 255, 0) if ok else (0, 0, 255), 1)
                            y_mc += 16

                # 检测耗时拆解 (ms)
                timing = det.get("_timing") 
                if timing:
                    total, t_mp, t_ell = timing
                    cv2.putText(img, f"T: {total:.0f}ms  MP: {t_mp:.0f}ms  Ell: {t_ell:.0f}ms",
                               (img.shape[1] - 320, img.shape[0] - 8),
                               0, 0.4, (200, 200, 200), 1)

                # 调用方追加的额外文字
                for text, pos, color in self._disp_extra_texts:
                    cv2.putText(img, text, pos, 0, 0.45, color, 1)

                # 显示所有 MediaPipe 关键点
                if self.show_all_landmarks and hasattr(self, '_last_lms_smooth') and self._last_lms_smooth is not None:
                    lms = self._last_lms_smooth
                    for i in range(lms.shape[0]):
                        px, py = int(lms[i, 0]), int(lms[i, 1])
                        cv2.circle(img, (px, py), 1, (180, 180, 255), -1)
                        cv2.putText(img, str(i), (px + 2, py - 1), 0, 0.22, (180, 180, 255), 1)

                cv2.imshow("EllSeg Detect", img)

                # 检测开关 HUD
                status_str = f"[1]MP:{'ON' if self.enable_mp else 'OFF'}  [2]ELL:{'ON' if self.enable_ellseg else 'OFF'}  [3]LMS:{'ON' if self.show_all_landmarks else 'OFF'}"
                cv2.putText(img, status_str, (10, img.shape[0] - 8),
                           0, 0.4, (200, 200, 200), 1)

                key = cv2.waitKey(1) & 0xFF
                if key:
                    self._last_key = key
                if key == ord('1'):
                    self.toggle_mp()
                elif key == ord('2'):
                    self.toggle_ellseg()
                elif key == ord('3'):
                    self.show_all_landmarks = not self.show_all_landmarks
                    print(f"[Detector] Show Landmarks: {'ON' if self.show_all_landmarks else 'OFF'}")
                elif key in (ord('q'), ord('Q'), 27):
                    self._user_key = key

    def start_display(self):
        """启动显示线程（幂等，已运行则跳过）"""
        if self._display_thread and self._display_thread.is_alive():
            return
        self._stop_event.clear()
        self._user_key = None
        self._last_key = None
        self._display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self._display_thread.start()

    def stop_display(self):
        """停止显示线程"""
        self._stop_event.set()
        if self._display_thread and self._display_thread.is_alive():
            self._display_thread.join(timeout=2)

    def update_hud(self, extra_texts=None):
        """更新HUD额外文字 (主逻辑线程调用)

        Args:
            extra_texts: list of (text: str, position: tuple, color: tuple)
                         e.g. [("[Iter 5/50]", (10, 100), (200,200,200))]
        """
        if extra_texts is not None:
            self._disp_extra_texts = extra_texts

    @property
    def user_pressed_stop(self):
        """用户是否按了 Q/Esc"""
        return self._user_key is not None

    # ---- 内部 ----
    def _make_result(self, frame, reason=""):
        return {"qualified": False, "offset": {},
                "left_iris": None, "right_iris": None, "nose": None,
                "frame": frame, "reason": reason}

    def _calc_offset(self, det):
        """计算与基线的偏移。返回 {"left": (dx,dy,dist), "right": ..., "nose": ...}"""
        if not self.baseline or self.baseline.get("nose") is None:
            return {}
        bl = self.baseline
        offset = {}
        key_map = {"left": "iris_left", "right": "iris_right", "nose": "nose"}
        for key, cur_pos in [("left", det.get("left_iris")),
                              ("right", det.get("right_iris")),
                              ("nose",   det.get("nose"))]:
            if cur_pos is None:
                continue
            base_pos = bl.get(key_map[key])
            if base_pos is None:
                continue
            dx = cur_pos[0] - base_pos[0]
            dy = cur_pos[1] - base_pos[1]
            dist = np.sqrt(dx * dx + dy * dy)
            offset[key] = (dx, dy, dist)
        return offset

    def _is_qualified(self, offset):
        if not offset:
            return False
        for k in ("left", "right"):
            if k in offset and offset[k][2] > TOLERANCE:
                return False
        return "left" in offset and "right" in offset

    # ---- 检测开关 ----
    def toggle_mp(self):
        """切换 MediaPipe 检测开/关"""
        self.enable_mp = not self.enable_mp
        print(f"[Detector] MediaPipe: {'ON' if self.enable_mp else 'OFF'}")

    def toggle_ellseg(self):
        """切换 EllSeg 检测开/关"""
        self.enable_ellseg = not self.enable_ellseg
        print(f"[Detector] EllSeg: {'ON' if self.enable_ellseg else 'OFF'}")

    # ---- 公开接口 ----
    def capture(self):
        ok, frame = self.cap.read()
        if not ok:
            return False, None

        # 统计实际捕获帧率
        self._cap_fps_counter += 1
        now = time.time()
        elapsed = now - self._cap_fps_time
        if elapsed >= 1.0:
            self._capture_fps = round(self._cap_fps_counter / elapsed, 1)
            self._cap_fps_counter = 0
            self._cap_fps_time = now

        if self.flip_horizontal:
            frame = cv2.flip(frame, 1)
        return True, frame

    def detect(self, frame):
        t_total = time.perf_counter()
        h, w = frame.shape[:2]
        self._timestamp += 33

        # ---------- MediaPipe 开关 ----------
        if not self.enable_mp:
            self.lm_smoother.reset()
            self.ell_smooth_L.reset()
            self.ell_smooth_R.reset()
            r = self._make_result(frame, "MP off")
            r["_timing"] = (0, 0, 0)
            self.last_result = r
            return r

        t0 = time.perf_counter()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        mp_result = self.landmarker.detect_for_video(mp_img, self._timestamp)
        t_mp = 1000 * (time.perf_counter() - t0)

        if not mp_result.face_landmarks:
            self.lm_smoother.reset()
            self.ell_smooth_L.reset()
            self.ell_smooth_R.reset()
            r = self._make_result(frame, "No face")
            self.last_result = r
            return r

        face = mp_result.face_landmarks[0]
        lms_px = np.array([[lm.x * w, lm.y * h] for lm in face])
        lms_smooth = self.lm_smoother.update(lms_px)

        out = self._make_result(frame)

        # 鼻子
        out["nose"] = (int(lms_smooth[NOSE_IDX, 0]), int(lms_smooth[NOSE_IDX, 1]))

        # ---------- EllSeg 开关 ----------
        t_ell = 0
        if self.enable_ellseg:
            for side, eye_idx, ell_sm in [
                ("left", LEFT_EYE_IDX, self.ell_smooth_L),
                ("right", RIGHT_EYE_IDX, self.ell_smooth_R),
            ]:
                crop, (x1, y1) = crop_eye(frame, lms_smooth, eye_idx)
                if crop.size == 0:
                    continue
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                e0 = time.perf_counter()
                ells = run_ellseg(gray)
                t_ell += 1000 * (time.perf_counter() - e0)
                iris = ells.get("iris")
                if iris is None:
                    ell_sm.reset()
                    continue
                smooth_iris = ell_sm.update(iris)
                (scx, scy), _, _ = smooth_iris
                out[f"{side}_iris"] = (int(scx + x1), int(scy + y1))

        # 与基线比较
        offset = self._calc_offset(out)
        out["offset"] = offset

        if offset and self.baseline:
            out["qualified"] = self._is_qualified(offset)
        else:
            out["qualified"] = False

        t_total = 1000 * (time.perf_counter() - t_total)
        out["_timing"] = (t_total, t_mp, t_ell)

        # 保存 lms_smooth 供 get_eyelid_signal() 使用
        self._last_lms_smooth = lms_smooth

        self.last_result = out
        return out

    def capture_and_score(self, stabilize_frames=None):
        """
        采集多帧，统计通过率。
        返回 (passed: bool, pass_rate: float, details_dict)
          passed   : 是否通过 (≥50% 帧qualified为True)
          pass_rate: 合格帧占比 (0.0~1.0)
          details  : 偏移信息等
        """
        if stabilize_frames is None:
            stabilize_frames = self.stabilize_frames

        qualified_count = 0
        total_count = 0
        last_det = None

        for _ in range(stabilize_frames):
            ok, frame = self.capture()
            if not ok:
                continue
            det = self.detect(frame)
            total_count += 1
            last_det = det
            if det.get("qualified"):
                qualified_count += 1

        if total_count == 0:
            empty = {"passed": False, "pass_rate": 0.0,
                     "qualified": False,
                     "L_dX": 999, "L_dY": 999, "R_dX": 999, "R_dY": 999,
                     "L_dist": 999, "R_dist": 999}
            return False, 0.0, empty

        pass_rate = qualified_count / total_count
        passed = pass_rate >= 0.5

        details = self._build_region_scores(last_det) if last_det else {}
        details["passed"] = passed
        details["pass_rate"] = pass_rate

        return passed, pass_rate, details

    def _build_region_scores(self, det):
        rs = {"qualified": det.get("qualified", False)}
        offset = det.get("offset", {})
        for side, prefix in [("left", "L"), ("right", "R")]:
            off = offset.get(side)
            if off is not None:
                dx, dy, dist = off
                rs[f"{prefix}_dX"] = float(dx)
                rs[f"{prefix}_dY"] = float(dy)
                rs[f"{prefix}_dist"] = float(dist)
            else:
                rs[f"{prefix}_dX"] = 999.0
                rs[f"{prefix}_dY"] = 999.0
                rs[f"{prefix}_dist"] = 999.0
        return rs

    def get_guide_signal(self):
        """
        返回舵机引导信号 (dict 或 None)。
        偏移的反方向 = 舵机应调整的方向。

        例: L_dX=+5 表示左虹膜在基线右侧5px
            L_adj_X=-5 表示舵机应往左调5px (反方向)
        """
        if self.last_result is None:
            return None
        offset = self.last_result.get("offset")
        if not offset:
            return None

        guide = {}
        total_dist = 0.0
        for k, prefix in [("left", "L"), ("right", "R"), ("nose", "Nose")]:
            if k not in offset:
                continue
            dx, dy, dist = offset[k]
            total_dist += dist
            guide[f"{prefix}_dX"] = float(dx)
            guide[f"{prefix}_dY"] = float(dy)
            guide[f"{prefix}_dist"] = float(dist)
            guide[f"{prefix}_adj_X"] = float(-dx)  # 反方向
            guide[f"{prefix}_adj_Y"] = float(-dy)
        guide["total_dist"] = float(total_dist)
        return guide

    def save_current_baseline(self, det=None):
        """手动保存当前检测结果为基线"""
        if det is None:
            det = self.last_result
        if det is None or det["nose"] is None \
                or det["left_iris"] is None or det["right_iris"] is None:
            print("[Baseline] Cannot save: points missing")
            return False
        self.baseline = {
            "nose": det["nose"],
            "iris_left": det["left_iris"],
            "iris_right": det["right_iris"],
        }
        save_baseline(
            nose_pos=self.baseline["nose"],
            left_iris_pos=self.baseline["iris_left"],
            right_iris_pos=self.baseline["iris_right"],
        )
        return True

    # ==================================================================
    # 眼皮 EAR 计算与信号
    # ==================================================================

    @staticmethod
    def _calc_ear(lms_smooth, pairs, corners):
        """
        计算单眼 Eye Aspect Ratio (高宽比)。

        EAR = mean(vertical_pair_distances) / horizontal_distance

        Args:
            lms_smooth: (N, 2) 平滑后的关键点像素坐标
            pairs: list of (top_idx, bottom_idx) 元组
            corners: (outer_idx, inner_idx) 眼角索引

        Returns:
            float: EAR 值 (0 ~ 1)
        """
        vert_dists = []
        for top_i, btm_i in pairs:
            dy = lms_smooth[btm_i, 1] - lms_smooth[top_i, 1]
            dx = lms_smooth[btm_i, 0] - lms_smooth[top_i, 0]
            vert_dists.append(np.sqrt(dx * dx + dy * dy))

        outer_i, inner_i = corners
        hor_dist = abs(lms_smooth[inner_i, 0] - lms_smooth[outer_i, 0])

        if hor_dist < 1e-6:
            return 0.0

        ear = np.mean(vert_dists) / hor_dist
        return float(ear)

    @staticmethod
    def _calc_ears(lms_smooth):
        """计算双眼 EAR。

        Returns:
            dict: {"left": float, "right": float} 或 None（无检测数据时）
        """
        from eye_constants import (
            LEFT_EAR_PAIRS, LEFT_EYE_CORNERS,
            RIGHT_EAR_PAIRS, RIGHT_EYE_CORNERS,
        )
        return {
            "left": EllSegDetector._calc_ear(
                lms_smooth, LEFT_EAR_PAIRS, LEFT_EYE_CORNERS),
            "right": EllSegDetector._calc_ear(
                lms_smooth, RIGHT_EAR_PAIRS, RIGHT_EYE_CORNERS),
        }

    @staticmethod
    def _calc_ebhr(lms_smooth, brow_points, corners):
        """
        计算单眉 Eyebrow Height Ratio。
        EBHR = (眉弓Y均值 - 眼角Y均值) / 眼宽
        """
        brow_y = np.mean([lms_smooth[i, 1] for i in brow_points])
        corner_y = np.mean([lms_smooth[i, 1] for i in corners])
        outer_i, inner_i = corners
        eye_width = abs(lms_smooth[inner_i, 0] - lms_smooth[outer_i, 0])
        if eye_width < 1e-6:
            return 0.0
        return float((brow_y - corner_y) / eye_width)

    @staticmethod
    def _calc_ebhrs(lms_smooth):
        """计算双眼 EBHR。"""
        from eye_constants import (
            LEFT_BROW_POINTS, LEFT_BROW_CORNERS,
            RIGHT_BROW_POINTS, RIGHT_BROW_CORNERS,
        )
        return {
            "left": EllSegDetector._calc_ebhr(
                lms_smooth, LEFT_BROW_POINTS, LEFT_BROW_CORNERS),
            "right": EllSegDetector._calc_ebhr(
                lms_smooth, RIGHT_BROW_POINTS, RIGHT_BROW_CORNERS),
        }

    def get_eyelid_signal(self, lms_smooth=None):
        """
        返回眼皮 EAR 信号 — 对比基线 EAR。

        Args:
            lms_smooth: 平滑后的关键点坐标。若为 None，从最近一次 detect 结果获取。


        Returns:
            dict 或 None:
              left_ear, right_ear: 当前帧 EAR 值
              left_delta, right_delta: 与基线的偏差
                (正值 = 比基线更开, 负值 = 比基线更闭)
              left_qualified, right_qualified: 是否在容差范围内
        """
        if lms_smooth is None:
            # 尝试从 detect 内部获取 lms_smooth
            if not hasattr(self, '_last_lms_smooth') or self._last_lms_smooth is None:
                return None
            lms_smooth = self._last_lms_smooth

        if self.eyelid_baseline is None:
            return None

        ears = self._calc_ears(lms_smooth)

        bl = self.eyelid_baseline
        from eye_constants import EYELID_EAR_TOLERANCE
        tol = EYELID_EAR_TOLERANCE

        left_delta = ears["left"] - bl["left_ear"]
        right_delta = ears["right"] - bl["right_ear"]

        return {
            "left_ear": ears["left"],
            "right_ear": ears["right"],
            "left_delta": left_delta,
            "right_delta": right_delta,
            "left_qualified": abs(left_delta) <= tol,
            "right_qualified": abs(right_delta) <= tol,
        }

    def save_current_eyelid_baseline(self, lms_smooth=None):
        """手动保存当前帧的 EAR 值为眼皮基线"""
        if lms_smooth is None:
            if not hasattr(self, '_last_lms_smooth') or self._last_lms_smooth is None:
                print("[Eyelid Baseline] Cannot save: no landmark data")
                return False
            lms_smooth = self._last_lms_smooth

        ears = self._calc_ears(lms_smooth)
        self.eyelid_baseline = {
            "left_ear": ears["left"],
            "right_ear": ears["right"],
        }
        save_eyelid_baseline(left_ear=ears["left"], right_ear=ears["right"])
        return True

    # ==================================================================
    # 眉毛 EBHR 计算与信号
    # ==================================================================

    def get_eyebrow_signal(self, lms_smooth=None):
        """
        返回眉毛 EBHR 信号 — 对比基线 EBHR。

        Args:
            lms_smooth: 平滑后的关键点坐标。若为 None，从最近一次 detect 结果获取。

        Returns:
            dict 或 None:
              left_ebhr, right_ebhr: 当前帧 EBHR 值
              left_delta, right_delta: 与基线的偏差
              left_qualified, right_qualified: 是否在容差范围内
        """
        if lms_smooth is None:
            if not hasattr(self, '_last_lms_smooth') or self._last_lms_smooth is None:
                return None
            lms_smooth = self._last_lms_smooth

        if self.eyebrow_baseline is None:
            return None

        ebhrs = self._calc_ebhrs(lms_smooth)
        bl = self.eyebrow_baseline
        from eye_constants import EYEBROW_EBHR_TOLERANCE
        tol = EYEBROW_EBHR_TOLERANCE

        left_delta = ebhrs["left"] - bl["left_ebhr"]
        right_delta = ebhrs["right"] - bl["right_ebhr"]

        return {
            "left_ebhr": ebhrs["left"],
            "right_ebhr": ebhrs["right"],
            "left_delta": left_delta,
            "right_delta": right_delta,
            "left_qualified": abs(left_delta) <= tol,
            "right_qualified": abs(right_delta) <= tol,
        }

    def save_current_eyebrow_baseline(self, lms_smooth=None):
        """手动保存当前帧的 EBHR 值为眉毛基线"""
        if lms_smooth is None:
            if not hasattr(self, '_last_lms_smooth') or self._last_lms_smooth is None:
                print("[Eyebrow Baseline] Cannot save: no landmark data")
                return False
            lms_smooth = self._last_lms_smooth

        ebhrs = self._calc_ebhrs(lms_smooth)
        self.eyebrow_baseline = {
            "left_ebhr": ebhrs["left"],
            "right_ebhr": ebhrs["right"],
        }
        save_eyebrow_baseline(left_ebhr=ebhrs["left"], right_ebhr=ebhrs["right"])
        return True

    # ==================================================================
    # 上唇 ULR 计算与信号 — A16
    # ==================================================================

    @staticmethod
    def _calc_ulr(lms_smooth):
        """
        计算 Upper Lip Ratio (上唇中央比例)。
        ULR = dist(mp0, 眼球中心) / inter_eye_width

        以眼球中心为原点，计算 mp0(人中) 到眼球中点的欧氏距离。
        用眼间距做分母归一化。
        """
        from eye_constants import (
            UL_LIP_TOP,
            UL_IRIS_LEFT, UL_IRIS_RIGHT,
            UL_NORM_LEFT, UL_NORM_RIGHT,
            UL_NORM_LEFT2, UL_NORM_RIGHT2,
        )
        import numpy as np

        # mp0(人中) 到眼球中点(左右虹膜均值) 的欧氏距离
        mp0 = lms_smooth[UL_LIP_TOP]
        iris_left = lms_smooth[UL_IRIS_LEFT]
        iris_right = lms_smooth[UL_IRIS_RIGHT]
        eyeball_mid = (iris_left + iris_right) / 2.0
        dist = float(np.linalg.norm(mp0 - eyeball_mid))

        # 水平归一化: 取两对参考点的均值，更稳定
        w1 = abs(lms_smooth[UL_NORM_RIGHT][0] - lms_smooth[UL_NORM_LEFT][0])
        w2 = abs(lms_smooth[UL_NORM_RIGHT2][0] - lms_smooth[UL_NORM_LEFT2][0])
        eye_width = (w1 + w2) / 2.0

        if eye_width < 1e-6:
            return 0.0

        return dist / eye_width

    def get_upper_lip_signal(self, lms_smooth=None):
        """
        返回 ULR 信号 — 对比基线 ULR。

        Returns:
            dict 或 None: {ulr, delta, qualified}
        """
        if lms_smooth is None:
            if not hasattr(self, '_last_lms_smooth') or self._last_lms_smooth is None:
                return None
            lms_smooth = self._last_lms_smooth

        if self.upper_lip_baseline is None:
            return None

        ulr = self._calc_ulr(lms_smooth)
        bl = self.upper_lip_baseline
        from eye_constants import UPPER_LIP_ULR_TOLERANCE
        tol = UPPER_LIP_ULR_TOLERANCE
        delta = ulr - bl["ulr"]

        return {
            "ulr": ulr,
            "delta": delta,
            "qualified": abs(delta) <= tol,
        }

    def save_current_upper_lip_baseline(self, lms_smooth=None):
        """手动保存当前帧的 ULR 值为上唇基线"""
        if lms_smooth is None:
            if not hasattr(self, '_last_lms_smooth') or self._last_lms_smooth is None:
                print("[UpperLip Baseline] Cannot save: no landmark data")
                return False
            lms_smooth = self._last_lms_smooth

        ulr = self._calc_ulr(lms_smooth)
        self.upper_lip_baseline = {"ulr": ulr}
        save_upper_lip_baseline(ulr=ulr)
        return True

    # ==================================================================
    # 嘴角偏移计算与信号 — A22-A25
    # ==================================================================

    def get_mouth_corner_signal(self, lms_smooth=None):
        """
        返回嘴角像素偏移信号 — 对比嘴角基线 (nose + left_corner + right_corner)。

        复用眼球偏移逻辑：当前像素 vs 基线像素，反方向 = 舵机调整方向。

        Returns:
            dict 或 None:
              left:  {dx, dy, dist, qualified, adj_x, adj_y}
              right: {dx, dy, dist, qualified, adj_x, adj_y}
              all_qualified: bool
        """
        if lms_smooth is None:
            if not hasattr(self, '_last_lms_smooth') or self._last_lms_smooth is None:
                return None
            lms_smooth = self._last_lms_smooth

        if self.mouth_corners_baseline is None:
            return None

        from eye_constants import (
            MOUTH_CORNER_LEFT_IDX, MOUTH_CORNER_RIGHT_IDX,
            MOUTH_CORNER_TOLERANCE,
        )
        bl = self.mouth_corners_baseline
        tol = MOUTH_CORNER_TOLERANCE

        result = {}
        for side, idx, bl_key in [
            ("left", MOUTH_CORNER_LEFT_IDX, "corner_left"),
            ("right", MOUTH_CORNER_RIGHT_IDX, "corner_right"),
        ]:
            cur = lms_smooth[idx]  # [x, y]
            base = bl[bl_key]      # (x, y)
            dx = float(cur[0] - base[0])
            dy = float(cur[1] - base[1])
            dist = float(np.sqrt(dx * dx + dy * dy))
            qualified = dist <= tol
            result[side] = {
                "dx": dx, "dy": dy, "dist": dist,
                "qualified": qualified,
                "adj_x": -dx, "adj_y": -dy,  # 反方向 = 舵机调整方向
            }

        result["all_qualified"] = result["left"]["qualified"] and result["right"]["qualified"]
        return result

    def save_current_mouth_corners_baseline(self, lms_smooth=None, n_frames=60, trim=20):
        """多帧采集取中位 → 保存嘴角基线 (抗抖动)。

        Args:
            n_frames: 采集帧数 (默认 60)
            trim:     去掉最大/最小帧数 (默认 20)
        """
        import numpy as np
        from eye_constants import (
            MOUTH_CORNER_LEFT_IDX, MOUTH_CORNER_RIGHT_IDX,
        )

        if lms_smooth is not None:
            # 单帧快捷路径 (用于一键保存全部基线 [A])
            nose = (int(lms_smooth[NOSE_IDX, 0]), int(lms_smooth[NOSE_IDX, 1]))
            self.mouth_corners_baseline = {
                "nose": nose,
                "corner_left": (int(lms_smooth[MOUTH_CORNER_LEFT_IDX, 0]),
                                int(lms_smooth[MOUTH_CORNER_LEFT_IDX, 1])),
                "corner_right": (int(lms_smooth[MOUTH_CORNER_RIGHT_IDX, 0]),
                                 int(lms_smooth[MOUTH_CORNER_RIGHT_IDX, 1])),
            }
            save_mouth_corners_baseline(
                nose_pos=nose,
                left_corner_pos=self.mouth_corners_baseline["corner_left"],
                right_corner_pos=self.mouth_corners_baseline["corner_right"],
            )
            return True

        # ---- 多帧采集抗抖动路径 ----
        if not hasattr(self, '_last_lms_smooth') or self._last_lms_smooth is None:
            print("[MouthCorners Baseline] Cannot save: no landmark data")
            return False

        mp_was_on = self.enable_mp
        self.enable_mp = True
        self.enable_ellseg = False

        frames_data = []  # [(nose, left, right), ...]
        collected = 0
        no_face_count = 0

        print(f"[MouthCorners Baseline] 采集 {n_frames} 帧 (trim={trim})...")
        while collected < n_frames:
            ok, frame = self.capture()
            if not ok:
                time.sleep(0.001)
                continue

            self.detect(frame)
            lms = self._last_lms_smooth
            if lms is None:
                no_face_count += 1
                if no_face_count > 10:
                    break
                time.sleep(0.001)
                continue
            no_face_count = 0

            nose_px = (int(lms[NOSE_IDX, 0]), int(lms[NOSE_IDX, 1]))
            left_px = (int(lms[MOUTH_CORNER_LEFT_IDX, 0]),
                       int(lms[MOUTH_CORNER_LEFT_IDX, 1]))
            right_px = (int(lms[MOUTH_CORNER_RIGHT_IDX, 0]),
                        int(lms[MOUTH_CORNER_RIGHT_IDX, 1]))
            frames_data.append((nose_px, left_px, right_px))
            collected += 1

        self.enable_mp = mp_was_on

        if len(frames_data) < 2 * trim + 1:
            print(f"[MouthCorners Baseline] 有效帧不足: {len(frames_data)}, 需要 >= {2*trim+1}")
            return False

        # 计算均值
        noses = np.array([f[0] for f in frames_data], dtype=np.float64)
        lefts = np.array([f[1] for f in frames_data], dtype=np.float64)
        rights = np.array([f[2] for f in frames_data], dtype=np.float64)
        mean_n = noses.mean(axis=0)
        mean_l = lefts.mean(axis=0)
        mean_r = rights.mean(axis=0)

        # 按偏离排序
        devs = []
        for i, (n, l, r) in enumerate(frames_data):
            d = np.linalg.norm(np.array(n) - mean_n) \
              + np.linalg.norm(np.array(l) - mean_l) \
              + np.linalg.norm(np.array(r) - mean_r)
            devs.append((d, i))
        devs.sort(key=lambda x: x[0])

        # 修剪取中位数
        trimmed_indices = [idx for _, idx in devs[trim:-trim]]
        median_idx = trimmed_indices[len(trimmed_indices) // 2]
        nose_px, left_px, right_px = frames_data[median_idx]

        self.mouth_corners_baseline = {
            "nose": nose_px,
            "corner_left": left_px,
            "corner_right": right_px,
        }
        save_mouth_corners_baseline(
            nose_pos=nose_px,
            left_corner_pos=left_px,
            right_corner_pos=right_px,
        )

        # 打印统计
        all_d = [d for d, _ in devs]
        med_d = all_d[len(all_d) // 2]
        print(f"  [MouthCorners Baseline] {len(frames_data)}帧→trim{trim}→ "
              f"中位={nose_px}/{left_px}/{right_px} "
              f"(偏差: min={all_d[0]:.1f} med={med_d:.1f} max={all_d[-1]:.1f})")
        return True

    # ==================================================================
    # 嘴部 MAR 计算与信号
    # ==================================================================

    @staticmethod
    def _calc_mar(lms_smooth):
        """
        计算 Mouth Aspect Ratio (嘴部高宽比)。
        MAR = mouth_vertical / inter_eye_width

        用眼间距做分母，不受嘴角不对称影响。
        """
        from eye_constants import (
            MOUTH_TOP, MOUTH_BOTTOM,
            MOUTH_NORM_LEFT, MOUTH_NORM_RIGHT,
        )

        # 垂直: 上唇上缘 → 下唇下缘 (纯垂直分量)
        top = lms_smooth[MOUTH_TOP]
        bottom = lms_smooth[MOUTH_BOTTOM]
        vert = abs(bottom[1] - top[1])

        # 水平归一化: 左眼外角 → 右眼外角（稳定，不受嘴变形影响）
        left_eye = lms_smooth[MOUTH_NORM_LEFT]
        right_eye = lms_smooth[MOUTH_NORM_RIGHT]
        eye_width = abs(right_eye[0] - left_eye[0])

        if eye_width < 1e-6:
            return 0.0

        return float(vert / eye_width)

    def get_mouth_signal(self, lms_smooth=None):
        """
        返回 MAR 信号 — 对比基线 MAR。

        Returns:
            dict 或 None: {mar, delta, qualified}
        """
        if lms_smooth is None:
            if not hasattr(self, '_last_lms_smooth') or self._last_lms_smooth is None:
                return None
            lms_smooth = self._last_lms_smooth

        if self.mouth_baseline is None:
            return None

        mar = self._calc_mar(lms_smooth)
        bl = self.mouth_baseline
        from eye_constants import MOUTH_MAR_TOLERANCE
        tol = MOUTH_MAR_TOLERANCE
        delta = mar - bl["mar"]

        return {
            "mar": mar,
            "delta": delta,
            "qualified": abs(delta) <= tol,
        }

    def save_current_mouth_baseline(self, lms_smooth=None):
        """手动保存当前帧的 MAR 值为嘴部基线"""
        if lms_smooth is None:
            if not hasattr(self, '_last_lms_smooth') or self._last_lms_smooth is None:
                print("[Mouth Baseline] Cannot save: no landmark data")
                return False
            lms_smooth = self._last_lms_smooth

        mar = self._calc_mar(lms_smooth)
        self.mouth_baseline = {"mar": mar}
        save_mouth_baseline(mar=mar)
        return True

    # ==================================================================
    # 下唇 LLR 计算与信号
    # ==================================================================

    @staticmethod
    def _calc_llr(lms_smooth):
        """
        计算 Lower Lip Ratio (下唇垂直比例)。
        LLR = lower_lip_vertical / inter_eye_width

        用眼间距做分母，不受嘴角不对称影响。
        """
        from eye_constants import (
            LL_LIP_UPPER, LL_LIP_LOWER,
            LL_NORM_LEFT, LL_NORM_RIGHT,
        )

        # 垂直: 下唇上缘 → 下唇下缘 (纯垂直分量)
        upper = lms_smooth[LL_LIP_UPPER]
        lower = lms_smooth[LL_LIP_LOWER]
        vert = abs(lower[1] - upper[1])

        # 水平归一化: 左眼外角 → 右眼外角
        left_eye = lms_smooth[LL_NORM_LEFT]
        right_eye = lms_smooth[LL_NORM_RIGHT]
        eye_width = abs(right_eye[0] - left_eye[0])

        if eye_width < 1e-6:
            return 0.0

        return float(vert / eye_width)

    def get_lower_lip_signal(self, lms_smooth=None):
        """
        返回 LLR 信号 — 对比基线 LLR。

        Returns:
            dict 或 None: {llr, delta, qualified}
        """
        if lms_smooth is None:
            if not hasattr(self, '_last_lms_smooth') or self._last_lms_smooth is None:
                return None
            lms_smooth = self._last_lms_smooth

        if self.lower_lip_baseline is None:
            return None

        llr = self._calc_llr(lms_smooth)
        bl = self.lower_lip_baseline
        from eye_constants import LOWER_LIP_LLR_TOLERANCE
        tol = LOWER_LIP_LLR_TOLERANCE
        delta = llr - bl["llr"]

        return {
            "llr": llr,
            "delta": delta,
            "qualified": abs(delta) <= tol,
        }

    def save_current_lower_lip_baseline(self, lms_smooth=None):
        """手动保存当前帧的 LLR 值为下唇基线"""
        if lms_smooth is None:
            if not hasattr(self, '_last_lms_smooth') or self._last_lms_smooth is None:
                print("[LowerLip Baseline] Cannot save: no landmark data")
                return False
            lms_smooth = self._last_lms_smooth

        llr = self._calc_llr(lms_smooth)
        self.lower_lip_baseline = {"llr": llr}
        save_lower_lip_baseline(llr=llr)
        return True

    def close(self):
        self.stop_display()
        if hasattr(self, 'landmarker') and self.landmarker:
            self.landmarker.close()
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def __del__(self):
        self.close()


if __name__ == "__main__":
    import sys
    raw_test = "--raw-fps" in sys.argv

    detector = EllSegDetector()
    detector.start_display()
    print("\n[S] Save eyeball  [E] Save eyelid  [B] Save eyebrow  [M] Save mouth  [L] Save lower lip  [U] Save upper lip  [C] Save mouth corners  [A] Save ALL  [1] Toggle MP  [2] Toggle EllSeg  [3] Toggle Landmarks  [Q] Quit")

    if raw_test:
        print("\n[Raw FPS Test] 仅测量摄像头原始读取帧率 (跳过 MediaPipe + EllSeg)...")
        print("按 Q 退出\n")

    try:
        while True:
            ok, frame = detector.capture()
            if not ok:
                time.sleep(0.001)
                continue

            if raw_test:
                # 只更新画面，不跑检测管线
                detector.last_result = {"frame": frame, "qualified": False, "offset": {}}
            else:
                detector.detect(frame)

            if detector.user_pressed_stop:
                break
            # 检测 S 键保存基线
            last = detector._last_key
            if last in (ord('s'), ord('S')):
                detector.save_current_baseline()
                detector._last_key = None
            # 检测 E 键保存眼皮基线
            if last in (ord('e'), ord('E')):
                detector.save_current_eyelid_baseline()
                detector._last_key = None
            # 检测 B 键保存眉毛基线
            if last in (ord('b'), ord('B')):
                detector.save_current_eyebrow_baseline()
                detector._last_key = None
            # 检测 M 键保存嘴部基线
            if last in (ord('m'), ord('M')):
                detector.save_current_mouth_baseline()
                detector._last_key = None
            # 检测 L 键保存下唇基线
            if last in (ord('l'), ord('L')):
                detector.save_current_lower_lip_baseline()
                detector._last_key = None
            # 检测 U 键保存上唇基线
            if last in (ord('u'), ord('U')):
                detector.save_current_upper_lip_baseline()
                detector._last_key = None
            # 检测 C 键保存嘴角基线
            if last in (ord('c'), ord('C')):
                detector.save_current_mouth_corners_baseline()
                detector._last_key = None
            # 检测 A 键一键保存全部基线
            if last in (ord('a'), ord('A')):
                print("\n=== 一键保存全部基线 ===")
                detector.save_current_baseline()
                detector.save_current_eyelid_baseline()
                detector.save_current_eyebrow_baseline()
                detector.save_current_mouth_baseline()
                detector.save_current_lower_lip_baseline()
                detector.save_current_upper_lip_baseline()
                detector.save_current_mouth_corners_baseline()
                print("=== 全部基线保存完成 ===\n")
                detector._last_key = None

    except KeyboardInterrupt:
        print("\n[Interrupted]")
    finally:
        detector.stop_display()
        detector.close()
