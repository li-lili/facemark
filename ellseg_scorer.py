"""
EllSeg Scorer — 基线标定 + 虹膜偏移检测 → 舵机引导
======================================================
核心逻辑：
  1. 标定基线：保存鼻子+左右虹膜中心像素位置 → face_point.json
  2. 偏移检测：实时对比当前帧与基线 (TOLERANCE=2px)
  3. 通过判定：规定帧数内，≥50% 帧合格(qualified)则通过
  4. 舵机引导：偏移方向反馈给舵机，指导眼球调整
"""

import os, sys, subprocess, urllib.request, cv2, numpy as np, json, torch, threading, time
from camera_tuning import draw_camera_hud, snapshot as camera_snapshot
from landmark_snapshot import LandmarkSnapshots, mediapipe_points, timestamp
from reference_head_overlay import draw_reference

# ============================================================
# 配置
# ============================================================
DIR = os.path.dirname(os.path.abspath(__file__))
ELLSEG_REPO = os.path.join(DIR, "EllSeg")
WEIGHTS_PATH = os.path.join(ELLSEG_REPO, "weights", "all.git_ok")
FACE_POINT_FILE = os.path.join(DIR, "face_point.json")
LEGACY_FACE_POINTS_FILE = os.path.join(DIR, "face_points.json")
FACE_POINTS_FILE = FACE_POINT_FILE
EYELID_BASELINE_FILE = os.path.join(DIR, "eyelid_baseline.json")
EYEBROW_BASELINE_FILE = os.path.join(DIR, "eyebrow_baseline.json")
EYEBROW_BASELINE_MALE_FILE = os.path.join(DIR, "eyebrow_baseline_male.json")
EYEBROW_BASELINE_FEMALE_FILE = os.path.join(DIR, "eyebrow_baseline_female.json")
MOUTH_BASELINE_FILE = os.path.join(DIR, "mouth_baseline.json")
LOWER_LIP_BASELINE_FILE = os.path.join(DIR, "lower_lip_baseline.json")
UPPER_LIP_BASELINE_FILE = os.path.join(DIR, "upper_lip_baseline.json")
MOUTH_CORNERS_BASELINE_FILE = os.path.join(DIR, "mouth_corners_baseline.json")
HEAD_POSITION_BASELINE_FILE = os.path.join(DIR, "head_position_baseline.json")
BASELINE_SECTIONS = (
    "eyelid", "eyebrow", "mouth", "lower_lip", "upper_lip",
    "mouth_corners", "head_position", "head_adapter", "side_profile_slopes",
)
DEFAULT_TEMPLATE_NAME = "default"

from eye_constants import (
    EYEBALL_LOWER_LID_CONTACT_SAMPLE_FRAMES,
    EYEBALL_LOWER_LID_CONTACT_TRIM_FRAMES,
    EYEBALL_OFFSET_TOLERANCE,
    EYELID_EAR_TOLERANCE,
    LOWER_LIP_SIDE_CAMERA_INDEX,
    LOWER_LIP_SIDE_FLIP_VERTICAL,
    LOWER_LIP_SIDE_ROI,
    LOWER_LIP_SIDE_X_TOLERANCE,
    LOWER_LIP_SIDE_Y_TOLERANCE,
    MOUTH_CORNER_CURL_MIN_TARGET,
    MOUTH_CORNER_CURL_SYMMETRY_TOLERANCE,
    MOUTH_CORNER_CURL_TARGET,
    MOUTH_CORNER_CURL_TOLERANCE,
    HEAD_ADAPTER_POINT_TOLERANCE,
    SIDE_PROFILE_CHIN_LINE_TOLERANCE_PX,
    SIDE_PROFILE_CROP_SIZE,
    SIDE_PROFILE_FACE_DIRECTION,
    SIDE_PROFILE_LIP_CHIN_TARGET_SLOPE,
    SIDE_PROFILE_LOWER_LINE_TOLERANCE_PX,
    SIDE_PROFILE_NOSE_UPPER_TARGET_SLOPE,
    SIDE_PROFILE_NOSE_EDGE_CONTRAST_MIN,
    SIDE_PROFILE_NOSE_X_RANGE,
    SIDE_PROFILE_NOSE_Y_RANGE,
    SIDE_PROFILE_PREVIEW_FILTER_FRAMES,
    SIDE_PROFILE_PREVIEW_MAX_JUMP_PX,
    SIDE_PROFILE_SAMPLE_FRAMES,
    SIDE_PROFILE_SLOPE_TOLERANCE,
    SIDE_PROFILE_TRIM_FRAMES,
    UPPER_LIP_SIDE_CAMERA_INDEX,
    UPPER_LIP_SIDE_FLIP_VERTICAL,
    UPPER_LIP_SIDE_ROI,
    UPPER_LIP_SIDE_X_TOLERANCE,
    UPPER_LIP_SIDE_Y_TOLERANCE,
)
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
from face_point_store import (
    DEFAULT_TEMPLATE_NAME,
    FACE_POINT_FILE,
    FACE_POINTS_FILE,
    get_active_template,
    get_template_names,
    load_baseline,
    load_eyebrow_baseline,
    load_eyelid_baseline,
    load_face_point_data,
    load_head_adapter_baseline,
    load_head_position_baseline,
    load_lower_lip_baseline,
    load_mouth_baseline,
    load_mouth_corner_curl_baseline,
    load_mouth_corners_baseline,
    load_side_profile_slope_baseline,
    load_side_nose_roi,
    load_side_chin_roi,
    save_side_chin_roi,
    save_side_nose_roi,
    load_upper_lip_baseline,
    save_baseline,
    save_current_as_template,
    save_eyebrow_baseline,
    save_eyelid_baseline,
    save_face_point_data,
    save_head_adapter_baseline,
    save_head_position_baseline,
    save_lower_lip_baseline,
    save_lower_lip_side_roi,
    save_mouth_baseline,
    save_mouth_corner_curl_baseline,
    save_mouth_corners_baseline,
    save_side_profile_slope_baseline,
    save_upper_lip_baseline,
    save_upper_lip_side_roi,
    set_active_template,
    template_section_exists,
)


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
                 flip_horizontal=False, flip_vertical=False,
                 stabilize_frames=8, rotate_180=True):
        self.width = width
        self.height = height
        self.flip_horizontal = False
        self.flip_vertical = False
        self.rotate_180 = bool(rotate_180)  # 正面采集方向；检测、显示和快照共用
        self.stabilize_frames = stabilize_frames
        self.camera_index = camera_index
        self.landmark_snapshots = LandmarkSnapshots()

        # 检测模块开关
        self.enable_mp = False        # MediaPipe 人脸关键点检测
        self.enable_ellseg = False    # EllSeg 虹膜椭圆检测
        self.show_all_landmarks = False  # 显示所有 MediaPipe 关键点
        self.show_baseline_overlay = False  # 叠加显示标准眼球位置基线
        self.show_eye_line_offset = False  # 显示参考线垂直偏移 HUD
        self.show_iris_cover_overlay = False  # 显示虹膜下弧遮盖计算线
        self.show_face_overlays = True  # 总开关：人物脸上的点、线和框（HUD 不受影响）
        self.show_side_profile_preview = False  # Qt 侧面四端点观察开关状态
        self.show_side_profile_expected_slope = False  # Qt 侧面期望斜率虚线状态
        self.show_side_profile_coordinate_grid = False  # Qt 侧面中心框局部坐标网格

        print("[Detector] Initializing MediaPipe...")
        self.landmarker = _init_mediapipe()

        print(f"[Detector] Opening camera {camera_index}...")
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_index}")

        # 打印摄像头当前参数
        #self._print_camera_capabilities()

        # DirectShow 设置 FPS/尺寸可能重置格式，因此最后请求 MJPG。
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # 避免缓冲区堆积旧帧

        fourcc_code = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        actual_fourcc = "".join(chr((fourcc_code >> (8 * i)) & 255) for i in range(4))
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Detector] Camera {camera_index}: {actual_width}x{actual_height} "
              f"FOURCC={actual_fourcc!r} (requested MJPG, 30 FPS)")
        if actual_fourcc != "MJPG" or (actual_width, actual_height) != (width, height):
            self.cap.release()
            self.cap = None
            self.landmarker.close()
            self.landmarker = None
            raise RuntimeError(
                f"Front camera mode mismatch: got {actual_width}x{actual_height} "
                f"FOURCC={actual_fourcc!r}; expected {width}x{height} MJPG")

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
            print(f"[Eyelid Baseline] Loaded from {FACE_POINT_FILE} ({get_active_template()})")
            print(f"  L_EAR = {self.eyelid_baseline['left_ear']:.4f}")
            print(f"  R_EAR = {self.eyelid_baseline['right_ear']:.4f}")
        else:
            print("[Eyelid Baseline] No eyelid baseline found.")

        # 眉毛基线
        self.eyebrow_baseline = load_eyebrow_baseline()
        if self.eyebrow_baseline:
            print(f"[Eyebrow Baseline] Loaded from {FACE_POINT_FILE} ({get_active_template()})")
            print(f"  L_S = {self.eyebrow_baseline['left_slope']:.4f}  R_S = {self.eyebrow_baseline['right_slope']:.4f}")
        else:
            print("[Eyebrow Baseline] No custom eyebrow baseline found; eyebrow check disabled.")

        # 嘴部基线
        self.mouth_baseline = load_mouth_baseline()
        if self.mouth_baseline:
            print(f"[Mouth Baseline] Loaded from {FACE_POINT_FILE} ({get_active_template()})")
            print(f"  MAR = {self.mouth_baseline['mar']:.4f}")
        else:
            print("[Mouth Baseline] No mouth baseline found.")

        # 下唇基线
        self.lower_lip_baseline = load_lower_lip_baseline()
        if self.lower_lip_baseline:
            print(f"[LowerLip Baseline] Loaded from {FACE_POINT_FILE} ({get_active_template()})")
            if "llr" in self.lower_lip_baseline:
                print(f"  LLR = {self.lower_lip_baseline['llr']:.4f}")
        else:
            print("[LowerLip Baseline] No lower lip baseline found.")

        # 上唇基线
        self.upper_lip_baseline = load_upper_lip_baseline()
        if self.upper_lip_baseline:
            print(f"[UpperLip Baseline] Loaded from {FACE_POINT_FILE} ({get_active_template()})")
            print(f"  ULR = {self.upper_lip_baseline['ulr']:.4f}")
        else:
            print("[UpperLip Baseline] No upper lip baseline found.")

        # 嘴角基线
        self.mouth_corners_baseline = load_mouth_corners_baseline()
        if self.mouth_corners_baseline:
            print(f"[MouthCorners Baseline] Loaded from {FACE_POINT_FILE} ({get_active_template()})")
            print(f"  nose   = {self.mouth_corners_baseline['nose']}")
            print(f"  L_corner = {self.mouth_corners_baseline['corner_left']}")
            print(f"  R_corner = {self.mouth_corners_baseline['corner_right']}")
        else:
            print("[MouthCorners Baseline] No mouth corners baseline found.")

        self.mouth_corner_curl_baseline = load_mouth_corner_curl_baseline()
        if self.mouth_corner_curl_baseline:
            print(f"[MouthCornerCurl Baseline] Loaded from {FACE_POINT_FILE} ({get_active_template()})")
            print(f"  L_curl = {self.mouth_corner_curl_baseline['left_curl']:+.4f}")
            print(f"  R_curl = {self.mouth_corner_curl_baseline['right_curl']:+.4f}")
        else:
            print("[MouthCornerCurl Baseline] No mouth corner curl baseline found.")

        # 头部定位基线
        self.head_position_baseline = load_head_position_baseline()
        if self.head_position_baseline:
            print(f"[HeadPosition Baseline] Loaded from {FACE_POINT_FILE} ({get_active_template()})")
            print(f"  nose   = {self.head_position_baseline['nose_px']}")
            print(f"  L_eye  = {self.head_position_baseline['eye_left']}")
            print(f"  R_eye  = {self.head_position_baseline['eye_right']}")
            print(f"  eye_dist = {self.head_position_baseline['eye_distance']:.1f}")
            print(f"  eye_angle = {self.head_position_baseline['eye_line_angle_deg']:+.3f}")
            print(f"  eye_slope = {self.head_position_baseline['eye_line_slope']}")
        else:
            print("[HeadPosition Baseline] No head position baseline found.")

        self.head_adapter_baseline = load_head_adapter_baseline()
        if self.head_adapter_baseline:
            print(f"[HeadAdapter Baseline] Loaded from {FACE_POINT_FILE} ({get_active_template()})")
            print(f"  nose_offset = {self.head_adapter_baseline.get('nose_offset')}")
            print(f"  point_max_offset = {self.head_adapter_baseline.get('point_max_offset')}")
        else:
            print("[HeadAdapter Baseline] No head adapter found.")

        self.side_nose_roi = load_side_nose_roi()
        self.side_chin_roi = load_side_chin_roi()
        self.side_profile_slope_baseline = load_side_profile_slope_baseline()
        if self.side_profile_slope_baseline:
            print(
                f"[SideProfile Slopes] N-U="
                f"{self.side_profile_slope_baseline['nose_upper_slope']:.4f} "
                f"U-C={self.side_profile_slope_baseline['lip_chin_slope']:.4f}")
        else:
            print("[SideProfile Slopes] Using default target slopes.")

        self._timestamp = 0
        self.last_result = None

        # 显示线程 (独立线程刷新 cv2.imshow，不阻塞主逻辑)
        self._disp_extra_texts = []       # 额外HUD文字 [(text, pos, color), ...]
        self._stop_event = threading.Event()
        self._display_thread = None
        self._reference_head_display = (None, False)
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

    def _draw_lower_iris_cover_overlay(self, img, det):
        """绘制虹膜下弧遮盖判断所用的椭圆、眼皮曲线与深度线。"""
        signal = det.get("lower_iris_cover")
        if not signal:
            return

        for side in ("left", "right"):
            measurement = signal.get(side) or {}
            ellipse = det.get(f"{side}_iris_ellipse")
            if ellipse is not None:
                (cx, cy), (axis_w, axis_h), angle_deg = ellipse
                rotated_rect = (
                    (int(round(cx)), int(round(cy))),
                    (max(1, int(round(axis_w))), max(1, int(round(axis_h)))),
                    float(angle_deg),
                )
                # 暗黄色：EllSeg 当前完整虹膜拟合椭圆。
                cv2.ellipse(img, rotated_rect, (0, 180, 220), 1, cv2.LINE_AA)

            arc_points = measurement.get("arc_points")
            lid_points = measurement.get("lid_curve_points")
            contact_mask = measurement.get("contact_mask")
            over_cover_mask = measurement.get("over_cover_mask")
            if arc_points is None or lid_points is None:
                continue

            arc_px = np.rint(np.asarray(arc_points)).astype(np.int32)
            lid_px = np.rint(np.asarray(lid_points)).astype(np.int32)
            if len(arc_px) >= 2:
                # 黄色：用于计算的理论虹膜下圆弧。
                cv2.polylines(img, [arc_px], False, (0, 255, 255), 1, cv2.LINE_AA)
            if len(lid_px) >= 2:
                # 青色：同一批 X 位置插值得到的下眼皮曲线。
                cv2.polylines(img, [lid_px], False, (255, 255, 0), 1, cv2.LINE_AA)

            if over_cover_mask is not None and len(arc_px) >= 2:
                mask = np.asarray(over_cover_mask, dtype=bool)
                # 红色：下眼皮遮住过多的理论下圆弧段。
                for index in range(min(len(mask), len(arc_px)) - 1):
                    if mask[index] and mask[index + 1]:
                        cv2.line(
                            img,
                            tuple(arc_px[index]),
                            tuple(arc_px[index + 1]),
                            (0, 0, 255), 3, cv2.LINE_AA,
                        )

            if contact_mask is not None and len(arc_px) >= 2:
                mask = np.asarray(contact_mask, dtype=bool)
                # 绿色：理论下圆弧与下眼皮处于贴合容差内的弧段。
                for index in range(min(len(mask), len(arc_px)) - 1):
                    if mask[index] and mask[index + 1]:
                        cv2.line(
                            img,
                            tuple(arc_px[index]),
                            tuple(arc_px[index + 1]),
                            (0, 255, 0), 3, cv2.LINE_AA,
                        )

            depth_segment = measurement.get("depth_segment")
            if depth_segment is not None:
                lid_point, arc_point = depth_segment
                if measurement.get("qualified", False):
                    depth_color = (0, 255, 0)
                elif (measurement.get("contact_offset_px") or 0.0) > 0.0:
                    depth_color = (0, 0, 255)
                else:
                    depth_color = (0, 165, 255)
                # 绿/红/橙色：第 90 百分位有符号贴合偏差的代表线段。
                cv2.line(
                    img,
                    tuple(np.rint(lid_point).astype(np.int32)),
                    tuple(np.rint(arc_point).astype(np.int32)),
                    depth_color, 2, cv2.LINE_AA,
                )

    def _display_loop(self):
        """显示线程主循环: 持续读取 last_result 并刷新窗口"""
        while not self._stop_event.is_set():
            det = self.last_result
            if det and det.get("frame") is not None:
                img = det["frame"].copy()

                eye_line_signal = self.get_eye_line_screen_signal()
                iris_overlay_visible = (
                    self.show_face_overlays and self.show_iris_cover_overlay)
                if self.show_face_overlays:
                    # 关键点圆
                    if det.get("nose"):
                        cv2.circle(img, det["nose"], 5, (0, 255, 0), -1)
                    if det.get("left_iris"):
                        cv2.circle(img, det["left_iris"], 4, (0, 255, 255), -1)
                    if det.get("right_iris"):
                        cv2.circle(img, det["right_iris"], 4, (0, 255, 255), -1)

                    if iris_overlay_visible:
                        self._draw_lower_iris_cover_overlay(img, det)

                    if eye_line_signal:
                        eye_l = eye_line_signal["left"]
                        eye_r = eye_line_signal["right"]
                        ref_y = eye_line_signal["mid_y"]
                        # 蓝色眼角线 + 同高度屏幕水平参考线。
                        cv2.line(img,
                                 (int(eye_l[0]), int(ref_y)),
                                 (int(eye_r[0]), int(ref_y)),
                                 (255, 220, 160), 1, cv2.LINE_AA)
                        cv2.line(img,
                                 (int(eye_l[0]), int(eye_l[1])),
                                 (int(eye_r[0]), int(eye_r[1])),
                                 (255, 150, 0), 2, cv2.LINE_AA)

                    # 标准眼球位置基线叠加 (G 键开关)
                    if self.show_baseline_overlay and self.baseline:
                        bl = self.baseline
                        overlay = img.copy()
                        target_nose = self._eyeball_target_point("nose") or bl["nose"]
                        target_L = self._eyeball_target_point("left") or bl["iris_left"]
                        target_R = self._eyeball_target_point("right") or bl["iris_right"]
                        bl_nose = (int(target_nose[0]), int(target_nose[1]))
                        bl_L = (int(target_L[0]), int(target_L[1]))
                        bl_R = (int(target_R[0]), int(target_R[1]))
                        cv2.circle(overlay, bl_nose, 16, (0, 255, 0), 2)
                        cv2.circle(overlay, bl_L, 16, (255, 255, 0), 2)
                        cv2.circle(overlay, bl_R, 16, (255, 255, 0), 2)
                        cv2.line(overlay, bl_nose, bl_L, (0, 255, 0), 1, cv2.LINE_AA)
                        cv2.line(overlay, bl_nose, bl_R, (0, 255, 0), 1, cv2.LINE_AA)
                        cv2.line(overlay, bl_L, bl_R, (0, 255, 0), 1, cv2.LINE_AA)
                        img = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)
                        label = "Target" if self._eyeball_target_mode() == "head_adapter" else "Baseline"
                        cv2.putText(img, label, (bl_nose[0] + 20, bl_nose[1] - 10),
                                   0, 0.4, (0, 255, 0), 1)

                # PASS/FAIL + 常驻 HUD。
                # 右侧信息栏统一承载检测参数，左侧留给 auto_adjust 的临时提示。
                q = det.get("qualified", False)
                cv2.putText(img, f"{'PASS' if q else 'FAIL'}", (10, 22), 0, 0.55,
                           (0, 255, 0) if q else (0, 0, 255), 1)
                x_off = max(340, img.shape[1] - 430)
                panel_bottom = min(img.shape[0] - 40, 788)
                panel = img.copy()
                cv2.rectangle(panel,
                              (max(0, x_off - 10), 30),
                              (img.shape[1] - 8, panel_bottom),
                              (20, 20, 20),
                              -1)
                cv2.addWeighted(panel, 0.58, img, 0.42, 0, img)
                cv2.rectangle(img,
                              (max(0, x_off - 10), 30),
                              (img.shape[1] - 8, panel_bottom),
                              (80, 80, 80),
                              1)
                y_off = 44
                raw_off = det.get("offset", {})
                off = det.get("target_offset", raw_off)
                target_mode = det.get("eyeball_target_mode", "raw")
                for label, key, o in [("L", "left", off.get("left")), ("R", "right", off.get("right"))]:
                    if o:
                        dx, dy, d = o
                        if target_mode == "head_adapter":
                            raw = raw_off.get(key)
                            raw_text = f" raw={raw[0]:+.1f},{raw[1]:+.1f}" if raw else ""
                            text = f"{label}*:dX={dx:+.1f} dY={dy:+.1f} d={d:.1f}px{raw_text}"
                        else:
                            text = f"{label}:dX={dx:+.1f} dY={dy:+.1f} d={d:.1f}px"
                        cv2.putText(img,
                                   text,
                                   (x_off, y_off), 0, 0.35,
                                   (0, 255, 0) if d <= TOLERANCE else (0, 0, 255), 1)
                    y_off += 18

                # 虹膜中心在本眼上下眼皮之间的位置比例（上=0%，下=100%）。
                # 仅用于观察，不参与眼球 dx/dy 或合格判断。
                lid_pos = self.get_eyeball_lid_position_signal(det=det)
                if lid_pos:
                    def _ratio_text(value):
                        return "--" if value is None else f"{value * 100.0:.1f}%"

                    cv2.putText(
                        img,
                        f"IrisY L={_ratio_text(lid_pos['left'])} "
                        f"R={_ratio_text(lid_pos['right'])} (Top0 Bottom100)",
                        (x_off, 80), 0, 0.35, (200, 220, 255), 1)

                cover_signal = det.get("lower_iris_cover")
                if cover_signal:
                    def _cover_text(label, value):
                        if not value or value.get("contact_offset_px") is None:
                            return f"{label}:--"
                        return (
                            f"{label}:{value['contact_offset_px']:+.1f}/"
                            f"{value['contact_arc_ratio'] * 100.0:.0f}%"
                            f"{value['state']}"
                        )

                    cover_ok = all(
                        cover_signal.get(side, {}).get("qualified", False)
                        for side in ("left", "right")
                    )
                    cv2.putText(
                        img,
                        f"IrisLid {_cover_text('L', cover_signal.get('left'))} "
                        f"{_cover_text('R', cover_signal.get('right'))} "
                        f"T:0+/-{cover_signal['contact_tolerance_px']:.1f}/"
                        f"{cover_signal['contact_arc_min_ratio'] * 100.0:.0f}% "
                        f"A{EYEBALL_LOWER_LID_CONTACT_SAMPLE_FRAMES}t"
                        f"{EYEBALL_LOWER_LID_CONTACT_TRIM_FRAMES} "
                        f"{'ON' if iris_overlay_visible else 'OFF'}",
                        (x_off, 98), 0, 0.29,
                        (0, 255, 0) if cover_ok else (0, 0, 255), 1)

                if eye_line_signal:
                    color = (0, 255, 0) if eye_line_signal["horizontal_ok"] else (0, 0, 255)
                    text = f"EyeLine H={eye_line_signal['angle_deg']:+.1f}°"
                    if self.show_eye_line_offset and off:
                        text += f" dY={off.get('_dy_eye_line', 0.0):+.1f}px"
                    cv2.putText(img, text, (x_off, 630), 0, 0.38, color, 1)

                # Eye control parameters: reserved rows, also in standalone mode.
                from eyeball_adaptive import hud_lines
                for text, y in zip(hud_lines(getattr(self, "eyeball_control_mode", "fixed")), (580, 596)):
                    cv2.putText(img, text, (x_off, y), 0, 0.29, (200, 200, 200), 1)

                # mp0(人中) 实时坐标 — 右侧
                if hasattr(self, '_last_lms_smooth') and self._last_lms_smooth is not None:
                    mp0 = self._last_lms_smooth[0]
                    cv2.putText(img, f"mp0:({mp0[0]:.0f},{mp0[1]:.0f})", (x_off, 612), 0, 0.35, (255, 200, 0), 1)

                # FPS 显示 (来自主采集线程的实际捕获速率)
                cap_fps = self._capture_fps
                cv2.putText(img, f"Cap FPS: {cap_fps:.1f}", (img.shape[1] - 170, 22),
                           0, 0.45, (200, 200, 200), 1)

                # EAR 眼皮高宽比 (如果有基线) — 放到右侧
                if self.eyelid_baseline and hasattr(self, '_last_lms_smooth') and self._last_lms_smooth is not None:
                    signal = self.get_eyelid_signal()
                    if signal:
                        bl = self.eyelid_baseline
                        l_ear = signal["left_ear"]
                        r_ear = signal["right_ear"]
                        l_ok = signal["left_qualified"]
                        r_ok = signal["right_qualified"]
                        e_sym = signal["eyelid_symmetry"]
                        e_sym_color = (0, 255, 0) if signal["symmetry_qualified"] else (0, 0, 255)
                        x_eye = x_off
                        y_eye = 122
                        cv2.putText(img, f"L_EAR={l_ear:.3f} (bl={bl['left_ear']:.3f})",
                                   (x_eye, y_eye), 0, 0.35,
                                   (0, 255, 0) if l_ok else (0, 0, 255), 1)
                        cv2.putText(img, f"R_EAR={r_ear:.3f} (bl={bl['right_ear']:.3f})",
                                   (x_eye, y_eye + 16), 0, 0.35,
                                   (0, 255, 0) if r_ok else (0, 0, 255), 1)
                        cv2.putText(img, f"E_SYM={e_sym:.4f} <= {EYELID_EAR_TOLERANCE}",
                                   (x_eye, y_eye + 32), 0, 0.35,
                                   e_sym_color, 1)

                # Eyebrow BIG + slope status.
                if hasattr(self, '_last_lms_smooth') and self._last_lms_smooth is not None:
                    signal = self.get_eyebrow_signal()
                    if signal:
                        l_ok = signal["left_qualified"]
                        r_ok = signal["right_qualified"]
                        x_brow = x_off
                        y_brow = 180
                        left_color = (0, 255, 0) if l_ok else (0, 0, 255)
                        right_color = (0, 255, 0) if r_ok else (0, 0, 255)
                        sym_color = (0, 255, 0) if signal["big_symmetry_ok"] else (0, 0, 255)
                        slope_color = (0, 255, 0) if signal["slope_range_qualified"] else (0, 0, 255)
                        cv2.putText(img, f"L_BIG={signal['left_brow_iris_gap']:+.3f} [{signal['left_position_min']:+.2f},{signal['left_position_max']:+.2f}]",
                                   (x_brow, y_brow), 0, 0.35,
                                   left_color, 1)
                        cv2.putText(img, f"R_BIG={signal['right_brow_iris_gap']:+.3f} [{signal['right_position_min']:+.2f},{signal['right_position_max']:+.2f}]",
                                   (x_brow, y_brow + 16), 0, 0.35,
                                   right_color, 1)
                        cv2.putText(img, f"B_SYM={signal['big_symmetry']:.3f} <= {signal['big_symmetry_tolerance']:.3f}",
                                   (x_brow, y_brow + 32), 0, 0.35,
                                   sym_color, 1)
                        cv2.putText(img, f"S L={signal['left_slope']:+.2f} R={signal['right_slope']:+.2f} {'OK' if signal['slope_range_qualified'] else 'BAD'}",
                                   (x_brow, y_brow + 48), 0, 0.35,
                                   slope_color, 1)

                # MAR 嘴部高宽比 (如果有基线) — 右侧
                if self.mouth_baseline and hasattr(self, '_last_lms_smooth') and self._last_lms_smooth is not None:
                    signal = self.get_mouth_signal()
                    if signal:
                        bl = self.mouth_baseline
                        mar = signal["mar"]
                        ok = signal["qualified"]
                        y_mouth = 250
                        cv2.putText(img, f"MAR={mar:.3f} (bl={bl['mar']:.3f})",
                                   (x_off, y_mouth), 0, 0.35,
                                   (0, 255, 0) if ok else (0, 0, 255), 1)

                # LLR 下唇比例 (如果有基线) — 右侧
                if self.lower_lip_baseline and hasattr(self, '_last_lms_smooth') and self._last_lms_smooth is not None:
                    signal = self.get_lower_lip_signal()
                    if signal:
                        bl = self.lower_lip_baseline
                        ok = signal["qualified"]
                        y_ll = 268
                        cv2.putText(img, f"LLR={signal['llr']:.3f} (bl={bl['llr']:.3f})",
                                   (x_off, y_ll), 0, 0.35,
                                   (0, 255, 0) if ok else (0, 0, 255), 1)

                # ULR 上唇比例 (如果有基线) — 右侧
                if self.upper_lip_baseline and hasattr(self, '_last_lms_smooth') and self._last_lms_smooth is not None:
                    signal = self.get_upper_lip_signal()
                    if signal:
                        bl = self.upper_lip_baseline
                        ok = signal["qualified"]
                        y_ul = 286
                        cv2.putText(img, f"ULR={signal['ulr']:.3f} (bl={bl['ulr']:.3f})",
                                   (x_off, y_ul), 0, 0.35,
                                   (0, 255, 0) if ok else (0, 0, 255), 1)

                # 嘴角偏移 HUD (如果有基线) — 右侧
                if self.mouth_corners_baseline and hasattr(self, '_last_lms_smooth') and self._last_lms_smooth is not None:
                    signal = self.get_mouth_corner_signal()
                    if signal:
                        y_mc = 314
                        cv2.putText(img, "MouthCorner:",
                                   (x_off, y_mc), 0, 0.35, (255, 200, 0), 1)
                        y_mc += 16
                        for label, key in [("L", "left"), ("R", "right")]:
                            d = signal.get(key, {})
                            dx, dy, dist = d.get("dx", 0), d.get("dy", 0), d.get("dist", 0)
                            ok = d.get("qualified", False)
                            cv2.putText(img,
                                       f"  {label}:dX={dx:+.1f} dY={dy:+.1f} d={dist:.1f}px",
                                       (x_off, y_mc), 0, 0.35,
                                       (0, 255, 0) if ok else (0, 0, 255), 1)
                            y_mc += 16

                curl_signal = self.get_mouth_corner_curl_signal()
                if curl_signal:
                    ok = curl_signal.get("all_qualified", False)
                    cv2.putText(img,
                               f"Curl L={curl_signal['left']['curl']:+.3f}/{curl_signal['left']['target']:+.3f} R={curl_signal['right']['curl']:+.3f}/{curl_signal['right']['target']:+.3f}",
                               (x_off, 362), 0, 0.35,
                               (0, 255, 0) if ok else (0, 0, 255), 1)
                    min_label = (f"Curl min>={curl_signal['min_target']:.3f}"
                                 if curl_signal.get("min_required", False) else "Curl saved")
                    cv2.putText(img,
                               f"{min_label} Sym={curl_signal['symmetry_delta']:.3f}<={curl_signal['symmetry_tolerance']:.3f}",
                               (x_off, 378), 0, 0.35,
                               (0, 255, 0) if curl_signal.get("symmetry_qualified", False) else (0, 0, 255), 1)

                # 头部定位 HUD — 鼻尖+倾斜+对称 (如果有基线) — 右侧
                if self.head_position_baseline and hasattr(self, '_last_lms_smooth') and self._last_lms_smooth is not None:
                    signal = self.get_head_signal()
                    if signal:
                        y_hd = 404
                        cv2.putText(img, "HeadPosition:",
                                   (x_off, y_hd), 0, 0.35, (255, 180, 0), 1)
                        y_hd += 16
                        # 鼻尖偏移
                        nose_ok = signal["nose_ok"]
                        cv2.putText(img,
                                   f"  Nose: dX={signal['nose_dx']:+.1f} dY={signal['nose_dy']:+.1f}",
                                   (x_off, y_hd), 0, 0.35,
                                   (0, 255, 0) if nose_ok else (0, 0, 255), 1)
                        y_hd += 16
                        # 倾斜
                        tilt_ok = signal["tilt_ok"]
                        cv2.putText(img,
                                   f"  Tilt: {signal['tilt_deg']:+.1f}°",
                                   (x_off, y_hd), 0, 0.35,
                                   (0, 255, 0) if tilt_ok else (0, 0, 255), 1)
                        y_hd += 16
                        # 对称
                        cv2.putText(img,
                                   f"  Ang={signal['eye_line_angle_deg']:+.2f} bl={signal['baseline_eye_line_angle_deg']:+.2f} d={signal['eye_line_angle_delta_deg']:+.2f}",
                                   (x_off, y_hd), 0, 0.35,
                                   (0, 255, 0) if tilt_ok else (0, 0, 255), 1)
                        y_hd += 16
                        slope = signal["eye_line_slope"]
                        bl_slope = signal["baseline_eye_line_slope"]
                        slope_delta = signal["eye_line_slope_delta"]
                        slope_text = "inf" if slope is None else f"{slope:+.4f}"
                        bl_slope_text = "inf" if bl_slope is None else f"{bl_slope:+.4f}"
                        slope_delta_text = "inf" if slope_delta is None else f"{slope_delta:+.4f}"
                        cv2.putText(img,
                                   f"  Slope={slope_text} bl={bl_slope_text} d={slope_delta_text}",
                                   (x_off, y_hd), 0, 0.35,
                                   (0, 255, 0) if tilt_ok else (0, 0, 255), 1)
                        y_hd += 16
                        cv2.putText(img,
                                   f"  Dist={signal['eye_distance']:.1f} bl={signal['baseline_eye_distance']:.1f} d={signal['eye_distance_delta']:+.1f}",
                                   (x_off, y_hd), 0, 0.35, (200, 220, 255), 1)
                        y_hd += 16
                        cv2.putText(img,
                                   f"  Ctr dX={signal['eye_center_dx']:+.1f} dY={signal['eye_center_dy']:+.1f}",
                                   (x_off, y_hd), 0, 0.35, (200, 220, 255), 1)
                        y_hd += 16
                        points_ok = signal["points_ok"]
                        cv2.putText(img,
                                   f"  RawPts N={signal['nose_dist']:.1f} L={signal['eye_left_dist']:.1f} R={signal['eye_right_dist']:.1f} <= {signal['point_tolerance']:.1f}",
                                   (x_off, y_hd), 0, 0.35,
                                   (0, 255, 0) if points_ok else (0, 0, 255), 1)
                        y_hd += 16
                        adapter_active = signal.get("adapter_active", False)
                        adapter_ok = signal.get("adapter_points_ok", False)
                        if adapter_active:
                            adapter_text = (
                                f"  AdpPts N={signal['adapter_nose_dist']:.1f} "
                                f"L={signal['adapter_eye_left_dist']:.1f} "
                                f"R={signal['adapter_eye_right_dist']:.1f} <= {signal['adapter_point_tolerance']:.1f}"
                            )
                            adapter_color = (0, 255, 0) if adapter_ok else (0, 0, 255)
                        else:
                            adapter_text = "  Adapter: none"
                            adapter_color = (200, 220, 255)
                        cv2.putText(img, adapter_text,
                                   (x_off, y_hd), 0, 0.35, adapter_color, 1)
                        y_hd += 16
                        eye_left = signal["eye_left"]
                        eye_right = signal["eye_right"]
                        cv2.putText(img,
                                   f"  eye_left:  ({eye_left[0]:.0f},{eye_left[1]:.0f})",
                                   (x_off, y_hd), 0, 0.35, (200, 220, 255), 1)
                        y_hd += 16
                        cv2.putText(img,
                                   f"  eye_right: ({eye_right[0]:.0f},{eye_right[1]:.0f})",
                                   (x_off, y_hd), 0, 0.35, (200, 220, 255), 1)
                        # 画面中心十字线 (画面正中央)
                        # 淡黄色虚十字
                        # 基线眼线位置 (半透明白色)
                        if self.show_face_overlays:
                            bl = self.head_position_baseline
                            cv2.line(img,
                                     (int(bl["eye_left"][0]), int(bl["eye_left"][1])),
                                     (int(bl["eye_right"][0]), int(bl["eye_right"][1])),
                                     (255, 255, 200), 1, cv2.LINE_AA)
                            # 鼻尖到基线鼻尖的偏移指示线 (红色)
                            if abs(signal["nose_dx"]) > 0.5 or abs(signal["nose_dy"]) > 0.5:
                                nose_cur = self._last_lms_smooth[1]  # HEAD_NOSE_IDX=1
                                nose_bl = (int(bl["nose_px"][0]), int(bl["nose_px"][1]))
                                cv2.line(img,
                                         (int(nose_cur[0]), int(nose_cur[1])),
                                         nose_bl,
                                         (0, 0, 255), 1)

                # 上唇侧面自动调整参数 — 右侧常驻显示
                roi = UPPER_LIP_SIDE_ROI
                if isinstance(self.upper_lip_baseline, dict) and self.upper_lip_baseline.get("side_roi") is not None:
                    roi = self.upper_lip_baseline["side_roi"]
                cv2.putText(img,
                           f"ULSide tol: X<={UPPER_LIP_SIDE_X_TOLERANCE:.1f}px Y<={UPPER_LIP_SIDE_Y_TOLERANCE:.1f}px",
                           (x_off, 648), 0, 0.35, (200, 220, 255), 1)
                cv2.putText(img,
                           f"ULSide cam={UPPER_LIP_SIDE_CAMERA_INDEX} VFlip={UPPER_LIP_SIDE_FLIP_VERTICAL} roi=[{roi[0]},{roi[1]},{roi[2]},{roi[3]}]",
                           (x_off, 664), 0, 0.35, (200, 220, 255), 1)

                # 下唇侧面自动调整参数 — 右侧常驻显示
                lower_roi = LOWER_LIP_SIDE_ROI
                if isinstance(self.lower_lip_baseline, dict) and self.lower_lip_baseline.get("side_roi") is not None:
                    lower_roi = self.lower_lip_baseline["side_roi"]
                cv2.putText(img,
                           f"LLSide tol: X<={LOWER_LIP_SIDE_X_TOLERANCE:.1f}px Y<={LOWER_LIP_SIDE_Y_TOLERANCE:.1f}px",
                           (x_off, 682), 0, 0.35, (200, 220, 255), 1)
                cv2.putText(img,
                           f"LLSide cam={LOWER_LIP_SIDE_CAMERA_INDEX} VFlip={LOWER_LIP_SIDE_FLIP_VERTICAL} roi=[{lower_roi[0]},{lower_roi[1]},{lower_roi[2]},{lower_roi[3]}]",
                           (x_off, 698), 0, 0.35, (200, 220, 255), 1)
                cv2.putText(img,
                           f"FrontCam HFlip={self.flip_horizontal} VFlip={self.flip_vertical} Rot180={self.rotate_180}",
                           (x_off, 716), 0, 0.35, (200, 220, 255), 1)
                cv2.putText(img,
                           f"SideProfile center={SIDE_PROFILE_CROP_SIZE}x{SIDE_PROFILE_CROP_SIZE} "
                           f"dir={SIDE_PROFILE_FACE_DIRECTION} "
                           f"NROI={self.side_nose_roi or 'SELECT'} "
                           f"edge>={SIDE_PROFILE_NOSE_EDGE_CONTRAST_MIN:.0f}",
                           (x_off, 734), 0, 0.35, (200, 220, 255), 1)
                cv2.putText(img,
                           "Nose fit: rows>=15 gap<=2 step<=4 win10 res<=2.5",
                           (x_off, 782), 0, 0.35, (200, 220, 255), 1)
                cv2.putText(img,
                           "Nose: peak1.5 turn>=2 border2 vertex3; missing=--",
                           (x_off, 798), 0, 0.35, (200, 220, 255), 1)
                cv2.putText(img, f"Upper ROI={(self.upper_lip_baseline or {}).get('side_roi', 'SELECT')}",
                           (x_off, 814), 0, 0.35, (200, 220, 255), 1)
                cv2.putText(img, f"Chin TURN ROI={self.side_chin_roi or 'SELECT'}",
                           (x_off, 830), 0, 0.35, (200, 220, 255), 1)
                cv2.putText(img, "Lip/Chin: rows15 gap2 step4 win10/20 med5",
                           (x_off, 846), 0, 0.35, (200, 220, 255), 1)
                cv2.putText(img, "LC: hold6px confirm5 miss5; chin tangent45 win10",
                           (x_off, 862), 0, 0.35, (200, 220, 255), 1)
                cv2.putText(img, f"Lower ROI={(self.lower_lip_baseline or {}).get('side_roi', 'SELECT')}",
                           (x_off, 878), 0, 0.35, (200, 220, 255), 1)
                cv2.putText(img, "Lower: color edge12 full-track peak1.5 turn2",
                           (x_off, 894), 0, 0.35, (200, 220, 255), 1)
                from eye_constants import LIP_PREVIEW_HOLD_PX, LIP_PREVIEW_CONFIRM_FRAMES, LIP_PREVIEW_CLUSTER_PX
                cv2.putText(img, f"Lip hold={LIP_PREVIEW_HOLD_PX:.0f}px confirm={LIP_PREVIEW_CONFIRM_FRAMES} cluster={LIP_PREVIEW_CLUSTER_PX:.0f}px",
                           (x_off, 910), 0, 0.35, (200, 220, 255), 1)
                slope_baseline = (
                    self.side_profile_slope_baseline
                    if isinstance(self.side_profile_slope_baseline, dict) else {})
                nose_upper_target = float(slope_baseline.get(
                    "nose_upper_slope", SIDE_PROFILE_NOSE_UPPER_TARGET_SLOPE))
                lip_chin_target = float(slope_baseline.get(
                    "lip_chin_slope", SIDE_PROFILE_LIP_CHIN_TARGET_SLOPE))
                slope_mode = "saved" if slope_baseline else "default"
                cv2.putText(img,
                           f"SP slope N-U={nose_upper_target:.2f} "
                           f"U-C={lip_chin_target:.2f} "
                           f"tol={SIDE_PROFILE_SLOPE_TOLERANCE:.2f} {slope_mode}",
                           (x_off, 750), 0, 0.35, (200, 220, 255), 1)
                cv2.putText(img,
                           f"SP line L<={SIDE_PROFILE_LOWER_LINE_TOLERANCE_PX:.1f}px "
                           f"C<={SIDE_PROFILE_CHIN_LINE_TOLERANCE_PX:.1f}px "
                           f"A{SIDE_PROFILE_SAMPLE_FRAMES}t{SIDE_PROFILE_TRIM_FRAMES} "
                           f"F{SIDE_PROFILE_PREVIEW_FILTER_FRAMES} J{SIDE_PROFILE_PREVIEW_MAX_JUMP_PX:.0f}",
                           (x_off, 766), 0, 0.35, (200, 220, 255), 1)

                # Camera settings and calibration gates remain visible in standalone mode.
                if time.monotonic() - getattr(self, '_camera_hud_time', 0.0) >= 1.0:
                    self._camera_hud_settings = camera_snapshot(self.cap)
                    ref_fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
                    self._reference_fourcc = "".join(chr((ref_fourcc >> (8 * i)) & 255) for i in range(4))
                    self._camera_hud_time = time.monotonic()
                draw_camera_hud(img, getattr(self, '_camera_hud_settings', {}))

                # Only the copied display image is changed; never modify det/targets.
                ref_context = {
                    "front_camera_index": self.camera_index,
                    "width": img.shape[1], "height": img.shape[0],
                    "fourcc": getattr(self, '_reference_fourcc', None),
                    "rotate_180": self.rotate_180,
                    "flip_horizontal": self.flip_horizontal,
                    "flip_vertical": self.flip_vertical,
                    "current_dots_visible": self.show_face_overlays,
                }
                ref_lines = draw_reference(
                    img, det, getattr(self, '_reference_head_display', (None, False)), ref_context)
                for ref_y, ref_text in zip((1022, 1038, 1054), ref_lines):
                    cv2.putText(img, ref_text, (x_off, ref_y), 0, 0.35, (100, 255, 100), 1)

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
                if (self.show_face_overlays and self.show_all_landmarks
                        and hasattr(self, '_last_lms_smooth')
                        and self._last_lms_smooth is not None):
                    lms = self._last_lms_smooth
                    for i in range(lms.shape[0]):
                        px, py = int(lms[i, 0]), int(lms[i, 1])
                        cv2.circle(img, (px, py), 1, (180, 180, 255), -1)
                        cv2.putText(img, str(i), (px + 2, py - 1), 0,  0.4, (180, 180, 255), 1)

                # 检测开关 HUD
                status_str = (
                    f"[1]MP:{'ON' if self.enable_mp else 'OFF'}  "
                    f"[2]ELL:{'ON' if self.enable_ellseg else 'OFF'}  "
                    f"[3]LMS:{'ON' if self.show_all_landmarks else 'OFF'}  "
                    f"FaceDraw={'ON' if self.show_face_overlays else 'OFF'}  "
                    f"SP4={'ON' if self.show_side_profile_preview else 'OFF'}  "
                    f"SPDash={'ON' if self.show_side_profile_expected_slope else 'OFF'}  "
                    f"SPGrid={'ON' if self.show_side_profile_coordinate_grid else 'OFF'}")
                cv2.putText(img, status_str, (10, img.shape[0] - 8),
                           0, 0.4, (200, 200, 200), 1)

                cv2.imshow("EllSeg Detect", img)

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
                elif key == ord('4'):
                    self.save_landmark_snapshot()
                elif key == ord('g') or key == ord('G'):
                    # 但注意不能和 [C] 等冲突: ord('G')=71, ord('g')=103
                    if not self.show_all_landmarks:
                        self.show_baseline_overlay = not self.show_baseline_overlay
                        print(f"[Detector] Show Baseline Overlay: {'ON' if self.show_baseline_overlay else 'OFF'}")
                elif key == ord('y') or key == ord('Y'):
                    self.show_eye_line_offset = not self.show_eye_line_offset
                    print(f"[Detector] Show EyeLine Offset: {'ON' if self.show_eye_line_offset else 'OFF'}")
                elif key in (ord('q'), ord('Q'), 27):
                    self._user_key = key

    def save_landmark_snapshot(self, logger=None):
        if not self.cap.isOpened():
            self.landmark_snapshots.invalidate("front", "camera_not_open")
        elif not self.enable_mp:
            self.landmark_snapshots.invalidate("front", "mediapipe_disabled")
        return self.landmark_snapshots.request_save(logger=logger)

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
        # __init__ 可能在创建显示线程状态前失败。
        stop_event = getattr(self, '_stop_event', None)
        if stop_event is not None:
            stop_event.set()
        display_thread = getattr(self, '_display_thread', None)
        if display_thread and display_thread.is_alive():
            display_thread.join(timeout=2)

    def request_user_stop(self, reason="external"):
        """让外部控制面板复用 Q/Esc 的中断通道。"""
        self._user_key = reason

    def clear_user_stop(self):
        """清除外部中断标记，避免路线中断后关闭摄像头。"""
        self._user_key = None

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
                "left_iris": None, "right_iris": None,
                "left_iris_ellipse": None, "right_iris_ellipse": None,
                "lower_iris_cover": None, "nose": None,
                "frame": frame, "reason": reason}

    def _calc_offset(self, det):
        """计算与基线的偏移。返回 {"left": (dx,dy,dist), "right": ..., "nose": ..., "_dy_eye_line": float}"""
        if not self.baseline or self.baseline.get("nose") is None:
            return {}
        bl = self.baseline
        offset = {}

        # 计算参考线垂直偏移 (左右虹膜 Y 均值差)
        dy_eye_line = 0.0
        bl_eye_line = bl.get("eye_line_y")
        if bl_eye_line is not None:
            li = det.get("left_iris")
            ri = det.get("right_iris")
            if li is not None and ri is not None:
                cur_line = (li[1] + ri[1]) / 2.0
                dy_eye_line = cur_line - bl_eye_line
        offset["_dy_eye_line"] = dy_eye_line

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

    def _eyeball_head_adapter_offset(self, key):
        # 暂停眼球目标的 head_adapter 修正：只追保存的绝对像素基准。
        # 原实现保留注释，其他 head_position 功能仍可读取原 adapter。
        # adapter = self.head_adapter_baseline if isinstance(getattr(self, "head_adapter_baseline", None), dict) else None
        # if not adapter:
        #     return (0.0, 0.0)
        # adapter_key = {
        #     "left": "eye_right_offset",
        #     "right": "eye_left_offset",
        #     "nose": "nose_offset",
        # }.get(key)
        # if adapter_key is None:
        #     return (0.0, 0.0)
        # value = adapter.get(adapter_key)
        # if value is None:
        #     return (0.0, 0.0)
        # return (float(value[0]), float(value[1]))
        return (0.0, 0.0)

    def _calc_target_offset(self, det):
        raw = self._calc_offset(det)
        if not raw:
            return {}
        target_offset = {}
        for key, value in raw.items():
            if key.startswith("_"):
                target_offset[key] = value
                continue
            dx, dy, _dist = value
            ox, oy = self._eyeball_head_adapter_offset(key)
            adj_dx = dx - ox
            adj_dy = dy - oy
            target_offset[key] = (adj_dx, adj_dy, float(np.sqrt(adj_dx * adj_dx + adj_dy * adj_dy)))
        return target_offset

    def _eyeball_target_mode(self):
        # 暂停适配目标模式，HUD 和结果统一标记为未修正的基准模式。
        # return "head_adapter" if isinstance(getattr(self, "head_adapter_baseline", None), dict) else "raw"
        return "raw"

    def _eyeball_target_point(self, key):
        if not self.baseline:
            return None
        key_map = {"left": "iris_left", "right": "iris_right", "nose": "nose"}
        base_pos = self.baseline.get(key_map.get(key))
        if base_pos is None:
            return None
        ox, oy = self._eyeball_head_adapter_offset(key)
        return (float(base_pos[0]) + ox, float(base_pos[1]) + oy)

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
            self.landmark_snapshots.invalidate("front", "camera_read_failed")
            return False, None

        # 在所有检测和绘制之前统一旋转，保证关键点与 JSON 使用同一坐标系。
        if self.rotate_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        # 统计实际捕获帧率
        self._cap_fps_counter += 1
        now = time.time()
        elapsed = now - self._cap_fps_time
        if elapsed >= 1.0:
            self._capture_fps = round(self._cap_fps_counter / elapsed, 1)
            self._cap_fps_counter = 0
            self._cap_fps_time = now

        return True, frame

    def detect(self, frame):
        t_total = time.perf_counter()
        detected_at = timestamp()
        h, w = frame.shape[:2]
        self._timestamp += 33

        # ---------- MediaPipe 开关 ----------
        if not self.enable_mp:
            self.landmark_snapshots.invalidate("front", "mediapipe_disabled")
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
            self.landmark_snapshots.invalidate("front", "no_face")
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
                ("right", LEFT_EYE_IDX, self.ell_smooth_R),
                ("left", RIGHT_EYE_IDX, self.ell_smooth_L),
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
                (scx, scy), (saw, sah), sang = smooth_iris
                out[f"{side}_iris"] = (int(scx + x1), int(scy + y1))
                out[f"{side}_iris_ellipse"] = (
                    (float(scx + x1), float(scy + y1)),
                    (float(saw), float(sah)),
                    float(sang),
                )

        out["lower_iris_cover"] = self.get_lower_iris_cover_signal(
            det=out, lms_smooth=lms_smooth)

        # 与基线比较
        offset = self._calc_offset(out)
        target_offset = self._calc_target_offset(out)
        out["offset"] = offset
        out["target_offset"] = target_offset
        out["eyeball_target_mode"] = self._eyeball_target_mode()

        if target_offset and self.baseline:
            out["qualified"] = self._is_qualified(target_offset)
        else:
            out["qualified"] = False

        t_total = 1000 * (time.perf_counter() - t_total)
        out["_timing"] = (t_total, t_mp, t_ell)

        # 保存 lms_smooth 供 get_eyelid_signal() 使用
        self._last_lms_smooth = lms_smooth

        self.landmark_snapshots.publish("front", {
            "status": "detected", "camera_index": self.camera_index,
            "detected_at": detected_at, "model_timestamp_ms": self._timestamp,
            "frame_size": {"width": w, "height": h},
            "coordinate_system": "detector_input_frame; x right, y down; raw z is MediaPipe normalized depth",
            "flip_horizontal": self.flip_horizontal, "flip_vertical": self.flip_vertical,
            "rotate_180": self.rotate_180,
            "detector": "MediaPipe FaceLandmarker", "face_index": 0,
            "point_count": len(face),
            "landmarks": mediapipe_points(face, lms_smooth, w, h),
        })
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
        offset = det.get("target_offset", det.get("offset", {}))
        rs["eyeball_target_mode"] = det.get("eyeball_target_mode", "raw")
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
        offset = self.last_result.get("target_offset", self.last_result.get("offset"))
        if not offset:
            return None

        guide = {}
        guide["eyeball_target_mode"] = self.last_result.get("eyeball_target_mode", "raw")
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
        eye_line_y = (det["left_iris"][1] + det["right_iris"][1]) / 2.0
        self.baseline = {
            "nose": det["nose"],
            "iris_left": det["left_iris"],
            "iris_right": det["right_iris"],
            "eye_line_y": eye_line_y,
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
                lms_smooth, RIGHT_EAR_PAIRS, RIGHT_EYE_CORNERS),
            "right": EllSegDetector._calc_ear(
                lms_smooth, LEFT_EAR_PAIRS, LEFT_EYE_CORNERS),
        }

    def get_eyeball_lid_position_signal(self, det=None, lms_smooth=None):
        """返回虹膜中心在上下眼皮之间的比例位置（上=0，下=1）。

        L/R 遵循用户标记和舵机语义：L 是画面右侧眼，R 是画面左侧眼。
        比例沿两组上眼皮点均值到两组下眼皮点均值的局部轴投影计算，
        因此头部轻微倾斜时不会直接把屏幕 Y 值误当成眼内垂直位置。
        此观察量不做截断，也不参与原始眼球 dx/dy 或通过判定。
        """
        if lms_smooth is None:
            lms_smooth = getattr(self, "_last_lms_smooth", None)
        if det is None:
            det = self.last_result
        if lms_smooth is None or not det:
            return None

        from eye_constants import LEFT_EAR_PAIRS, RIGHT_EAR_PAIRS

        def position_ratio(iris_center, pairs):
            if iris_center is None:
                return None
            upper = np.mean(
                [np.asarray(lms_smooth[top_i], dtype=np.float64)
                 for top_i, _ in pairs], axis=0)
            lower = np.mean(
                [np.asarray(lms_smooth[bottom_i], dtype=np.float64)
                 for _, bottom_i in pairs], axis=0)
            lid_axis = lower - upper
            axis_len_sq = float(np.dot(lid_axis, lid_axis))
            if axis_len_sq < 1e-6:
                return None
            iris = np.asarray(iris_center, dtype=np.float64)
            return float(np.dot(iris - upper, lid_axis) / axis_len_sq)

        return {
            "left": position_ratio(det.get("left_iris"), RIGHT_EAR_PAIRS),
            "right": position_ratio(det.get("right_iris"), LEFT_EAR_PAIRS),
        }

    def get_lower_iris_cover_signal(self, det=None, lms_smooth=None):
        """判断虹膜理论下圆弧是否与下眼皮贴合，供 HUD 和独立上下调整使用。

        EllSeg 椭圆用于推算完整虹膜边界，MediaPipe 下眼皮关键点通过
        分段线性插值得到眼皮曲线。对椭圆下半弧逐点计算
        ``理论圆弧 y - 下眼皮 y``。第 90 百分位作为有符号贴合偏差：
        负值表示仍有空隙，0 表示贴合，正值表示下眼皮遮住过多。
        通过条件要求偏差接近 0，且足够比例的下弧处在双向贴合容差内，
        避免单个约 1px 的跳点误判。

        L/R 遵循用户和舵机语义：L 是画面右侧眼，R 是画面左侧眼。
        本信号不修改虹膜中心、原始 dx/dy 或眼球合格判断。
        """
        if lms_smooth is None:
            lms_smooth = getattr(self, "_last_lms_smooth", None)
        if det is None:
            det = self.last_result
        if lms_smooth is None or not det:
            return None

        from eye_constants import (
            EYEBALL_LOWER_LID_CONTACT_ARC_MIN_RATIO,
            EYEBALL_LOWER_LID_CONTACT_TOLERANCE,
            LEFT_LOWER_LID_POINTS,
            RIGHT_LOWER_LID_POINTS,
        )

        contact_tol = float(EYEBALL_LOWER_LID_CONTACT_TOLERANCE)
        contact_arc_min = float(EYEBALL_LOWER_LID_CONTACT_ARC_MIN_RATIO)

        def empty_measurement():
            return {
                "contact_offset_px": None,
                "contact_offset_ratio": None,
                "contact_arc_ratio": None,
                "qualified": False,
                "state": "--",
                "arc_points": None,
                "lid_curve_points": None,
                "contact_mask": None,
                "over_cover_mask": None,
                "depth_segment": None,
            }

        def measure(ellipse, lower_lid_indices):
            if ellipse is None:
                return empty_measurement()

            try:
                (cx, cy), (axis_w, axis_h), angle_deg = ellipse
                cx, cy = float(cx), float(cy)
                radius_a = float(axis_w) * 0.5
                radius_b = float(axis_h) * 0.5
                angle = np.deg2rad(float(angle_deg))
            except (TypeError, ValueError):
                return empty_measurement()
            if radius_a < 1e-6 or radius_b < 1e-6:
                return empty_measurement()

            theta = np.linspace(0.0, 2.0 * np.pi, 361, endpoint=False)
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            ellipse_x = cx + radius_a * cos_t * cos_a - radius_b * sin_t * sin_a
            ellipse_y = cy + radius_a * cos_t * sin_a + radius_b * sin_t * cos_a

            lid_points = np.asarray(
                [lms_smooth[index] for index in lower_lid_indices],
                dtype=np.float64,
            )
            order = np.argsort(lid_points[:, 0])
            lid_x_sorted = lid_points[order, 0]
            lid_y_sorted = lid_points[order, 1]
            lid_x, inverse = np.unique(lid_x_sorted, return_inverse=True)
            if lid_x.size < 2:
                return empty_measurement()
            lid_y = np.zeros(lid_x.size, dtype=np.float64)
            lid_counts = np.zeros(lid_x.size, dtype=np.float64)
            np.add.at(lid_y, inverse, lid_y_sorted)
            np.add.at(lid_counts, inverse, 1.0)
            lid_y /= np.maximum(lid_counts, 1.0)

            lower_arc = (
                (ellipse_y >= cy)
                & (ellipse_x >= lid_x[0])
                & (ellipse_x <= lid_x[-1])
            )
            if int(np.count_nonzero(lower_arc)) < 8:
                return empty_measurement()

            arc_x = ellipse_x[lower_arc]
            arc_y = ellipse_y[lower_arc]
            lower_lid_y = np.interp(arc_x, lid_x, lid_y)
            arc_order = np.argsort(arc_x)
            arc_x = arc_x[arc_order]
            arc_y = arc_y[arc_order]
            lower_lid_y = lower_lid_y[arc_order]
            depths = arc_y - lower_lid_y
            iris_height = float(np.max(ellipse_y) - np.min(ellipse_y))
            if iris_height < 1e-6:
                return empty_measurement()

            robust_depth = float(np.percentile(depths, 90.0))
            contact_mask = np.abs(depths) <= contact_tol
            contact_arc_ratio = float(np.mean(contact_mask))
            qualified = (
                abs(robust_depth) <= contact_tol
                and contact_arc_ratio >= contact_arc_min
            )
            if qualified:
                state = "FIT"
            elif robust_depth < -contact_tol:
                state = "GAP"
            elif robust_depth > contact_tol:
                state = "COVER"
            else:
                state = "NARROW"
            representative_index = int(np.argmin(np.abs(depths - robust_depth)))
            return {
                # 有符号：负=仍有空隙，0=贴合，正=下眼皮遮住过多。
                "contact_offset_px": robust_depth,
                "contact_offset_ratio": robust_depth / iris_height,
                "contact_arc_ratio": contact_arc_ratio,
                "qualified": qualified,
                "state": state,
                "arc_points": np.column_stack((arc_x, arc_y)),
                "lid_curve_points": np.column_stack((arc_x, lower_lid_y)),
                "contact_mask": contact_mask,
                "over_cover_mask": depths > contact_tol,
                "depth_segment": (
                    (float(arc_x[representative_index]),
                     float(lower_lid_y[representative_index])),
                    (float(arc_x[representative_index]),
                     float(arc_y[representative_index])),
                ),
            }

        return {
            "left": measure(
                det.get("left_iris_ellipse"), RIGHT_LOWER_LID_POINTS),
            "right": measure(
                det.get("right_iris_ellipse"), LEFT_LOWER_LID_POINTS),
            "contact_tolerance_px": contact_tol,
            "contact_arc_min_ratio": contact_arc_min,
        }

    @staticmethod
    def _calc_brow_geometry(lms_smooth, brow_points, corners):
        """Calculate one eyebrow's mirrored slope."""
        outer_i, inner_i = corners
        outer = lms_smooth[outer_i]
        inner = lms_smooth[inner_i]
        eye_width = abs(inner[0] - outer[0])
        if eye_width < 1e-6:
            return {"slope": 0.0}

        brow_xy = np.array([lms_smooth[i] for i in brow_points], dtype=float)
        d_inner = np.linalg.norm(brow_xy - inner, axis=1)
        d_outer = np.linalg.norm(brow_xy - outer, axis=1)
        inner_brow = brow_xy[int(np.argmin(d_inner))]
        outer_brow = brow_xy[int(np.argmin(d_outer))]
        return {"slope": float((outer_brow[1] - inner_brow[1]) / eye_width)}

    @staticmethod
    def _calc_brow_iris_gaps(lms_smooth):
        """
        Project each brow-to-iris vector onto the axis perpendicular to the
        two iris centers, then normalize by that eye's corner-to-corner width.
        """
        from eye_constants import (
            LEFT_BROW_IRIS_POINTS, RIGHT_BROW_IRIS_POINTS,
            LEFT_BROW_IRIS_WEIGHTS, RIGHT_BROW_IRIS_WEIGHTS,
            LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER,
            LEFT_BROW_CORNERS, RIGHT_BROW_CORNERS,
        )

        left_iris = lms_smooth[LEFT_IRIS_CENTER]
        right_iris = lms_smooth[RIGHT_IRIS_CENTER]
        eye_axis = right_iris - left_iris
        axis_norm = float(np.linalg.norm(eye_axis))
        if axis_norm < 1e-6:
            return {"left": 0.0, "right": 0.0}

        vertical_axis = np.array([-eye_axis[1], eye_axis[0]], dtype=float) / axis_norm

        def normalized_gap(brow_indices, brow_weights, iris_idx, corners):
            outer_i, inner_i = corners
            eye_width = float(np.linalg.norm(
                lms_smooth[inner_i] - lms_smooth[outer_i]))
            if eye_width < 1e-6:
                return 0.0
            weighted_sum = 0.0
            weight_total = 0.0
            for brow_idx, weight in zip(brow_indices, brow_weights):
                brow_to_iris = lms_smooth[brow_idx] - lms_smooth[iris_idx]
                value = float(np.dot(brow_to_iris, vertical_axis) / eye_width)
                weighted_sum += value * float(weight)
                weight_total += float(weight)
            if weight_total < 1e-6:
                return 0.0
            return float(weighted_sum / weight_total)

        return {
            "left": normalized_gap(
                RIGHT_BROW_IRIS_POINTS, RIGHT_BROW_IRIS_WEIGHTS,
                RIGHT_IRIS_CENTER, RIGHT_BROW_CORNERS),
            "right": normalized_gap(
                LEFT_BROW_IRIS_POINTS, LEFT_BROW_IRIS_WEIGHTS,
                LEFT_IRIS_CENTER, LEFT_BROW_CORNERS),
        }

    @staticmethod
    def _calc_eyebrow_metrics(lms_smooth):
        """Calculate eyebrow slope and Brow-Iris Gap."""
        from eye_constants import (
            LEFT_BROW_POINTS, LEFT_BROW_CORNERS,
            RIGHT_BROW_POINTS, RIGHT_BROW_CORNERS,
        )
        left = EllSegDetector._calc_brow_geometry(
            lms_smooth, RIGHT_BROW_POINTS, RIGHT_BROW_CORNERS)
        right = EllSegDetector._calc_brow_geometry(
            lms_smooth, LEFT_BROW_POINTS, LEFT_BROW_CORNERS)
        brow_iris_gaps = EllSegDetector._calc_brow_iris_gaps(lms_smooth)
        return {
            "left_slope": left["slope"],
            "right_slope": right["slope"],
            "slope_symmetry": abs(left["slope"] - right["slope"]),
            "left_brow_iris_gap": brow_iris_gaps["left"],
            "right_brow_iris_gap": brow_iris_gaps["right"],
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
              eyelid_symmetry: 左右当前 EAR 高宽比差值
              left_qualified, right_qualified: 是否在容差范围内
              symmetry_qualified: 左右当前 EAR 是否对称
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
        eyelid_symmetry = abs(ears["left"] - ears["right"])

        return {
            "left_ear": ears["left"],
            "right_ear": ears["right"],
            "left_delta": left_delta,
            "right_delta": right_delta,
            "eyelid_symmetry": eyelid_symmetry,
            "left_qualified": abs(left_delta) <= tol,
            "right_qualified": abs(right_delta) <= tol,
            "symmetry_qualified": eyelid_symmetry <= tol,
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
    # Eyebrow BIG + slope signal.
    # ==================================================================

    def get_eyebrow_signal(self, lms_smooth=None):
        """
        Return eyebrow status using only BIG (Brow-Iris Gap) and slope.
        BIG judges brow height against iris center; slope judges brow shape.
        """
        if lms_smooth is None:
            if not hasattr(self, '_last_lms_smooth') or self._last_lms_smooth is None:
                return None
            lms_smooth = self._last_lms_smooth

        metrics = self._calc_eyebrow_metrics(lms_smooth)
        from eye_constants import (
            EYEBROW_BIG_SYMMETRY_TOLERANCE,
            EYEBROW_CUSTOM_BIG_TOLERANCE,
            EYEBROW_CUSTOM_SLOPE_TOLERANCE,
        )

        bl = self.eyebrow_baseline
        if not (
            bl
            and "left_brow_iris_gap" in bl
            and "right_brow_iris_gap" in bl
            and "left_slope" in bl
            and "right_slope" in bl
        ):
            return None

        left_range = (
            float(bl["left_brow_iris_gap"]) - EYEBROW_CUSTOM_BIG_TOLERANCE,
            float(bl["left_brow_iris_gap"]) + EYEBROW_CUSTOM_BIG_TOLERANCE,
        )
        right_range = (
            float(bl["right_brow_iris_gap"]) - EYEBROW_CUSTOM_BIG_TOLERANCE,
            float(bl["right_brow_iris_gap"]) + EYEBROW_CUSTOM_BIG_TOLERANCE,
        )
        left_slope_range = (
            float(bl["left_slope"]) - EYEBROW_CUSTOM_SLOPE_TOLERANCE,
            float(bl["left_slope"]) + EYEBROW_CUSTOM_SLOPE_TOLERANCE,
        )
        right_slope_range = (
            float(bl["right_slope"]) - EYEBROW_CUSTOM_SLOPE_TOLERANCE,
            float(bl["right_slope"]) + EYEBROW_CUSTOM_SLOPE_TOLERANCE,
        )

        def range_delta(value, value_range):
            lo, hi = value_range
            if value < lo:
                return value - lo
            if value > hi:
                return value - hi
            return 0.0

        left_big = metrics["left_brow_iris_gap"]
        right_big = metrics["right_brow_iris_gap"]
        left_position_delta = range_delta(left_big, left_range)
        right_position_delta = range_delta(right_big, right_range)
        left_position_ok = left_position_delta == 0.0
        right_position_ok = right_position_delta == 0.0

        left_slope = metrics["left_slope"]
        right_slope = metrics["right_slope"]
        left_slope_range_ok = left_slope_range[0] <= left_slope <= left_slope_range[1]
        right_slope_range_ok = right_slope_range[0] <= right_slope <= right_slope_range[1]
        slope_range_ok = left_slope_range_ok and right_slope_range_ok

        big_symmetry = abs(left_big - right_big)
        big_symmetry_ok = big_symmetry <= EYEBROW_BIG_SYMMETRY_TOLERANCE
        left_side_ok = left_position_ok and left_slope_range_ok
        right_side_ok = right_position_ok and right_slope_range_ok

        return {
            "left_slope": left_slope,
            "right_slope": right_slope,
            "slope_symmetry": metrics["slope_symmetry"],
            "left_brow_iris_gap": left_big,
            "right_brow_iris_gap": right_big,
            "left_position": left_big,
            "right_position": right_big,
            "left_position_target": (left_range[0] + left_range[1]) / 2.0,
            "right_position_target": (right_range[0] + right_range[1]) / 2.0,
            "left_position_min": left_range[0],
            "left_position_max": left_range[1],
            "right_position_min": right_range[0],
            "right_position_max": right_range[1],
            "left_position_delta": left_position_delta,
            "right_position_delta": right_position_delta,
            "big_symmetry": big_symmetry,
            "big_symmetry_ok": big_symmetry_ok,
            "big_symmetry_tolerance": EYEBROW_BIG_SYMMETRY_TOLERANCE,
            "left_slope_min": left_slope_range[0],
            "left_slope_max": left_slope_range[1],
            "right_slope_min": right_slope_range[0],
            "right_slope_max": right_slope_range[1],
            "left_slope_range_qualified": left_slope_range_ok,
            "right_slope_range_qualified": right_slope_range_ok,
            "slope_range_qualified": slope_range_ok,
            "left_qualified": left_side_ok,
            "right_qualified": right_side_ok,
            "symmetry_qualified": big_symmetry_ok,
            "qualified": left_side_ok and right_side_ok and big_symmetry_ok,
            "baseline": bl,
        }

    def save_current_eyebrow_baseline(self, lms_smooth=None):
        """Save the current BIG and slope as the custom eyebrow baseline."""
        if lms_smooth is None:
            if not hasattr(self, '_last_lms_smooth') or self._last_lms_smooth is None:
                print("[Eyebrow Baseline] Cannot save: no landmark data")
                return False
            lms_smooth = self._last_lms_smooth

        metrics = self._calc_eyebrow_metrics(lms_smooth)
        self.eyebrow_baseline = {
            "left_slope": metrics["left_slope"],
            "right_slope": metrics["right_slope"],
            "slope_symmetry": metrics["slope_symmetry"],
            "left_brow_iris_gap": metrics["left_brow_iris_gap"],
            "right_brow_iris_gap": metrics["right_brow_iris_gap"],
            "has_brow_iris_gap": True,
        }
        save_eyebrow_baseline(metrics=metrics)
        return True

    # ==================================================================
    # Upper Lip ULR 计算与信号 — A16
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

        if self.upper_lip_baseline is None or "ulr" not in self.upper_lip_baseline:
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
        side_upper_tip = None
        side_roi = None
        if isinstance(self.upper_lip_baseline, dict):
            side_upper_tip = self.upper_lip_baseline.get("side_upper_tip")
            side_roi = self.upper_lip_baseline.get("side_roi")
        self.upper_lip_baseline = {"ulr": ulr}
        if side_upper_tip is not None:
            self.upper_lip_baseline["side_upper_tip"] = side_upper_tip
        if side_roi is not None:
            self.upper_lip_baseline["side_roi"] = side_roi
        save_upper_lip_baseline(ulr=ulr, side_upper_tip=side_upper_tip, side_roi=side_roi)
        return True

    # ==================================================================
    # 头部定位信号 — 鼻尖+眼线
    # ==================================================================

    def get_eye_line_screen_signal(self, lms_smooth=None):
        """
        当前左右眼外角连线相对屏幕水平线的角度。

        正值表示从左眼外角到右眼外角向下倾斜，屏幕坐标系里相当于顺时针。
        """
        if lms_smooth is None:
            if not hasattr(self, '_last_lms_smooth') or self._last_lms_smooth is None:
                return None
            lms_smooth = self._last_lms_smooth

        from eye_constants import HEAD_EYE_LEFT_IDX, HEAD_EYE_RIGHT_IDX, HEAD_TILT_TOLERANCE

        eye_left = lms_smooth[HEAD_EYE_LEFT_IDX]
        eye_right = lms_smooth[HEAD_EYE_RIGHT_IDX]
        dx = float(eye_right[0] - eye_left[0])
        dy = float(eye_right[1] - eye_left[1])
        if abs(dx) <= 1e-6 and abs(dy) <= 1e-6:
            return None

        angle_deg = float(np.degrees(np.arctan2(dy, dx)))
        if angle_deg > 90.0:
            angle_deg -= 180.0
        elif angle_deg <= -90.0:
            angle_deg += 180.0

        return {
            "left": (float(eye_left[0]), float(eye_left[1])),
            "right": (float(eye_right[0]), float(eye_right[1])),
            "mid_y": float((eye_left[1] + eye_right[1]) / 2.0),
            "angle_deg": angle_deg,
            "tolerance": HEAD_TILT_TOLERANCE,
            "horizontal_ok": abs(angle_deg) <= HEAD_TILT_TOLERANCE,
        }

    def get_head_signal(self, lms_smooth=None):
        """
        返回头部定位检查信号 — 基于鼻尖基线 + 眼线基准。

        返回 dict:
            nose_dx, nose_dy:   鼻尖像素偏移 (当前-基线)
            nose_dist:          鼻尖欧氏偏移距离 (px)
            tilt_deg:           眼线倾斜角 (度), 正值=顺时针
            symmetry:           左右眼到鼻尖距离比 (1.0=完美对称)
            nose_ok, tilt_ok, symmetry_ok: 单项合格标志
            all_ok:             全部合格
        """
        if self.head_position_baseline is None:
            return None
        if lms_smooth is None:
            if not hasattr(self, '_last_lms_smooth') or self._last_lms_smooth is None:
                return None
            lms_smooth = self._last_lms_smooth

        from eye_constants import (
            HEAD_NOSE_IDX, HEAD_EYE_LEFT_IDX, HEAD_EYE_RIGHT_IDX,
            HEAD_ADAPTER_POINT_TOLERANCE,
            HEAD_POSITION_POINT_TOLERANCE,
            HEAD_NOSE_DX_TOLERANCE, HEAD_NOSE_DY_TOLERANCE,
            HEAD_TILT_TOLERANCE, HEAD_SYMMETRY_TOLERANCE,
        )
        bl = self.head_position_baseline
        adapter = self.head_adapter_baseline if isinstance(getattr(self, "head_adapter_baseline", None), dict) else None

        # --- 鼻尖偏移 ---
        nose_cur = lms_smooth[HEAD_NOSE_IDX]
        eye_L_cur = lms_smooth[HEAD_EYE_LEFT_IDX]
        eye_R_cur = lms_smooth[HEAD_EYE_RIGHT_IDX]
        eye_L_bl = np.array(bl["eye_left"], dtype=np.float64)
        eye_R_bl = np.array(bl["eye_right"], dtype=np.float64)
        nose_delta = np.asarray(nose_cur[:2], dtype=np.float64) - np.array(bl["nose_px"], dtype=np.float64)
        eye_left_delta = np.asarray(eye_L_cur[:2], dtype=np.float64) - eye_L_bl
        eye_right_delta = np.asarray(eye_R_cur[:2], dtype=np.float64) - eye_R_bl
        nose_dx = float(nose_delta[0])
        nose_dy = float(nose_delta[1])
        nose_dist = float(np.linalg.norm(nose_delta))
        eye_left_dist = float(np.linalg.norm(eye_left_delta))
        eye_right_dist = float(np.linalg.norm(eye_right_delta))
        point_max_dist = float(max(nose_dist, eye_left_dist, eye_right_dist))
        points_ok = point_max_dist <= HEAD_POSITION_POINT_TOLERANCE
        nose_ok = abs(nose_dx) <= HEAD_NOSE_DX_TOLERANCE and abs(nose_dy) <= HEAD_NOSE_DY_TOLERANCE

        # --- 眼线倾斜角 ---
        eye_L_cur = lms_smooth[HEAD_EYE_LEFT_IDX]
        eye_R_cur = lms_smooth[HEAD_EYE_RIGHT_IDX]
        eye_L_bl = np.array(bl["eye_left"], dtype=np.float64)
        eye_R_bl = np.array(bl["eye_right"], dtype=np.float64)

        # 基线眼线角度
        bl_vec = eye_R_bl - eye_L_bl
        bl_angle = np.degrees(np.arctan2(bl_vec[1], bl_vec[0]))
        if bl_angle > 90.0:
            bl_angle -= 180.0
        elif bl_angle <= -90.0:
            bl_angle += 180.0
        bl_slope = float(bl_vec[1] / bl_vec[0]) if abs(float(bl_vec[0])) > 1e-6 else None
        bl_eye_dist = float(bl.get("eye_distance") or np.linalg.norm(bl_vec))
        bl_eye_center = np.array(
            bl.get("eye_center") or ((eye_L_bl + eye_R_bl) * 0.5),
            dtype=np.float64,
        )
        # 当前眼线角度
        cur_vec = np.array([eye_R_cur[0] - eye_L_cur[0],
                            eye_R_cur[1] - eye_L_cur[1]], dtype=np.float64)
        cur_angle = np.degrees(np.arctan2(cur_vec[1], cur_vec[0]))
        if cur_angle > 90.0:
            cur_angle -= 180.0
        elif cur_angle <= -90.0:
            cur_angle += 180.0
        cur_slope = float(cur_vec[1] / cur_vec[0]) if abs(float(cur_vec[0])) > 1e-6 else None
        cur_eye_dist = float(np.linalg.norm(cur_vec))
        cur_eye_center = (np.asarray(eye_L_cur, dtype=np.float64) + np.asarray(eye_R_cur, dtype=np.float64)) * 0.5
        eye_center_delta = cur_eye_center - bl_eye_center
        # 眼线倾斜: 正值=顺时针 (相对于基线)
        tilt_deg = cur_angle - bl_angle
        # 归一化到 (-180, 180]
        if tilt_deg > 180:
            tilt_deg -= 360
        elif tilt_deg <= -180:
            tilt_deg += 360
        tilt_ok = abs(tilt_deg) <= HEAD_TILT_TOLERANCE

        # --- 左右对称 ---
        dist_L = np.linalg.norm(np.array([eye_L_cur[0] - nose_cur[0],
                                          eye_L_cur[1] - nose_cur[1]]))
        dist_R = np.linalg.norm(np.array([eye_R_cur[0] - nose_cur[0],
                                          eye_R_cur[1] - nose_cur[1]]))
        if dist_R > 0.5:
            symmetry = dist_L / dist_R
        else:
            symmetry = 0.0
        symmetry_ok = abs(1.0 - symmetry) <= HEAD_SYMMETRY_TOLERANCE
        slope_delta = None if cur_slope is None or bl_slope is None else cur_slope - bl_slope
        adapter_active = adapter is not None
        adapter_nose_delta = nose_delta
        adapter_eye_left_delta = eye_left_delta
        adapter_eye_right_delta = eye_right_delta
        adapter_eye_center_delta = eye_center_delta
        adapter_eye_distance_delta = cur_eye_dist - bl_eye_dist
        adapter_angle_delta = tilt_deg
        adapter_slope_delta = slope_delta
        if adapter_active:
            adapter_nose_delta = nose_delta - np.array(adapter.get("nose_offset", (0.0, 0.0)), dtype=np.float64)
            adapter_eye_left_delta = eye_left_delta - np.array(adapter.get("eye_left_offset", (0.0, 0.0)), dtype=np.float64)
            adapter_eye_right_delta = eye_right_delta - np.array(adapter.get("eye_right_offset", (0.0, 0.0)), dtype=np.float64)
            adapter_eye_center_delta = eye_center_delta - np.array(adapter.get("eye_center_offset", (0.0, 0.0)), dtype=np.float64)
            adapter_eye_distance_delta = (cur_eye_dist - bl_eye_dist) - float(adapter.get("eye_distance_offset", 0.0) or 0.0)
            adapter_angle_delta = tilt_deg - float(adapter.get("eye_line_angle_offset_deg", 0.0) or 0.0)
            slope_offset = adapter.get("eye_line_slope_offset")
            adapter_slope_delta = None if slope_delta is None or slope_offset is None else slope_delta - float(slope_offset)

        adapter_nose_dist = float(np.linalg.norm(adapter_nose_delta))
        adapter_eye_left_dist = float(np.linalg.norm(adapter_eye_left_delta))
        adapter_eye_right_dist = float(np.linalg.norm(adapter_eye_right_delta))
        adapter_point_max_dist = float(max(adapter_nose_dist, adapter_eye_left_dist, adapter_eye_right_dist))
        adapter_points_ok = adapter_point_max_dist <= HEAD_ADAPTER_POINT_TOLERANCE

        return {
            "nose_dx": float(nose_dx),
            "nose_dy": float(nose_dy),
            "nose_dist": float(nose_dist),
            "eye_center": (float(cur_eye_center[0]), float(cur_eye_center[1])),
            "baseline_eye_center": (float(bl_eye_center[0]), float(bl_eye_center[1])),
            "eye_center_dx": float(eye_center_delta[0]),
            "eye_center_dy": float(eye_center_delta[1]),
            "eye_distance": cur_eye_dist,
            "baseline_eye_distance": bl_eye_dist,
            "eye_distance_delta": cur_eye_dist - bl_eye_dist,
            "eye_line_angle_deg": float(cur_angle),
            "baseline_eye_line_angle_deg": float(bl_angle),
            "eye_line_angle_delta_deg": float(tilt_deg),
            "eye_line_slope": cur_slope,
            "baseline_eye_line_slope": bl_slope,
            "eye_line_slope_delta": slope_delta,
            "eye_left_dist": eye_left_dist,
            "eye_right_dist": eye_right_dist,
            "eye_left": (float(eye_L_cur[0]), float(eye_L_cur[1])),
            "eye_right": (float(eye_R_cur[0]), float(eye_R_cur[1])),
            "point_max_dist": point_max_dist,
            "point_tolerance": HEAD_POSITION_POINT_TOLERANCE,
            "adapter_active": adapter_active,
            "adapter_point_tolerance": HEAD_ADAPTER_POINT_TOLERANCE,
            "adapter_nose_dx": float(adapter_nose_delta[0]),
            "adapter_nose_dy": float(adapter_nose_delta[1]),
            "adapter_nose_dist": adapter_nose_dist,
            "adapter_eye_center_dx": float(adapter_eye_center_delta[0]),
            "adapter_eye_center_dy": float(adapter_eye_center_delta[1]),
            "adapter_eye_distance_delta": float(adapter_eye_distance_delta),
            "adapter_eye_line_angle_delta_deg": float(adapter_angle_delta),
            "adapter_eye_line_slope_delta": adapter_slope_delta,
            "adapter_eye_left_dist": adapter_eye_left_dist,
            "adapter_eye_right_dist": adapter_eye_right_dist,
            "adapter_point_max_dist": adapter_point_max_dist,
            "adapter_points_ok": adapter_points_ok,
            "tilt_deg": float(tilt_deg),
            "symmetry": float(symmetry),
            "nose_ok": nose_ok,
            "points_ok": points_ok,
            "tilt_ok": tilt_ok,
            "symmetry_ok": symmetry_ok,
            "all_ok": (adapter_points_ok if adapter_active else points_ok) and nose_ok and tilt_ok and symmetry_ok,
        }

    def save_current_head_adapter_baseline(self, lms_smooth=None):
        """Save the current head shape offset relative to the template head_position."""
        if self.head_position_baseline is None:
            print("[HeadAdapter Baseline] Cannot save: no head_position baseline")
            return False
        signal = self.get_head_signal(lms_smooth)
        if signal is None:
            print("[HeadAdapter Baseline] Cannot save: no landmark data")
            return False

        adapter = {
            "nose_offset": (signal["nose_dx"], signal["nose_dy"]),
            "eye_left_offset": (
                signal["eye_left"][0] - self.head_position_baseline["eye_left"][0],
                signal["eye_left"][1] - self.head_position_baseline["eye_left"][1],
            ),
            "eye_right_offset": (
                signal["eye_right"][0] - self.head_position_baseline["eye_right"][0],
                signal["eye_right"][1] - self.head_position_baseline["eye_right"][1],
            ),
            "eye_center_offset": (signal["eye_center_dx"], signal["eye_center_dy"]),
            "eye_distance_offset": signal["eye_distance_delta"],
            "eye_line_angle_offset_deg": signal["eye_line_angle_delta_deg"],
            "eye_line_slope_offset": signal["eye_line_slope_delta"],
            "point_max_offset": signal["point_max_dist"],
        }
        self.head_adapter_baseline = adapter
        save_head_adapter_baseline(adapter)
        self.head_adapter_baseline = load_head_adapter_baseline()
        return True

    def save_current_head_position_baseline(self, lms_smooth=None, n_frames=60, trim=20):
        """多帧采集取中位 → 保存头部位置基线 (抗抖动)。

        Args:
            n_frames: 采集帧数 (默认 60)
            trim:     去掉最大/最小帧数 (默认 20)
        """
        from eye_constants import (
            HEAD_NOSE_IDX, HEAD_EYE_LEFT_IDX, HEAD_EYE_RIGHT_IDX,
        )

        if lms_smooth is not None:
            # 单帧快捷路径 (用于一键保存全部基线 [A])
            nose = (int(lms_smooth[HEAD_NOSE_IDX, 0]), int(lms_smooth[HEAD_NOSE_IDX, 1]))
            eye_l = (int(lms_smooth[HEAD_EYE_LEFT_IDX, 0]), int(lms_smooth[HEAD_EYE_LEFT_IDX, 1]))
            eye_r = (int(lms_smooth[HEAD_EYE_RIGHT_IDX, 0]), int(lms_smooth[HEAD_EYE_RIGHT_IDX, 1]))
            self.head_position_baseline = {
                "nose_px": nose,
                "eye_left": eye_l,
                "eye_right": eye_r,
                "frame_width": self.width,
                "frame_height": self.height,
                "eye_distance": round(float(np.linalg.norm(
                    np.array(eye_r, dtype=np.float64) - np.array(eye_l, dtype=np.float64))), 1),
            }
            save_head_position_baseline(
                nose_px=nose, eye_left_px=eye_l, eye_right_px=eye_r,
                frame_width=self.width, frame_height=self.height,
            )
            self.head_position_baseline = load_head_position_baseline()
            self.head_adapter_baseline = load_head_adapter_baseline()
            return True

        # ---- 多帧采集抗抖动路径 ----
        if not hasattr(self, '_last_lms_smooth') or self._last_lms_smooth is None:
            print("[HeadPosition Baseline] Cannot save: no landmark data")
            return False

        mp_was_on = self.enable_mp
        self.enable_mp = True
        self.enable_ellseg = False

        frames_data = []  # [(nose, eye_l, eye_r), ...]
        collected = 0
        no_face_count = 0

        print(f"[HeadPosition Baseline] 采集 {n_frames} 帧 (trim={trim})...")
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

            nose_px = (int(lms[HEAD_NOSE_IDX, 0]), int(lms[HEAD_NOSE_IDX, 1]))
            left_px = (int(lms[HEAD_EYE_LEFT_IDX, 0]), int(lms[HEAD_EYE_LEFT_IDX, 1]))
            right_px = (int(lms[HEAD_EYE_RIGHT_IDX, 0]), int(lms[HEAD_EYE_RIGHT_IDX, 1]))
            frames_data.append((nose_px, left_px, right_px))
            collected += 1

        self.enable_mp = mp_was_on

        if len(frames_data) < 2 * trim + 1:
            print(f"[HeadPosition Baseline] 有效帧不足: {len(frames_data)}, 需要 >= {2*trim+1}")
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

        eye_dist = round(float(np.linalg.norm(
            np.array(right_px, dtype=np.float64) - np.array(left_px, dtype=np.float64))), 1)

        self.head_position_baseline = {
            "nose_px": nose_px,
            "eye_left": left_px,
            "eye_right": right_px,
            "frame_width": self.width,
            "frame_height": self.height,
            "eye_distance": eye_dist,
        }
        save_head_position_baseline(
            nose_px=nose_px, eye_left_px=left_px, eye_right_px=right_px,
            frame_width=self.width, frame_height=self.height,
        )
        self.head_position_baseline = load_head_position_baseline()
        self.head_adapter_baseline = load_head_adapter_baseline()

        all_d = [d for d, _ in devs]
        med_d = all_d[len(all_d) // 2]
        print(f"  [HeadPosition Baseline] {len(frames_data)}帧→trim{trim}→ "
              f"中位={nose_px}/{left_px}/{right_px} eye_dist={eye_dist:.1f} "
              f"(偏差: min={all_d[0]:.1f} med={med_d:.1f} max={all_d[-1]:.1f})")
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

        # 嘴角 L/R 与眼球保持一致：按用户标记/机器语义，不按 MediaPipe 原始左右。
        result = {}
        for side, idx, bl_key in [
            ("left", MOUTH_CORNER_RIGHT_IDX, "corner_left"),
            ("right", MOUTH_CORNER_LEFT_IDX, "corner_right"),
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

    def get_mouth_corner_curl_signal(self, lms_smooth=None):
        """
        返回嘴角相对本侧内口裂点的上翘比例，不依赖保存的嘴角像素基线。

        curl = (本侧内口裂点Y - 本侧外嘴角Y) / 嘴宽，数值越大表示嘴角越向上翘。
        L/R 与眼球一致：按用户标记/机器语义，不按 MediaPipe 原始左右。
        """
        if lms_smooth is None:
            if not hasattr(self, '_last_lms_smooth') or self._last_lms_smooth is None:
                return None
            lms_smooth = self._last_lms_smooth

        from eye_constants import (
            MOUTH_CORNER_LEFT_IDX, MOUTH_CORNER_RIGHT_IDX,
            MOUTH_CORNER_CURL_MIN_TARGET, MOUTH_CORNER_CURL_SYMMETRY_TOLERANCE,
            MOUTH_CORNER_CURL_TARGET, MOUTH_CORNER_CURL_TOLERANCE,
        )

        # MediaPipe inner mouth corners are local side references, not a shared fitted centerline.
        raw_left_inner = lms_smooth[78]
        raw_right_inner = lms_smooth[308]
        mouth_width = float(np.linalg.norm(raw_right_inner[:2] - raw_left_inner[:2]))
        if mouth_width <= 1e-6:
            return None

        baseline = getattr(self, "mouth_corner_curl_baseline", None)
        left_target = MOUTH_CORNER_CURL_TARGET
        right_target = MOUTH_CORNER_CURL_TARGET
        min_required = True
        if isinstance(baseline, dict):
            left_target = float(baseline.get("left_curl", left_target))
            right_target = float(baseline.get("right_curl", right_target))
            min_required = False

        result = {}
        for side, idx, local_ref in [
            ("left", MOUTH_CORNER_RIGHT_IDX, raw_right_inner),
            ("right", MOUTH_CORNER_LEFT_IDX, raw_left_inner),
        ]:
            corner = lms_smooth[idx]
            curl_px = float(local_ref[1] - corner[1])
            curl = curl_px / mouth_width
            target = left_target if side == "left" else right_target
            delta = curl - target
            min_qualified = (curl >= MOUTH_CORNER_CURL_MIN_TARGET) if min_required else True
            if min_required:
                target_qualified = abs(delta) <= MOUTH_CORNER_CURL_TOLERANCE
            else:
                target_qualified = round(curl, 3) == round(target, 3)
            result[side] = {
                "curl": curl,
                "curl_px": curl_px,
                "delta": delta,
                "target": target,
                "ref": (float(local_ref[0]), float(local_ref[1])),
                "min_qualified": min_qualified,
                "target_qualified": target_qualified,
                "qualified": min_qualified and target_qualified,
            }

        symmetry_delta = abs(result["left"]["curl"] - result["right"]["curl"])
        symmetry_qualified = symmetry_delta <= MOUTH_CORNER_CURL_SYMMETRY_TOLERANCE
        result["mouth_width"] = mouth_width
        result["min_target"] = MOUTH_CORNER_CURL_MIN_TARGET
        result["min_required"] = min_required
        result["tolerance"] = MOUTH_CORNER_CURL_TOLERANCE
        result["symmetry_delta"] = symmetry_delta
        result["symmetry_tolerance"] = MOUTH_CORNER_CURL_SYMMETRY_TOLERANCE
        result["symmetry_qualified"] = symmetry_qualified
        result["all_qualified"] = result["left"]["qualified"] and result["right"]["qualified"] and symmetry_qualified
        return result

    def save_current_mouth_corner_curl_baseline(self, lms_smooth=None):
        """Save the current mouth-corner curl ratio as the target for this template."""
        signal = self.get_mouth_corner_curl_signal(lms_smooth)
        if signal is None:
            print("[MouthCornerCurl Baseline] Cannot save: no landmark data")
            return False

        left_curl = float(signal["left"]["curl"])
        right_curl = float(signal["right"]["curl"])
        self.mouth_corner_curl_baseline = {
            "left_curl": left_curl,
            "right_curl": right_curl,
        }
        save_mouth_corner_curl_baseline(left_curl=left_curl, right_curl=right_curl)
        return True

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
                "corner_left": (int(lms_smooth[MOUTH_CORNER_RIGHT_IDX, 0]),
                                int(lms_smooth[MOUTH_CORNER_RIGHT_IDX, 1])),
                "corner_right": (int(lms_smooth[MOUTH_CORNER_LEFT_IDX, 0]),
                                 int(lms_smooth[MOUTH_CORNER_LEFT_IDX, 1])),
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
            left_px = (int(lms[MOUTH_CORNER_RIGHT_IDX, 0]),
                       int(lms[MOUTH_CORNER_RIGHT_IDX, 1]))
            right_px = (int(lms[MOUTH_CORNER_LEFT_IDX, 0]),
                        int(lms[MOUTH_CORNER_LEFT_IDX, 1]))
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

        if self.lower_lip_baseline is None or "llr" not in self.lower_lip_baseline:
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
        side_lower_tip = None
        side_roi = None
        if isinstance(self.lower_lip_baseline, dict):
            side_lower_tip = self.lower_lip_baseline.get("side_lower_tip")
            side_roi = self.lower_lip_baseline.get("side_roi")
        self.lower_lip_baseline = {"llr": llr}
        if side_lower_tip is not None:
            self.lower_lip_baseline["side_lower_tip"] = side_lower_tip
        if side_roi is not None:
            self.lower_lip_baseline["side_roi"] = side_roi
        save_lower_lip_baseline(llr=llr, side_lower_tip=side_lower_tip, side_roi=side_roi)
        return True

    def adjust_baseline_by_vertical_offset(self, det=None):
        """用当前帧的参考线垂直偏移，永久平移基线所有Y坐标并保存"""
        if det is None:
            det = self.last_result
        if det is None or not self.baseline:
            print("[V-Offset] No baseline or detection available")
            return False
        bl = self.baseline
        bl_eye_line = bl.get("eye_line_y")
        if bl_eye_line is None:
            print("[V-Offset] Baseline has no eye_line_y, re-save baseline (S) first")
            return False
        li = det.get("left_iris")
        ri = det.get("right_iris")
        if li is None or ri is None:
            print("[V-Offset] No iris detected, cannot calculate offset")
            return False
        cur_line = (li[1] + ri[1]) / 2.0
        dy = cur_line - bl_eye_line
        print(f"[V-Offset] Vertical offset: {dy:+.2f} px")

        # 平移基线所有Y坐标
        self.baseline["nose"] = (self.baseline["nose"][0], self.baseline["nose"][1] + dy)
        self.baseline["iris_left"] = (self.baseline["iris_left"][0], self.baseline["iris_left"][1] + dy)
        self.baseline["iris_right"] = (self.baseline["iris_right"][0], self.baseline["iris_right"][1] + dy)
        self.baseline["eye_line_y"] = cur_line

        save_baseline(
            nose_pos=self.baseline["nose"],
            left_iris_pos=self.baseline["iris_left"],
            right_iris_pos=self.baseline["iris_right"],
        )
        print(f"[V-Offset] Baseline adjusted and saved. New eye_line_y={cur_line:.2f}")
        return True

    def close(self):
        self.stop_display()
        if hasattr(self, 'landmark_snapshots'):
            self.landmark_snapshots.invalidate("front", "camera_closed")
        # detach first so explicit close followed by __del__ is safe.
        landmarker = getattr(self, 'landmarker', None)
        cap = getattr(self, 'cap', None)
        self.landmarker = None
        self.cap = None
        try:
            if landmarker is not None:
                landmarker.close()
        finally:
            try:
                if cap is not None:
                    cap.release()
            finally:
                cv2.destroyAllWindows()

    def __del__(self):
        try:
            self.close()
        except Exception:
            # 析构不得覆盖原始初始化异常或在解释器退出时再打印异常。
            pass


if __name__ == "__main__":
    import sys
    raw_test = "--raw-fps" in sys.argv

    detector = EllSegDetector()
    detector.start_display()
    print("\n[S] Save eyeball  [E] Save eyelid  [B] Save eyebrow  [M] Save mouth  [L] Save lower lip  [U] Save upper lip  [C] Save mouth corners  [K] Save mouth curl  [D] Save head position  [O] Save head offset  [V] Adjust baseline by eyeLine  [A] Save ALL  [G] Toggle baseline overlay  [Y] Toggle eyeLine HUD  [1] Toggle MP  [2] Toggle EllSeg  [3] Toggle Landmarks  [Q] Quit")

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
            if last in (ord('k'), ord('K')):
                detector.save_current_mouth_corner_curl_baseline()
                detector._last_key = None
            # 检测 D 键保存头部定位基线
            if last in (ord('d'), ord('D')):
                detector.save_current_head_position_baseline()
                detector._last_key = None
            if last in (ord('o'), ord('O')):
                detector.save_current_head_adapter_baseline()
                detector._last_key = None
            # 检测 V 键：用参考线垂直偏移永久调整基线
            if last in (ord('v'), ord('V')):
                detector.adjust_baseline_by_vertical_offset()
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
                detector.save_current_mouth_corner_curl_baseline()
                detector.save_current_head_position_baseline()
                print("=== 全部基线保存完成 ===\n")
                detector._last_key = None

    except KeyboardInterrupt:
        print("\n[Interrupted]")
    finally:
        detector.stop_display()
        detector.close()
