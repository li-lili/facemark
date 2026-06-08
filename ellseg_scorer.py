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

TOLERANCE = 2        # 合格阈值 (像素)
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

                # FPS 显示 (来自主采集线程的实际捕获速率)
                cap_fps = self._capture_fps
                cv2.putText(img, f"Cap FPS: {cap_fps:.1f}", (img.shape[1] - 170, 22),
                           0, 0.45, (200, 200, 200), 1)

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

                cv2.imshow("EllSeg Detect", img)

                # 检测开关 HUD
                status_str = f"[1]MP:{'ON' if self.enable_mp else 'OFF'}  [2]ELL:{'ON' if self.enable_ellseg else 'OFF'}"
                cv2.putText(img, status_str, (10, img.shape[0] - 8),
                           0, 0.4, (200, 200, 200), 1)

                key = cv2.waitKey(1) & 0xFF
                if key:
                    self._last_key = key
                if key == ord('1'):
                    self.toggle_mp()
                elif key == ord('2'):
                    self.toggle_ellseg()
                elif key in (ord('q'), ord('Q'), 27):
                    self._user_key = key

    def start_display(self):
        """启动显示线程"""
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
        save_baseline(**self.baseline)
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
    print("\n[S] Save baseline  [1] Toggle MP  [2] Toggle EllSeg  [Q] Quit")

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

    except KeyboardInterrupt:
        print("\n[Interrupted]")
    finally:
        detector.stop_display()
        detector.close()
