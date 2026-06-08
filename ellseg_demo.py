"""
EllSeg Demo — MediaPipe 自动检测眼部 + EllSeg 虹膜中心
======================================================
MediaPipe 检测眼睛 → 裁眼 → EllSeg 分割虹膜 → 显示中心
按 [Q] 退出
按 [S] 保存鼻子+双眼虹膜中心到 face_points.json

依赖：opencv-python, numpy, torch, mediapipe
"""

import os, sys, subprocess, urllib.request, cv2, numpy as np, json

# ---- 自动安装 EllSeg ----
DIR = os.path.dirname(os.path.abspath(__file__))
ELLSEG = os.path.join(DIR, "EllSeg")
WEIGHTS = os.path.join(ELLSEG, "weights", "all.git_ok")
if not os.path.exists(ELLSEG):
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/RSKothari/EllSeg.git", ELLSEG], check=True)
if not os.path.exists(WEIGHTS):
    os.makedirs(os.path.dirname(WEIGHTS), exist_ok=True)
    urllib.request.urlretrieve(
        "https://github.com/RSKothari/EllSeg/raw/refs/heads/master/weights/all.git_ok",
        WEIGHTS)
sys.path.insert(0, ELLSEG)

# ---- MediaPipe 模型 ----
MP_DIR = os.path.join(DIR, "models")
MP_MODEL = os.path.join(MP_DIR, "face_landmarker.task")
if not os.path.exists(MP_MODEL):
    os.makedirs(MP_DIR, exist_ok=True)
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
        MP_MODEL)

import torch
from modelSummary import model_dict
from utils import get_predictions
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import RunningMode

# ---- 加载 EllSeg ----
net = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
ellseg = model_dict["ritnet_v3"]
ellseg.load_state_dict(net["state_dict"], strict=True)
ellseg.eval()
print("EllSeg OK")

# ---- 加载 MediaPipe ----
mp_opts = vision.FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MP_MODEL),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
)
landmarker = vision.FaceLandmarker.create_from_options(mp_opts)
print("MediaPipe OK")

# ---- MediaPipe 眼部关键点索引 ----
LEFT_EYE = [33, 246, 161, 160, 159, 158, 157, 173,
            133, 155, 154, 153, 145, 144, 163, 7]
RIGHT_EYE = [362, 398, 384, 385, 386, 387, 388, 466,
             263, 249, 390, 373, 374, 380, 381, 382]
EYE_PAD_X = 0.5   # 水平扩充比例
EYE_PAD_Y = 0.8   # 垂直扩充比例

# EMA 平滑参数（越低越平滑，1.0=不平滑）
EMA_LM = 0.35   # MediaPipe 关键点
EMA_ELL = 0.3   # EllSeg 椭圆参数

class LMASmoother:
    """EMA 平滑 MediaPipe 关键点"""
    def __init__(self, alpha=0.35):
        self.alpha = alpha
        self._lms = None
    def update(self, lms):
        if self._lms is None:
            self._lms = lms.copy()
        else:
            self._lms = self.alpha * lms + (1 - self.alpha) * self._lms
        return self._lms.copy()
    def reset(self):
        self._lms = None

class EllipseSmoother:
    """EMA 平滑椭圆参数（处理角度环绕）"""
    def __init__(self, alpha=0.3):
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
            if diff > 90: ang -= 180
            elif diff < -90: ang += 180
            cur[4] = ang
            a = self.alpha
            self._v = a * cur + (1 - a) * self._v
            self._v[4] = self._v[4] % 360
        v = self._v
        return ((v[0], v[1]), (v[2], v[3]), v[4])
    def reset(self):
        self._v = None

lm_smoother = LMASmoother(EMA_LM)
ell_smooth_L = EllipseSmoother(EMA_ELL)
ell_smooth_R = EllipseSmoother(EMA_ELL)

# ---- 保存状态 ----
SAVE_FILE = os.path.join(DIR, "face_points.json")
cur_iris = {"L": None, "R": None}
cur_nose = None
save_msg = ""
save_msg_timer = 0

# ---- 基准与检查模式 ----
TOLERANCE = 1   # 允许的最大偏移（像素）
baseline = {"nose": None, "iris_left": None, "iris_right": None}
check_mode = False

if os.path.isfile(SAVE_FILE):
    try:
        with open(SAVE_FILE) as f:
            _d = json.load(f)
        baseline["nose"] = tuple(_d["nose"])
        baseline["iris_left"] = tuple(_d["iris_left"])
        baseline["iris_right"] = tuple(_d["iris_right"])
        print(f"[Loaded] baseline from {SAVE_FILE}")
    except Exception:
        pass

# ---- 摄像头 ----
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


def crop_eye(frame, lms_px, eye_idx):
    """从 MediaPipe 关键点裁出眼部区域"""
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
    crop = frame[y1:y2, x1:x2]
    return crop, (x1, y1)


def run_ellseg(eye_gray):
    """EllSeg 推理，返回虹膜/瞳孔椭圆（裁剪坐标）"""
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
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()

    with torch.no_grad():
        x4, x3, x2, x1, x = ellseg.enc(tensor)
        seg = get_predictions(ellseg.dec(x4, x3, x2, x1, x))

    seg_map = seg.squeeze().numpy()
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


ts = 0
print("=" * 50)
print("  EllSeg + MediaPipe Demo  [Q] Quit")
print("=" * 50)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    ts += 33

    # MediaPipe 检测
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_img, ts)

    if not result.face_landmarks:
        lm_smoother.reset()
        ell_smooth_L.reset()
        ell_smooth_R.reset()

    if result.face_landmarks:
        for face in result.face_landmarks:
            lms_px = np.array([[lm.x * w, lm.y * h] for lm in face])
            # EMA 平滑关键点
            lms_smooth = lm_smoother.update(lms_px)

            for side, eye_idx, label, ell_s in [
                ("L", LEFT_EYE, "Left", ell_smooth_L),
                ("R", RIGHT_EYE, "Right", ell_smooth_R),
            ]:
                crop, (x1, y1) = crop_eye(frame, lms_smooth, eye_idx)
                if crop.size == 0:
                    continue

                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                ells = run_ellseg(gray)

                iris = ells.get("iris")
                if iris is None:
                    ell_s.reset()
                    continue

                # EMA 平滑椭圆
                iris = ell_s.update(iris)

                (cx, cy), (aw, ah), ang = iris
                # 映射回原图
                px, py = int(cx + x1), int(cy + y1)

                # 椭圆
                cv2.ellipse(frame, (px, py), (int(aw / 2), int(ah / 2)),
                            ang, 0, 360, (0, 165, 255), 2, cv2.LINE_AA)
                # 中心点
                cv2.circle(frame, (px, py), 4, (0, 255, 255), -1, cv2.LINE_AA)
                # 坐标
                cv2.putText(frame, f"{side}_iris {px},{py}",
                            (px + 10, py - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1,
                            cv2.LINE_AA)
                # 记录当前虹膜中心
                cur_iris[side] = (px, py)

            # ---- 鼻子中心 ----
            NOSE_IDX = 1   # MediaPipe 鼻尖
            nx, ny = int(lms_smooth[NOSE_IDX, 0]), int(lms_smooth[NOSE_IDX, 1])
            cur_nose = (nx, ny)
            cv2.circle(frame, (nx, ny), 5, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.putText(frame, f"Nose {nx},{ny}",
                        (nx + 10, ny - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

            # ---- 合格检查（[C] 开启后） ----
            if check_mode and all(b is not None for b in baseline.values()):
                checks = []
                labels_pts = [
                    ("L_iris", cur_iris["L"], baseline["iris_left"]),
                    ("R_iris", cur_iris["R"], baseline["iris_right"]),
                    ("Nose",   cur_nose,      baseline["nose"]),
                ]
                all_pass = True
                for name, cur, base in labels_pts:
                    if cur is None:
                        checks.append(f"{name}: --")
                        all_pass = False
                        continue
                    dx = cur[0] - base[0]
                    dy = cur[1] - base[1]
                    ok = abs(dx) <= TOLERANCE and abs(dy) <= TOLERANCE
                    if not ok:
                        all_pass = False
                    checks.append(f"{name}: dx={dx:+d} dy={dy:+d} {'OK' if ok else 'NG'}")

                status = "PASS" if all_pass else "FAIL"
                color = (0, 255, 0) if all_pass else (0, 0, 255)
                cv2.putText(frame, f"[{status}]", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
                for i, line in enumerate(checks):
                    c = (0, 255, 0) if "OK" in line else (0, 0, 255)
                    cv2.putText(frame, line, (10, 120 + i * 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)

    cv2.putText(frame, "EllSeg [Q] Quit  [S] Save  [C] Check", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    # 保存提示
    if save_msg_timer > 0:
        cv2.putText(frame, save_msg, (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        save_msg_timer -= 1

    cv2.imshow("EllSeg Demo", frame)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), ord("Q"), 27):
        break
    elif key in (ord("s"), ord("S")):
        if cur_nose is not None and all(v is not None for v in cur_iris.values()):
            data = {
                "nose": cur_nose,
                "iris_left": cur_iris["L"],
                "iris_right": cur_iris["R"],
            }
            with open(SAVE_FILE, "w") as f:
                json.dump(data, f, indent=2)
            baseline["nose"] = cur_nose
            baseline["iris_left"] = cur_iris["L"]
            baseline["iris_right"] = cur_iris["R"]
            save_msg = f"[Saved] {SAVE_FILE}"
            save_msg_timer = 60
            print(save_msg)
            print(f"  Nose: {cur_nose}")
            print(f"  Iris L: {cur_iris['L']}")
            print(f"  Iris R: {cur_iris['R']}")
    elif key in (ord("c"), ord("C")):
        if not all(b is not None for b in baseline.values()):
            save_msg = "[Error] 请先按 [S] 保存基准"
            save_msg_timer = 60
        else:
            check_mode = not check_mode
            save_msg = f"[Check] {'ON' if check_mode else 'OFF'}"
            save_msg_timer = 30
        print(save_msg)

landmarker.close()
cap.release()
cv2.destroyAllWindows()
