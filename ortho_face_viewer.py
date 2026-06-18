"""
正交投影人脸查看器
====================
将摄像头画面中的面部关键点映射到"正视平面"，去除透视/倾斜/偏移，
让面部特征在标准化的正交投影空间中展示。

按键：
  [Q/Esc] 退出
  [S]     保存当前帧
  [R]     重置参考关键点为一帧

依赖：mediapipe, opencv-python, numpy
"""

import os, cv2, numpy as np, urllib.request, time

DIR = os.path.dirname(os.path.abspath(__file__))

# ─── MediaPipe 模型路径 ───
MP_DIR = os.path.join(DIR, "models")
MP_MODEL_PATH = os.path.join(MP_DIR, "face_landmarker.task")
if not os.path.exists(MP_MODEL_PATH):
    os.makedirs(MP_DIR, exist_ok=True)
    print("[MediaPipe] Downloading model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
        MP_MODEL_PATH)

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import RunningMode

# ─── 正交投影目标关键点（标准正面脸布局，归一化坐标 0~1） ───
# 使用 4 点：额头中心(10)、左眼外角(33)、右眼外角(263)、下巴(152)
# 形成更稳定的面部四边形
CANONICAL_PTS = np.array([
    [0.50, 0.15],   # 额头中心
    [0.28, 0.35],   # 左眼外角
    [0.72, 0.35],   # 右眼外角
    [0.50, 0.85],   # 下巴
], dtype=np.float32)

# 源关键点索引
SRC_IDX = [10, 33, 263, 152]  # 额头, 左眼, 右眼, 下巴

# 正交画布尺寸
ORTHO_SIZE = 600

# EMA 平滑
EMA_ALPHA = 0.35


def init_mediapipe():
    opts = vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MP_MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return vision.FaceLandmarker.create_from_options(opts)


def compute_rigid_transform(src_pts, dst_pts):
    """
    刚性变换 (只旋转+平移, 不缩放)，保持原始面部比例不变。
    使用 OpenCV estimateAffinePartial2D 通过 RANSAC 求解。
    """
    src = np.array(src_pts, dtype=np.float32)
    dst = np.array(dst_pts, dtype=np.float32)
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                              ransacReprojThreshold=5.0)
    if M is None:
        # 退路: 手动最小二乘
        M = np.eye(2, 3, dtype=np.float32)
        # 只算质心平移 + 方向旋转
        mu_s = src.mean(axis=0)
        mu_d = dst.mean(axis=0)
        M[:2, 2] = mu_d - mu_s
    return M


class OrthoFaceViewer:
    """正交投影人脸查看器"""

    def __init__(self, camera_idx=0, width=1920, height=1080):
        self.width = width
        self.height = height
        self.ortho_w = ORTHO_SIZE
        self.ortho_h = ORTHO_SIZE

        print("[Ortho] Initializing MediaPipe...")
        self.landmarker = init_mediapipe()

        print(f"[Ortho] Opening camera {camera_idx}...")
        self.cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._timestamp = 0
        self._lms_smooth = None
        self._M_smooth = None  # EMA 平滑后的变换矩阵

    def capture(self):
        ok, frame = self.cap.read()
        if not ok:
            return False, None
        frame = cv2.flip(frame, 1)
        return True, frame

    def detect(self, frame):
        self._timestamp += 1
        h, w = frame.shape[:2]

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = self.landmarker.detect_for_video(mp_img, self._timestamp)

        if not result.face_landmarks:
            return None

        lm = result.face_landmarks[0]
        lms_px = np.array([[pt.x * w, pt.y * h] for pt in lm], dtype=np.float32)

        # EMA 平滑
        if self._lms_smooth is None:
            self._lms_smooth = lms_px.copy()
        else:
            self._lms_smooth = EMA_ALPHA * lms_px + (1 - EMA_ALPHA) * self._lms_smooth

        # 正交变换
        src = self._lms_smooth[SRC_IDX]
        dst = CANONICAL_PTS * np.array([self.ortho_w, self.ortho_h], dtype=np.float32)
        M = compute_rigid_transform(src, dst)

        # EMA 平滑变换矩阵
        if self._M_smooth is None:
            self._M_smooth = M
        else:
            self._M_smooth = EMA_ALPHA * M + (1 - EMA_ALPHA) * self._M_smooth

        # 把全图映射到正交空间
        ortho = cv2.warpAffine(frame, self._M_smooth,
                               (self.ortho_w, self.ortho_h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(40, 40, 40))

        return {
            "lms_px": self._lms_smooth,
            "ortho": ortho,
            "M": self._M_smooth.copy(),
        }

    def run(self):
        print("\n[Q/Esc] Quit  [S] Save screenshot  [R] Reset reference\n")

        while True:
            ok, frame = self.capture()
            if not ok:
                time.sleep(0.001)
                continue

            det = self.detect(frame)
            if det is None:
                # 没人脸时只显示原始
                display = cv2.resize(frame, (self.width // 2, self.height // 2))
                cv2.imshow("Ortho Face Viewer", display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    break
                continue

            ortho = det["ortho"]

            # ── 拼合显示: 左侧=原始, 右侧=正交投影 ──
            # 缩放到并排尺寸
            disp_w = self.width // 2
            disp_h = self.height // 2
            frame_small = cv2.resize(frame, (disp_w, disp_h))
            ortho_small = cv2.resize(ortho, (disp_w, disp_h))

            # 在原图小窗上画关键点
            lms = det["lms_px"]
            colors = [(255, 150, 0), (0, 255, 255), (0, 255, 255), (255, 150, 0)]  # 额头/下巴橙, 双眼黄
            for i, idx in enumerate(SRC_IDX):
                pt = lms[idx]
                fx = pt[0] * disp_w / self.width
                fy = pt[1] * disp_h / self.height
                cv2.circle(frame_small, (int(fx), int(fy)), 4, colors[i], -1)
            # 四边形连线 (额头→左眼→下巴→右眼→额头)
            pts_small = [(int(lms[i][0] * disp_w / self.width),
                          int(lms[i][1] * disp_h / self.height)) for i in SRC_IDX]
            # 额头→左眼, 左眼→下巴, 下巴→右眼, 右眼→额头
            cv2.line(frame_small, pts_small[0], pts_small[1], (0, 255, 0), 1)  # 额头→左眼
            cv2.line(frame_small, pts_small[1], pts_small[3], (0, 255, 0), 1)  # 左眼→下巴
            cv2.line(frame_small, pts_small[3], pts_small[2], (0, 255, 0), 1)  # 下巴→右眼
            cv2.line(frame_small, pts_small[2], pts_small[0], (0, 255, 0), 1)  # 右眼→额头

            # 标签
            cv2.putText(frame_small, "Original", (8, 22), 0, 0.6, (0, 255, 0), 2)
            cv2.putText(ortho_small, "Orthographic", (8, 22), 0, 0.6, (255, 200, 0), 2)

            display = np.hstack([frame_small, ortho_small])

            cv2.imshow("Ortho Face Viewer", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
            if key in (ord('s'), ord('S')):
                ts = time.strftime("%Y%m%d_%H%M%S")
                fname = os.path.join(DIR, f"ortho_{ts}.png")
                cv2.imwrite(fname, display)
                print(f"[Save] {fname}")
            if key in (ord('r'), ord('R')):
                self._lms_smooth = None
                self._M_smooth = None
                print("[Reset] 参考关键点已重置")

        self.cap.release()
        cv2.destroyAllWindows()

    def close(self):
        if hasattr(self, 'landmarker') and self.landmarker:
            self.landmarker.close()
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    viewer = OrthoFaceViewer()
    try:
        viewer.run()
    except KeyboardInterrupt:
        print("\n[Interrupted]")
    finally:
        viewer.close()
