"""
Eye Score Calculator — 纯几何约束面部评分
==============================================
不依赖任何预录制的标准脸，仅通过 MediaPipe 关键点
的几何特征来评价眼睛质量。

3 个独立指标:
  1. 眼睛开放度 (Openness)    — 高宽比是否在正常范围
  2. 虹膜居中度 (Centeredness) — 虹膜是否在眼眶中心
  (3. 左右对称性 (Symmetry) — 已注释)
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import RunningMode

from expression_match import (
    ensure_model, FaceEMA,
    L_EYE, R_EYE, L_IRIS, R_IRIS,
)
from eye_constants import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, FLIP_HORIZONTAL,
)

# ============================================================
# 几何评分参数
# ============================================================

# 指标1: 眼睛开放度 — 高宽比理想区间
OPENNESS_IDEAL_MIN = 0.57
OPENNESS_IDEAL_MAX = 0.58

# 指标2: 虹膜居中度 — 偏移占眼宽的阈值
IRIS_OFFSET_THRESHOLD = 0.033

# 虹膜质心外移补偿 (占眼宽的比例)
IRIS_CENTER_OUTWARD = 0.05

# 指标4: 形状规整度
SHAPE_CV_IDEAL = 0.50

# 综合得分权重
GEOMETRY_WEIGHTS = {
    "openness": 0.50,
    "centered": 0.50,
}


# ============================================================
# 纯几何评分函数 (无状态，可独立测试)
# ============================================================

def polygon_centroid(pts):
    """Compute the centroid (center of mass) of a closed polygon.

    Unlike arithmetic mean (sum/N), the polygon centroid accounts
    for the actual shape of the eye opening — wider in the middle,
    narrower at the corners — so it naturally sits where the iris
    resides in a neutral gaze.

    Args:
        pts: np.ndarray (N, 2) polygon vertices

    Returns:
        np.ndarray (2,) centroid coordinates [cx, cy]
    """
    n = len(pts)
    if n < 3:
        return pts.mean(axis=0)
    A = 0.0
    cx = 0.0
    cy = 0.0
    for i in range(n):
        j = (i + 1) % n
        cross = pts[i, 0] * pts[j, 1] - pts[j, 0] * pts[i, 1]
        A += cross
        cx += (pts[i, 0] + pts[j, 0]) * cross
        cy += (pts[i, 1] + pts[j, 1]) * cross
    A *= 0.5
    if abs(A) < 1e-6:
        return pts.mean(axis=0)
    cx /= (6.0 * A)
    cy /= (6.0 * A)
    return np.array([cx, cy])

def _eye_openness_score(kps, eye_indices):
    """
    指标1: 眼睛开放度得分 0~100

    用眼眶轮廓点的 Y 跨度 / X 跨度 = 高宽比。

    Returns:
        (score, height, width, ratio) — 方便调试日志
    """
    pts = kps[list(eye_indices)]
    y_coords = pts[:, 1]
    x_coords = pts[:, 0]

    # 去极值的跨度 (p10~p90)
    eye_height = np.percentile(y_coords, 90) - np.percentile(y_coords, 10)
    eye_width = np.ptp(x_coords)

    if eye_width < 1e-6:
        return 0.0, 0.0, 0.0, 0.0

    ratio = eye_height / eye_width

    if OPENNESS_IDEAL_MIN <= ratio <= OPENNESS_IDEAL_MAX:
        score = 100.0
    elif ratio < OPENNESS_IDEAL_MIN:
        score = max(0.0, 100.0 * (ratio / OPENNESS_IDEAL_MIN))
    else:
        score = max(0.0, 100.0 * (1.0 - (ratio - OPENNESS_IDEAL_MAX) / OPENNESS_IDEAL_MAX))

    return score, float(eye_height), float(eye_width), float(ratio)


def _iris_centeredness_score(kps, eye_indices, iris_indices, nose_x=None):
    """
    指标2: 虹膜居中度得分 0~100

    使用多边形质心（而非算术均值）作为眼眶几何中心，
    更准确地反映眼眶形状。质心可沿水平方向外移以补偿
    多边形质心偏向鼻侧的偏差。

    Args:
        kps: (478, 2) 归一化关键点
        eye_indices: 眼眶轮廓索引列表
        iris_indices: 虹膜轮廓索引列表
        nose_x: 鼻尖 X 坐标（归一化），用于确定外移方向

    Returns:
        (score, offset, eye_width, normalized_offset, eye_center, iris_center)
        eye_center / iris_center 均为归一化坐标 (x, y)，用于预览绘制
    """
    eye_pts = kps[list(eye_indices)]
    iris_pts = kps[list(iris_indices)]

    # 多边形质心作为眼眶几何中心
    eye_center = polygon_centroid(eye_pts)
    iris_center = iris_pts.mean(axis=0)

    # 外移补偿：多边形质心偏向鼻侧，向远离鼻子的方向偏移
    eye_width = np.ptp(eye_pts[:, 0])
    if nose_x is not None and IRIS_CENTER_OUTWARD != 0 and eye_width > 1e-6:
        offset = IRIS_CENTER_OUTWARD * eye_width
        if eye_center[0] < nose_x:
            # 左眼 → 外移 = 向左 → 减小 x
            eye_center[0] -= offset
        else:
            # 右眼 → 外移 = 向右 → 增大 x
            eye_center[0] += offset

    offset = np.linalg.norm(iris_center - eye_center)

    if eye_width < 1e-6:
        return 0.0, 0.0, 0.0, 0.0, eye_center, iris_center

    normalized_offset = offset / eye_width

    if normalized_offset <= IRIS_OFFSET_THRESHOLD:
        score = 100.0
    else:
        score = max(0.0, 100.0 * (1.0 - (normalized_offset - IRIS_OFFSET_THRESHOLD) / IRIS_OFFSET_THRESHOLD))

    return score, float(offset), float(eye_width), float(normalized_offset), eye_center, iris_center


def calc_geometry_eye_score(current_kps):
    """
    纯几何眼睛质量综合评分 — 不依赖标准脸

    Args:
        current_kps: (478, 2) 归一化关键点坐标 (来自 FaceEMA.norm_coords())

    Returns:
        (total_score, detail_scores): 总分 0~100 和 各项细分字典
    """
    d = {}

    # 鼻尖 X 坐标（用于虹膜质心外移补偿）
    nose_x = float(current_kps[1, 0])

    # --- 指标1: 眼睛开放度 (30%) ---
    l_opn_score, l_h, l_w, l_ratio = _eye_openness_score(current_kps, L_EYE)
    r_opn_score, r_h, r_w, r_ratio = _eye_openness_score(current_kps, R_EYE)

    #print(f"Left Eye Openness Score: {l_opn_score}, Right Eye Openness Score: {r_opn_score}")
    d["L_Openness"] = l_opn_score
    d["R_Openness"] = r_opn_score
    # --- 指标2: 虹膜居中度 (30%) ---
    l_cen_score, l_offset, l_ew, l_noff, l_eye_ctr, l_iris_ctr = _iris_centeredness_score(current_kps, L_EYE, L_IRIS, nose_x)
    r_cen_score, r_offset, r_ew, r_noff, r_eye_ctr, r_iris_ctr = _iris_centeredness_score(current_kps, R_EYE, R_IRIS, nose_x)
    d["L_Centered"] = l_cen_score
    d["R_Centered"] = r_cen_score

    d["_debug"] = {
        "L_H": l_h, "L_W": l_w, "L_Ratio": l_ratio,
        "R_H": r_h, "R_W": r_w, "R_Ratio": r_ratio,
        "L_IrisOffset": l_offset, "L_IrisWidth": l_ew, "L_IrisNormOff": l_noff,
        "R_IrisOffset": r_offset, "R_IrisWidth": r_ew, "R_IrisNormOff": r_noff,
        "L_EyeCenter": l_eye_ctr, "L_IrisCenter": l_iris_ctr,
        "R_EyeCenter": r_eye_ctr, "R_IrisCenter": r_iris_ctr,
    }


    # === 加权综合 ===
    w = GEOMETRY_WEIGHTS
    total = (
        w["openness"] * 0.5 * (d["L_Openness"] + d["R_Openness"]) +
        w["centered"] * 0.5 * (d["L_Centered"] + d["R_Centered"])
        # + w["symmetry"] * avg_symmetry
    )

    d["_Total"] = float(np.clip(total, 0, 100))
    return d["_Total"], d


# ============================================================
# 采集器类
# ============================================================

class EyeScoreCalculator:
    """
    眼睛质量计算器 — 封装摄像头采集 + 纯几何评分

    不再需要标准脸文件。换任何头都能直接运行。
    """

    def __init__(self):
        self.model_path = ensure_model()

        # 初始化 MediaPipe
        options = vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.model_path),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

        # 初始化摄像头
        self.cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {CAMERA_INDEX}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        # EMA 平滑器
        self.ema = FaceEMA(alpha=0.5, max_dis=3)
        self.timestamp = 0

        print(f"[INFO] EyeScoreCalculator 初始化完成 (纯几何模式)")
        print(f"[INFO] 无需标准脸数据 — 换头即用")

    def capture_and_score(self, stabilize_frames: int = 200,
                          show_preview: bool = False):
        """
        采集当前帧并返回纯几何评分

        Returns:
            (eye_score, region_scores):
              eye_score: float 综合几何得分 0~100
              region_scores: dict 各项细分得分 (L_Openness, R_Openness, ...)
        """
        scores_collected = []
        detail_sum = {}

        for _ in range(stabilize_frames):
            ok, frame = self.cap.read()
            if not ok:
                continue

            if FLIP_HORIZONTAL:
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            self.timestamp += 33
            result = self.landmarker.detect_for_video(mp_img, self.timestamp)
            face_lms_list = result.face_landmarks or []

            if face_lms_list:
                self.ema.update(face_lms_list[0])
            else:
                self.ema.skip()

            if self.ema.ok:
                current_kps = self.ema.norm_coords()
                if current_kps is not None:
                    total, detail = calc_geometry_eye_score(current_kps)
                    scores_collected.append(total)
                    for key, val in detail.items():
                        detail_sum.setdefault(key, []).append(val)

                    
                        self._draw_preview(frame, current_kps, total, detail, w, h)
                        cv2.imshow("Eye Tuner Preview", frame)
                        cv2.waitKey(1)

        # 保留最新的 debug 数据（不做 mean 聚合）
        last_debug = {}
        if scores_collected:
            avg_total = float(np.mean(scores_collected[-5:]))
            avg_detail = {
                key: float(np.mean(vals[-5:]))
                for key, vals in detail_sum.items()
                if len(vals) >= 5 and not key.startswith("_")
            }
            # _debug 是原始 dict，取最近一次
            if "_debug" in detail_sum and detail_sum["_debug"]:
                last_debug = detail_sum["_debug"][-1]
            avg_detail.update(last_debug)
            return avg_total, avg_detail

        return 0.0, {}

    def capture_and_score_eyelid(self, stabilize_frames: int = 10,
                                  show_preview: bool = False):
        """
        眼皮阶段专用评分 — 重点看 L/R_Openness + Symmetry

        Returns:
            (eyelid_scores, raw_detail):
              eyelid_scores: {"L_Eye": score, "R_Eye": score, "average": score}
              raw_detail: 全部几何细分指标
        """
        _, detail = self.capture_and_score(stabilize_frames, show_preview=show_preview)

        if not detail:
            return {"L_Eye": 0.0, "R_Eye": 0.0, "average": 0.0}, {}

        l_open = detail.get("L_Openness", 0.0)
        r_open = detail.get("R_Openness", 0.0)

        # 眼皮阶段: 仅看开放度
        l_eye = l_open
        r_eye = r_open
        avg = (l_eye + r_eye) / 2

        eyelid_scores = {
            "L_Eye": l_eye,
            "R_Eye": r_eye,
            "average": avg,
        }
        return eyelid_scores, detail

    # ------------------------------------------------------------------
    # 预览绘制
    # ------------------------------------------------------------------

    def _draw_eyelid_preview(self, image, current_kps, l_eye_score, r_eye_score, avg_score, w, h):
        """在预览窗口上绘制眼皮调试信息"""
        lms_px = (current_kps * np.array([w, h])).astype(np.int32)

        for idx in list(L_EYE):
            px, py = int(lms_px[idx, 0]), int(lms_px[idx, 1])
            cv2.circle(image, (px, py), 3, (0, 255, 0), -1)
        for idx in list(R_EYE):
            px, py = int(lms_px[idx, 0]), int(lms_px[idx, 1])
            cv2.circle(image, (px, py), 3, (255, 100, 0), -1)

        y_offset = 40
        cv2.putText(image, f"=== Eyelid Phase (A8/A9) ===", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        l_color = (0, 255, 0) if l_eye_score >= 80 else (0, 215, 255)
        r_color = (0, 255, 0) if r_eye_score >= 80 else (0, 215, 255)
        a_color = (0, 255, 0) if avg_score >= 80 else (0, 215, 255)

        cv2.putText(image, f"L_Eye: {l_eye_score:.1f}%", (10, y_offset + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, l_color, 2)
        cv2.putText(image, f"R_Eye: {r_eye_score:.1f}%", (10, y_offset + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, r_color, 2)
        cv2.putText(image, f"Avg:   {avg_score:.1f}%", (10, y_offset + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, a_color, 2)
        cv2.putText(image, "Press [Q] to stop", (10, y_offset + 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

    def _draw_preview(self, image, current_kps, eye_score, detail, w, h):
        """在预览窗口上绘制几何评分详情"""
        lms_px = (current_kps * np.array([w, h])).astype(np.int32)

        # 眼眶和虹膜关键点（黄色小圆点）
        for idx in list(L_EYE) + list(R_EYE) + list(L_IRIS) + list(R_IRIS):
            px, py = int(lms_px[idx, 0]), int(lms_px[idx, 1])
            cv2.circle(image, (px, py), 2, (0, 255, 255), -1)

        color = (0, 255, 0) if eye_score >= 80 else (0, 215, 255) if eye_score >= 50 else (0, 0, 255)
        cv2.putText(image, f"Geo Score: {eye_score:.1f}%", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        # 显示4个子分
        if detail:
            sub_keys = ["L_Openness", "R_Openness", "L_Centered", "R_Centered",
                        "Symmetry_Eye"]
            for i, key in enumerate(sub_keys[:4]):
                v = detail.get(key, 0)
                c = (0, 200, 0) if v >= 70 else (0, 180, 255)
                short = key.replace("_", "\n")[:7]
                cv2.putText(image, f"{short}: {v:.0f}", (10, 75 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)

        cv2.putText(image, "Press [Q] to stop", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

    def close(self):
        """释放资源"""
        self.landmarker.close()
        self.cap.release()
        cv2.destroyAllWindows()


# ============================================================
# 独立运行入口
# ============================================================

def main():
    """直接运行打分系统 — 无串口、无舵机，仅摄像头+几何评分"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Eye Score Calculator — 纯几何面部评分 (独立模式)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python eye_scorer.py
  python eye_scorer.py --no-preview
  python eye_scorer.py --cam 1
        """
    )
    parser.add_argument("--no-preview", action="store_true",
                       help="禁用预览窗口")
    parser.add_argument("--cam", type=int, default=CAMERA_INDEX,
                       help=f"摄像头索引 (默认: {CAMERA_INDEX})")

    args = parser.parse_args()

    print("""
+==============================================================+
||     Eye Score Calculator — 独立打分模式                     |
+--------------------------------------------------------------+
||  无串口 / 无舵机 / 仅摄像头 + MediaPipe 几何评分            ||
||                                                               ||
||  4项指标: 开放度 | 居中度 | 对称性 | 规整度                  ||
||                                                               ||
||  按 [Q] 或 [Esc] 退出                                         |
+==============================================================
""")

    scorer = EyeScoreCalculator()

    frame_count = 0
    try:
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break

            frame_count += 1
            eye_score, detail = scorer.capture_and_score(
                stabilize_frames=3,
                show_preview=not args.no_preview
            )

            if frame_count % 5 == 0 and detail:
                bar_len = int(eye_score / 4)
                bar = "#" * bar_len + "-" * (25 - bar_len)

                labels = [
                    ("L_Openness", "L_Opn"),
                    ("R_Openness", "R_Opn"),
                    ("L_Centered", "L_Cen"),
                    ("R_Centered", "R_Cen"),
                    ("Symmetry_Eye", "Sym"),
                ]
                subs = []
                for full, short in labels[:4]:
                    v = detail.get(full, 0)
                    subs.append(f"{short}:{v:.0f}")

                print(
                    f"  [{frame_count:>5d}] {eye_score:>5.1f}% |{bar}| "
                    + "  ".join(subs),
                    end="\r",
                )

                # # 实时打印眼睛高/宽/高宽比（直接从 flatten 后的 key 读取）
                # if "L_H" in detail:
                #     print()
                #     print(
                #         f"         [DEBUG] L: H={detail['L_H']:.4f}  W={detail['L_W']:.4f}  "
                #         f"H/W={detail['L_Ratio']:.3f}  |  "
                #         f"R: H={detail['R_H']:.4f}  W={detail['R_W']:.4f}  "
                #         f"H/W={detail['R_Ratio']:.3f}  "
                #         f"(理想区间: {OPENNESS_IDEAL_MIN:.2f} ~ {OPENNESS_IDEAL_MAX:.2f})"
                #     )

                # 实时打印虹膜居中相关参数
                # if "L_IrisOffset" in detail:
                #     print()
                #     print(
                #         f"         [IRIS] L: offset={detail['L_IrisOffset']:.4f}  "
                #         f"eyeW={detail['L_IrisWidth']:.4f}  "
                #         f"normOff={detail['L_IrisNormOff']:.4f}  "
                #         f"|  R: offset={detail['R_IrisOffset']:.4f}  "
                #         f"eyeW={detail['R_IrisWidth']:.4f}  "
                #         f"normOff={detail['R_IrisNormOff']:.4f}  "
                #         f"(阈值: {IRIS_OFFSET_THRESHOLD})"
                #     )
            elif frame_count % 5 == 0:
                bar_len = int(eye_score / 4)
                bar = "#" * bar_len + "-" * (25 - bar_len)
                print(f"  [{frame_count:>5d}] {eye_score:>5.1f}% |{bar}| (检测中...)",
                      end="\r")

    except KeyboardInterrupt:
        pass
    finally:
        print(f"\n\n  打分结束，共 {frame_count} 帧")
        scorer.close()


if __name__ == "__main__":
    main()
