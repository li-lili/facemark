"""
Eye Auto Tuner — 常量定义与数据类型
========================================
包含舵机通道、角度范围、几何评分参数等常量，
以及调优结果数据结构。
"""

from dataclasses import dataclass, field
from typing import Dict, List


# ============================================================
# 舵机通道定义
# ============================================================
EYE_SERVO_CHANNELS = [8, 9, 10, 11, 12, 13]
EYE_SERVO_NAMES = ["A8", "A9", "A10", "A11", "A12", "A13"]

# 分组常量
EYELID_CHANNELS = [8, 9]       # A8,A9 眼皮
EYEBALL_CHANNELS = [10, 11, 12, 13]  # A10-A13 眼球
EYEBROW_CHANNELS = [0, 1]      # A0(左眉), A1(右眉)
EYEBROW_NAMES = ["A0", "A1"]

# ============================================================
# 眼球偏移容差（像素）
# ============================================================
EYEBALL_OFFSET_TOLERANCE = 2   # 虹膜偏移合格阈值 (px)

# ============================================================
# 眼皮 EAR (Eye Aspect Ratio) 参数
# ============================================================
EYELID_BASELINE_FILE = "eyelid_baseline.json"
EYELID_EAR_TOLERANCE = 0.003   # EAR 偏差容差（高宽比无量纲）

# EAR 计算用关键点 (MediaPipe 478 关键点索引)
# 左眼: 上下点对 + 眼角
LEFT_EAR_PAIRS = [(159, 145), (158, 153)]  # (上睑, 下睑) 点对
LEFT_EYE_CORNERS = (33, 133)               # (外眼角, 内眼角)   
# 右眼: 上下点对 + 眼角
RIGHT_EAR_PAIRS = [(386, 374), (385, 380)]
RIGHT_EYE_CORNERS = (362, 263)

# 眼皮自动调整参数
EYELID_WAIT_SECONDS = 1.0      # 每次调整后等待秒数
EYELID_MAX_ITERATIONS = 30     # 最大调整轮次

# ============================================================
# 眉毛 EBHR (Eyebrow Height Ratio) 参数
# ============================================================
EYEBROW_BASELINE_FILE = "eyebrow_baseline.json"
EYEBROW_EBHR_TOLERANCE = 0.01          # EBHR 偏差容差
EYEBROW_WAIT_SECONDS = 1.0             # 每次调整后等待秒数
EYEBROW_MAX_ITERATIONS = 30            # 最大调整轮次

# EBHR 计算用关键点 (MediaPipe 478 关键点索引)
# 左眉: 眉弓上缘8个关键点 (更稳健)
LEFT_BROW_POINTS = [52, 53, 55, 63, 65, 66, 105, 107]
LEFT_BROW_CORNERS = (33, 133)          # (外眼角, 内眼角)
# 右眉: 眉弓上缘8个关键点
RIGHT_BROW_POINTS = [282, 283, 285, 293, 295, 296, 334, 336]
RIGHT_BROW_CORNERS = (362, 263)

# ============================================================
# 嘴部 MAR (Mouth Aspect Ratio) 参数 — A26 下巴
# ============================================================
MOUTH_BASELINE_FILE = "mouth_baseline.json"
MOUTH_MAR_TOLERANCE = 0.015          # MAR 偏差容差
MOUTH_WAIT_SECONDS = 1.0
MOUTH_MAX_ITERATIONS = 30
MOUTH_CHIN_CHANNEL = 26              # A26 下巴

# MAR 计算用关键点 (MediaPipe 478)
MOUTH_TOP = 1         # 鼻尖(鼻子中间)
MOUTH_BOTTOM = 152    # 下巴点
MOUTH_NORM_LEFT = 33   # 左眼外角 — 归一化参考
MOUTH_NORM_RIGHT = 263 # 右眼外角 — 归一化参考

# ============================================================
# 下唇 LLR (Lower Lip Ratio) 参数 — A19 中下嘴唇
# ============================================================
LOWER_LIP_BASELINE_FILE = "lower_lip_baseline.json"
LOWER_LIP_LLR_TOLERANCE = 0.005      # LLR 偏差容差
LOWER_LIP_WAIT_SECONDS = 1.0
LOWER_LIP_MAX_ITERATIONS = 30
LOWER_LIP_CHANNEL = 19               # A19 中下嘴唇

# LLR 计算用关键点 (MediaPipe 478)
LL_LIP_UPPER = 14      # 下唇上缘 (唇红边界)
LL_LIP_LOWER = 17      # 下唇下缘
LL_NORM_LEFT = 33       # 左眼外角 — 归一化参考
LL_NORM_RIGHT = 263     # 右眼外角 — 归一化参考

# ============================================================
# 上唇 ULR (Upper Lip Ratio) 参数 — A16 中上嘴唇
# ============================================================
UPPER_LIP_BASELINE_FILE = "upper_lip_baseline.json"
UPPER_LIP_ULR_TOLERANCE = 0.002      # ULR 偏差容差
UPPER_LIP_WAIT_SECONDS = 1.0
UPPER_LIP_MAX_ITERATIONS = 30
UPPER_LIP_CHANNEL = 16               # A16 中上嘴唇

# ULR 计算用关键点 (MediaPipe 478)
UL_LIP_TOP = 0         # 上唇上缘(人中) — mp0
UL_IRIS_LEFT = 468     # 左眼虹膜中心
UL_IRIS_RIGHT = 473    # 右眼虹膜中心
UL_NORM_LEFT = 33      # 左眼外角 — 归一化参考1
UL_NORM_RIGHT = 263    # 右眼外角 — 归一化参考1
UL_NORM_LEFT2 = 37     # 左脸轮廓(更宽) — 归一化参考2
UL_NORM_RIGHT2 = 267   # 右脸轮廓(更宽) — 归一化参考2

# ============================================================
# 默认角度范围（度）
# ============================================================
DEFAULT_ANGLE_MIN = 0
DEFAULT_ANGLE_MAX = 270
DEFAULT_ANGLE_STEP = 5  # 默认搜索步长

# ============================================================
# 嘴角调优参数 — A22-A25
# ============================================================
MOUTH_CORNERS_BASELINE_FILE = "mouth_corners_baseline.json"
MOUTH_CORNER_LEFT_IDX = 61        # 左嘴角 MediaPipe 索引
MOUTH_CORNER_RIGHT_IDX = 291      # 右嘴角 MediaPipe 索引
MOUTH_CORNER_TOLERANCE = 1      # 水平像素容差 (px)
MOUTH_CORNER_Y_TOLERANCE = 1    # 垂直像素容差 (px) — 更严格
MOUTH_CORNER_WAIT_SECONDS = 1.0
MOUTH_CORNER_MAX_ITERATIONS = 30
# A22=右嘴角水平(前→后), A23=左嘴角垂直(上→下), A24=左嘴角水平(前→后), A25=右嘴角垂直(上→下)
MOUTH_CORNER_H_CHANNELS = [24, 22]     # [左水平A24, 右水平A22]
MOUTH_CORNER_V_CHANNELS = [23, 25]     # [左垂直A23, 右垂直A25]

# ============================================================
# 摄像头配置
# ============================================================
CAMERA_INDEX = 0
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
FLIP_HORIZONTAL = True

# ============================================================
# 数据类型
# ============================================================

@dataclass
class TuningResult:
    """调优结果数据类"""
    best_angles: List[int]
    best_score: float
    best_eye_score: float
    iteration: int
    total_iterations: int
    score_history: List[float] = field(default_factory=list)
    region_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self):
        # 将 region_scores 中的 numpy 类型转为 Python 原生类型
        safe_scores = {}
        for k, v in self.region_scores.items():
            try:
                safe_scores[k] = float(v)
            except (TypeError, ValueError):
                safe_scores[k] = str(v)

        return {
            "best_angles": {name: angle for name, angle in zip(EYE_SERVO_NAMES, self.best_angles)},
            "best_score": self.best_score,
            "best_eye_score": self.best_eye_score,
            "iteration": self.iteration,
            "total_iterations": self.total_iterations,
            "region_scores": safe_scores,
        }

    def save(self, filepath: str):
        import json
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"[INFO] 调优结果已保存 → {filepath}")
