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
# 默认角度范围（度）
# ============================================================
DEFAULT_ANGLE_MIN = 0
DEFAULT_ANGLE_MAX = 270
DEFAULT_ANGLE_STEP = 5  # 默认搜索步长

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
