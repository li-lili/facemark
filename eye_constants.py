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

# ============================================================
# 几何评分参数（纯约束，不依赖标准脸）
# ============================================================

# 指标1: 眼睛开放度 — 高宽比理想区间
OPENNESS_IDEAL_MIN = 0.61   # 正常眼睛高宽比下限
OPENNESS_IDEAL_MAX = 0.62   # 正常眼睛高宽比上限

# 指标2: 虹膜居中度 — 偏移占眼宽的阈值
IRIS_OFFSET_THRESHOLD = 0.033  # 偏移 < 10%眼宽 → 满分

# 虹膜质心外移补偿 (占眼宽的比例)
# 多边形质心偏向鼻侧，需向远离鼻子的方向微调
# 0.00 = 不补偿（纯多边形质心）
# 0.05 = 外移 5% 眼宽（推荐起始值）
IRIS_CENTER_OUTWARD = 0.05

# 指标3: 对称性 — (已注释)
# SYMMETRY_THRESHOLD = 0.10    # 偏差 < 10%半眼距 → 满分

# 指标4: 形状规整度
SHAPE_CV_IDEAL = 0.50        # 点距变异系数 < 0.5 为佳

# 综合得分权重 (总和=1.0)
GEOMETRY_WEIGHTS = {
    "openness":   0.50,   # 左右眼开放度各半
    "centered":   0.50,   # 左右眼虹膜居中度各半
    # "symmetry":   0.20,   # (已注释)
}

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
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
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
