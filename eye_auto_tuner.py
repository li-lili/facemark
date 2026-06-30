"""
Eye Auto Tuner — 眼睛舵机调整工具 (主入口)
==========================================
核心功能: 初始化硬件 + 发送舵机角度 + 采集检测 + 保存结果

模块结构:
  eye_constants.py   - 常量 + TuningResult 数据类
  ellseg_scorer.py   - 摄像头采集 + EllSeg虹膜偏移检测 (EllSegDetector)

检测方式:
  - 眼球模式 (eyeball): 用 EllSeg 检测虹膜位置，与基线对比计算像素偏移
  - 眼皮模式 (eyelid):  用 MediaPipe 关键点计算 EAR，与基线对比

使用方式:
  由 qt_calibration_panel.py 初始化并调用；本模块不再提供命令行入口。
"""

import os
import time
import json
from typing import List, Tuple

from Motor import FaceController
from Communication import UARTDevice
from utility import read_yaml, write_yaml
from eye_auto_lip_side import LipSideSamplingMixin
from eye_constants import (
    EYE_SERVO_CHANNELS, EYE_SERVO_NAMES,
    EYEBALL_CHANNELS, EYELID_CHANNELS,     EYEBROW_CHANNELS, EYEBROW_NAMES,
    EYEBROW_CUSTOM_BIG_TOLERANCE,
    EYEBROW_BIG_SYMMETRY_TOLERANCE,
    EYEBROW_WAIT_SECONDS, EYEBROW_MAX_ITERATIONS,
    DEFAULT_ANGLE_MIN, DEFAULT_ANGLE_MAX,
    CAMERA_INDEX, FLIP_HORIZONTAL,
    EYELID_EAR_TOLERANCE, EYELID_WAIT_SECONDS, EYELID_MAX_ITERATIONS,
    MOUTH_MAR_TOLERANCE, MOUTH_WAIT_SECONDS, MOUTH_MAX_ITERATIONS,
    MOUTH_CHIN_CHANNEL,
    LOWER_LIP_LLR_TOLERANCE,
    LOWER_LIP_WAIT_SECONDS, LOWER_LIP_MAX_ITERATIONS,
    LOWER_LIP_CHANNEL, LOWER_LIP_SIDE_CHANNEL,
    LOWER_LIP_SIDE_CAMERA_INDEX, LOWER_LIP_SIDE_ROI,
    LOWER_LIP_SIDE_FACE_DIRECTION, LOWER_LIP_SIDE_FLIP_VERTICAL, LOWER_LIP_SIDE_X_TOLERANCE,
    LOWER_LIP_SIDE_Y_TOLERANCE, LOWER_LIP_SIDE_SCORE_PCT,
    LOWER_LIP_SIDE_A_DELTA, LOWER_LIP_SIDE_MIN_SAT,
    LOWER_LIP_SIDE_SPLIT_PCT, LOWER_LIP_SIDE_MIN_AREA,
    UPPER_LIP_ULR_TOLERANCE,
    UPPER_LIP_WAIT_SECONDS, UPPER_LIP_MAX_ITERATIONS,
    UPPER_LIP_CHANNEL, UPPER_LIP_SIDE_CHANNEL,
    UPPER_LIP_SIDE_CAMERA_INDEX, UPPER_LIP_SIDE_ROI,
    UPPER_LIP_SIDE_FACE_DIRECTION, UPPER_LIP_SIDE_FLIP_VERTICAL, UPPER_LIP_SIDE_X_TOLERANCE,
    UPPER_LIP_SIDE_Y_TOLERANCE, UPPER_LIP_SIDE_SCORE_PCT,
    UPPER_LIP_SIDE_A_DELTA, UPPER_LIP_SIDE_MIN_SAT,
    UPPER_LIP_SIDE_SPLIT_PCT, UPPER_LIP_SIDE_MIN_AREA,
    MOUTH_CORNER_TOLERANCE,
    MOUTH_CORNER_Y_TOLERANCE,
    MOUTH_CORNER_WAIT_SECONDS, MOUTH_CORNER_MAX_ITERATIONS,
    MOUTH_CORNER_H_CHANNELS, MOUTH_CORNER_V_CHANNELS,
)
from ellseg_scorer import EllSegDetector, TOLERANCE

# 眼球舵机名称 (A10-A13)
EYEBALL_NAMES = [f"A{ch}" for ch in EYEBALL_CHANNELS]
# 眼皮舵机名称 (A8-A9)
EYELID_NAMES = [f"A{ch}" for ch in EYELID_CHANNELS]


class EyeAutoTuner(LipSideSamplingMixin):
    """眼睛舵机调整工具"""

    def __init__(
        self,
        yaml_file: str = "29_servo_config(13).yaml",
        port: str = "COM5",
        baudrate: int = 115200,
        servo_num: int = 27,
        stabilize_frames: int = 8,
        settle_time_ms: int = 300,
        detector: EllSegDetector = None,
    ):
        self.yaml_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), yaml_file)
        self.port = port
        self.baudrate = baudrate
        self.servo_num = servo_num
        self.stabilize_frames = stabilize_frames
        self.settle_time_ms = settle_time_ms

        # 外部传入的 detector（qt_calibration_panel 共享相机）
        self.external_detector = detector

        # 组件引用（延迟初始化）
        self.controller = None
        self.scorer = None
        self._owns_scorer = False
        self.angle_ranges = [(DEFAULT_ANGLE_MIN, DEFAULT_ANGLE_MAX)] * len(EYE_SERVO_CHANNELS)
        self.side_cap = None
        self._owns_side_cap = False

    # ------------------------------------------------------------------
    # 安全读取 list_temp_deg (list 没有 .get 方法)
    # ------------------------------------------------------------------
    def _temp_deg(self, ch, default=None):
        """安全读取 controller.list_temp_deg[ch]，越界返回 default"""
        try:
            return self.controller.list_temp_deg[ch]
        except (IndexError, TypeError):
            return default

    # ==================================================================
    # 初始化
    # ==================================================================

    def initialize(self):
        """初始化所有硬件和模型"""
        print("=" * 60)
        print("  EyeAutoTuner - 眼睛舵机调整工具")
        print("=" * 60)

        # 1. 初始化舵机控制器
        print("\n[1/2] 初始化舵机控制器...")
        interface = UARTDevice(self.port, self.baudrate)
        self.controller = FaceController(interface, self.servo_num, self.yaml_file)
        self.controller.open()
        self.controller.init_data()

        # 计算每个眼睛舵机的范围
        print("\n    眼睛舵机 (A8-A13) 范围:")
        for i, ch in enumerate(EYE_SERVO_CHANNELS):
            start = self.controller.list_start_deg[ch]
            end = self.controller.list_end_deg[ch]
            lo = min(start, end)
            hi = max(start, end)
            self.angle_ranges[i] = (lo, hi)
            center = round(start + (end - start) * 0.5)
            print(f"    A{ch} ({EYE_SERVO_NAMES[i]}): range=[{lo}, {hi}] center={center}°")

        # 眉毛舵机 (A0-A1) 范围
        print("\n    眉毛舵机 (A0-A1) 范围:")
        self.eyebrow_ranges = {}
        for idx, ch in enumerate(EYEBROW_CHANNELS):
            start = self.controller.list_start_deg[ch]
            end = self.controller.list_end_deg[ch]
            lo = min(start, end)
            hi = max(start, end)
            self.eyebrow_ranges[ch] = (lo, hi)
            print(f"    A{ch} ({EYEBROW_NAMES[idx]}): range=[{lo}, {hi}]")

        # 2. 初始化 EllSeg 虹膜检测器
        if self.external_detector is not None:
            self.scorer = self.external_detector
            self._owns_scorer = False
            print("  (使用外部共享相机)")
        else:
            print("\n[2/2] 初始化 EllSeg 虹膜检测系统...")
            self.scorer = EllSegDetector(
                camera_index=CAMERA_INDEX,
                width=1920,
                height=1080,
                flip_horizontal=FLIP_HORIZONTAL,
                stabilize_frames=self.stabilize_frames,
            )
            self._owns_scorer = True

        print("\n" + "=" * 60)
        print("  初始化完成！")
        print("=" * 60 + "\n")

    # ==================================================================
    # 舵机操作
    # ==================================================================

    def send_servo(self, channel: int, angle: int,
                   angle_range: Tuple[int, int] = None):
        """发送单个舵机角度，自动 clamp 到范围，并同步当前内存角度。"""
        if angle_range:
            angle = max(angle_range[0], min(angle_range[1], angle))
        self.controller.set_servo_angle_time_32(
            [angle], [channel], 200, servo_num=self.controller.servo_num
        )
        # 记录已发送角度，避免下一次 UI 手动/自动动作又从 YAML temp_deg 起步。
        if 0 <= channel < len(self.controller.list_temp_deg):
            self.controller.list_temp_deg[channel] = angle

    def send_servos(self, channels: List[int], angles: List[int],
                    angle_ranges: List[Tuple[int, int]] = None):
        """发送多个舵机角度"""
        for ch, ang in zip(channels, angles):
            idx = channels.index(ch) if angle_ranges else None
            ar = angle_ranges[idx] if (angle_ranges and idx is not None) else None
            self.send_servo(ch, ang, ar)

    def capture_and_score(self):
        """等待舵机稳定后采集检测（EllSeg 虹膜偏移）"""
        time.sleep(self.settle_time_ms / 1000.0)
        return self.scorer.capture_and_score(
            stabilize_frames=self.stabilize_frames,
        )

    def _angle_range(self, channel: int) -> Tuple[int, int]:
        start = self.controller.list_start_deg[channel]
        end = self.controller.list_end_deg[channel]
        return min(start, end), max(start, end)

    # ==================================================================
    # 多帧采集 + 修剪平均 — 眼球
    # ==================================================================

    def collect_guide_samples(self, n_frames: int = 60, trim: int = 10):
        """
        打开 MP + EllSeg → 采集 n 帧偏移 → 关闭 → 修剪平均。

        流程:
          1. 开启 MP 和 EllSeg 检测
          2. 连续采集 n_frames 帧有效引导信号
          3. 关闭 MP 和 EllSeg
          4. 按 total_dist 排序，去掉最高的 trim 帧和最低的 trim 帧
          5. 对中间帧求平均值
          6. 返回平均偏移信号（结构兼容 get_guide_signal）

        Returns:
            dict 或 None: 平均后的引导信号
        """
        # ---- 1. 开启检测 ----
        self.scorer.enable_mp = True
        self.scorer.enable_ellseg = True

        samples = []  # list of (total_dist, offsets_dict)
        collected = 0
        no_face_count = 0

        while collected < n_frames:
            if self.scorer.user_pressed_stop:
                break

            ok, frame = self.scorer.capture()
            if not ok:
                time.sleep(0.001)
                continue

            self.scorer.detect(frame)
            signal = self.scorer.get_guide_signal()

            if signal is None:
                no_face_count += 1
                if no_face_count > 10:  # 连续10帧无脸 → 放弃
                    break
                time.sleep(0.001)
                continue
            no_face_count = 0

            offsets = {
                "L_dX": signal.get("L_dX", 0),
                "L_dY": signal.get("L_dY", 0),
                "L_dist": signal.get("L_dist", 0),
                "R_dX": signal.get("R_dX", 0),
                "R_dY": signal.get("R_dY", 0),
                "R_dist": signal.get("R_dist", 0),
            }
            total_dist = offsets["L_dist"] + offsets["R_dist"]
            samples.append((total_dist, offsets))
            collected += 1

            self.scorer.update_hud([
                (f"Sampling: {collected}/{n_frames}", (10, 120), (200, 200, 200)),
            ])

        # ---- 2. 关闭检测 ----
        self.scorer.enable_mp = False
        self.scorer.enable_ellseg = False

        if len(samples) < 2 * trim + 1:
            print(f"  [Collect] 有效样本不足: {len(samples)}, 需要 ≥ {2*trim+1}")
            return None

        # ---- 3. 排序 + 修剪 ----
        samples.sort(key=lambda x: x[0])
        trimmed = samples[trim:-trim]

        # ---- 4. 平均 ----
        n = len(trimmed)
        avg = {}
        for key in ["L_dX", "L_dY", "L_dist", "R_dX", "R_dY", "R_dist"]:
            avg[key] = sum(s[1][key] for s in trimmed) / n
        # 反方向 = 舵机调整方向
        avg["L_adj_X"] = -avg["L_dX"]
        avg["L_adj_Y"] = -avg["L_dY"]
        avg["R_adj_X"] = -avg["R_dX"]
        avg["R_adj_Y"] = -avg["R_dY"]
        avg["total_dist"] = avg["L_dist"] + avg["R_dist"]

        # 统计
        all_dists = [s[0] for s in samples]
        print(f"  [Collect] {len(samples)} frames, trim {trim}: "
              f"avg_dist={avg['total_dist']:.1f}px "
              f"(min={all_dists[0]:.1f}, max={all_dists[-1]:.1f})")

        return avg

    # ==================================================================
    # 引导式自动调整: 采集60帧取平均 → 移动眼球(A10-A13) → 等1s → 循环
    # ==================================================================

    def auto_adjust(
        self,
        pixel_to_degree: float = 2.0,
        max_iterations: int = 50,
        wait_seconds: float = 1.0,
        keep_display: bool = False,
    ) -> Tuple[bool, int, dict]:
        """
        引导式自动调整 — 只调整 A10-A13 眼球舵机。

        流程（每轮迭代）:
          1. 打开 MP + EllSeg
          2. 采集 60 帧 → 去最高/最低各 10 帧 → 中间 40 帧求平均偏移
          3. 关闭 MP + EllSeg
          4. 若平均偏移 ≤ TOLERANCE → 通过
          5. 否则按平均偏移方向移动舵机 → 等待 → 下一轮

        Args:
            pixel_to_degree:   像素→角度换算系数
            max_iterations:    最大循环次数
            wait_seconds:      每次移动后等待秒数

        Returns:
            (passed: bool, iterations: int, final_region_scores: dict)
        """
        if self.scorer.baseline is None:
            print("[WARN] 未加载基线! 请先运行 ellseg_scorer.py 按 [S] 保存基线到 face_point.json")
            return False, 0, {}

        print(f"{'='*60}")
        print(f"  引导式自动调整 (仅眼球 A10-A13)")
        print(f"  采集: 60帧, 修剪上下10帧, 中间40帧取平均")
        print(f"  系数: {pixel_to_degree} °/px | 最大迭代: {max_iterations} | "
              f"等待: {wait_seconds}s")
        print(f"  目标: 平均偏移 ≤ {TOLERANCE}px")
        print(f"{'='*60}\n")

        self.scorer.start_display()

        current_angles = {ch: self.controller.list_temp_deg[ch] for ch in EYEBALL_CHANNELS}
        print(f"  初始角度: {dict(zip(EYEBALL_NAMES, [current_angles[c] for c in EYEBALL_CHANNELS]))}")

        iteration = 0
        sleep_chunk = 0.1

        try:
            while iteration < max_iterations:
                iteration += 1

                if self.scorer.user_pressed_stop:
                    print("\n[USER] 收到停止信号")
                    break

                self.scorer.update_hud([
                    (f"[Iter {iteration}/{max_iterations}]", (10, 100), (200, 200, 200)),
                ])

                # ---- 步骤 1+2: 采集60帧 → 修剪平均 ----
                avg_signal = self.collect_guide_samples(n_frames=60, trim=10)

                if avg_signal is None:
                    self.scorer.update_hud([
                        (f"[Iter {iteration}/{max_iterations}]", (10, 100), (200, 200, 200)),
                        (f"Signal: None (no valid frames)", (10, 118), (0, 160, 255)),
                    ])
                    print(f"  [Iter {iteration:3d}] 无法获取有效帧")

                    _waited = 0
                    while _waited < wait_seconds:
                        time.sleep(min(sleep_chunk, wait_seconds - _waited))
                        _waited += sleep_chunk
                        if self.scorer.user_pressed_stop:
                            break
                    continue

                l_dx = avg_signal["L_dX"]
                l_dy = avg_signal["L_dY"]
                r_dx = avg_signal["R_dX"]
                r_dy = avg_signal["R_dY"]
                l_dist = avg_signal["L_dist"]
                r_dist = avg_signal["R_dist"]

                # ---- 步骤 3: 已关闭 MP/EllSeg (collect_guide_samples 已关) ----

                # ---- 步骤 4: 检查是否已合格 ----
                qualified = (l_dist <= TOLERANCE and r_dist <= TOLERANCE)

                dist_s = f"L={l_dist:.1f}px R={r_dist:.1f}px"
                print(f"  [Iter {iteration:3d}] 平均偏移: {dist_s} | "
                      f"L(dX={l_dx:+.1f} dY={l_dy:+.1f}) "
                      f"R(dX={r_dx:+.1f} dY={r_dy:+.1f})")

                if qualified:
                    print(f"\n{'='*60}")
                    print(f"  ★★ 通过! 共迭代 {iteration} 次")
                    print(f"      平均偏移: L={l_dist:.1f}px R={r_dist:.1f}px")
                    final_ang = " ".join([f"{n}={current_angles[c]}°"
                                         for n, c in zip(EYEBALL_NAMES, EYEBALL_CHANNELS)])
                    print(f"      角度: {final_ang}")
                    print(f"{'='*60}\n")

                    all_angles = [current_angles.get(ch, 0) for ch in EYE_SERVO_CHANNELS]
                    self.save_result(all_angles)
                    self.export_angle_config(all_angles)

                    self.scorer.update_hud([("★★ PASSED ★★", (120, 80), (0, 255, 0))])
                    time.sleep(1)
                    return True, iteration, {}

                # ---- 步骤 5: 按平均偏移移动舵机 ----
                l_adj_x = avg_signal["L_adj_X"]
                r_adj_x = avg_signal["R_adj_X"]
                l_adj_y = avg_signal["L_adj_Y"]
                r_adj_y = avg_signal["R_adj_Y"]

                # A10: L_vertical 上下 (减=下)  |  A11: R_vertical 上下 (减=下, 物理反向)
                # A12: L_horizontal 内外 (减=外) |  A13: R_horizontal 内外 (减=外, 物理反向)
                adj_map = {
                    EYEBALL_CHANNELS[0]:  1.0 if l_adj_y < 0 else (-1.0 if l_adj_y > 0 else 0.0),
                    EYEBALL_CHANNELS[1]: -1.0 if r_adj_y < 0 else ( 1.0 if r_adj_y > 0 else 0.0),
                    EYEBALL_CHANNELS[2]: -1.0 if l_adj_x < 0 else ( 1.0 if l_adj_x > 0 else 0.0),
                    EYEBALL_CHANNELS[3]: -1.0 if r_adj_x < 0 else ( 1.0 if r_adj_x > 0 else 0.0),
                }

                moved = []
                for idx, ch in enumerate(EYEBALL_CHANNELS):
                    delta = adj_map[ch]
                    if delta == 0:
                        continue
                    new_angle = round(current_angles[ch] + delta)
                    lo, hi = self.angle_ranges[EYE_SERVO_CHANNELS.index(ch)]
                    new_angle = max(lo, min(hi, new_angle))
                    self.send_servo(ch, new_angle, (lo, hi))
                    current_angles[ch] = new_angle
                    moved.append(f"{EYEBALL_NAMES[idx]}={new_angle}°(Δ{delta:+.1f})")

                if not moved:
                    print(f"  [Iter {iteration:3d}] 无需移动 (偏移为0)")
                else:
                    print(f"  [Iter {iteration:3d}] 舵机移动: {' '.join(moved)}")

                # ---- 步骤 6: 等待 ----
                _waited = 0
                while _waited < wait_seconds:
                    time.sleep(min(sleep_chunk, wait_seconds - _waited))
                    _waited += sleep_chunk
                    self.scorer.update_hud([
                        (f"[Iter {iteration}/{max_iterations}]", (10, 100), (200, 200, 200)),
                        (f"Waiting... {_waited:.1f}/{wait_seconds}s", (10, 118), (0, 255, 255)),
                    ])
                    if self.scorer.user_pressed_stop:
                        break

            # 达到最大迭代未通过
            print(f"\n{'='*60}")
            print(f"  ✘ 未在 {max_iterations} 次迭代内通过")
            print(f"{'='*60}\n")
            return False, iteration, {}

        finally:
            if not keep_display:
                self.scorer.stop_display()
    def auto_adjust_upper_lip_front_side(
        self,
        step: float = 1.0,
        max_iterations: int = None,
        wait_seconds: float = None,
        keep_display: bool = False,
    ) -> tuple:
        """Adjust upper lip with front ULR and side-view front-point checks.

        A16 follows the front ULR error. A17 follows the side upper-lip x error.
        Both channels are clamped to their YAML ranges and sent in the same loop.
        """
        if max_iterations is None:
            max_iterations = UPPER_LIP_MAX_ITERATIONS
        if wait_seconds is None:
            wait_seconds = UPPER_LIP_WAIT_SECONDS

        bl = self.scorer.upper_lip_baseline
        if bl is None or "ulr" not in bl:
            print("[WARN] Missing upper lip front ULR baseline.")
            return False, 0, {}
        if "side_upper_tip" not in bl:
            print("[WARN] Missing upper lip side baseline. Save side upper lip point first.")
            return False, 0, {}

        ch_front = UPPER_LIP_CHANNEL
        ch_side = UPPER_LIP_SIDE_CHANNEL
        front_range = self._angle_range(ch_front)
        side_range = self._angle_range(ch_side)
        front_angle = self._temp_deg(ch_front, default=round(sum(front_range) / 2))
        side_angle = self._temp_deg(ch_side, default=round(sum(side_range) / 2))
        front_sign = getattr(self, "_upper_lip_flip", 1)
        side_sign = getattr(self, "_upper_lip_side_flip", 1)

        print("=" * 60)
        print("  Upper lip auto adjust: front ULR + side front point")
        print(f"  A16 range={front_range}, A17 range={side_range}, step={step}")
        print(
            f"  Target: ULR tol={UPPER_LIP_ULR_TOLERANCE}, "
            f"side x tol={UPPER_LIP_SIDE_X_TOLERANCE}px, "
            f"side y tol={UPPER_LIP_SIDE_Y_TOLERANCE}px"
        )
        print(f"  Baseline: ULR={bl['ulr']:.4f}, side_upper_tip={bl['side_upper_tip']}")
        print("=" * 60)

        self.scorer.start_display()
        iteration = 0
        sleep_chunk = 0.1
        final_data = {}

        try:
            while iteration < max_iterations:
                iteration += 1
                if self.scorer.user_pressed_stop:
                    break

                self.scorer.update_hud([
                    (f"[UpperLip FS Iter {iteration}/{max_iterations}]", (10, 100), (200, 200, 200)),
                    (f"A16={front_angle} A17={side_angle}", (10, 118), (0, 255, 255)),
                ])

                front = self.collect_upper_lip_samples(n_frames=30, trim=5)
                side = self.collect_upper_lip_side_samples(n_frames=30, trim=5)
                final_data = {"front": front, "side": side}
                if front is None or side is None:
                    print(f"  [UpperLip FS Iter {iteration:3d}] missing sample")
                    time.sleep(wait_seconds)
                    continue

                front_delta = front["delta"]
                side_dx = side.get("dx", 0.0)
                side_dy = side.get("dy", 0.0)
                front_ok = abs(front_delta) <= UPPER_LIP_ULR_TOLERANCE
                side_ok = side.get("qualified", False)
                side_y_out = abs(side_dy) > UPPER_LIP_SIDE_Y_TOLERANCE

                print(
                    f"  [UpperLip FS Iter {iteration:3d}] "
                    f"front ULR={front['ulr']:.4f} d={front_delta:+.4f} ok={front_ok} | "
                    f"side tip=({side['tip'][0]:.1f},{side['tip'][1]:.1f}) "
                    f"dx={side_dx:+.1f} dy={side_dy:+.1f} ok={side_ok}"
                )

                if front_ok and side_ok:
                    self.scorer.update_hud([
                        ("UPPER LIP FRONT+SIDE PASSED", (100, 80), (0, 255, 0)),
                    ])
                    all_angles = [self.controller.list_temp_deg[c] for c in EYE_SERVO_CHANNELS]
                    self.export_angle_config(
                        all_angles,
                        chin_angle=self._temp_deg(MOUTH_CHIN_CHANNEL),
                        lower_lip_angle=self._temp_deg(LOWER_LIP_CHANNEL),
                        upper_lip_angle=front_angle,
                    )
                    return True, iteration, final_data

                moved = False
                if not front_ok:
                    front_dir = -1 if front_delta > UPPER_LIP_ULR_TOLERANCE else 1
                    new_front = round(front_angle + front_sign * front_dir * step)
                    new_front = max(front_range[0], min(front_range[1], new_front))
                    if new_front != front_angle:
                        front_angle = new_front
                        moved = True
                elif side_y_out:
                    # 正面达标但侧面 y 超差时允许 A16 小步补偿；若下一轮 ULR 出界，会优先按正面纠正。
                    front_dir = -1 if side_dy > UPPER_LIP_SIDE_Y_TOLERANCE else 1
                    new_front = round(front_angle + front_dir * step)
                    new_front = max(front_range[0], min(front_range[1], new_front))
                    if new_front != front_angle:
                        front_angle = new_front
                        moved = True
                        final_data["side_dy_front_adjust"] = {
                            "side_dy": side_dy,
                            "front_delta": front_delta,
                        }

                if not side_ok and abs(side_dx) > UPPER_LIP_SIDE_X_TOLERANCE:
                    # A17: 前=165, 后=200. side_dx>0 表示上唇前点太靠前，需增大角度往后。
                    side_dir = 1 if side_dx > UPPER_LIP_SIDE_X_TOLERANCE else -1
                    new_side = round(side_angle + side_sign * side_dir * step)
                    new_side = max(side_range[0], min(side_range[1], new_side))
                    if new_side != side_angle:
                        side_angle = new_side
                        moved = True

                if moved:
                    self.send_servo(ch_front, front_angle, front_range)
                    self.send_servo(ch_side, side_angle, side_range)
                    print(f"  [UpperLip FS Iter {iteration:3d}] move A16={front_angle}, A17={side_angle}")
                else:
                    warning = (
                        f"[UpperLip FS Iter {iteration:3d}] no legal move available "
                        f"(front_ok={front_ok}, side_ok={side_ok}, "
                        f"front_delta={front_delta:+.4f}, side_dx={side_dx:+.1f}, side_dy={side_dy:+.1f})"
                    )
                    print(f"  {warning}")
                    self.scorer.update_hud([
                        ("UPPER LIP STOP: NO LEGAL MOVE", (10, 136), (0, 0, 255)),
                    ])
                    final_data["warning"] = warning
                    final_data["stop_reason"] = "no_legal_move"
                    return False, iteration, final_data

                waited = 0.0
                while waited < wait_seconds:
                    time.sleep(min(sleep_chunk, wait_seconds - waited))
                    waited += sleep_chunk
                    if self.scorer.user_pressed_stop:
                        break

            return False, iteration, final_data
        finally:
            if not keep_display:
                self.scorer.stop_display()

    def collect_eyelid_samples(self, n_frames: int = 60, trim: int = 10):
        """
        打开 MP（无需 EllSeg）→ 采集 n 帧 EAR → 关闭 → 修剪平均。

        流程:
          1. 开启 MP
          2. 连续采集 n_frames 帧 EAR 信号
          3. 关闭 MP
          4. 按 abs(L_delta) + abs(R_delta) 排序，修剪上下
          5. 返回平均 EAR 信号

        Returns:
            dict 或 None: {"left_ear", "right_ear", "left_delta", "right_delta",
                           "eyelid_symmetry", "left_qualified",
                           "right_qualified", "symmetry_qualified"}
        """
        self.scorer.enable_mp = True
        self.scorer.enable_ellseg = False

        samples = []  # list of (total_deviation, signal_dict)
        collected = 0
        no_face_count = 0

        while collected < n_frames:
            if self.scorer.user_pressed_stop:
                break

            ok, frame = self.scorer.capture()
            if not ok:
                time.sleep(0.001)
                continue

            self.scorer.detect(frame)
            signal = self.scorer.get_eyelid_signal()

            if signal is None:
                no_face_count += 1
                if no_face_count > 10:
                    break
                time.sleep(0.001)
                continue
            no_face_count = 0

            total_dev = abs(signal["left_delta"]) + abs(signal["right_delta"])
            samples.append((total_dev, {
                "left_ear": signal["left_ear"],
                "right_ear": signal["right_ear"],
                "left_delta": signal["left_delta"],
                "right_delta": signal["right_delta"],
                "eyelid_symmetry": signal["eyelid_symmetry"],
            }))
            collected += 1

            # 采集进度（基本参数由 cv2.putText 常驻显示）
            self.scorer.update_hud([
                (f"Eyelid Sampling: {collected}/{n_frames}", (10, 136), (200, 200, 200)),
            ])

        self.scorer.enable_mp = False

        if len(samples) < 2 * trim + 1:
            print(f"  [Eyelid Collect] 有效样本不足: {len(samples)}, 需要 >= {2*trim+1}")
            return None

        samples.sort(key=lambda x: x[0])
        trimmed = samples[trim:-trim]

        n = len(trimmed)
        avg = {}
        for key in ["left_ear", "right_ear", "left_delta", "right_delta"]:
            avg[key] = sum(s[1][key] for s in trimmed) / n

        tol = EYELID_EAR_TOLERANCE
        avg["eyelid_symmetry"] = abs(avg["left_ear"] - avg["right_ear"])
        avg["left_qualified"] = abs(avg["left_delta"]) <= tol
        avg["right_qualified"] = abs(avg["right_delta"]) <= tol
        avg["symmetry_qualified"] = avg["eyelid_symmetry"] <= tol

        all_devs = [s[0] for s in samples]
        print(f"  [Eyelid Collect] {len(samples)} frames, trim {trim}: "
              f"L_delta={avg['left_delta']:+.4f} R_delta={avg['right_delta']:+.4f} "
              f"Sym={avg['eyelid_symmetry']:.4f} "
              f"(min={all_devs[0]:.4f}, max={all_devs[-1]:.4f})")

        return avg

    # ==================================================================
    # 引导式自动调整 — 眼皮 (A8-A9)
    # ==================================================================

    def auto_adjust_eyelid(
        self,
        ear_step: float = 1.0,
        max_iterations: int = None,
        wait_seconds: float = None,
        keep_display: bool = False,
    ) -> tuple:
        """
        引导式自动调整 — 两阶段眼皮 A8/A9 舵机。

        Phase 1 (各自达标): 左右眼皮各自回到自己的 EAR 基线通过区间
        Phase 2 (对称): 高宽比更高的一侧向更低的一侧靠

        流程（每轮迭代）:
          1. 打开 MP（仅 MediaPipe 关键点）
          2. 采集 60 帧 → 去头尾求平均 EAR
          3. 关闭 MP
          4. P1: 若单侧 EAR 偏离基线容差 → 该侧单独开/合
          5. P1 通过后进入 P2: 若 |L_EAR - R_EAR| > 对称容差 → 合上 EAR 更高侧
          6. 全部达标 → 通过

        Args:
            ear_step:       每次调整的舵机角度步长 (默认 1°)
            max_iterations: 最大循环次数 (默认 EYELID_MAX_ITERATIONS)
            wait_seconds:   每次移动后等待秒数 (默认 EYELID_WAIT_SECONDS)

        Returns:
            (passed: bool, iterations: int, final_ear_data: dict)
        """
        if max_iterations is None:
            max_iterations = EYELID_MAX_ITERATIONS
        if wait_seconds is None:
            wait_seconds = EYELID_WAIT_SECONDS

        if self.scorer.eyelid_baseline is None:
            print("[WARN] 未加载眼皮基线! 请先在 ellseg_scorer.py 中按 [E] 保存到 face_point.json 当前性别")
            return False, 0, {}

        tol = EYELID_EAR_TOLERANCE
        sym_tol = EYELID_EAR_TOLERANCE  # 对称容差与 EAR 容差相同

        print(f"{'='*60}")
        print(f"  引导式自动调整 (仅眼皮 A8-A9) — 两阶段")
        print(f"  Phase 1: 各自达标 (|L_delta|、|R_delta| ≤ {tol})")
        print(f"  Phase 2: 左右对称 (|L_EAR - R_EAR| ≤ {sym_tol})")
        print(f"  采集: 60帧, 修剪上下10帧, 中间40帧取平均 EAR")
        print(f"  步长: {ear_step}度/次 | 最大迭代: {max_iterations} | 等待: {wait_seconds}s")
        bl = self.scorer.eyelid_baseline
        print(f"  基线: L_EAR={bl['left_ear']:.4f} R_EAR={bl['right_ear']:.4f}")
        print(f"{'='*60}\n")

        self.scorer.start_display()

        current_angles = {
            ch: self.controller.list_temp_deg[ch]
            for ch in EYELID_CHANNELS
        }
        print(f"  初始角度: {dict(zip(EYELID_NAMES, [current_angles[c] for c in EYELID_CHANNELS]))}")

        a8_sign = getattr(self, '_eyelid_a8_flip', -1)
        a9_sign = getattr(self, '_eyelid_a9_flip', 1)

        iteration = 0
        phase = 1  # start in Phase 1 (各自达标)
        sleep_chunk = 0.1

        try:
            while iteration < max_iterations:
                iteration += 1

                if self.scorer.user_pressed_stop:
                    print("\n[USER] 收到停止信号")
                    break

                self.scorer.update_hud([
                    (f"[Eyelid P{phase} Iter {iteration}/{max_iterations}]", (10, 100), (200, 200, 200)),
                ])

                # ---- 采集 EAR ----
                avg_signal = self.collect_eyelid_samples(n_frames=60, trim=10)

                if avg_signal is None:
                    self.scorer.update_hud([
                        (f"[Eyelid P{phase} Iter {iteration}/{max_iterations}]", (10, 100), (200, 200, 200)),
                        (f"Signal: None (no valid frames)", (10, 118), (0, 160, 255)),
                    ])
                    print(f"  [Eyelid P{phase} Iter {iteration:3d}] 无法获取有效帧")
                    _waited = 0
                    while _waited < wait_seconds:
                        time.sleep(min(sleep_chunk, wait_seconds - _waited))
                        _waited += sleep_chunk
                        if self.scorer.user_pressed_stop:
                            break
                    continue

                l_delta = avg_signal["left_delta"]
                r_delta = avg_signal["right_delta"]
                l_ear = avg_signal["left_ear"]
                r_ear = avg_signal["right_ear"]

                # 对称性: 左右当前 EAR 高宽比差异。
                ear_sym = avg_signal["eyelid_symmetry"]
                sym_ok = avg_signal["symmetry_qualified"]

                print(f"  [Eyelid P{phase} Iter {iteration:3d}] "
                      f"L_EAR={l_ear:.4f} (Δ={l_delta:+.4f}) "
                      f"R_EAR={r_ear:.4f} (Δ={r_delta:+.4f}) "
                      f"Sym={ear_sym:.4f}")

                # ---- update_hud: 只显示调整专有信息，基本参数由 cv2.putText 常驻 ----
                phase_label = "P1:各自" if phase == 1 else "P2:对称"

                self.scorer.update_hud([
                    (f"[Eyelid {phase_label} Iter {iteration}/{max_iterations}]", (10, 100), (200, 200, 200)),
                    (f"A8={current_angles[EYELID_CHANNELS[0]]}° A9={current_angles[EYELID_CHANNELS[1]]}°", (10, 118), (200, 200, 200)),
                ])

                # ---- 检查两阶段全部合格 ----
                l_ok = abs(l_delta) <= tol
                r_ok = abs(r_delta) <= tol
                if phase == 2 and sym_ok and l_ok and r_ok:
                    print(f"\n{'='*60}")
                    print(f"  ★★ 眼皮调整通过! 共迭代 {iteration} 次 (Phase 2)")
                    print(f"      L_EAR={l_ear:.4f}  R_EAR={r_ear:.4f}")
                    final_ang = " ".join([f"{n}={current_angles[c]}度"
                                         for n, c in zip(EYELID_NAMES, EYELID_CHANNELS)])
                    print(f"      角度: {final_ang}")
                    print(f"{'='*60}\n")

                    all_angles = []
                    for ch in EYE_SERVO_CHANNELS:
                        if ch in current_angles:
                            all_angles.append(current_angles[ch])
                        else:
                            all_angles.append(self.controller.list_temp_deg[ch])
                    self.save_result(all_angles)
                    self.export_angle_config(all_angles)

                    self.scorer.update_hud([("★★ EYELID PASSED ★★", (100, 80), (0, 255, 0))])
                    time.sleep(1)
                    return True, iteration, avg_signal

                # ================================================================
                # Phase 1: 左右各自达标 — 单侧 EAR 偏离基线就单独开/合
                # ================================================================
                if phase == 1:
                    if l_ok and r_ok:
                        # 两侧都已在各自基线容差内 → 进入 Phase 2
                        phase = 2
                        print(f"  [P1 → P2] L/R_EAR 均通过，进入对称调整 "
                              f"(当前 sym={ear_sym:.4f}, 容差={sym_tol:.4f})")
                        continue

                    moves = []
                    if not l_ok:
                        moves.append(("L", EYELID_CHANNELS[0], EYELID_NAMES[0], a8_sign, l_delta))
                    if not r_ok:
                        moves.append(("R", EYELID_CHANNELS[1], EYELID_NAMES[1], a9_sign, r_delta))

                    moved_sides = []
                    limit_notes = []
                    action_parts = []
                    for side, ch, name, sign, deviation in moves:
                        # deviation > 0: 这侧比基线更开 → 合
                        # deviation < 0: 这侧比基线更闭 → 开
                        direction = -1 if deviation > 0 else 1
                        delta_angle = sign * direction
                        old_angle = current_angles[ch]
                        lo, hi = self.angle_ranges[EYE_SERVO_CHANNELS.index(ch)]
                        new_angle = round(old_angle + delta_angle * ear_step)
                        new_angle = max(lo, min(hi, new_angle))

                        if new_angle == old_angle:
                            limit_note = f"{name}到极限({old_angle}度)"
                            limit_notes.append(limit_note)
                            print(f"  [Eyelid P{phase} Iter {iteration:3d}] {limit_note}")
                            continue

                        self.send_servo(ch, new_angle, (lo, hi))
                        current_angles[ch] = new_angle
                        moved_sides.append(side)
                        action = "合" if direction < 0 else "开"
                        action_parts.append(f"{side}{action} Δ={deviation:+.4f}")
                        print(f"  [Eyelid P{phase} Iter {iteration:3d}] {side}: {name}={new_angle}度 "
                              f"(delta={deviation:+.4f}, {'close' if direction < 0 else 'open'} to own range)")

                    if not moved_sides:
                        print(f"  [Eyelid P{phase} Iter {iteration:3d}] P1未移动: 未通过侧已到极限")
                    # P1 动作 HUD
                    action_text = (
                        "P1→" + " ".join(action_parts)
                        if action_parts else "P1未通过侧到极限"
                    )
                    display_phase_label = "P1:各自" if phase == 1 else "P2:对称"
                    self.scorer.update_hud([
                        (f"[Eyelid {display_phase_label} Iter {iteration}/{max_iterations}]", (10, 100), (200, 200, 200)),
                        (action_text, (10, 118), (0, 255, 200)),
                        ("; ".join(limit_notes), (10, 136), (0, 180, 255)),
                    ])

                # ================================================================
                # Phase 2: 左右对称 — EAR 高的一侧向低的一侧靠
                # ================================================================
                else:
                    if not (l_ok and r_ok):
                        phase = 1
                        print(f"  [Eyelid P2 Iter {iteration:3d}] 单侧 EAR 离开通过区间 "
                              f"(L_ok={l_ok} R_ok={r_ok})，回到P1各自达标")
                        self.scorer.update_hud([
                            (f"[Eyelid P1:各自 Iter {iteration}/{max_iterations}]", (10, 100), (200, 200, 200)),
                            ("P2单侧离区间→回P1", (10, 118), (0, 180, 255)),
                        ])
                        continue

                    if sym_ok:
                        print(f"\n{'='*60}")
                        print(f"  ★★ 眼皮调整通过! 共迭代 {iteration} 次 (Phase 2)")
                        print(f"      L_EAR={l_ear:.4f}  R_EAR={r_ear:.4f}  Sym={ear_sym:.4f}")
                        final_ang = " ".join([f"{n}={current_angles[c]}度"
                                             for n, c in zip(EYELID_NAMES, EYELID_CHANNELS)])
                        print(f"      角度: {final_ang}")
                        print(f"{'='*60}\n")

                        all_angles = []
                        for ch in EYE_SERVO_CHANNELS:
                            if ch in current_angles:
                                all_angles.append(current_angles[ch])
                            else:
                                all_angles.append(self.controller.list_temp_deg[ch])
                        self.save_result(all_angles)
                        self.export_angle_config(all_angles)

                        self.scorer.update_hud([("★★ EYELID PASSED ★★", (100, 80), (0, 255, 0))])
                        time.sleep(1)
                        return True, iteration, avg_signal

                    # 高宽比高的一侧闭合，向低的一侧靠。
                    if l_ear >= r_ear:
                        side, ch, name, base_sign = "L", EYELID_CHANNELS[0], EYELID_NAMES[0], a8_sign
                    else:
                        side, ch, name, base_sign = "R", EYELID_CHANNELS[1], EYELID_NAMES[1], a9_sign

                    old_angle = current_angles[ch]
                    lo, hi = self.angle_ranges[EYE_SERVO_CHANNELS.index(ch)]
                    new_angle = round(old_angle + base_sign * -1 * ear_step)
                    new_angle = max(lo, min(hi, new_angle))
                    limit_note = ""
                    if new_angle == old_angle:
                        limit_note = f"{name}闭合到极限({old_angle}deg)"
                        print(f"  [Eyelid P{phase} Iter {iteration:3d}] {limit_note}")
                    else:
                        self.send_servo(ch, new_angle, (lo, hi))
                        current_angles[ch] = new_angle
                        print(f"  [Eyelid P{phase} Iter {iteration:3d}] {side}: {name}={new_angle}deg "
                              f"(higher EAR closes toward lower, sym={ear_sym:.4f})")

                    self.scorer.update_hud([
                        (f"[Eyelid {phase_label} Iter {iteration}/{max_iterations}]", (10, 100), (200, 200, 200)),
                        (f"P2→合{side} 靠低EAR sym={ear_sym:.4f}", (10, 118), (255, 180, 0)),
                        (limit_note, (10, 136), (0, 180, 255)),
                    ])

                # ---- 等待 ----
                _waited = 0
                while _waited < wait_seconds:
                    time.sleep(min(sleep_chunk, wait_seconds - _waited))
                    _waited += sleep_chunk
                    p_label = "P1:各自" if phase == 1 else "P2:对称"
                    self.scorer.update_hud([
                        (f"[Eyelid {p_label} Iter {iteration}/{max_iterations}]", (10, 100), (200, 200, 200)),
                        (f"Waiting... {_waited:.1f}/{wait_seconds}s", (10, 118), (0, 255, 255)),
                    ])
                    if self.scorer.user_pressed_stop:
                        break

            # 达到最大迭代未通过
            print(f"\n{'='*60}")
            print(f"  ✘ 眼皮未在 {max_iterations} 次迭代内通过 (停在 Phase {phase})")
            print(f"{'='*60}\n")
            return False, iteration, {}

        finally:
            if not keep_display:
                self.scorer.stop_display()
    def collect_eyebrow_samples(self, n_frames: int = 60, trim: int = 10):
        """
        Collect eyebrow BIG and slope samples, trim outliers, and average.
        Eyebrow pass/fail uses only BIG range, BIG symmetry, and slope range.
        """
        self.scorer.enable_mp = True
        self.scorer.enable_ellseg = False

        samples = []
        collected = 0
        no_face_count = 0

        while collected < n_frames:
            if self.scorer.user_pressed_stop:
                break

            ok, frame = self.scorer.capture()
            if not ok:
                time.sleep(0.001)
                continue

            self.scorer.detect(frame)
            signal = self.scorer.get_eyebrow_signal()

            if signal is None:
                no_face_count += 1
                if no_face_count > 10:
                    break
                time.sleep(0.001)
                continue
            no_face_count = 0

            total_dev = abs(signal["left_position_delta"]) + abs(signal["right_position_delta"])
            samples.append((total_dev, {
                "left_slope": signal["left_slope"],
                "right_slope": signal["right_slope"],
                "left_brow_iris_gap": signal["left_brow_iris_gap"],
                "right_brow_iris_gap": signal["right_brow_iris_gap"],
                "left_position": signal["left_position"],
                "right_position": signal["right_position"],
                "left_position_target": signal["left_position_target"],
                "right_position_target": signal["right_position_target"],
                "left_position_min": signal["left_position_min"],
                "left_position_max": signal["left_position_max"],
                "right_position_min": signal["right_position_min"],
                "right_position_max": signal["right_position_max"],
                "left_position_delta": signal["left_position_delta"],
                "right_position_delta": signal["right_position_delta"],
                "left_slope_min": signal["left_slope_min"],
                "left_slope_max": signal["left_slope_max"],
                "right_slope_min": signal["right_slope_min"],
                "right_slope_max": signal["right_slope_max"],
                "big_symmetry": signal["big_symmetry"],
                "big_symmetry_tolerance": signal["big_symmetry_tolerance"],
            }))
            collected += 1

            self.scorer.update_hud([
                (f"Eyebrow Sampling: {collected}/{n_frames}", (10, 120), (200, 200, 200)),
            ])

        self.scorer.enable_mp = False

        if len(samples) < 2 * trim + 1:
            print(f"  [Eyebrow Collect] 有效样本不足: {len(samples)}, 需要 >= {2*trim+1}")
            return None

        samples.sort(key=lambda x: x[0])
        trimmed = samples[trim:-trim]
        n = len(trimmed)
        avg = {}
        for key in [
            "left_slope", "right_slope",
            "left_brow_iris_gap", "right_brow_iris_gap",
            "left_position", "right_position",
            "left_position_target", "right_position_target",
            "left_position_min", "left_position_max",
            "right_position_min", "right_position_max",
            "left_position_delta", "right_position_delta",
            "left_slope_min", "left_slope_max",
            "right_slope_min", "right_slope_max",
            "big_symmetry", "big_symmetry_tolerance",
        ]:
            avg[key] = sum(s[1][key] for s in trimmed) / n

        avg["left_slope_qualified"] = (
            avg["left_slope_min"] <= avg["left_slope"] <= avg["left_slope_max"])
        avg["right_slope_qualified"] = (
            avg["right_slope_min"] <= avg["right_slope"] <= avg["right_slope_max"])
        avg["slope_range_qualified"] = (
            avg["left_slope_qualified"] and avg["right_slope_qualified"])
        avg["symmetry_qualified"] = avg["big_symmetry"] <= avg["big_symmetry_tolerance"]
        avg["left_qualified"] = (
            avg["left_position_delta"] == 0.0 and avg["left_slope_qualified"])
        avg["right_qualified"] = (
            avg["right_position_delta"] == 0.0 and avg["right_slope_qualified"])
        avg["qualified"] = (
            avg["left_qualified"] and avg["right_qualified"] and avg["symmetry_qualified"])

        all_devs = [s[0] for s in samples]
        print(f"  [Eyebrow Collect] {len(samples)} frames, trim {trim}: "
              f"BIG L_delta={avg['left_position_delta']:+.4f} "
              f"R_delta={avg['right_position_delta']:+.4f} "
              f"Slope={'OK' if avg['slope_range_qualified'] else 'BAD'} "
              f"Sym={'OK' if avg['symmetry_qualified'] else 'BAD'} "
              f"(min={all_devs[0]:.4f}, max={all_devs[-1]:.4f})")

        return avg
    # ==================================================================
    # 多帧采集 + 修剪平均 — 嘴部 MAR
    # ==================================================================

    def collect_mouth_samples(self, n_frames: int = 60, trim: int = 10):
        """
        打开 MP（无需 EllSeg）→ 采集 n 帧 MAR → 关闭 → 修剪平均。

        流程:
          1. 开启 MP
          2. 连续采集 n_frames 帧 MAR 信号
          3. 关闭 MP
          4. 按 abs(delta) 排序，修剪上下
          5. 返回平均 MAR 信号

        Returns:
            dict 或 None: {"mar", "delta", "qualified"}
        """
        self.scorer.enable_mp = True
        self.scorer.enable_ellseg = False

        samples = []  # list of (abs_delta, signal_dict)
        collected = 0
        no_face_count = 0

        while collected < n_frames:
            if self.scorer.user_pressed_stop:
                break

            ok, frame = self.scorer.capture()
            if not ok:
                time.sleep(0.001)
                continue

            self.scorer.detect(frame)
            signal = self.scorer.get_mouth_signal()

            if signal is None:
                no_face_count += 1
                if no_face_count > 10:
                    break
                time.sleep(0.001)
                continue
            no_face_count = 0

            samples.append((abs(signal["delta"]), {
                "mar": signal["mar"],
                "delta": signal["delta"],
            }))
            collected += 1

            self.scorer.update_hud([
                (f"Mouth Sampling: {collected}/{n_frames}", (10, 120), (200, 200, 200)),
            ])

        self.scorer.enable_mp = False

        if len(samples) < 2 * trim + 1:
            print(f"  [Mouth Collect] 有效样本不足: {len(samples)}, 需要 >= {2*trim+1}")
            return None

        samples.sort(key=lambda x: x[0])
        trimmed = samples[trim:-trim]

        n = len(trimmed)
        avg = {
            "mar": sum(s[1]["mar"] for s in trimmed) / n,
            "delta": sum(s[1]["delta"] for s in trimmed) / n,
        }
        tol = MOUTH_MAR_TOLERANCE
        avg["qualified"] = abs(avg["delta"]) <= tol

        all_devs = [s[0] for s in samples]
        print(f"  [Mouth Collect] {len(samples)} frames, trim {trim}: "
              f"MAR={avg['mar']:.4f} Δ={avg['delta']:+.4f} "
              f"(min={all_devs[0]:.4f}, max={all_devs[-1]:.4f})")

        return avg

    # ==================================================================
    # 引导式自动调整 — 眉毛 (A0-A1)
    # ==================================================================

    def auto_adjust_eyebrow(
        self,
        ebhr_step: float = 1.0,
        max_iterations: int = None,
        wait_seconds: float = None,
        keep_display: bool = False,
        log_func=None,
    ) -> tuple:
        """
        Auto-adjust A0/A1 eyebrows using a two-phase strategy.

        Phase 1 — 对称调整:
          Physical coupling causes L_BIG to drag R_BIG when only one servo moves.
          Therefore we first make left/right BIG symmetric by moving ONLY ONE side
          toward the symmetry midpoint each iteration.

        Phase 2 — 同步升降:
          Once symmetric, move A0 and A1 together by the same amount to bring
          both BIG values into target range simultaneously.
        """
        if max_iterations is None:
            max_iterations = EYEBROW_MAX_ITERATIONS
        if wait_seconds is None:
            wait_seconds = EYEBROW_WAIT_SECONDS

        bl = self.scorer.eyebrow_baseline
        if not (
            bl
            and "left_brow_iris_gap" in bl
            and "right_brow_iris_gap" in bl
            and "left_slope" in bl
            and "right_slope" in bl
        ):
            print("[WARN] No eyebrow baseline. Save one first in ellseg_scorer.py with [B].")
            return False, 0, {}

        left_range = (
            float(bl["left_brow_iris_gap"]) - EYEBROW_CUSTOM_BIG_TOLERANCE,
            float(bl["left_brow_iris_gap"]) + EYEBROW_CUSTOM_BIG_TOLERANCE,
        )
        right_range = (
            float(bl["right_brow_iris_gap"]) - EYEBROW_CUSTOM_BIG_TOLERANCE,
            float(bl["right_brow_iris_gap"]) + EYEBROW_CUSTOM_BIG_TOLERANCE,
        )

        print(f"{'='*60}")
        print("  Auto eyebrow adjust (A0-A1) — 两阶段策略")
        print("  Phase 1: 对称调整 (单独调一侧)")
        print("  Phase 2: 同步升降 (两边一起)")
        print("  Step: {:.0f} deg | max iter: {} | wait: {}s".format(ebhr_step, max_iterations, wait_seconds))
        print("  Target: BIG in range, BIG symmetry OK, slope range OK")
        print("  custom baseline: "
              f"L_BIG=[{left_range[0]:+.4f},{left_range[1]:+.4f}] "
              f"R_BIG=[{right_range[0]:+.4f},{right_range[1]:+.4f}]")
        print(f"{'='*60}\n")

        self.scorer.start_display()
        current_angles = {ch: self.controller.list_temp_deg[ch] for ch in EYEBROW_CHANNELS}
        print(f"  Initial angles: {dict(zip(EYEBROW_NAMES, [current_angles[c] for c in EYEBROW_CHANNELS]))}")

        if log_func is None:
            log_func = print
        a0_sign = getattr(self, '_eyebrow_a0_flip', -1)
        a1_sign = getattr(self, '_eyebrow_a1_flip', -1)

        iteration = 0
        phase = 1  # start in Phase 1
        sleep_chunk = 0.1

        try:
            while iteration < max_iterations:
                iteration += 1
                if self.scorer.user_pressed_stop:
                    print("\n[USER] Stop requested")
                    break

                self.scorer.update_hud([
                    (f"[Eyebrow P{phase} Iter {iteration}/{max_iterations}]", (10, 100), (200, 200, 200)),
                ])

                avg_signal = self.collect_eyebrow_samples(n_frames=60, trim=10)
                if avg_signal is None:
                    print(f"  [Eyebrow Iter {iteration:3d}] no valid frames")
                    _waited = 0
                    while _waited < wait_seconds:
                        time.sleep(min(sleep_chunk, wait_seconds - _waited))
                        _waited += sleep_chunk
                        if self.scorer.user_pressed_stop:
                            break
                    continue

                l_big = avg_signal["left_brow_iris_gap"]
                r_big = avg_signal["right_brow_iris_gap"]
                l_delta = avg_signal["left_position_delta"]
                r_delta = avg_signal["right_position_delta"]
                symmetry = avg_signal["big_symmetry"]
                slope_ok = avg_signal.get("slope_range_qualified", True)
                sym_ok = symmetry <= EYEBROW_BIG_SYMMETRY_TOLERANCE

                print(f"  [Eyebrow P{phase} Iter {iteration:3d}] "
                      f"L={l_big:.4f} (d={l_delta:+.4f}) "
                      f"R={r_big:.4f} (d={r_delta:+.4f}) "
                      f"Sym={symmetry:.4f} "
                      f"Slope={'OK' if slope_ok else 'BAD'}")

                # 检查是否全部合格
                l_in_range = l_delta == 0.0
                r_in_range = r_delta == 0.0
                if l_in_range and r_in_range and sym_ok and slope_ok:
                    phase_str = "Phase 1" if phase == 1 else "Phase 2"
                    log_func(f"{phase_str} 结束: 全部达标 → 通过 "
                             f"(L={l_big:.3f} R={r_big:.3f})")
                    print(f"\n{'='*60}")
                    print(f"  EYEBROW PASSED in {iteration} iterations ({phase_str})")
                    print(f"      L_BIG={l_big:.4f}  R_BIG={r_big:.4f}")
                    final_ang = " ".join([f"{n}={current_angles[c]}deg"
                                         for n, c in zip(EYEBROW_NAMES, EYEBROW_CHANNELS)])
                    print(f"      Angles: {final_ang}")
                    print(f"{'='*60}\n")

                    all_angles = []
                    for ch in EYE_SERVO_CHANNELS:
                        all_angles.append(self.controller.list_temp_deg[ch])
                    self.save_result(all_angles)
                    self.export_angle_config(
                        all_angles,
                        eyebrow_angles={c: current_angles[c] for c in EYEBROW_CHANNELS},
                    )
                    self.scorer.update_hud([("EYEBROW PASSED", (100, 80), (0, 255, 0))])
                    time.sleep(1)
                    return True, iteration, avg_signal

                # ---- Phase 1: 对称调整 ----
                if phase == 1:
                    if sym_ok:
                        # 对称已达标 → 进入 Phase 2
                        phase = 2
                        log_func("Phase 1 结束: 对称调整完成")
                        log_func("Phase 2 开始: 同步升降")
                        print(f"  [Phase 1 → Phase 2] Symmetry OK ({symmetry:.4f} <= {EYEBROW_BIG_SYMMETRY_TOLERANCE:.3f})")
                        continue

                    # 找出偏差更大的那侧，往对称中点方向调
                    midpoint = (l_big + r_big) / 2.0
                    l_dev = l_big - midpoint
                    r_dev = r_big - midpoint

                    # 非空格的偏差更大 → 调它
                    if abs(l_dev) >= abs(r_dev):
                        side = "L"
                        ch = EYEBROW_CHANNELS[0]
                        name = EYEBROW_NAMES[0]
                        sign = a0_sign
                        deviation = l_dev
                    else:
                        side = "R"
                        ch = EYEBROW_CHANNELS[1]
                        name = EYEBROW_NAMES[1]
                        # A1 is mechanically opposite to A0:
                        # A0 raise = angle +, A1 raise = angle -.
                        sign = -a1_sign
                        deviation = r_dev

                    # deviation > 0 → 这侧更低 → 要抬高
                    # deviation < 0 → 这侧更高 → 要降低
                    delta_angle = sign * (-1 if deviation > 0 else 1)
                    new_angle = round(current_angles[ch] + delta_angle * ebhr_step)
                    lo, hi = self.eyebrow_ranges[ch]
                    new_angle = max(lo, min(hi, new_angle))
                    self.send_servo(ch, new_angle, (lo, hi))
                    current_angles[ch] = new_angle
                    print(f"  [Eyebrow P{phase} Iter {iteration:3d}] {side}: {name}={new_angle}deg "
                          f"(sym dev={deviation:+.4f}, toward midpoint)")

                # ---- Phase 2: 同步升降 ----
                else:
                    # 计算平均位置和平均目标，两边一起移
                    avg_current = (l_big + r_big) / 2.0
                    avg_target = (float(bl["left_brow_iris_gap"]) + float(bl["right_brow_iris_gap"])) / 2.0
                    avg_delta = avg_current - avg_target

                    if abs(avg_delta) <= EYEBROW_CUSTOM_BIG_TOLERANCE:
                        # 平均位置已达目标，slope/sym 问题忽略，记为通过
                        log_func(f"Phase 2 结束: 平均BIG达标 → 通过 "
                                 f"(L={l_big:.3f} R={r_big:.3f}, slope/sym忽略)")
                        print(f"  [Eyebrow P{phase} Iter {iteration:3d}] Avg BIG in range. "
                              f"(L={l_big:.4f} R={r_big:.4f}) → PASS (ignore slope/sym)")
                        print(f"\n{'='*60}")
                        print(f"  EYEBROW PASSED in {iteration} iterations (Phase 2 BEST FIT)")
                        print(f"      L_BIG={l_big:.4f}  R_BIG={r_big:.4f}")
                        final_ang = " ".join([f"{n}={current_angles[c]}deg"
                                             for n, c in zip(EYEBROW_NAMES, EYEBROW_CHANNELS)])
                        print(f"      Angles: {final_ang}")
                        print(f"{'='*60}\n")

                        all_angles = []
                        for ch in EYE_SERVO_CHANNELS:
                            all_angles.append(self.controller.list_temp_deg[ch])
                        self.save_result(all_angles)
                        self.export_angle_config(
                            all_angles,
                            eyebrow_angles={c: current_angles[c] for c in EYEBROW_CHANNELS},
                        )
                        self.scorer.update_hud([("EYEBROW PASSED", (100, 80), (0, 255, 0))])
                        time.sleep(1)
                        return True, iteration, avg_signal

                    if avg_delta > 0:
                        # 偏低 → 两边同时抬高 (delta负)
                        direction = -1
                    elif avg_delta < 0:
                        # 偏高 → 两边同时降低 (delta正)
                        direction = 1
                    else:
                        direction = 0

                    # Phase 2: A0 和 A1 机械方向相反 → 符号翻转
                    for side, ch, name, base_sign in [
                        ("L", EYEBROW_CHANNELS[0], EYEBROW_NAMES[0], a0_sign),
                        ("R", EYEBROW_CHANNELS[1], EYEBROW_NAMES[1], -a1_sign),
                    ]:
                        delta_angle = base_sign * direction
                        new_angle = round(current_angles[ch] + delta_angle * ebhr_step)
                        lo, hi = self.eyebrow_ranges[ch]
                        new_angle = max(lo, min(hi, new_angle))
                        self.send_servo(ch, new_angle, (lo, hi))
                        current_angles[ch] = new_angle

                    if direction != 0:
                        print(f"  [Eyebrow P{phase} Iter {iteration:3d}] Sync: L={current_angles[EYEBROW_CHANNELS[0]]}deg "
                              f"R={current_angles[EYEBROW_CHANNELS[1]]}deg "
                              f"(avg_delta={avg_delta:+.4f}, dir={direction:+.0f})")

                _waited = 0
                while _waited < wait_seconds:
                    time.sleep(min(sleep_chunk, wait_seconds - _waited))
                    _waited += sleep_chunk
                    self.scorer.update_hud([
                        (f"[Eyebrow P{phase} Iter {iteration}/{max_iterations}]", (10, 100), (200, 200, 200)),
                        (f"Waiting... {_waited:.1f}/{wait_seconds}s", (10, 118), (0, 255, 255)),
                    ])
                    if self.scorer.user_pressed_stop:
                        break

            print(f"\n{'='*60}")
            print(f"  Eyebrow failed after {max_iterations} iterations")
            print(f"{'='*60}\n")
            return False, iteration, {}

        finally:
            if not keep_display:
                self.scorer.stop_display()
    def auto_adjust_mouth_chin(
        self,
        mar_step: float = 1.0,
        max_iterations: int = None,
        wait_seconds: float = None,
        keep_display: bool = False,
    ) -> tuple:
        """
        引导式自动调整 — 只调整 A26 下巴舵机。

        流程（每轮迭代）:
          1. 打开 MP → 采集 60 帧 MAR → 关闭 MP
          2. 若 MAR 偏差 ≤ 容差 → 通过
          3. 否则移动 A26 → 等待 → 下一轮

        Args:
            mar_step:       每次调整的舵机角度步长 (默认 1°)
            max_iterations: 最大迭代次数
            wait_seconds:   每轮等待秒数
            keep_display:   完成后是否保持窗口

        Returns:
            (passed: bool, iterations: int, final_data: dict)
        """
        if max_iterations is None:
            max_iterations = MOUTH_MAX_ITERATIONS
        if wait_seconds is None:
            wait_seconds = MOUTH_WAIT_SECONDS

        if self.scorer.mouth_baseline is None:
            print("[WARN] 未加载嘴部基线! 请在 ellseg_scorer.py 中按 [M] 保存到 face_point.json 当前性别")
            return False, 0, {}

        tol = MOUTH_MAR_TOLERANCE

        print(f"{'='*60}")
        print(f"  引导式自动调整 (仅下巴 A26)")
        print(f"  采集: 60帧, 修剪上下10帧, 中间40帧取平均 MAR")
        print(f"  步长: {mar_step}度/次 | 最大迭代: {max_iterations} | 等待: {wait_seconds}s")
        print(f"  目标: MAR 偏差 ≤ {tol}")
        bl = self.scorer.mouth_baseline
        print(f"  基线: MAR={bl['mar']:.4f}")
        print(f"{'='*60}\n")

        self.scorer.start_display()

        # 获取 A26 的范围
        ch = MOUTH_CHIN_CHANNEL
        start = self.controller.list_start_deg[ch]
        end = self.controller.list_end_deg[ch]
        chin_range = (min(start, end), max(start, end))
        chin_sign = getattr(self, '_mouth_chin_flip', 1)  # 正方向: MAR大→合(角度减小)

        current_angle = self.controller.list_temp_deg[ch]
        print(f"  初始角度: A26={current_angle}° (range=[{chin_range[0]}, {chin_range[1]}])")

        iteration = 0
        sleep_chunk = 0.1

        try:
            while iteration < max_iterations:
                iteration += 1

                if self.scorer.user_pressed_stop:
                    print("\n[USER] 收到停止信号")
                    break

                self.scorer.update_hud([
                    (f"[Mouth Iter {iteration}/{max_iterations}]", (10, 100), (200, 200, 200)),
                ])

                # ---- 采集 MAR ----
                avg_signal = self.collect_mouth_samples(n_frames=60, trim=10)

                if avg_signal is None:
                    print(f"  [Mouth Iter {iteration:3d}] 无法获取有效帧")
                    _waited = 0
                    while _waited < wait_seconds:
                        time.sleep(min(sleep_chunk, wait_seconds - _waited))
                        _waited += sleep_chunk
                        if self.scorer.user_pressed_stop:
                            break
                    continue

                mar = avg_signal["mar"]
                delta = avg_signal["delta"]

                print(f"  [Mouth Iter {iteration:3d}] MAR={mar:.4f} (Δ={delta:+.4f})")

                # ---- 检查是否已合格 ----
                if abs(delta) <= tol:
                    print(f"\n{'='*60}")
                    print(f"  ★★ 下巴调整通过! 共迭代 {iteration} 次")
                    print(f"      MAR={mar:.4f} (基线={bl['mar']:.4f})")
                    print(f"      A26={current_angle}°")
                    print(f"{'='*60}\n")

                    # 保存结果
                    all_angles = [self.controller.list_temp_deg[c] for c in EYE_SERVO_CHANNELS]
                    self.save_result(all_angles)
                    self.export_angle_config(all_angles, chin_angle=current_angle)

                    self.scorer.update_hud([("★★ MOUTH PASSED ★★", (100, 80), (0, 255, 0))])
                    time.sleep(1)
                    return True, iteration, avg_signal

                # ---- 移动 A26 ----
                # delta>0 → MAR太大(嘴太开) → 合(-1); delta<0 → MAR太小(嘴太闭) → 开(+1)
                delta_sign = -1 if delta > tol else (1 if delta < -tol else 0)
                move = chin_sign * delta_sign * mar_step
                new_angle = round(current_angle + move)
                new_angle = max(chin_range[0], min(chin_range[1], new_angle))
                self.send_servo(ch, new_angle, chin_range)
                current_angle = new_angle
                print(f"  [Mouth Iter {iteration:3d}] 移动 A26={new_angle}° "
                      f"(MAR Δ={delta:+.4f}, Δangle={move:+.1f})")

                # ---- 等待 ----
                _waited = 0
                while _waited < wait_seconds:
                    time.sleep(min(sleep_chunk, wait_seconds - _waited))
                    _waited += sleep_chunk
                    self.scorer.update_hud([
                        (f"[Mouth Iter {iteration}/{max_iterations}]", (10, 100), (200, 200, 200)),
                        (f"Waiting... {_waited:.1f}/{wait_seconds}s", (10, 118), (0, 255, 255)),
                    ])
                    if self.scorer.user_pressed_stop:
                        break

            # 达到最大迭代未通过
            print(f"\n{'='*60}")
            print(f"  ✘ 下巴未在 {max_iterations} 次迭代内通过")
            print(f"{'='*60}\n")
            return False, iteration, {}

        finally:
            if not keep_display:
                self.scorer.stop_display()
    def collect_lower_lip_samples(self, n_frames: int = 60, trim: int = 10):
        """
        打开 MP（无需 EllSeg）→ 采集 n 帧 LLR → 关闭 → 修剪平均。

        Returns:
            dict 或 None: {"llr", "delta", "qualified"}
        """
        self.scorer.enable_mp = True
        self.scorer.enable_ellseg = False

        samples = []  # list of (abs_delta, signal_dict)
        collected = 0
        no_face_count = 0

        while collected < n_frames:
            if self.scorer.user_pressed_stop:
                break

            ok, frame = self.scorer.capture()
            if not ok:
                time.sleep(0.001)
                continue

            self.scorer.detect(frame)
            signal = self.scorer.get_lower_lip_signal()

            if signal is None:
                no_face_count += 1
                if no_face_count > 10:
                    break
                time.sleep(0.001)
                continue
            no_face_count = 0

            samples.append((abs(signal["delta"]), {
                "llr": signal["llr"],
                "delta": signal["delta"],
            }))
            collected += 1

            self.scorer.update_hud([
                (f"LowerLip Sampling: {collected}/{n_frames}", (10, 120), (200, 200, 200)),
            ])

        self.scorer.enable_mp = False

        if len(samples) < 2 * trim + 1:
            print(f"  [LowerLip Collect] 有效样本不足: {len(samples)}, 需要 >= {2*trim+1}")
            return None

        samples.sort(key=lambda x: x[0])
        trimmed = samples[trim:-trim]

        n = len(trimmed)
        avg = {
            "llr": sum(s[1]["llr"] for s in trimmed) / n,
            "delta": sum(s[1]["delta"] for s in trimmed) / n,
        }
        tol = LOWER_LIP_LLR_TOLERANCE
        avg["qualified"] = abs(avg["delta"]) <= tol

        all_devs = [s[0] for s in samples]
        print(f"  [LowerLip Collect] {len(samples)} frames, trim {trim}: "
              f"LLR={avg['llr']:.4f} Δ={avg['delta']:+.4f} "
              f"(min={all_devs[0]:.4f}, max={all_devs[-1]:.4f})")

        return avg

    # ==================================================================
    # 引导式自动调整 — 下唇 (A19)
    # ==================================================================

    def auto_adjust_lower_lip_front_side(
        self,
        step: float = 1.0,
        max_iterations: int = None,
        wait_seconds: float = None,
        keep_display: bool = False,
    ) -> tuple:
        """Adjust lower lip with front LLR and side-view front-point checks."""
        if max_iterations is None:
            max_iterations = LOWER_LIP_MAX_ITERATIONS
        if wait_seconds is None:
            wait_seconds = LOWER_LIP_WAIT_SECONDS

        bl = self.scorer.lower_lip_baseline
        if bl is None or "llr" not in bl:
            print("[WARN] Missing lower lip front LLR baseline.")
            return False, 0, {}
        if "side_lower_tip" not in bl:
            print("[WARN] Missing lower lip side baseline. Save lower lip front+side first.")
            return False, 0, {}

        ch_front = LOWER_LIP_CHANNEL
        ch_side = LOWER_LIP_SIDE_CHANNEL
        front_range = self._angle_range(ch_front)
        side_range = self._angle_range(ch_side)
        front_angle = self._temp_deg(ch_front, default=round(sum(front_range) / 2))
        side_angle = self._temp_deg(ch_side, default=round(sum(side_range) / 2))
        front_sign = getattr(self, "_lower_lip_flip", -1)
        side_sign = getattr(self, "_lower_lip_side_flip", 1)

        print("=" * 60)
        print("  Lower lip auto adjust: front LLR + side front point")
        print(f"  A19 range={front_range}, A18 range={side_range}, step={step}")
        print(
            f"  Target: LLR tol={LOWER_LIP_LLR_TOLERANCE}, "
            f"side x tol={LOWER_LIP_SIDE_X_TOLERANCE}px, "
            f"side y tol={LOWER_LIP_SIDE_Y_TOLERANCE}px"
        )
        print(f"  Baseline: LLR={bl['llr']:.4f}, side_lower_tip={bl['side_lower_tip']}")
        print("=" * 60)

        self.scorer.start_display()
        iteration = 0
        sleep_chunk = 0.1
        final_data = {}

        try:
            while iteration < max_iterations:
                iteration += 1
                if self.scorer.user_pressed_stop:
                    break

                self.scorer.update_hud([
                    (f"[LowerLip FS Iter {iteration}/{max_iterations}]", (10, 100), (200, 200, 200)),
                    (f"A19={front_angle} A18={side_angle}", (10, 118), (0, 255, 255)),
                ])

                front = self.collect_lower_lip_samples(n_frames=30, trim=5)
                side = self.collect_lower_lip_side_samples(n_frames=30, trim=5)
                final_data = {"front": front, "side": side}
                if front is None or side is None:
                    print(f"  [LowerLip FS Iter {iteration:3d}] missing sample")
                    time.sleep(wait_seconds)
                    continue

                front_delta = front["delta"]
                side_dx = side.get("dx", 0.0)
                side_dy = side.get("dy", 0.0)
                front_ok = abs(front_delta) <= LOWER_LIP_LLR_TOLERANCE
                side_ok = side.get("qualified", False)
                side_y_out = abs(side_dy) > LOWER_LIP_SIDE_Y_TOLERANCE

                print(
                    f"  [LowerLip FS Iter {iteration:3d}] "
                    f"front LLR={front['llr']:.4f} d={front_delta:+.4f} ok={front_ok} | "
                    f"side tip=({side['tip'][0]:.1f},{side['tip'][1]:.1f}) "
                    f"dx={side_dx:+.1f} dy={side_dy:+.1f} ok={side_ok}"
                )

                if front_ok and side_ok:
                    self.scorer.update_hud([
                        ("LOWER LIP FRONT+SIDE PASSED", (100, 80), (0, 255, 0)),
                    ])
                    all_angles = [self.controller.list_temp_deg[c] for c in EYE_SERVO_CHANNELS]
                    self.export_angle_config(
                        all_angles,
                        chin_angle=self._temp_deg(MOUTH_CHIN_CHANNEL),
                        lower_lip_angle=front_angle,
                        upper_lip_angle=self._temp_deg(UPPER_LIP_CHANNEL),
                    )
                    return True, iteration, final_data

                moved = False
                if not front_ok:
                    front_dir = -1 if front_delta > LOWER_LIP_LLR_TOLERANCE else 1
                    new_front = round(front_angle + front_sign * front_dir * step)
                    new_front = max(front_range[0], min(front_range[1], new_front))
                    if new_front != front_angle:
                        front_angle = new_front
                        moved = True
                elif side_y_out:
                    # 正面达标但侧面 y 超差时允许 A19 小步补偿；若下一轮 LLR 出界，会优先按正面纠正。
                    front_dir = 1 if side_dy > LOWER_LIP_SIDE_Y_TOLERANCE else -1
                    new_front = round(front_angle + front_dir * step)
                    new_front = max(front_range[0], min(front_range[1], new_front))
                    if new_front != front_angle:
                        front_angle = new_front
                        moved = True
                        final_data["side_dy_front_adjust"] = {
                            "side_dy": side_dy,
                            "front_delta": front_delta,
                        }

                if not side_ok and abs(side_dx) > LOWER_LIP_SIDE_X_TOLERANCE:
                    # A18: 前=240, 后=180. side_dx>0 表示下唇前点太靠前，需减小角度往后。
                    side_dir = -1 if side_dx > LOWER_LIP_SIDE_X_TOLERANCE else 1
                    new_side = round(side_angle + side_sign * side_dir * step)
                    new_side = max(side_range[0], min(side_range[1], new_side))
                    if new_side != side_angle:
                        side_angle = new_side
                        moved = True

                if moved:
                    self.send_servo(ch_front, front_angle, front_range)
                    self.send_servo(ch_side, side_angle, side_range)
                    print(f"  [LowerLip FS Iter {iteration:3d}] move A19={front_angle}, A18={side_angle}")
                else:
                    warning = (
                        f"[LowerLip FS Iter {iteration:3d}] no legal move available "
                        f"(front_ok={front_ok}, side_ok={side_ok}, "
                        f"front_delta={front_delta:+.4f}, side_dx={side_dx:+.1f}, side_dy={side_dy:+.1f})"
                    )
                    print(f"  {warning}")
                    self.scorer.update_hud([
                        ("LOWER LIP STOP: NO LEGAL MOVE", (10, 136), (0, 0, 255)),
                    ])
                    final_data["warning"] = warning
                    final_data["stop_reason"] = "no_legal_move"
                    return False, iteration, final_data

                waited = 0.0
                while waited < wait_seconds:
                    time.sleep(min(sleep_chunk, wait_seconds - waited))
                    waited += sleep_chunk
                    if self.scorer.user_pressed_stop:
                        break

            return False, iteration, final_data
        finally:
            if not keep_display:
                self.scorer.stop_display()
    def collect_mouth_corner_samples(self, n_frames: int = 60, trim: int = 10):
        """
        打开 MP（无需 EllSeg）→ 采集 n 帧嘴角像素偏移 → 关闭 → 修剪平均。

        Returns:
            dict 或 None: {"left": {dx,dy,dist,qualified,adj_x,adj_y},
                           "right": {dx,dy,dist,qualified,adj_x,adj_y},
                           "all_qualified": bool}
        """
        self.scorer.enable_mp = True
        self.scorer.enable_ellseg = False

        samples = []  # list of (total_dist, signal_dict)
        collected = 0
        no_face_count = 0

        while collected < n_frames:
            if self.scorer.user_pressed_stop:
                break

            ok, frame = self.scorer.capture()
            if not ok:
                time.sleep(0.001)
                continue

            self.scorer.detect(frame)
            signal = self.scorer.get_mouth_corner_signal()

            if signal is None:
                no_face_count += 1
                if no_face_count > 10:
                    break
                time.sleep(0.001)
                continue
            no_face_count = 0

            total_dist = signal["left"]["dist"] + signal["right"]["dist"]
            samples.append((total_dist, {
                "left":  dict(signal["left"]),
                "right": dict(signal["right"]),
            }))
            collected += 1

            self.scorer.update_hud([
                (f"MouthCorner Sampling: {collected}/{n_frames}", (10, 120), (200, 200, 200)),
            ])

        self.scorer.enable_mp = False

        if len(samples) < 2 * trim + 1:
            print(f"  [MouthCorner Collect] 有效样本不足: {len(samples)}, 需要 >= {2*trim+1}")
            return None

        samples.sort(key=lambda x: x[0])
        trimmed = samples[trim:-trim]

        n = len(trimmed)
        avg = {"left": {}, "right": {}}
        for side in ("left", "right"):
            for key in ("dx", "dy", "dist"):
                avg[side][key] = sum(s[1][side][key] for s in trimmed) / n
            avg[side]["adj_x"] = -avg[side]["dx"]
            avg[side]["adj_y"] = -avg[side]["dy"]
            avg[side]["qualified"] = avg[side]["dist"] <= MOUTH_CORNER_TOLERANCE

        avg["all_qualified"] = avg["left"]["qualified"] and avg["right"]["qualified"]
        avg["total_dist"] = avg["left"]["dist"] + avg["right"]["dist"]

        all_dists = [s[0] for s in samples]
        print(f"  [MouthCorner Collect] {len(samples)} frames, trim {trim}: "
              f"avg_L=({avg['left']['dx']:+.1f},{avg['left']['dy']:+.1f}) "
              f"avg_R=({avg['right']['dx']:+.1f},{avg['right']['dy']:+.1f}) "
              f"(min={all_dists[0]:.1f}, max={all_dists[-1]:.1f})")

        return avg

    # ==================================================================
    # 引导式自动调整 — 嘴角 (A22-A25)  竖向优先 → 水平随后
    # ==================================================================

    def auto_adjust_mouth_corners(
        self,
        step: float = 1.0,
        max_iterations: int = None,
        wait_seconds: float = None,
        keep_display: bool = False,
    ) -> tuple:
        """
        两阶段串行调整 A22-A25 嘴角舵机：
          Phase 1: 竖向 A23(左)+A25(右)  dy ≤ 0.1px
          Phase 2: 水平 A24(左)+A22(右)  dx ≤ 1.0px

        Args:
            step:           每次调整的舵机角度步长 (默认 1°)
            max_iterations: 最大迭代次数
            wait_seconds:   每轮等待秒数
            keep_display:   完成后是否保持窗口

        Returns:
            (passed: bool, iterations: int, final_data: dict)
        """
        if max_iterations is None:
            max_iterations = MOUTH_CORNER_MAX_ITERATIONS
        if wait_seconds is None:
            wait_seconds = MOUTH_CORNER_WAIT_SECONDS

        if self.scorer.mouth_corners_baseline is None:
            print("[WARN] 未加载嘴角基线! 请在 ellseg_scorer.py 中按 [C] 保存到 face_point.json 当前性别")
            return False, 0, {}

        tol_h = MOUTH_CORNER_TOLERANCE       # 1.0 px — 水平
        tol_v = MOUTH_CORNER_Y_TOLERANCE     # 0.1 px — 垂直

        # 通道: A24=左水平, A22=右水平, A23=左垂直, A25=右垂直
        ch_Lh = MOUTH_CORNER_H_CHANNELS[0]   # A24
        ch_Rh = MOUTH_CORNER_H_CHANNELS[1]   # A22
        ch_Lv = MOUTH_CORNER_V_CHANNELS[0]   # A23
        ch_Rv = MOUTH_CORNER_V_CHANNELS[1]   # A25

        ranges = {
            ch: (min(self.controller.list_start_deg[ch], self.controller.list_end_deg[ch]),
                 max(self.controller.list_start_deg[ch], self.controller.list_end_deg[ch]))
            for ch in [ch_Lh, ch_Rh, ch_Lv, ch_Rv]
        }

        h_flip = -1 if getattr(self, '_corner_horiz_flip', False) else 1
        v_flip = -1 if getattr(self, '_corner_vert_flip', False) else 1

        current = {ch: self._temp_deg(ch) for ch in [ch_Lh, ch_Rh, ch_Lv, ch_Rv]}

        print(f"{'='*60}")
        print(f"  嘴角两阶段调整 (A22-A25)")
        print(f"  Phase 1: 竖向 Y (A23+A25) ≤ {tol_v}px")
        print(f"  Phase 2: 水平 X (A24+A22) ≤ {tol_h}px")
        print(f"  步长: {step}°/次 | 最大迭代: {max_iterations} | 等待: {wait_seconds}s")
        bl = self.scorer.mouth_corners_baseline
        print(f"  基线: L=({bl['corner_left'][0]},{bl['corner_left'][1]}) "
              f"R=({bl['corner_right'][0]},{bl['corner_right'][1]})")
        print(f"  方向翻转: horiz={h_flip==-1} vert={v_flip==-1}")
        print(f"  初始: A22={current[ch_Rh]}° A23={current[ch_Lv]}° "
              f"A24={current[ch_Lh]}° A25={current[ch_Rv]}°")
        print(f"{'='*60}\n")

        self.scorer.start_display()

        iteration = 0
        sleep_chunk = 0.1
        vert_done = False

        def _wait():
            _w = 0
            while _w < wait_seconds:
                time.sleep(min(sleep_chunk, wait_seconds - _w))
                _w += sleep_chunk
                if self.scorer.user_pressed_stop:
                    break

        try:
            while iteration < max_iterations:
                iteration += 1
                if self.scorer.user_pressed_stop:
                    print("\n[USER] 收到停止信号")
                    break

                phase = "VERT" if not vert_done else "HORIZ"
                self.scorer.update_hud([
                    (f"[Corner-{phase} {iteration}/{max_iterations}]", (10, 100), (200, 200, 200)),
                ])

                avg = self.collect_mouth_corner_samples(n_frames=60, trim=10)
                if avg is None:
                    print(f"  [Corner-{phase} {iteration:3d}] 无效帧")
                    _wait()
                    continue

                L_dx, L_dy = avg["left"]["dx"], avg["left"]["dy"]
                R_dx, R_dy = avg["right"]["dx"], avg["right"]["dy"]

                L_dy_ok = abs(L_dy) <= tol_v
                R_dy_ok = abs(R_dy) <= tol_v
                L_dx_ok = abs(L_dx) <= tol_h
                R_dx_ok = abs(R_dx) <= tol_h
                vert_ok = L_dy_ok and R_dy_ok
                horiz_ok = L_dx_ok and R_dx_ok

                mark = lambda ok: "✓" if ok else "✗"
                print(f"  [Corner-{phase} {iteration:3d}] "
                      f"L=(dX={L_dx:+.1f}{mark(L_dx_ok)} dY={L_dy:+.1f}{mark(L_dy_ok)}) "
                      f"R=(dX={R_dx:+.1f}{mark(R_dx_ok)} dY={R_dy:+.1f}{mark(R_dy_ok)})")

                # ---- 全部通过 ----
                if vert_ok and horiz_ok:
                    print(f"\n{'='*60}")
                    print(f"  ★★ 嘴角通过! {iteration} 次")
                    print(f"      A22={current[ch_Rh]}° A23={current[ch_Lv]}° "
                          f"A24={current[ch_Lh]}° A25={current[ch_Rv]}°")
                    print(f"{'='*60}\n")
                    all_angles = [self._temp_deg(c) for c in EYE_SERVO_CHANNELS]
                    self.save_result(all_angles)
                    self.export_angle_config(
                        all_angles,
                        chin_angle=self._temp_deg(MOUTH_CHIN_CHANNEL),
                        lower_lip_angle=self._temp_deg(LOWER_LIP_CHANNEL),
                        upper_lip_angle=self._temp_deg(UPPER_LIP_CHANNEL),
                        corner_angles={
                            "A22": {"channel": ch_Rh, "angle": current[ch_Rh]},
                            "A23": {"channel": ch_Lv, "angle": current[ch_Lv]},
                            "A24": {"channel": ch_Lh, "angle": current[ch_Lh]},
                            "A25": {"channel": ch_Rv, "angle": current[ch_Rv]},
                        },
                    )
                    self.scorer.update_hud([("★★ MOUTH CORNERS PASSED ★★", (100, 80), (0, 255, 0))])
                    time.sleep(1)
                    return True, iteration, avg

                moved = []
                if not vert_done:
                    # ---- Phase 1: 竖向 ----
                    if vert_ok:
                        vert_done = True
                        print(f"  [Corner-VERT {iteration:3d}] ★ 竖向收敛，进入水平阶段")
                        continue
                    if not L_dy_ok:
                        s = v_flip if L_dy > 0 else -v_flip
                        self._move_one(ch_Lv, s * step, ranges, current, moved, "A23")
                    if not R_dy_ok:
                        s = -v_flip if R_dy > 0 else v_flip
                        self._move_one(ch_Rv, s * step, ranges, current, moved, "A25")
                else:
                    # ---- Phase 2: 水平 ----
                    if not L_dx_ok:
                        s = h_flip if L_dx > 0 else -h_flip
                        self._move_one(ch_Lh, s * step, ranges, current, moved, "A24")
                    if not R_dx_ok:
                        s = h_flip if R_dx > 0 else -h_flip
                        self._move_one(ch_Rh, s * step, ranges, current, moved, "A22")

                if moved:
                    print(f"  [Corner-{phase} {iteration:3d}] -> " + " ".join(moved))
                else:
                    print(f"  [Corner-{phase} {iteration:3d}] 已达极限")
                _wait()

            print(f"\n{'='*60}")
            print(f"  ✘ 嘴角未通过 ({'竖向' if not vert_done else '水平'})")
            print(f"{'='*60}\n")
            return False, iteration, {}

        finally:
            if not keep_display:
                self.scorer.stop_display()

    def _move_one(self, ch: int, delta: float, ranges: dict, current: dict,
                  moved: list, label: str):
        """单通道移动，自动 clamp。返回新角度；已到极限则不动。"""
        lo, hi = ranges[ch]
        new_angle = round(current[ch] + delta)
        new_angle = max(lo, min(hi, new_angle))
        if new_angle == current[ch]:
            return current[ch]
        self.send_servo(ch, new_angle, (lo, hi))
        current[ch] = new_angle
        moved.append(f"{label}={new_angle}°")
        return new_angle

    # ==================================================================
    # 多帧采集 + 修剪平均 — 上唇 ULR
    # ==================================================================

    def collect_upper_lip_samples(self, n_frames: int = 60, trim: int = 10):
        """
        打开 MP（无需 EllSeg）→ 采集 n 帧 ULR → 关闭 → 修剪平均。

        Returns:
            dict 或 None: {"ulr", "delta", "qualified"}
        """
        self.scorer.enable_mp = True
        self.scorer.enable_ellseg = False

        samples = []  # list of (abs_delta, signal_dict)
        collected = 0
        no_face_count = 0

        while collected < n_frames:
            if self.scorer.user_pressed_stop:
                break

            ok, frame = self.scorer.capture()
            if not ok:
                time.sleep(0.001)
                continue

            self.scorer.detect(frame)
            signal = self.scorer.get_upper_lip_signal()

            if signal is None:
                no_face_count += 1
                if no_face_count > 10:
                    break
                time.sleep(0.001)
                continue
            no_face_count = 0

            samples.append((abs(signal["delta"]), {
                "ulr": signal["ulr"],
                "delta": signal["delta"],
            }))
            collected += 1

            self.scorer.update_hud([
                (f"UpperLip Sampling: {collected}/{n_frames}", (10, 120), (200, 200, 200)),
            ])

        self.scorer.enable_mp = False

        if len(samples) < 2 * trim + 1:
            print(f"  [UpperLip Collect] 有效样本不足: {len(samples)}, 需要 >= {2*trim+1}")
            return None

        samples.sort(key=lambda x: x[0])
        trimmed = samples[trim:-trim]

        n = len(trimmed)
        avg = {
            "ulr": sum(s[1]["ulr"] for s in trimmed) / n,
            "delta": sum(s[1]["delta"] for s in trimmed) / n,
        }
        tol = UPPER_LIP_ULR_TOLERANCE
        avg["qualified"] = abs(avg["delta"]) <= tol

        all_devs = [s[0] for s in samples]
        print(f"  [UpperLip Collect] {len(samples)} frames, trim {trim}: "
              f"ULR={avg['ulr']:.4f} Δ={avg['delta']:+.4f} "
              f"(min={all_devs[0]:.4f}, max={all_devs[-1]:.4f})")

        return avg

    def save_result(self, angles: List[int], region_scores: dict = None,
                    save_path: str = None):
        """兼容旧调用：不再保存眼睛结果 JSON。"""
        return None

    def export_angle_config(self, angles: List[int], eyebrow_angles: dict = None,
                            chin_angle: int = None, lower_lip_angle: int = None,
                            upper_lip_angle: int = None, corner_angles: dict = None):
        """兼容旧调用：不再保存角度 JSON。"""
        return None

    def export_full_best_config(self):
        """全量 27 个舵机最优配置：从 YAML 读原始 temp_deg，与当前值对比后保存。

        保存为 best_servo_config.json，每通道含:
          name, channel, info, range, original_deg, tuned_deg, delta
        """
        yaml_path = self.controller.yaml_file
        original = read_yaml(yaml_path)
        servo_info = original.get("SERVO_INFO", original)

        servos = []
        for name in servo_info:
            si = servo_info[name]
            ch = si["channel_idx"]
            orig = si["temp_deg"]
            tuned = self.controller.list_temp_deg[ch]
            servos.append({
                "name": name,
                "channel": ch,
                "info": si.get("info", name),
                "range": [si["start_deg"], si["end_deg"]],
                "original_deg": orig,
                "tuned_deg": tuned,
                "delta": tuned - orig,
            })

        out = {
            "description": "全量 27 通道舵机调优最佳配置",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "yaml_source": os.path.basename(yaml_path),
            "servos": servos,
        }

        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "best_servo_config.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"[INFO] 全量最优配置已导出 → {path}")

        # 打印 delta 摘要
        changed = [s for s in servos if s["delta"] != 0]
        if changed:
            print("  变化列表:")
            for s in changed:
                arrow = "↑" if s["delta"] > 0 else "↓"
                print(f"    {s['name']}: {s['original_deg']}° → {s['tuned_deg']}° ({arrow}{abs(s['delta'])})")
        else:
            print("  (无变化)")

    def export_adjusted_yaml(self, output_path: str = None) -> str:
        """按原 YAML 格式导出当前舵机角度，仅覆盖 temp_deg。"""
        yaml_path = self.controller.yaml_file
        config = read_yaml(yaml_path)
        servo_info = config.get("SERVO_INFO", config)

        for servo_name, info in servo_info.items():
            channel_idx = info.get("channel_idx")
            if channel_idx is None or not (0 <= channel_idx < len(self.controller.list_temp_deg)):
                continue
            current_angle = int(round(self.controller.list_temp_deg[channel_idx]))
            low = min(int(info["start_deg"]), int(info["end_deg"]))
            high = max(int(info["start_deg"]), int(info["end_deg"]))
            info["temp_deg"] = max(low, min(high, current_angle))

        if output_path is None:
            base, ext = os.path.splitext(yaml_path)
            output_path = f"{base}_adjusted{ext or '.yaml'}"
        write_yaml(config, output_path)
        print(f"[INFO] 调整后 YAML 已导出 → {output_path}")
        return output_path

    # ==================================================================
    # 清理
    # ==================================================================

    def cleanup(self):
        """清理资源"""
        if self.scorer and self._owns_scorer:
            try:
                self.scorer.close()  # 内部会 stop_display
            except Exception:
                pass
        if self.side_cap and self._owns_side_cap:
            try:
                self.side_cap.release()
            except Exception:
                pass
            self.side_cap = None
        if self.controller:
            try:
                self.controller.close()
            except Exception:
                pass
        print("[INFO] 资源已释放")
