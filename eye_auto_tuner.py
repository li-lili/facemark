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

运行方式:
  python eye_auto_tuner.py                  # 默认: 眼球 A10-A13
  python eye_auto_tuner.py --mode eyelid    # 眼皮 A8-A9
"""

import os
import time
import json
from typing import List, Tuple

from Motor import FaceController
from Communication import UARTDevice
from utility import read_yaml
from eye_constants import (
    EYE_SERVO_CHANNELS, EYE_SERVO_NAMES,
    EYEBALL_CHANNELS, EYELID_CHANNELS,     EYEBROW_CHANNELS, EYEBROW_NAMES,
    EYEBROW_EBHR_TOLERANCE, EYEBROW_WAIT_SECONDS, EYEBROW_MAX_ITERATIONS,
    DEFAULT_ANGLE_MIN, DEFAULT_ANGLE_MAX,
    CAMERA_INDEX, FLIP_HORIZONTAL,
    EYELID_EAR_TOLERANCE, EYELID_WAIT_SECONDS, EYELID_MAX_ITERATIONS,
    MOUTH_MAR_TOLERANCE, MOUTH_WAIT_SECONDS, MOUTH_MAX_ITERATIONS,
    MOUTH_CHIN_CHANNEL,
    LOWER_LIP_LLR_TOLERANCE,
    LOWER_LIP_WAIT_SECONDS, LOWER_LIP_MAX_ITERATIONS,
    LOWER_LIP_CHANNEL,
    UPPER_LIP_ULR_TOLERANCE,
    UPPER_LIP_WAIT_SECONDS, UPPER_LIP_MAX_ITERATIONS,
    UPPER_LIP_CHANNEL,
    MOUTH_CORNER_TOLERANCE,
    MOUTH_CORNER_Y_TOLERANCE,
    MOUTH_CORNER_WAIT_SECONDS, MOUTH_CORNER_MAX_ITERATIONS,
    MOUTH_CORNER_H_CHANNELS, MOUTH_CORNER_V_CHANNELS,
    TuningResult,
)
from ellseg_scorer import EllSegDetector, TOLERANCE

# 眼球舵机名称 (A10-A13)
EYEBALL_NAMES = [f"A{ch}" for ch in EYEBALL_CHANNELS]
# 眼皮舵机名称 (A8-A9)
EYELID_NAMES = [f"A{ch}" for ch in EYELID_CHANNELS]


class EyeAutoTuner:
    """眼睛舵机调整工具"""

    def __init__(
        self,
        yaml_file: str = "29_servo_config(13).yaml",
        port: str = "COM5",
        baudrate: int = 115200,
        servo_num: int = 27,
        stabilize_frames: int = 8,
        settle_time_ms: int = 300,
    ):
        self.yaml_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), yaml_file)
        self.port = port
        self.baudrate = baudrate
        self.servo_num = servo_num
        self.stabilize_frames = stabilize_frames
        self.settle_time_ms = settle_time_ms

        # 组件引用（延迟初始化）
        self.controller = None
        self.scorer = None
        self.angle_ranges = [(DEFAULT_ANGLE_MIN, DEFAULT_ANGLE_MAX)] * len(EYE_SERVO_CHANNELS)

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

        # 2. 初始化 EllSeg 虹膜检测器 (必须与标定基线时分辨率一致: 1920x1080)
        print("\n[2/2] 初始化 EllSeg 虹膜检测系统...")
        self.scorer = EllSegDetector(
            camera_index=CAMERA_INDEX,
            width=1920,
            height=1080,
            flip_horizontal=FLIP_HORIZONTAL,
            stabilize_frames=self.stabilize_frames,
        )

        print("\n" + "=" * 60)
        print("  初始化完成！")
        print("=" * 60 + "\n")

    # ==================================================================
    # 舵机操作
    # ==================================================================

    def send_servo(self, channel: int, angle: int,
                   angle_range: Tuple[int, int] = None):
        """发送单个舵机角度，自动 clamp 到范围"""
        if angle_range:
            angle = max(angle_range[0], min(angle_range[1], angle))
        self.controller.set_servo_angle_time_32(
            [angle], [channel], 200, servo_num=self.controller.servo_num
        )

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
            print("[WARN] 未加载基线! 请先运行 ellseg_scorer.py 按 [S] 保存基线到 face_points.json")
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

    # ==================================================================
    # 引导式自动调整 — 上唇 (A16)
    # ==================================================================

    def auto_adjust_upper_lip(
        self,
        ulr_step: float = 1.0,
        max_iterations: int = None,
        wait_seconds: float = None,
        keep_display: bool = False,
    ) -> tuple:
        """
        引导式自动调整 — 只调整 A16 中上嘴唇舵机。

        流程（每轮迭代）:
          1. 打开 MP → 采集 60 帧 ULR → 关闭 MP
          2. 若 ULR 偏差 ≤ 容差 → 通过
          3. 否则移动 A16 → 等待 → 下一轮

        Args:
            ulr_step:       每次调整的舵机角度步长 (默认 1°)
            max_iterations: 最大迭代次数
            wait_seconds:   每轮等待秒数
            keep_display:   完成后是否保持窗口

        Returns:
            (passed: bool, iterations: int, final_data: dict)
        """
        if max_iterations is None:
            max_iterations = UPPER_LIP_MAX_ITERATIONS
        if wait_seconds is None:
            wait_seconds = UPPER_LIP_WAIT_SECONDS

        if self.scorer.upper_lip_baseline is None:
            print("[WARN] 未加载上唇基线! 请在 ellseg_scorer.py 中按 [U] 保存上唇基线到 upper_lip_baseline.json")
            return False, 0, {}

        tol = UPPER_LIP_ULR_TOLERANCE

        print(f"{'='*60}")
        print(f"  引导式自动调整 (仅中上嘴唇 A16)")
        print(f"  采集: 60帧, 修剪上下10帧, 中间40帧取平均 ULR")
        print(f"  步长: {ulr_step}度/次 | 最大迭代: {max_iterations} | 等待: {wait_seconds}s")
        print(f"  目标: ULR 偏差 ≤ {tol}")
        bl = self.scorer.upper_lip_baseline
        print(f"  基线: ULR={bl['ulr']:.4f}")
        print(f"{'='*60}\n")

        self.scorer.start_display()

        # 获取 A16 的范围
        ch = UPPER_LIP_CHANNEL
        start = self.controller.list_start_deg[ch]
        end = self.controller.list_end_deg[ch]
        ul_range = (min(start, end), max(start, end))
        ul_sign = getattr(self, '_upper_lip_flip', 1)   # 默认: ULR小→增大角度

        current_angle = self.controller.list_temp_deg[ch]
        print(f"  初始角度: A16={current_angle}° (range=[{ul_range[0]}, {ul_range[1]}])")

        iteration = 0
        sleep_chunk = 0.1

        try:
            while iteration < max_iterations:
                iteration += 1

                if self.scorer.user_pressed_stop:
                    print("\n[USER] 收到停止信号")
                    break

                self.scorer.update_hud([
                    (f"[UpperLip Iter {iteration}/{max_iterations}]", (10, 100), (200, 200, 200)),
                ])

                # ---- 采集 ULR ----
                avg_signal = self.collect_upper_lip_samples(n_frames=60, trim=10)

                if avg_signal is None:
                    print(f"  [UpperLip Iter {iteration:3d}] 无法获取有效帧")
                    _waited = 0
                    while _waited < wait_seconds:
                        time.sleep(min(sleep_chunk, wait_seconds - _waited))
                        _waited += sleep_chunk
                        if self.scorer.user_pressed_stop:
                            break
                    continue

                ulr = avg_signal["ulr"]
                delta = avg_signal["delta"]

                print(f"  [UpperLip Iter {iteration:3d}] ULR={ulr:.4f} (Δ={delta:+.4f})")

                # ---- 检查是否已合格 ----
                if abs(delta) <= tol:
                    print(f"\n{'='*60}")
                    print(f"  ★★ 上唇调整通过! 共迭代 {iteration} 次")
                    print(f"      ULR={ulr:.4f} (基线={bl['ulr']:.4f})")
                    print(f"      A16={current_angle}°")
                    print(f"{'='*60}\n")

                    # 保存结果
                    all_angles = [self.controller.list_temp_deg[c] for c in EYE_SERVO_CHANNELS]
                    self.save_result(all_angles)
                    chin_angle = self._temp_deg(MOUTH_CHIN_CHANNEL)
                    ll_angle = self._temp_deg(LOWER_LIP_CHANNEL)
                    self.export_angle_config(all_angles,
                                            chin_angle=chin_angle,
                                            lower_lip_angle=ll_angle,
                                            upper_lip_angle=current_angle)

                    self.scorer.update_hud([("★★ UPPER LIP PASSED ★★", (100, 80), (0, 255, 0))])
                    time.sleep(1)
                    return True, iteration, avg_signal

                # ---- 移动 A16 ----
                # delta>0 → ULR太大(上唇太展) → 合(-1); delta<0 → ULR太小(上唇太收) → 开(+1)
                delta_sign = -1 if delta > tol else (1 if delta < -tol else 0)
                move = ul_sign * delta_sign * ulr_step
                new_angle = round(current_angle + move)
                new_angle = max(ul_range[0], min(ul_range[1], new_angle))
                self.send_servo(ch, new_angle, ul_range)
                current_angle = new_angle
                print(f"  [UpperLip Iter {iteration:3d}] 移动 A16={new_angle}° "
                      f"(ULR Δ={delta:+.4f}, Δangle={move:+.1f})")

                # ---- 等待 ----
                _waited = 0
                while _waited < wait_seconds:
                    time.sleep(min(sleep_chunk, wait_seconds - _waited))
                    _waited += sleep_chunk
                    self.scorer.update_hud([
                        (f"[UpperLip Iter {iteration}/{max_iterations}]", (10, 100), (200, 200, 200)),
                        (f"Waiting... {_waited:.1f}/{wait_seconds}s", (10, 118), (0, 255, 255)),
                    ])
                    if self.scorer.user_pressed_stop:
                        break

            # 达到最大迭代未通过
            print(f"\n{'='*60}")
            print(f"  ✘ 上唇未在 {max_iterations} 次迭代内通过")
            print(f"{'='*60}\n")
            return False, iteration, {}

        finally:
            if not keep_display:
                self.scorer.stop_display()

    # ==================================================================
    # 多帧采集 + 修剪平均 — 眼皮 EAR
    # ==================================================================

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
                           "left_qualified", "right_qualified"}
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
            }))
            collected += 1

            self.scorer.update_hud([
                (f"Eyelid Sampling: {collected}/{n_frames}", (10, 120), (200, 200, 200)),
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
        avg["left_qualified"] = abs(avg["left_delta"]) <= tol
        avg["right_qualified"] = abs(avg["right_delta"]) <= tol

        all_devs = [s[0] for s in samples]
        print(f"  [Eyelid Collect] {len(samples)} frames, trim {trim}: "
              f"L_delta={avg['left_delta']:+.4f} R_delta={avg['right_delta']:+.4f} "
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
        引导式自动调整 — 只调整 A8/A9 眼皮舵机。

        流程（每轮迭代）:
          1. 打开 MP（仅 MediaPipe 关键点）
          2. 采集 60 帧 → 去头尾求平均 EAR
          3. 关闭 MP
          4. 若左右眼 EAR 偏差均 ≤ 容差 → 通过
          5. 否则按偏差方向移动 A8/A9 → 等待 → 下一轮

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
            print("[WARN] 未加载眼皮基线! 请先在 ellseg_scorer.py 中按 [E] 保存眼皮基线到 eyelid_baseline.json")
            return False, 0, {}

        tol = EYELID_EAR_TOLERANCE

        print(f"{'='*60}")
        print(f"  引导式自动调整 (仅眼皮 A8-A9)")
        print(f"  采集: 60帧, 修剪上下10帧, 中间40帧取平均 EAR")
        print(f"  步长: {ear_step}度/次 | 最大迭代: {max_iterations} | 等待: {wait_seconds}s")
        print(f"  目标: 左右眼 EAR 偏差 ≤ {tol}")
        bl = self.scorer.eyelid_baseline
        print(f"  基线: L_EAR={bl['left_ear']:.4f} R_EAR={bl['right_ear']:.4f}")
        print(f"{'='*60}\n")

        self.scorer.start_display()

        current_angles = {
            ch: self.controller.list_temp_deg[ch]
            for ch in EYELID_CHANNELS
        }
        print(f"  初始角度: {dict(zip(EYELID_NAMES, [current_angles[c] for c in EYELID_CHANNELS]))}")

        iteration = 0
        sleep_chunk = 0.1

        try:
            while iteration < max_iterations:
                iteration += 1

                if self.scorer.user_pressed_stop:
                    print("\n[USER] 收到停止信号")
                    break

                self.scorer.update_hud([
                    (f"[Eyelid Iter {iteration}/{max_iterations}]", (10, 100), (200, 200, 200)),
                ])

                # ---- 采集 EAR ----
                avg_signal = self.collect_eyelid_samples(n_frames=60, trim=10)

                if avg_signal is None:
                    self.scorer.update_hud([
                        (f"[Eyelid Iter {iteration}/{max_iterations}]", (10, 100), (200, 200, 200)),
                        (f"Signal: None (no valid frames)", (10, 118), (0, 160, 255)),
                    ])
                    print(f"  [Eyelid Iter {iteration:3d}] 无法获取有效帧")
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

                # ---- 检查是否已合格 ----
                l_ok = abs(l_delta) <= tol
                r_ok = abs(r_delta) <= tol

                print(f"  [Eyelid Iter {iteration:3d}] "
                      f"L_EAR={l_ear:.4f} (Δ={l_delta:+.4f}) "
                      f"R_EAR={r_ear:.4f} (Δ={r_delta:+.4f})")

                if l_ok and r_ok:
                    print(f"\n{'='*60}")
                    print(f"  ★★ 眼皮调整通过! 共迭代 {iteration} 次")
                    print(f"      L_EAR={l_ear:.4f}  R_EAR={r_ear:.4f}")
                    final_ang = " ".join([f"{n}={current_angles[c]}度"
                                         for n, c in zip(EYELID_NAMES, EYELID_CHANNELS)])
                    print(f"      角度: {final_ang}")
                    print(f"{'='*60}\n")

                    # 合并眼球当前角度到完整配置
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

                # ---- 移动 A8/A9: 先调偏差大的眼 ----
                # adj_map: delta > 0 = 太开 → 关 (-1); delta < 0 = 太闭 → 开 (+1)
                a8_sign = getattr(self, '_eyelid_a8_flip', -1)
                a9_sign = getattr(self, '_eyelid_a9_flip', 1)

                l_need = abs(l_delta) > tol
                r_need = abs(r_delta) > tol

                if not l_need and not r_need:
                    print(f"  [Eyelid Iter {iteration:3d}] 无需移动 (偏差在容差内)")
                else:
                    # 选偏差大的那只眼单独调整
                    if not r_need or (l_need and abs(l_delta) >= abs(r_delta)):
                        side = "L"
                        ch = EYELID_CHANNELS[0]
                        name = EYELID_NAMES[0]
                        sign = a8_sign
                        d = l_delta
                    else:
                        side = "R"
                        ch = EYELID_CHANNELS[1]
                        name = EYELID_NAMES[1]
                        sign = a9_sign
                        d = r_delta

                    delta = sign * (-1 if d > tol else (1 if d < -tol else 0))
                    new_angle = round(current_angles[ch] + delta * ear_step)
                    lo, hi = self.angle_ranges[EYE_SERVO_CHANNELS.index(ch)]
                    new_angle = max(lo, min(hi, new_angle))
                    self.send_servo(ch, new_angle, (lo, hi))
                    current_angles[ch] = new_angle
                    print(f"  [Eyelid Iter {iteration:3d}] 移动 {side}: {name}={new_angle}度 "
                          f"(EAR Δ={d:+.4f}, Δangle={delta:+.0f})")

                # ---- 等待 ----
                _waited = 0
                while _waited < wait_seconds:
                    time.sleep(min(sleep_chunk, wait_seconds - _waited))
                    _waited += sleep_chunk
                    self.scorer.update_hud([
                        (f"[Eyelid Iter {iteration}/{max_iterations}]", (10, 100), (200, 200, 200)),
                        (f"Waiting... {_waited:.1f}/{wait_seconds}s", (10, 118), (0, 255, 255)),
                    ])
                    if self.scorer.user_pressed_stop:
                        break

            # 达到最大迭代未通过
            print(f"\n{'='*60}")
            print(f"  ✘ 眼皮未在 {max_iterations} 次迭代内通过")
            print(f"{'='*60}\n")
            return False, iteration, {}

        finally:
            if not keep_display:
                self.scorer.stop_display()

    # ==================================================================
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

    # ==================================================================
    # 引导式自动调整 — 上唇 (A16)
    # ==================================================================

    def auto_adjust_upper_lip(
        self,
        ulr_step: float = 1.0,
        max_iterations: int = None,
        wait_seconds: float = None,
        keep_display: bool = False,
    ) -> tuple:
        """
        引导式自动调整 — 只调整 A16 中上嘴唇舵机。

        流程（每轮迭代）:
          1. 打开 MP → 采集 60 帧 ULR → 关闭 MP
          2. 若 ULR 偏差 ≤ 容差 → 通过
          3. 否则移动 A16 → 等待 → 下一轮

        Args:
            ulr_step:       每次调整的舵机角度步长 (默认 1°)
            max_iterations: 最大迭代次数
            wait_seconds:   每轮等待秒数
            keep_display:   完成后是否保持窗口

        Returns:
            (passed: bool, iterations: int, final_data: dict)
        """
        if max_iterations is None:
            max_iterations = UPPER_LIP_MAX_ITERATIONS
        if wait_seconds is None:
            wait_seconds = UPPER_LIP_WAIT_SECONDS

        if self.scorer.upper_lip_baseline is None:
            print("[WARN] 未加载上唇基线! 请在 ellseg_scorer.py 中按 [U] 保存上唇基线到 upper_lip_baseline.json")
            return False, 0, {}

        tol = UPPER_LIP_ULR_TOLERANCE

        print(f"{'='*60}")
        print(f"  引导式自动调整 (仅中上嘴唇 A16)")
        print(f"  采集: 60帧, 修剪上下10帧, 中间40帧取平均 ULR")
        print(f"  步长: {ulr_step}度/次 | 最大迭代: {max_iterations} | 等待: {wait_seconds}s")
        print(f"  目标: ULR 偏差 ≤ {tol}")
        bl = self.scorer.upper_lip_baseline
        print(f"  基线: ULR={bl['ulr']:.4f}")
        print(f"{'='*60}\n")

        self.scorer.start_display()

        # 获取 A16 的范围
        ch = UPPER_LIP_CHANNEL
        start = self.controller.list_start_deg[ch]
        end = self.controller.list_end_deg[ch]
        ul_range = (min(start, end), max(start, end))
        ul_sign = getattr(self, '_upper_lip_flip', 1)   # 默认: ULR小→增大角度

        current_angle = self.controller.list_temp_deg[ch]
        print(f"  初始角度: A16={current_angle}° (range=[{ul_range[0]}, {ul_range[1]}])")

        iteration = 0
        sleep_chunk = 0.1

        try:
            while iteration < max_iterations:
                iteration += 1

                if self.scorer.user_pressed_stop:
                    print("\n[USER] 收到停止信号")
                    break

                self.scorer.update_hud([
                    (f"[UpperLip Iter {iteration}/{max_iterations}]", (10, 100), (200, 200, 200)),
                ])

                # ---- 采集 ULR ----
                avg_signal = self.collect_upper_lip_samples(n_frames=60, trim=10)

                if avg_signal is None:
                    print(f"  [UpperLip Iter {iteration:3d}] 无法获取有效帧")
                    _waited = 0
                    while _waited < wait_seconds:
                        time.sleep(min(sleep_chunk, wait_seconds - _waited))
                        _waited += sleep_chunk
                        if self.scorer.user_pressed_stop:
                            break
                    continue

                ulr = avg_signal["ulr"]
                delta = avg_signal["delta"]

                print(f"  [UpperLip Iter {iteration:3d}] ULR={ulr:.4f} (Δ={delta:+.4f})")

                # ---- 检查是否已合格 ----
                if abs(delta) <= tol:
                    print(f"\n{'='*60}")
                    print(f"  ★★ 上唇调整通过! 共迭代 {iteration} 次")
                    print(f"      ULR={ulr:.4f} (基线={bl['ulr']:.4f})")
                    print(f"      A16={current_angle}°")
                    print(f"{'='*60}\n")

                    # 保存结果
                    all_angles = [self.controller.list_temp_deg[c] for c in EYE_SERVO_CHANNELS]
                    self.save_result(all_angles)
                    chin_angle = self._temp_deg(MOUTH_CHIN_CHANNEL)
                    ll_angle = self._temp_deg(LOWER_LIP_CHANNEL)
                    self.export_angle_config(all_angles,
                                            chin_angle=chin_angle,
                                            lower_lip_angle=ll_angle,
                                            upper_lip_angle=current_angle)

                    self.scorer.update_hud([("★★ UPPER LIP PASSED ★★", (100, 80), (0, 255, 0))])
                    time.sleep(1)
                    return True, iteration, avg_signal

                # ---- 移动 A16 ----
                # delta>0 → ULR太大(上唇太展) → 合(-1); delta<0 → ULR太小(上唇太收) → 开(+1)
                delta_sign = -1 if delta > tol else (1 if delta < -tol else 0)
                move = ul_sign * delta_sign * ulr_step
                new_angle = round(current_angle + move)
                new_angle = max(ul_range[0], min(ul_range[1], new_angle))
                self.send_servo(ch, new_angle, ul_range)
                current_angle = new_angle
                print(f"  [UpperLip Iter {iteration:3d}] 移动 A16={new_angle}° "
                      f"(ULR Δ={delta:+.4f}, Δangle={move:+.1f})")

                # ---- 等待 ----
                _waited = 0
                while _waited < wait_seconds:
                    time.sleep(min(sleep_chunk, wait_seconds - _waited))
                    _waited += sleep_chunk
                    self.scorer.update_hud([
                        (f"[UpperLip Iter {iteration}/{max_iterations}]", (10, 100), (200, 200, 200)),
                        (f"Waiting... {_waited:.1f}/{wait_seconds}s", (10, 118), (0, 255, 255)),
                    ])
                    if self.scorer.user_pressed_stop:
                        break

            # 达到最大迭代未通过
            print(f"\n{'='*60}")
            print(f"  ✘ 上唇未在 {max_iterations} 次迭代内通过")
            print(f"{'='*60}\n")
            return False, iteration, {}

        finally:
            if not keep_display:
                self.scorer.stop_display()

    # ==================================================================
    # 多帧采集 + 修剪平均 — 眉毛 EBHR
    # ==================================================================

    def collect_eyebrow_samples(self, n_frames: int = 60, trim: int = 10):
        """
        打开 MP（无需 EllSeg）→ 采集 n 帧 EBHR → 关闭 → 修剪平均。

        流程:
          1. 开启 MP
          2. 连续采集 n_frames 帧 EBHR 信号
          3. 关闭 MP
          4. 按 abs(L_delta) + abs(R_delta) 排序，修剪上下
          5. 返回平均 EBHR 信号

        Returns:
            dict 或 None: {"left_ebhr", "right_ebhr", "left_delta", "right_delta",
                           "left_qualified", "right_qualified"}
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
            signal = self.scorer.get_eyebrow_signal()

            if signal is None:
                no_face_count += 1
                if no_face_count > 10:
                    break
                time.sleep(0.001)
                continue
            no_face_count = 0

            total_dev = abs(signal["left_delta"]) + abs(signal["right_delta"])
            samples.append((total_dev, {
                "left_ebhr": signal["left_ebhr"],
                "right_ebhr": signal["right_ebhr"],
                "left_delta": signal["left_delta"],
                "right_delta": signal["right_delta"],
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
        for key in ["left_ebhr", "right_ebhr", "left_delta", "right_delta"]:
            avg[key] = sum(s[1][key] for s in trimmed) / n

        tol = EYEBROW_EBHR_TOLERANCE
        avg["left_qualified"] = abs(avg["left_delta"]) <= tol
        avg["right_qualified"] = abs(avg["right_delta"]) <= tol

        all_devs = [s[0] for s in samples]
        print(f"  [Eyebrow Collect] {len(samples)} frames, trim {trim}: "
              f"L_delta={avg['left_delta']:+.4f} R_delta={avg['right_delta']:+.4f} "
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
    ) -> tuple:
        """
        引导式自动调整 — 只调整 A0/A1 眉毛舵机。

        流程（每轮迭代）:
          1. 打开 MP（仅 MediaPipe 关键点）
          2. 采集 60 帧 → 去头尾求平均 EBHR
          3. 关闭 MP
          4. 若左右眉 EBHR 偏差均 ≤ 容差 → 通过
          5. 否则按偏差方向移动 A0/A1 → 等待 → 下一轮

        Args:
            ebhr_step:     每次调整的舵机角度步长 (默认 1°)
            max_iterations: 最大循环次数 (默认 EYEBROW_MAX_ITERATIONS)
            wait_seconds:   每次移动后等待秒数 (默认 EYEBROW_WAIT_SECONDS)

        Returns:
            (passed: bool, iterations: int, final_ebhr_data: dict)
        """
        if max_iterations is None:
            max_iterations = EYEBROW_MAX_ITERATIONS
        if wait_seconds is None:
            wait_seconds = EYEBROW_WAIT_SECONDS

        if self.scorer.eyebrow_baseline is None:
            print("[WARN] 未加载眉毛基线! 请先在 ellseg_scorer.py 中按 [B] 保存眉毛基线到 eyebrow_baseline.json")
            return False, 0, {}

        tol = EYEBROW_EBHR_TOLERANCE

        print(f"{'='*60}")
        print(f"  引导式自动调整 (仅眉毛 A0-A1)")
        print(f"  采集: 60帧, 修剪上下10帧, 中间40帧取平均 EBHR")
        print(f"  步长: {ebhr_step}度/次 | 最大迭代: {max_iterations} | 等待: {wait_seconds}s")
        print(f"  目标: 左右眉 EBHR 偏差 ≤ {tol}")
        bl = self.scorer.eyebrow_baseline
        print(f"  基线: L_EBHR={bl['left_ebhr']:.4f} R_EBHR={bl['right_ebhr']:.4f}")
        print(f"{'='*60}\n")

        self.scorer.start_display()

        current_angles = {
            ch: self.controller.list_temp_deg[ch]
            for ch in EYEBROW_CHANNELS
        }
        print(f"  初始角度: {dict(zip(EYEBROW_NAMES, [current_angles[c] for c in EYEBROW_CHANNELS]))}")

        iteration = 0
        sleep_chunk = 0.1

        try:
            while iteration < max_iterations:
                iteration += 1

                if self.scorer.user_pressed_stop:
                    print("\n[USER] 收到停止信号")
                    break

                self.scorer.update_hud([
                    (f"[Eyebrow Iter {iteration}/{max_iterations}]", (10, 100), (200, 200, 200)),
                ])

                # ---- 采集 EBHR ----
                avg_signal = self.collect_eyebrow_samples(n_frames=60, trim=10)

                if avg_signal is None:
                    self.scorer.update_hud([
                        (f"[Eyebrow Iter {iteration}/{max_iterations}]", (10, 100), (200, 200, 200)),
                        (f"Signal: None (no valid frames)", (10, 118), (0, 160, 255)),
                    ])
                    print(f"  [Eyebrow Iter {iteration:3d}] 无法获取有效帧")
                    _waited = 0
                    while _waited < wait_seconds:
                        time.sleep(min(sleep_chunk, wait_seconds - _waited))
                        _waited += sleep_chunk
                        if self.scorer.user_pressed_stop:
                            break
                    continue

                l_delta = avg_signal["left_delta"]
                r_delta = avg_signal["right_delta"]
                l_ebhr = avg_signal["left_ebhr"]
                r_ebhr = avg_signal["right_ebhr"]

                # ---- 检查是否已合格 ----
                l_ok = abs(l_delta) <= tol
                r_ok = abs(r_delta) <= tol

                print(f"  [Eyebrow Iter {iteration:3d}] "
                      f"L_EBHR={l_ebhr:.4f} (Δ={l_delta:+.4f}) "
                      f"R_EBHR={r_ebhr:.4f} (Δ={r_delta:+.4f})")

                if l_ok and r_ok:
                    print(f"\n{'='*60}")
                    print(f"  ★★ 眉毛调整通过! 共迭代 {iteration} 次")
                    print(f"      L_EBHR={l_ebhr:.4f}  R_EBHR={r_ebhr:.4f}")
                    final_ang = " ".join([f"{n}={current_angles[c]}度"
                                         for n, c in zip(EYEBROW_NAMES, EYEBROW_CHANNELS)])
                    print(f"      角度: {final_ang}")
                    print(f"{'='*60}\n")

                    # 合并所有通道当前角度
                    # 眼睛通道 (A8-A13) 取控制器当前值
                    all_angles = []
                    for ch in EYE_SERVO_CHANNELS:
                        all_angles.append(self.controller.list_temp_deg[ch])
                    self.save_result(all_angles)
                    self.export_angle_config(all_angles,
                                            eyebrow_angles={c: current_angles[c] for c in EYEBROW_CHANNELS})

                    self.scorer.update_hud([("★★ EYEBROW PASSED ★★", (100, 80), (0, 255, 0))])
                    time.sleep(1)
                    return True, iteration, avg_signal

                # ---- 移动 A0/A1: 先调偏差大的眉 ----
                a0_sign = getattr(self, '_eyebrow_a0_flip', -1)
                a1_sign = getattr(self, '_eyebrow_a1_flip', -1)

                l_need = abs(l_delta) > tol
                r_need = abs(r_delta) > tol

                if not l_need and not r_need:
                    print(f"  [Eyebrow Iter {iteration:3d}] 无需移动 (偏差在容差内)")
                else:
                    # 选偏差大的那只眉单独调整
                    if not r_need or (l_need and abs(l_delta) >= abs(r_delta)):
                        side = "L"
                        ch = EYEBROW_CHANNELS[0]
                        name = EYEBROW_NAMES[0]
                        sign = a0_sign
                        d = l_delta
                    else:
                        side = "R"
                        ch = EYEBROW_CHANNELS[1]
                        name = EYEBROW_NAMES[1]
                        sign = a1_sign
                        d = r_delta

                    # delta>0 眉毛太高 → 降低(-1); delta<0 太低 → 抬高(+1)
                    delta_sign = -1 if d > tol else (1 if d < -tol else 0)
                    delta = sign * delta_sign
                    new_angle = round(current_angles[ch] + delta * ebhr_step)
                    lo, hi = self.eyebrow_ranges[ch]
                    new_angle = max(lo, min(hi, new_angle))
                    self.send_servo(ch, new_angle, (lo, hi))
                    current_angles[ch] = new_angle
                    print(f"  [Eyebrow Iter {iteration:3d}] 移动 {side}: {name}={new_angle}度 "
                          f"(EBHR Δ={d:+.4f}, Δangle={delta:+.0f})")

                # ---- 等待 ----
                _waited = 0
                while _waited < wait_seconds:
                    time.sleep(min(sleep_chunk, wait_seconds - _waited))
                    _waited += sleep_chunk
                    self.scorer.update_hud([
                        (f"[Eyebrow Iter {iteration}/{max_iterations}]", (10, 100), (200, 200, 200)),
                        (f"Waiting... {_waited:.1f}/{wait_seconds}s", (10, 118), (0, 255, 255)),
                    ])
                    if self.scorer.user_pressed_stop:
                        break

            # 达到最大迭代未通过
            print(f"\n{'='*60}")
            print(f"  ✘ 眉毛未在 {max_iterations} 次迭代内通过")
            print(f"{'='*60}\n")
            return False, iteration, {}

        finally:
            if not keep_display:
                self.scorer.stop_display()

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

    # ==================================================================
    # 引导式自动调整 — 上唇 (A16)
    # ==================================================================

    def auto_adjust_upper_lip(
        self,
        ulr_step: float = 1.0,
        max_iterations: int = None,
        wait_seconds: float = None,
        keep_display: bool = False,
    ) -> tuple:
        """
        引导式自动调整 — 只调整 A16 中上嘴唇舵机。

        流程（每轮迭代）:
          1. 打开 MP → 采集 60 帧 ULR → 关闭 MP
          2. 若 ULR 偏差 ≤ 容差 → 通过
          3. 否则移动 A16 → 等待 → 下一轮

        Args:
            ulr_step:       每次调整的舵机角度步长 (默认 1°)
            max_iterations: 最大迭代次数
            wait_seconds:   每轮等待秒数
            keep_display:   完成后是否保持窗口

        Returns:
            (passed: bool, iterations: int, final_data: dict)
        """
        if max_iterations is None:
            max_iterations = UPPER_LIP_MAX_ITERATIONS
        if wait_seconds is None:
            wait_seconds = UPPER_LIP_WAIT_SECONDS

        if self.scorer.upper_lip_baseline is None:
            print("[WARN] 未加载上唇基线! 请在 ellseg_scorer.py 中按 [U] 保存上唇基线到 upper_lip_baseline.json")
            return False, 0, {}

        tol = UPPER_LIP_ULR_TOLERANCE

        print(f"{'='*60}")
        print(f"  引导式自动调整 (仅中上嘴唇 A16)")
        print(f"  采集: 60帧, 修剪上下10帧, 中间40帧取平均 ULR")
        print(f"  步长: {ulr_step}度/次 | 最大迭代: {max_iterations} | 等待: {wait_seconds}s")
        print(f"  目标: ULR 偏差 ≤ {tol}")
        bl = self.scorer.upper_lip_baseline
        print(f"  基线: ULR={bl['ulr']:.4f}")
        print(f"{'='*60}\n")

        self.scorer.start_display()

        # 获取 A16 的范围
        ch = UPPER_LIP_CHANNEL
        start = self.controller.list_start_deg[ch]
        end = self.controller.list_end_deg[ch]
        ul_range = (min(start, end), max(start, end))
        ul_sign = getattr(self, '_upper_lip_flip', 1)   # 默认: ULR小→增大角度

        current_angle = self.controller.list_temp_deg[ch]
        print(f"  初始角度: A16={current_angle}° (range=[{ul_range[0]}, {ul_range[1]}])")

        iteration = 0
        sleep_chunk = 0.1

        try:
            while iteration < max_iterations:
                iteration += 1

                if self.scorer.user_pressed_stop:
                    print("\n[USER] 收到停止信号")
                    break

                self.scorer.update_hud([
                    (f"[UpperLip Iter {iteration}/{max_iterations}]", (10, 100), (200, 200, 200)),
                ])

                # ---- 采集 ULR ----
                avg_signal = self.collect_upper_lip_samples(n_frames=60, trim=10)

                if avg_signal is None:
                    print(f"  [UpperLip Iter {iteration:3d}] 无法获取有效帧")
                    _waited = 0
                    while _waited < wait_seconds:
                        time.sleep(min(sleep_chunk, wait_seconds - _waited))
                        _waited += sleep_chunk
                        if self.scorer.user_pressed_stop:
                            break
                    continue

                ulr = avg_signal["ulr"]
                delta = avg_signal["delta"]

                print(f"  [UpperLip Iter {iteration:3d}] ULR={ulr:.4f} (Δ={delta:+.4f})")

                # ---- 检查是否已合格 ----
                if abs(delta) <= tol:
                    print(f"\n{'='*60}")
                    print(f"  ★★ 上唇调整通过! 共迭代 {iteration} 次")
                    print(f"      ULR={ulr:.4f} (基线={bl['ulr']:.4f})")
                    print(f"      A16={current_angle}°")
                    print(f"{'='*60}\n")

                    # 保存结果
                    all_angles = [self.controller.list_temp_deg[c] for c in EYE_SERVO_CHANNELS]
                    self.save_result(all_angles)
                    chin_angle = self._temp_deg(MOUTH_CHIN_CHANNEL)
                    ll_angle = self._temp_deg(LOWER_LIP_CHANNEL)
                    self.export_angle_config(all_angles,
                                            chin_angle=chin_angle,
                                            lower_lip_angle=ll_angle,
                                            upper_lip_angle=current_angle)

                    self.scorer.update_hud([("★★ UPPER LIP PASSED ★★", (100, 80), (0, 255, 0))])
                    time.sleep(1)
                    return True, iteration, avg_signal

                # ---- 移动 A16 ----
                # delta>0 → ULR太大(上唇太展) → 合(-1); delta<0 → ULR太小(上唇太收) → 开(+1)
                delta_sign = -1 if delta > tol else (1 if delta < -tol else 0)
                move = ul_sign * delta_sign * ulr_step
                new_angle = round(current_angle + move)
                new_angle = max(ul_range[0], min(ul_range[1], new_angle))
                self.send_servo(ch, new_angle, ul_range)
                current_angle = new_angle
                print(f"  [UpperLip Iter {iteration:3d}] 移动 A16={new_angle}° "
                      f"(ULR Δ={delta:+.4f}, Δangle={move:+.1f})")

                # ---- 等待 ----
                _waited = 0
                while _waited < wait_seconds:
                    time.sleep(min(sleep_chunk, wait_seconds - _waited))
                    _waited += sleep_chunk
                    self.scorer.update_hud([
                        (f"[UpperLip Iter {iteration}/{max_iterations}]", (10, 100), (200, 200, 200)),
                        (f"Waiting... {_waited:.1f}/{wait_seconds}s", (10, 118), (0, 255, 255)),
                    ])
                    if self.scorer.user_pressed_stop:
                        break

            # 达到最大迭代未通过
            print(f"\n{'='*60}")
            print(f"  ✘ 上唇未在 {max_iterations} 次迭代内通过")
            print(f"{'='*60}\n")
            return False, iteration, {}

        finally:
            if not keep_display:
                self.scorer.stop_display()

    # ==================================================================
    # 引导式自动调整 — 嘴部下巴 (A26)
    # ==================================================================

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
            print("[WARN] 未加载嘴部基线! 请在 ellseg_scorer.py 中按 [M] 保存嘴部基线到 mouth_baseline.json")
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

    # ==================================================================
    # 引导式自动调整 — 上唇 (A16)
    # ==================================================================

    def auto_adjust_upper_lip(
        self,
        ulr_step: float = 1.0,
        max_iterations: int = None,
        wait_seconds: float = None,
        keep_display: bool = False,
    ) -> tuple:
        """
        引导式自动调整 — 只调整 A16 中上嘴唇舵机。

        流程（每轮迭代）:
          1. 打开 MP → 采集 60 帧 ULR → 关闭 MP
          2. 若 ULR 偏差 ≤ 容差 → 通过
          3. 否则移动 A16 → 等待 → 下一轮

        Args:
            ulr_step:       每次调整的舵机角度步长 (默认 1°)
            max_iterations: 最大迭代次数
            wait_seconds:   每轮等待秒数
            keep_display:   完成后是否保持窗口

        Returns:
            (passed: bool, iterations: int, final_data: dict)
        """
        if max_iterations is None:
            max_iterations = UPPER_LIP_MAX_ITERATIONS
        if wait_seconds is None:
            wait_seconds = UPPER_LIP_WAIT_SECONDS

        if self.scorer.upper_lip_baseline is None:
            print("[WARN] 未加载上唇基线! 请在 ellseg_scorer.py 中按 [U] 保存上唇基线到 upper_lip_baseline.json")
            return False, 0, {}

        tol = UPPER_LIP_ULR_TOLERANCE

        print(f"{'='*60}")
        print(f"  引导式自动调整 (仅中上嘴唇 A16)")
        print(f"  采集: 60帧, 修剪上下10帧, 中间40帧取平均 ULR")
        print(f"  步长: {ulr_step}度/次 | 最大迭代: {max_iterations} | 等待: {wait_seconds}s")
        print(f"  目标: ULR 偏差 ≤ {tol}")
        bl = self.scorer.upper_lip_baseline
        print(f"  基线: ULR={bl['ulr']:.4f}")
        print(f"{'='*60}\n")

        self.scorer.start_display()

        # 获取 A16 的范围
        ch = UPPER_LIP_CHANNEL
        start = self.controller.list_start_deg[ch]
        end = self.controller.list_end_deg[ch]
        ul_range = (min(start, end), max(start, end))
        ul_sign = getattr(self, '_upper_lip_flip', 1)   # 默认: ULR小→增大角度

        current_angle = self.controller.list_temp_deg[ch]
        print(f"  初始角度: A16={current_angle}° (range=[{ul_range[0]}, {ul_range[1]}])")

        iteration = 0
        sleep_chunk = 0.1

        try:
            while iteration < max_iterations:
                iteration += 1

                if self.scorer.user_pressed_stop:
                    print("\n[USER] 收到停止信号")
                    break

                self.scorer.update_hud([
                    (f"[UpperLip Iter {iteration}/{max_iterations}]", (10, 100), (200, 200, 200)),
                ])

                # ---- 采集 ULR ----
                avg_signal = self.collect_upper_lip_samples(n_frames=60, trim=10)

                if avg_signal is None:
                    print(f"  [UpperLip Iter {iteration:3d}] 无法获取有效帧")
                    _waited = 0
                    while _waited < wait_seconds:
                        time.sleep(min(sleep_chunk, wait_seconds - _waited))
                        _waited += sleep_chunk
                        if self.scorer.user_pressed_stop:
                            break
                    continue

                ulr = avg_signal["ulr"]
                delta = avg_signal["delta"]

                print(f"  [UpperLip Iter {iteration:3d}] ULR={ulr:.4f} (Δ={delta:+.4f})")

                # ---- 检查是否已合格 ----
                if abs(delta) <= tol:
                    print(f"\n{'='*60}")
                    print(f"  ★★ 上唇调整通过! 共迭代 {iteration} 次")
                    print(f"      ULR={ulr:.4f} (基线={bl['ulr']:.4f})")
                    print(f"      A16={current_angle}°")
                    print(f"{'='*60}\n")

                    # 保存结果
                    all_angles = [self.controller.list_temp_deg[c] for c in EYE_SERVO_CHANNELS]
                    self.save_result(all_angles)
                    chin_angle = self._temp_deg(MOUTH_CHIN_CHANNEL)
                    ll_angle = self._temp_deg(LOWER_LIP_CHANNEL)
                    self.export_angle_config(all_angles,
                                            chin_angle=chin_angle,
                                            lower_lip_angle=ll_angle,
                                            upper_lip_angle=current_angle)

                    self.scorer.update_hud([("★★ UPPER LIP PASSED ★★", (100, 80), (0, 255, 0))])
                    time.sleep(1)
                    return True, iteration, avg_signal

                # ---- 移动 A16 ----
                # delta>0 → ULR太大(上唇太展) → 合(-1); delta<0 → ULR太小(上唇太收) → 开(+1)
                delta_sign = -1 if delta > tol else (1 if delta < -tol else 0)
                move = ul_sign * delta_sign * ulr_step
                new_angle = round(current_angle + move)
                new_angle = max(ul_range[0], min(ul_range[1], new_angle))
                self.send_servo(ch, new_angle, ul_range)
                current_angle = new_angle
                print(f"  [UpperLip Iter {iteration:3d}] 移动 A16={new_angle}° "
                      f"(ULR Δ={delta:+.4f}, Δangle={move:+.1f})")

                # ---- 等待 ----
                _waited = 0
                while _waited < wait_seconds:
                    time.sleep(min(sleep_chunk, wait_seconds - _waited))
                    _waited += sleep_chunk
                    self.scorer.update_hud([
                        (f"[UpperLip Iter {iteration}/{max_iterations}]", (10, 100), (200, 200, 200)),
                        (f"Waiting... {_waited:.1f}/{wait_seconds}s", (10, 118), (0, 255, 255)),
                    ])
                    if self.scorer.user_pressed_stop:
                        break

            # 达到最大迭代未通过
            print(f"\n{'='*60}")
            print(f"  ✘ 上唇未在 {max_iterations} 次迭代内通过")
            print(f"{'='*60}\n")
            return False, iteration, {}

        finally:
            if not keep_display:
                self.scorer.stop_display()

    # ==================================================================
    # 多帧采集 + 修剪平均 — 下唇 LLR
    # ==================================================================

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

    def auto_adjust_lower_lip(
        self,
        step: float = 1.0,
        max_iterations: int = None,
        wait_seconds: float = None,
        keep_display: bool = False,
    ) -> tuple:
        """
        引导式自动调整 — 只调整 A19 中下嘴唇舵机。

        流程（每轮迭代）:
          1. 打开 MP → 采集 60 帧 LLR → 关闭 MP
          2. 若 LLR 偏差 ≤ 容差 → 通过
          3. 否则移动 A19 → 等待 → 下一轮

        Args:
            step:           每次调整的舵机角度步长 (默认 1°)
            max_iterations: 最大迭代次数
            wait_seconds:   每轮等待秒数
            keep_display:   完成后是否保持窗口

        Returns:
            (passed: bool, iterations: int, final_data: dict)
        """
        if max_iterations is None:
            max_iterations = LOWER_LIP_MAX_ITERATIONS
        if wait_seconds is None:
            wait_seconds = LOWER_LIP_WAIT_SECONDS

        if self.scorer.lower_lip_baseline is None:
            print("[WARN] 未加载下唇基线! 请在 ellseg_scorer.py 中按 [L] 保存下唇基线到 lower_lip_baseline.json")
            return False, 0, {}

        tol = LOWER_LIP_LLR_TOLERANCE

        print(f"{'='*60}")
        print(f"  引导式自动调整 (仅中下嘴唇 A19)")
        print(f"  采集: 60帧, 修剪上下10帧, 中间40帧取平均 LLR")
        print(f"  步长: {step}度/次 | 最大迭代: {max_iterations} | 等待: {wait_seconds}s")
        print(f"  目标: LLR 偏差 ≤ {tol}")
        bl = self.scorer.lower_lip_baseline
        print(f"  基线: LLR={bl['llr']:.4f}")
        print(f"{'='*60}\n")

        self.scorer.start_display()

        # 获取 A19 的范围
        ch = LOWER_LIP_CHANNEL
        start = self.controller.list_start_deg[ch]
        end = self.controller.list_end_deg[ch]
        ll_range = (min(start, end), max(start, end))
        ll_sign = getattr(self, '_lower_lip_flip', -1)  # 默认: LLR大→合

        current_angle = self.controller.list_temp_deg[ch]
        print(f"  初始角度: A19={current_angle}° (range=[{ll_range[0]}, {ll_range[1]}])")

        iteration = 0
        sleep_chunk = 0.1

        try:
            while iteration < max_iterations:
                iteration += 1

                if self.scorer.user_pressed_stop:
                    print("\n[USER] 收到停止信号")
                    break

                self.scorer.update_hud([
                    (f"[LowerLip Iter {iteration}/{max_iterations}]", (10, 100), (200, 200, 200)),
                ])

                # ---- 采集 LLR ----
                avg_signal = self.collect_lower_lip_samples(n_frames=60, trim=10)

                if avg_signal is None:
                    print(f"  [LowerLip Iter {iteration:3d}] 无法获取有效帧")
                    _waited = 0
                    while _waited < wait_seconds:
                        time.sleep(min(sleep_chunk, wait_seconds - _waited))
                        _waited += sleep_chunk
                        if self.scorer.user_pressed_stop:
                            break
                    continue

                llr = avg_signal["llr"]
                delta = avg_signal["delta"]

                print(f"  [LowerLip Iter {iteration:3d}] LLR={llr:.4f} (Δ={delta:+.4f})")

                # ---- 检查是否已合格 ----
                if abs(delta) <= tol:
                    print(f"\n{'='*60}")
                    print(f"  ★★ 下唇调整通过! 共迭代 {iteration} 次")
                    print(f"      LLR={llr:.4f} (基线={bl['llr']:.4f})")
                    print(f"      A19={current_angle}°")
                    print(f"{'='*60}\n")

                    # 保存结果
                    all_angles = [self.controller.list_temp_deg[c] for c in EYE_SERVO_CHANNELS]
                    self.save_result(all_angles)
                    self.export_angle_config(all_angles, chin_angle=(
                        self._temp_deg(MOUTH_CHIN_CHANNEL)
                    ), lower_lip_angle=current_angle)

                    self.scorer.update_hud([("★★ LOWER LIP PASSED ★★", (100, 80), (0, 255, 0))])
                    time.sleep(1)
                    return True, iteration, avg_signal

                # ---- 移动 A19 ----
                # delta>0 → LLR太大(下唇太展) → 合(-1); delta<0 → LLR太小(下唇太收) → 开(+1)
                delta_sign = -1 if delta > tol else (1 if delta < -tol else 0)
                move = ll_sign * delta_sign * step
                new_angle = round(current_angle + move)
                new_angle = max(ll_range[0], min(ll_range[1], new_angle))
                self.send_servo(ch, new_angle, ll_range)
                current_angle = new_angle
                print(f"  [LowerLip Iter {iteration:3d}] 移动 A19={new_angle}° "
                      f"(LLR Δ={delta:+.4f}, Δangle={move:+.1f})")

                # ---- 等待 ----
                _waited = 0
                while _waited < wait_seconds:
                    time.sleep(min(sleep_chunk, wait_seconds - _waited))
                    _waited += sleep_chunk
                    self.scorer.update_hud([
                        (f"[LowerLip Iter {iteration}/{max_iterations}]", (10, 100), (200, 200, 200)),
                        (f"Waiting... {_waited:.1f}/{wait_seconds}s", (10, 118), (0, 255, 255)),
                    ])
                    if self.scorer.user_pressed_stop:
                        break

            # 达到最大迭代未通过
            print(f"\n{'='*60}")
            print(f"  ✘ 下唇未在 {max_iterations} 次迭代内通过")
            print(f"{'='*60}\n")
            return False, iteration, {}

        finally:
            if not keep_display:
                self.scorer.stop_display()

    # ==================================================================
    # 多帧采集 + 修剪平均 — 嘴角偏移 (A22-A25)
    # ==================================================================

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
            print("[WARN] 未加载嘴角基线! 请在 ellseg_scorer.py 中按 [C] 保存嘴角基线到 mouth_corners_baseline.json")
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

    # ==================================================================
    # 引导式自动调整 — 上唇 (A16)
    # ==================================================================

    def auto_adjust_upper_lip(
        self,
        ulr_step: float = 1.0,
        max_iterations: int = None,
        wait_seconds: float = None,
        keep_display: bool = False,
    ) -> tuple:
        """
        引导式自动调整 — 只调整 A16 中上嘴唇舵机。

        流程（每轮迭代）:
          1. 打开 MP → 采集 60 帧 ULR → 关闭 MP
          2. 若 ULR 偏差 ≤ 容差 → 通过
          3. 否则移动 A16 → 等待 → 下一轮

        Args:
            ulr_step:       每次调整的舵机角度步长 (默认 1°)
            max_iterations: 最大迭代次数
            wait_seconds:   每轮等待秒数
            keep_display:   完成后是否保持窗口

        Returns:
            (passed: bool, iterations: int, final_data: dict)
        """
        if max_iterations is None:
            max_iterations = UPPER_LIP_MAX_ITERATIONS
        if wait_seconds is None:
            wait_seconds = UPPER_LIP_WAIT_SECONDS

        if self.scorer.upper_lip_baseline is None:
            print("[WARN] 未加载上唇基线! 请在 ellseg_scorer.py 中按 [U] 保存上唇基线到 upper_lip_baseline.json")
            return False, 0, {}

        tol = UPPER_LIP_ULR_TOLERANCE

        print(f"{'='*60}")
        print(f"  引导式自动调整 (仅中上嘴唇 A16)")
        print(f"  采集: 60帧, 修剪上下10帧, 中间40帧取平均 ULR")
        print(f"  步长: {ulr_step}度/次 | 最大迭代: {max_iterations} | 等待: {wait_seconds}s")
        print(f"  目标: ULR 偏差 ≤ {tol}")
        bl = self.scorer.upper_lip_baseline
        print(f"  基线: ULR={bl['ulr']:.4f}")
        print(f"{'='*60}\n")

        self.scorer.start_display()

        # 获取 A16 的范围
        ch = UPPER_LIP_CHANNEL
        start = self.controller.list_start_deg[ch]
        end = self.controller.list_end_deg[ch]
        ul_range = (min(start, end), max(start, end))
        ul_sign = getattr(self, '_upper_lip_flip', 1)   # 默认: ULR小→增大角度

        current_angle = self.controller.list_temp_deg[ch]
        print(f"  初始角度: A16={current_angle}° (range=[{ul_range[0]}, {ul_range[1]}])")

        iteration = 0
        sleep_chunk = 0.1

        try:
            while iteration < max_iterations:
                iteration += 1

                if self.scorer.user_pressed_stop:
                    print("\n[USER] 收到停止信号")
                    break

                self.scorer.update_hud([
                    (f"[UpperLip Iter {iteration}/{max_iterations}]", (10, 100), (200, 200, 200)),
                ])

                # ---- 采集 ULR ----
                avg_signal = self.collect_upper_lip_samples(n_frames=60, trim=10)

                if avg_signal is None:
                    print(f"  [UpperLip Iter {iteration:3d}] 无法获取有效帧")
                    _waited = 0
                    while _waited < wait_seconds:
                        time.sleep(min(sleep_chunk, wait_seconds - _waited))
                        _waited += sleep_chunk
                        if self.scorer.user_pressed_stop:
                            break
                    continue

                ulr = avg_signal["ulr"]
                delta = avg_signal["delta"]

                print(f"  [UpperLip Iter {iteration:3d}] ULR={ulr:.4f} (Δ={delta:+.4f})")

                # ---- 检查是否已合格 ----
                if abs(delta) <= tol:
                    print(f"\n{'='*60}")
                    print(f"  ★★ 上唇调整通过! 共迭代 {iteration} 次")
                    print(f"      ULR={ulr:.4f} (基线={bl['ulr']:.4f})")
                    print(f"      A16={current_angle}°")
                    print(f"{'='*60}\n")

                    # 保存结果
                    all_angles = [self.controller.list_temp_deg[c] for c in EYE_SERVO_CHANNELS]
                    self.save_result(all_angles)
                    chin_angle = self._temp_deg(MOUTH_CHIN_CHANNEL)
                    ll_angle = self._temp_deg(LOWER_LIP_CHANNEL)
                    self.export_angle_config(all_angles,
                                            chin_angle=chin_angle,
                                            lower_lip_angle=ll_angle,
                                            upper_lip_angle=current_angle)

                    self.scorer.update_hud([("★★ UPPER LIP PASSED ★★", (100, 80), (0, 255, 0))])
                    time.sleep(1)
                    return True, iteration, avg_signal

                # ---- 移动 A16 ----
                # delta>0 → ULR太大(上唇太展) → 合(-1); delta<0 → ULR太小(上唇太收) → 开(+1)
                delta_sign = -1 if delta > tol else (1 if delta < -tol else 0)
                move = ul_sign * delta_sign * ulr_step
                new_angle = round(current_angle + move)
                new_angle = max(ul_range[0], min(ul_range[1], new_angle))
                self.send_servo(ch, new_angle, ul_range)
                current_angle = new_angle
                print(f"  [UpperLip Iter {iteration:3d}] 移动 A16={new_angle}° "
                      f"(ULR Δ={delta:+.4f}, Δangle={move:+.1f})")

                # ---- 等待 ----
                _waited = 0
                while _waited < wait_seconds:
                    time.sleep(min(sleep_chunk, wait_seconds - _waited))
                    _waited += sleep_chunk
                    self.scorer.update_hud([
                        (f"[UpperLip Iter {iteration}/{max_iterations}]", (10, 100), (200, 200, 200)),
                        (f"Waiting... {_waited:.1f}/{wait_seconds}s", (10, 118), (0, 255, 255)),
                    ])
                    if self.scorer.user_pressed_stop:
                        break

            # 达到最大迭代未通过
            print(f"\n{'='*60}")
            print(f"  ✘ 上唇未在 {max_iterations} 次迭代内通过")
            print(f"{'='*60}\n")
            return False, iteration, {}

        finally:
            if not keep_display:
                self.scorer.stop_display()

    # ==================================================================
    # 结果保存
    # ==================================================================

    def save_result(self, angles: List[int], region_scores: dict = None,
                    save_path: str = "best_eye_config.json"):
        """保存当前角度结果到 JSON"""
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_path)
        result = TuningResult(
            best_angles=angles,
            best_score=region_scores.get("average", 0) if region_scores else 0,
            best_eye_score=region_scores.get("average", 0) if region_scores else 0,
            iteration=0,
            total_iterations=0,
            score_history=[],
            region_scores=region_scores or {},
        )
        result.save(save_path)
        print(f"[INFO] 结果已保存 → {save_path}")
        return result

    def export_angle_config(self, angles: List[int], eyebrow_angles: dict = None,
                            chin_angle: int = None, lower_lip_angle: int = None,
                            upper_lip_angle: int = None, corner_angles: dict = None):
        """导出角度配置到 eye_angles_best.json"""
        config = {
            "description": "EyeAutoTuner 眼睛舵机角度配置",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "servo_angles": {
                name: {"channel": ch, "angle": ang}
                for name, ch, ang in zip(EYE_SERVO_NAMES, EYE_SERVO_CHANNELS, angles)
            },
            "quick_apply": {
                "indexes": list(EYE_SERVO_CHANNELS),
                "angles": angles,
            }
        }
        # 眉毛通道 (如有)
        if eyebrow_angles:
            config["eyebrow_angles"] = {
                EYEBROW_NAMES[i]: {"channel": EYEBROW_CHANNELS[i], "angle": eyebrow_angles[c]}
                for i, c in enumerate(EYEBROW_CHANNELS)
            }
            config["quick_apply"]["indexes"] = config["quick_apply"]["indexes"] + list(EYEBROW_CHANNELS)
            config["quick_apply"]["angles"] = config["quick_apply"]["angles"] + [
                eyebrow_angles[c] for c in EYEBROW_CHANNELS
            ]
        # 下巴通道 (如有)
        if chin_angle is not None:
            config["chin_angle"] = {"channel": MOUTH_CHIN_CHANNEL, "angle": chin_angle}
            config["quick_apply"]["indexes"].append(MOUTH_CHIN_CHANNEL)
            config["quick_apply"]["angles"].append(chin_angle)
        # 下唇通道 (如有)
        if lower_lip_angle is not None:
            config["lower_lip_angle"] = {"channel": LOWER_LIP_CHANNEL, "angle": lower_lip_angle}
            config["quick_apply"]["indexes"].append(LOWER_LIP_CHANNEL)
            config["quick_apply"]["angles"].append(lower_lip_angle)
        # 上唇通道 (如有)
        if upper_lip_angle is not None:
            config["upper_lip_angle"] = {"channel": UPPER_LIP_CHANNEL, "angle": upper_lip_angle}
            config["quick_apply"]["indexes"].append(UPPER_LIP_CHANNEL)
            config["quick_apply"]["angles"].append(upper_lip_angle)
        # 嘴角通道 (如有)
        if corner_angles:
            config["corner_angles"] = corner_angles
            for ch_data in corner_angles.values():
                if isinstance(ch_data, dict):
                    config["quick_apply"]["indexes"].append(ch_data["channel"])
                    config["quick_apply"]["angles"].append(ch_data["angle"])
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "eye_angles_best.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"[INFO] 角度配置已导出 → {path}")

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

    # ==================================================================
    # 清理
    # ==================================================================

    def cleanup(self):
        """清理资源"""
        if self.scorer:
            try:
                self.scorer.close()  # 内部会 stop_display
            except Exception:
                pass
        if self.controller:
            try:
                self.controller.close()
            except Exception:
                pass
        print("[INFO] 资源已释放")


# ==================================================================
# CLI 入口
# ==================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="眼睛舵机调整工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
运行模式:
  python eye_auto_tuner.py                  # 默认: 自动调整眼球 (A10-A13)
  python eye_auto_tuner.py --mode eyeball   # 同上
  python eye_auto_tuner.py --mode eyelid    # 自动调整眼皮 (A8-A9)
  python eye_auto_tuner.py --mode eyebrow   # 自动调整眉毛 (A0-A1)
        """,
    )

    parser.add_argument("--mode", type=str, default="eyeball",
                       choices=["eyeball", "eyelid", "eyebrow"],
                       help="调整模式: eyeball=眼球(A10-A13), eyelid=眼皮(A8-A9), eyebrow=眉毛(A0-A1) (默认: eyeball)")
    parser.add_argument("--port", type=str, default="COM5",
                       help="串口端口号 (默认: COM5)")
    parser.add_argument("--baud", type=int, default=115200,
                       help="波特率 (默认: 115200)")
    parser.add_argument("--stabilize", type=int, default=8,
                       help="每次采集稳定帧数 (默认: 8)")
    parser.add_argument("--settle", type=int, default=300,
                       help="舵机等待时间ms (默认: 300)")

    # 眼球模式参数
    parser.add_argument("--ratio", type=float, default=2.0,
                       help="像素→角度换算系数 (默认: 2.0 deg/px)")
    parser.add_argument("--max-iter", type=int, default=50,
                       help="自动调整最大迭代次数 (默认: 50)")
    parser.add_argument("--wait", type=float, default=1.0,
                       help="每次移动后等待秒数 (默认: 1.0)")

    # 眼皮模式参数
    parser.add_argument("--ear-step", type=float, default=1.0,
                       help="眼皮每次调整步长(deg) (默认: 1.0)")
    parser.add_argument("--eyelid-max-iter", type=int, default=300,
                       help="眼皮调整最大迭代次数 (默认: 30)")
    parser.add_argument("--eyelid-wait", type=float, default=1.0,
                       help="眼皮每轮等待秒数 (默认: 1.0)")
    parser.add_argument("--eyelid-a8-flip", action="store_true",
                       help="翻转 A8 舵机方向符号")
    parser.add_argument("--eyelid-a9-flip", action="store_true",
                       help="翻转 A9 舵机方向符号")

    # 眉毛模式参数
    parser.add_argument("--ebhr-step", type=float, default=1.0,
                       help="眉毛每次调整步长(deg) (默认: 1.0)")
    parser.add_argument("--eyebrow-max-iter", type=int, default=100,
                       help="眉毛调整最大迭代次数 (默认: 30)")
    parser.add_argument("--eyebrow-wait", type=float, default=1.0,
                       help="眉毛每轮等待秒数 (默认: 1.0)")
    parser.add_argument("--eyebrow-a0-flip", action="store_true",
                       help="翻转 A0 舵机方向符号")
    parser.add_argument("--eyebrow-a1-flip", action="store_true",
                       help="翻转 A1 舵机方向符号")

    # 嘴部下巴模式参数
    parser.add_argument("--mar-step", type=float, default=1.0,
                       help="下巴每次调整步长(deg) (默认: 1.0)")
    parser.add_argument("--mouth-max-iter", type=int, default=100,
                       help="下巴调整最大迭代次数 (默认: 30)")
    parser.add_argument("--mouth-wait", type=float, default=1.0,
                       help="下巴每轮等待秒数 (默认: 1.0)")
    parser.add_argument("--mouth-chin-flip", action="store_true",
                       help="翻转 A26 舵机方向符号")

    # 下唇模式参数
    parser.add_argument("--ll-step", type=float, default=1.0,
                       help="下唇每次调整步长(deg) (默认: 1.0)")
    parser.add_argument("--ll-max-iter", type=int, default=100,
                       help="下唇调整最大迭代次数 (默认: 30)")
    parser.add_argument("--ll-wait", type=float, default=1.0,
                       help="下唇每轮等待秒数 (默认: 1.0)")
    parser.add_argument("--ll-flip", action="store_true",
                       help="翻转 A19 舵机方向符号")

    # 嘴角模式参数 (A22-A25)
    parser.add_argument("--corner-step", type=float, default=1.0,
                       help="嘴角每次调整步长(deg) (默认: 1.0)")
    parser.add_argument("--corner-max-iter", type=int, default=100,
                       help="嘴角调整最大迭代次数 (默认: 30)")
    parser.add_argument("--corner-wait", type=float, default=1.0,
                       help="嘴角每轮等待秒数 (默认: 1.0)")
    parser.add_argument("--corner-horiz-flip", action="store_true",
                       help="翻转嘴角水平对(A22+A24)方向符号")
    parser.add_argument("--corner-vert-flip", action="store_true",
                       help="翻转嘴角垂直对(A23+A25)方向符号")

    # 上唇模式参数
    parser.add_argument("--ulr-step", type=float, default=1.0,
                       help="上唇每次调整步长(deg) (默认: 1.0)")
    parser.add_argument("--ul-max-iter", type=int, default=100,
                       help="上唇调整最大迭代次数 (默认: 30)")
    parser.add_argument("--ul-wait", type=float, default=1.0,
                       help="上唇每轮等待秒数 (默认: 1.0)")
    parser.add_argument("--ul-flip", action="store_true",
                       help="翻转 A16 舵机方向符号")

    args = parser.parse_args()

    mode = args.mode
    mode_desc = {"eyeball": "A10-A13 眼球", "eyelid": "A8-A9 眼皮", "eyebrow": "A0-A1 眉毛"}[mode]

    print(f"""
+==============================================================+
||       Eye Auto Tuner - 眼睛舵机调整工具                        |
+==============================================================+
||  功能: 引导式自动调整 ({mode_desc})                         |
||  流程: 采集60帧→去头尾求平均→关检测→移舵机→等1s→循环       |
||  Keys: [Q/Esc] Stop                                         |
+==============================================================+
""")

    tuner = EyeAutoTuner(
        yaml_file="29_servo_config(13).yaml",
        port=args.port,
        baudrate=args.baud,
        stabilize_frames=args.stabilize,
        settle_time_ms=args.settle,
    )

    try:
        tuner.initialize()

        if mode == "eyeball":
            has_upper_lip_bl = tuner.scorer.upper_lip_baseline is not None
            has_mouth_bl = tuner.scorer.mouth_baseline is not None
            has_corners_bl = tuner.scorer.mouth_corners_baseline is not None
            has_lower_lip_bl = tuner.scorer.lower_lip_baseline is not None
            has_eyebrow_bl = tuner.scorer.eyebrow_baseline is not None
            has_eyelid_bl = tuner.scorer.eyelid_baseline is not None
            has_after = has_upper_lip_bl or has_mouth_bl or has_corners_bl \
                        or has_lower_lip_bl or has_eyebrow_bl or has_eyelid_bl
            passed, iterations, _ = tuner.auto_adjust(
                pixel_to_degree=args.ratio,
                max_iterations=args.max_iter,
                wait_seconds=args.wait,
                keep_display=has_after,
            )
            if passed:
                print(f"\n[眼球完成] 共 {iterations} 次迭代，固定眼球舵机。")

                # ---- 上唇调整 (A16) ----
                passed_ul = False
                iter_ul = 0
                if not has_upper_lip_bl:
                    print("[WARN] 未加载上唇基线，跳过上唇调整。")
                else:
                    print("\n>>> 自动进入上唇调整 (A16) ...\n")
                    tuner._upper_lip_flip = 1 if args.ul_flip else 1
                    passed_ul, iter_ul, _ = tuner.auto_adjust_upper_lip(
                        ulr_step=args.ulr_step,
                        max_iterations=args.ul_max_iter,
                        wait_seconds=args.ul_wait,
                        keep_display=has_mouth_bl or has_corners_bl or has_lower_lip_bl or has_eyebrow_bl or has_eyelid_bl,
                    )
                    if passed_ul:
                        print(f"[上唇完成] A16, {iter_ul}次")
                    else:
                        print(f"[上唇未通过] A16, {iter_ul}次")

                # ---- 下巴调整 (A26) ----
                passed_chin = False
                iter_chin = 0
                if not has_mouth_bl:
                    print("[WARN] 未加载嘴部基线，跳过下巴调整。")
                else:
                    print("\n>>> 自动进入下巴调整 (A26) ...\n")
                    tuner._mouth_chin_flip = -1 if args.mouth_chin_flip else 1
                    passed_chin, iter_chin, _ = tuner.auto_adjust_mouth_chin(
                        mar_step=args.mar_step,
                        max_iterations=args.mouth_max_iter,
                        wait_seconds=args.mouth_wait,
                        keep_display=has_corners_bl or has_lower_lip_bl or has_eyebrow_bl or has_eyelid_bl,
                    )
                    if passed_chin:
                        print(f"[下巴完成] A26, {iter_chin}次")
                    else:
                        print(f"[下巴未通过] A26, {iter_chin}次")

                # ---- 嘴角调整 (A22-A25) ----
                passed_corner = False
                iter_corner = 0
                if not has_corners_bl:
                    print("[WARN] 未加载嘴角基线，跳过嘴角调整。")
                else:
                    print("\n>>> 自动进入嘴角调整 (A22-A25) ...\n")
                    tuner._corner_horiz_flip = args.corner_horiz_flip
                    tuner._corner_vert_flip = args.corner_vert_flip
                    passed_corner, iter_corner, _ = tuner.auto_adjust_mouth_corners(
                        step=args.corner_step,
                        max_iterations=args.corner_max_iter,
                        wait_seconds=args.corner_wait,
                        keep_display=has_lower_lip_bl or has_eyebrow_bl or has_eyelid_bl,
                    )
                    if passed_corner:
                        print(f"[嘴角完成] A22-A25, {iter_corner}次")
                    else:
                        print(f"[嘴角未通过] A22-A25, {iter_corner}次")

                # ---- 下唇调整 (A19) ----
                passed_ll = False
                iter_ll = 0
                if not has_lower_lip_bl:
                    print("[WARN] 未加载下唇基线，跳过下唇调整。")
                else:
                    print("\n>>> 自动进入下唇调整 (A19) ...\n")
                    tuner._lower_lip_flip = 1 if args.ll_flip else -1
                    passed_ll, iter_ll, _ = tuner.auto_adjust_lower_lip(
                        step=args.ll_step,
                        max_iterations=args.ll_max_iter,
                        wait_seconds=args.ll_wait,
                        keep_display=has_eyebrow_bl or has_eyelid_bl,
                    )
                    if passed_ll:
                        print(f"[下唇完成] A19, {iter_ll}次")
                    else:
                        print(f"[下唇未通过] A19, {iter_ll}次")

                # ---- 眉毛调整 (A0-A1) ----
                passed_eb = False
                iter_eb = 0
                if not has_eyebrow_bl:
                    print("[WARN] 未加载眉毛基线，跳过眉毛调整。")
                else:
                    print("\n>>> 自动进入眉毛调整 (A0-A1) ...\n")
                    tuner._eyebrow_a0_flip = 1 if args.eyebrow_a0_flip else -1
                    tuner._eyebrow_a1_flip = 1 if args.eyebrow_a1_flip else -1
                    passed_eb, iter_eb, _ = tuner.auto_adjust_eyebrow(
                        ebhr_step=args.ebhr_step,
                        max_iterations=args.eyebrow_max_iter,
                        wait_seconds=args.eyebrow_wait,
                        keep_display=has_eyelid_bl,
                    )
                    if passed_eb:
                        print(f"[眉毛完成] A0-A1, {iter_eb}次")
                    else:
                        print(f"[眉毛未通过] A0-A1, {iter_eb}次")

                # ---- 眼皮调整 (A8-A9) ----
                passed_el = False
                iter_el = 0
                if not has_eyelid_bl:
                    print("[WARN] 未加载眼皮基线，跳过眼皮调整。")
                else:
                    print("\n>>> 自动进入眼皮调整 (A8-A9) ...\n")
                    tuner._eyelid_a8_flip = 1 if args.eyelid_a8_flip else -1
                    tuner._eyelid_a9_flip = -1 if args.eyelid_a9_flip else 1
                    passed_el, iter_el, _ = tuner.auto_adjust_eyelid(
                        ear_step=args.ear_step,
                        max_iterations=args.eyelid_max_iter,
                        wait_seconds=args.eyelid_wait,
                        keep_display=True,
                    )
                    if passed_el:
                        print(f"[眼皮完成] A8-A9, {iter_el}次")
                    else:
                        print(f"[眼皮未通过] A8-A9, {iter_el}次")

                # ---- 汇总 ----
                parts = ["眼球"]
                iters = [iterations]
                if has_upper_lip_bl:
                    parts.append("上唇"); iters.append(iter_ul)
                if has_mouth_bl:
                    parts.append("下巴"); iters.append(iter_chin)
                if has_corners_bl:
                    parts.append("嘴角"); iters.append(iter_corner)
                if has_lower_lip_bl:
                    parts.append("下唇"); iters.append(iter_ll)
                if has_eyebrow_bl:
                    parts.append("眉毛"); iters.append(iter_eb)
                if has_eyelid_bl:
                    parts.append("眼皮"); iters.append(iter_el)
                all_ok = passed and (not has_upper_lip_bl or passed_ul) \
                         and (not has_mouth_bl or passed_chin) \
                         and (not has_corners_bl or passed_corner) \
                         and (not has_lower_lip_bl or passed_ll) \
                         and (not has_eyebrow_bl or passed_eb) \
                         and (not has_eyelid_bl or passed_el)
                iter_str = " ".join(f"{p}({i}次)" for p, i in zip(parts, iters))
                print(f"\n{'All DONE!' if all_ok else '部分完成'}: {iter_str}")

                # ---- 保存全量最优配置 ----
                tuner.export_full_best_config()

                # 完成，等用户按 Q 退出
                print("\n[提示] 所有调整完成，按 Q 退出...")
                tuner.scorer._user_key = None
                while not tuner.scorer.user_pressed_stop:
                    tuner.scorer.update_hud([("★★ ALL DONE ★★  Press Q to quit", (100, 60), (0, 255, 0))])
                    time.sleep(0.1)
                return
            else:
                print(f"\n调整未通过 (迭代 {iterations} 次)。")
        elif mode == "eyelid":
            tuner._eyelid_a8_flip = 1 if args.eyelid_a8_flip else -1
            tuner._eyelid_a9_flip = -1 if args.eyelid_a9_flip else 1
            passed, iterations, _ = tuner.auto_adjust_eyelid(
                ear_step=args.ear_step,
                max_iterations=args.eyelid_max_iter,
                wait_seconds=args.eyelid_wait,
            )
            if passed:
                print(f"\nDone! 调整通过，共 {iterations} 次迭代。")
            else:
                print(f"\n调整未通过 (迭代 {iterations} 次)。")
        elif mode == "eyebrow":
            tuner._eyebrow_a0_flip = 1 if args.eyebrow_a0_flip else -1
            tuner._eyebrow_a1_flip = 1 if args.eyebrow_a1_flip else -1
            passed, iterations, _ = tuner.auto_adjust_eyebrow(
                ebhr_step=args.ebhr_step,
                max_iterations=args.eyebrow_max_iter,
                wait_seconds=args.eyebrow_wait,
            )
            if passed:
                print(f"\nDone! 眉毛调整通过，共 {iterations} 次迭代。")
            else:
                print(f"\n眉毛调整未通过 (迭代 {iterations} 次)。")

    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] 用户中断")
    except Exception as e:
        print(f"\n[ERROR] 出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        tuner.cleanup()


if __name__ == "__main__":
    main()
