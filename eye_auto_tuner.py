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
from servo_motion_rules import ServoMotionRules, DEFAULT_SERVO_MOTION_RULES_FILE
from eye_constants import (
    EYE_SERVO_CHANNELS, EYE_SERVO_NAMES,
    EYEBALL_CHANNELS, EYELID_CHANNELS,     EYEBROW_CHANNELS, EYEBROW_NAMES,
    EYEBALL_LOWER_LID_CONTACT_ARC_MIN_RATIO,
    EYEBALL_LOWER_LID_CONTACT_SAMPLE_FRAMES,
    EYEBALL_LOWER_LID_CONTACT_TOLERANCE,
    EYEBALL_LOWER_LID_CONTACT_TRIM_FRAMES,
    EYEBROW_CUSTOM_BIG_TOLERANCE,
    EYEBROW_BIG_SYMMETRY_TOLERANCE,
    EYEBROW_WAIT_SECONDS, EYEBROW_MAX_ITERATIONS,
    DEFAULT_ANGLE_MIN, DEFAULT_ANGLE_MAX,
    CAMERA_INDEX, DEFAULT_SERVO_CONFIG_FILE, FLIP_HORIZONTAL, FLIP_VERTICAL,
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
    MOUTH_CORNER_CURL_MIN_TARGET,
    MOUTH_CORNER_CURL_SYMMETRY_TOLERANCE,
    MOUTH_CORNER_CURL_TARGET,
    MOUTH_CORNER_CURL_TOLERANCE,
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
        yaml_file: str = DEFAULT_SERVO_CONFIG_FILE,
        port: str = "COM5",
        baudrate: int = 115200,
        servo_num: int = 27,
        stabilize_frames: int = 8,
        settle_time_ms: int = 300,
        detector: EllSegDetector = None,
        motion_rules_file: str = DEFAULT_SERVO_MOTION_RULES_FILE,
    ):
        self.yaml_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), yaml_file)
        self.port = port
        self.baudrate = baudrate
        self.servo_num = servo_num
        self.stabilize_frames = stabilize_frames
        self.settle_time_ms = settle_time_ms
        self.motion_rules_file = motion_rules_file
        self.motion_rules = ServoMotionRules(motion_rules_file)

        # 外部传入的 detector（qt_calibration_panel 共享相机）
        self.external_detector = detector

        # 组件引用（延迟初始化）
        self.controller = None
        self.scorer = None
        self._owns_scorer = False
        self.angle_ranges = [(DEFAULT_ANGLE_MIN, DEFAULT_ANGLE_MAX)] * len(EYE_SERVO_CHANNELS)
        self.side_cap = None
        self._owns_side_cap = False
        # Qt 面板可注入回调，实时接收最后成功发送的全部舵机角度。
        self.servo_values_callback = None

    def _motion_delta(self, channel: int, desired_effect: str, step: float = 1.0) -> float:
        return self.motion_rules.angle_delta(channel, desired_effect, step)

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
                flip_vertical=FLIP_VERTICAL,
                stabilize_frames=self.stabilize_frames,
            )
            self._owns_scorer = True

        print("\n" + "=" * 60)
        print("  初始化完成！")
        print("=" * 60 + "\n")

    # ==================================================================
    # 舵机操作
    # ==================================================================

    def open_serial(self) -> bool:
        if self.controller is None:
            print("[SERVO][FAIL] controller is not initialized")
            return False
        ok = self.controller.open()
        if ok:
            print(f"[SERVO] opened {self.port}")
        else:
            print(f"[SERVO][FAIL] cannot open {self.port}")
        return bool(ok)

    def close_serial(self) -> None:
        if self.controller is None:
            return
        self.controller.close()
        print(f"[SERVO] closed {self.port}")

    def send_servo(self, channel: int, angle: int,
                   angle_range: Tuple[int, int] = None) -> bool:
        """发送单个舵机角度，自动 clamp 到范围，并同步当前内存角度。"""
        if angle_range:
            angle = max(angle_range[0], min(angle_range[1], angle))
        ok = self.controller.set_servo_angle_time_32(
            [angle], [channel], 200, servo_num=self.controller.servo_num
        )
        if not ok:
            print(f"[SERVO][FAIL] A{channel}={angle} port={self.port}")
            return False
        # 记录已发送角度，避免下一次 UI 手动/自动动作又从 YAML temp_deg 起步。
        if 0 <= channel < len(self.controller.list_temp_deg):
            self.controller.list_temp_deg[channel] = angle
        self._notify_servo_values()
        print(f"[SERVO][OK] A{channel}={angle} port={self.port}")
        return True

    def send_servos(self, channels: List[int], angles: List[int],
                    angle_ranges: List[Tuple[int, int]] = None) -> bool:
        """发送多个舵机角度"""
        clamped_angles = []
        for idx, (ch, ang) in enumerate(zip(channels, angles)):
            ar = angle_ranges[idx] if angle_ranges else None
            if ar:
                ang = max(ar[0], min(ar[1], ang))
            clamped_angles.append(ang)

        ok = self.controller.set_servo_angle_time_32(
            clamped_angles, channels, 200, servo_num=self.controller.servo_num
        )
        items = " ".join(f"A{ch}={ang}" for ch, ang in zip(channels, clamped_angles))
        if not ok:
            print(f"[SERVO][FAIL] {items} port={self.port}")
            return False

        for ch, ang in zip(channels, clamped_angles):
            if 0 <= ch < len(self.controller.list_temp_deg):
                self.controller.list_temp_deg[ch] = ang
        self._notify_servo_values()
        print(f"[SERVO][OK] {items} port={self.port}")
        return True

    def _notify_servo_values(self) -> None:
        callback = getattr(self, "servo_values_callback", None)
        if callback is None or self.controller is None:
            return
        try:
            callback(list(self.controller.list_temp_deg))
        except Exception as exc:
            print(f"[WARN] servo value callback failed: {exc}")

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
        l_x_locked = False
        r_x_locked = False

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
                l_qualified = l_dist <= TOLERANCE
                r_qualified = r_dist <= TOLERANCE
                qualified = (l_qualified and r_qualified)

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
                if l_qualified:
                    l_adj_x = 0.0
                    l_adj_y = 0.0
                if r_qualified:
                    r_adj_x = 0.0
                    r_adj_y = 0.0
                if abs(l_dx) <= TOLERANCE:
                    l_x_locked = True
                if abs(r_dx) <= TOLERANCE:
                    r_x_locked = True
                if l_x_locked:
                    l_adj_x = 0.0
                if abs(l_dy) <= TOLERANCE:
                    l_adj_y = 0.0
                if r_x_locked:
                    r_adj_x = 0.0
                if abs(r_dy) <= TOLERANCE:
                    r_adj_y = 0.0

                # 方向由 servo_motion_rules.yaml 决定，只改舵机动作，不改 raw dx/dy。
                def effect_delta(ch, effect, value):
                    return self._motion_delta(ch, effect) if value != 0 else 0.0

                adj_map = {
                    EYEBALL_CHANNELS[0]: effect_delta(
                        EYEBALL_CHANNELS[0],
                        "上" if l_adj_y < 0 else "下",
                        l_adj_y,
                    ),
                    EYEBALL_CHANNELS[1]: effect_delta(
                        EYEBALL_CHANNELS[1],
                        "上" if r_adj_y < 0 else "下",
                        r_adj_y,
                    ),
                    EYEBALL_CHANNELS[2]: effect_delta(
                        EYEBALL_CHANNELS[2],
                        "内" if l_adj_x < 0 else "外",
                        l_adj_x,
                    ),
                    EYEBALL_CHANNELS[3]: effect_delta(
                        EYEBALL_CHANNELS[3],
                        "外" if r_adj_x < 0 else "内",
                        r_adj_x,
                    ),
                }

                move_channels = []
                move_angles = []
                move_ranges = []
                moved = []
                for idx, ch in enumerate(EYEBALL_CHANNELS):
                    delta = adj_map[ch]
                    if delta == 0:
                        continue
                    new_angle = round(current_angles[ch] + delta)
                    lo, hi = self.angle_ranges[EYE_SERVO_CHANNELS.index(ch)]
                    new_angle = max(lo, min(hi, new_angle))
                    move_channels.append(ch)
                    move_angles.append(new_angle)
                    move_ranges.append((lo, hi))
                    moved.append(f"{EYEBALL_NAMES[idx]}={new_angle}°(Δ{delta:+.1f})")

                if moved:
                    sent = self.send_servos(move_channels, move_angles, move_ranges)
                    if sent:
                        for ch, angle in zip(move_channels, move_angles):
                            current_angles[ch] = angle
                    print(f"  [Iter {iteration:3d}] Eyeball serial sent={sent}")

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

    def auto_adjust_iris_lid_contact(
        self,
        step: float = 1.0,
        max_iterations: int = 100,
        wait_seconds: float = 1.0,
        sample_frames: int = EYEBALL_LOWER_LID_CONTACT_SAMPLE_FRAMES,
        trim_frames: int = EYEBALL_LOWER_LID_CONTACT_TRIM_FRAMES,
        keep_display: bool = False,
        log_func=None,
    ) -> Tuple[bool, int, dict]:
        """仅用左右眼各自的虹膜下弧贴合值调整 A10/A11。

        L 只控制 A10，R 只控制 A11；两侧分别做修剪平均，不互相混合。
        某一侧第一次达到贴合条件后即锁定，后续迭代不再移动该侧。
        """
        sample_frames = max(3, int(sample_frames))
        trim_frames = max(0, int(trim_frames))
        if sample_frames <= 2 * trim_frames:
            raise ValueError("sample_frames 必须大于 2 * trim_frames")

        vertical_channels = {"left": EYEBALL_CHANNELS[0], "right": EYEBALL_CHANNELS[1]}
        side_labels = {"left": "L", "right": "R"}
        current_angles = {}
        for side, channel in vertical_channels.items():
            low, high = self._angle_range(channel)
            current_angles[side] = self._temp_deg(channel, default=(low + high) / 2.0)

        locked = {"left": False, "right": False}
        final_measurements = {"left": None, "right": None}
        stop_reason = "max_iterations"
        sleep_chunk = 0.1

        def emit(message: str) -> None:
            print(message)
            if log_func is not None:
                log_func(message)

        def wait_after_move() -> bool:
            waited = 0.0
            while waited < wait_seconds:
                if self.scorer.user_pressed_stop:
                    return False
                duration = min(sleep_chunk, wait_seconds - waited)
                time.sleep(duration)
                waited += duration
            return True

        def collect_trimmed_signal():
            """左右眼分别排序、裁掉两端异常帧，再计算各自平均值。"""
            required_sides = [side for side in vertical_channels if not locked[side]]
            samples = {side: [] for side in required_sides}
            max_attempts = max(sample_frames * 6, sample_frames + 20)

            for _attempt in range(max_attempts):
                if self.scorer.user_pressed_stop:
                    return None
                ok, frame = self.scorer.capture()
                if not ok:
                    time.sleep(0.01)
                    continue
                detection = self.scorer.detect(frame)
                signal = detection.get("lower_iris_cover") if detection else None
                if signal:
                    for side in required_sides:
                        if len(samples[side]) >= sample_frames:
                            continue
                        measurement = signal.get(side) or {}
                        offset = measurement.get("contact_offset_px")
                        arc_ratio = measurement.get("contact_arc_ratio")
                        if offset is None or arc_ratio is None:
                            continue
                        samples[side].append({
                            "offset": float(offset),
                            "arc_ratio": float(arc_ratio),
                            "measurement": measurement,
                        })

                    progress_lines = [
                        (f"[IrisLid AVG {iteration}/{max_iterations}]", (10, 100), (200, 200, 200)),
                    ]
                    for row, side in enumerate(vertical_channels, start=1):
                        if locked[side]:
                            progress = "LOCK"
                            color = (0, 255, 0)
                        else:
                            progress = f"sampling {len(samples[side])}/{sample_frames}"
                            color = (200, 200, 200)
                        progress_lines.append((
                            f"{side_labels[side]} {progress}",
                            (10, 100 + row * 18),
                            color,
                        ))
                    self.scorer.update_hud(progress_lines)

                    if all(len(samples[side]) >= sample_frames for side in required_sides):
                        break
                time.sleep(0.01)

            if any(len(samples[side]) < sample_frames for side in required_sides):
                return None

            averaged = {}
            for side in vertical_channels:
                if locked[side]:
                    averaged[side] = final_measurements[side] or {}
                    continue

                ordered = sorted(samples[side], key=lambda item: item["offset"])
                kept = ordered[trim_frames:len(ordered) - trim_frames] if trim_frames else ordered
                avg_offset = sum(item["offset"] for item in kept) / len(kept)
                avg_arc_ratio = sum(item["arc_ratio"] for item in kept) / len(kept)
                qualified = (
                    abs(avg_offset) <= EYEBALL_LOWER_LID_CONTACT_TOLERANCE
                    and avg_arc_ratio >= EYEBALL_LOWER_LID_CONTACT_ARC_MIN_RATIO
                )
                if qualified:
                    state = "FIT"
                elif avg_offset < -EYEBALL_LOWER_LID_CONTACT_TOLERANCE:
                    state = "GAP"
                elif avg_offset > EYEBALL_LOWER_LID_CONTACT_TOLERANCE:
                    state = "COVER"
                else:
                    state = "NARROW"

                representative = min(kept, key=lambda item: abs(item["offset"] - avg_offset))
                measurement = dict(representative["measurement"])
                measurement.update({
                    "contact_offset_px": avg_offset,
                    "contact_arc_ratio": avg_arc_ratio,
                    "qualified": qualified,
                    "state": state,
                    "sample_count": sample_frames,
                    "trim_count_each_end": trim_frames,
                    "averaged_count": len(kept),
                })
                averaged[side] = measurement
            return averaged

        self.scorer.enable_mp = True
        self.scorer.enable_ellseg = True
        self.scorer.show_iris_cover_overlay = True
        self.scorer.start_display()
        emit(
            f">>> 调整眼球上下贴合：L=A10，R=A11；左右独立修剪平均 "
            f"{sample_frames} 帧，去掉最低/最高各 {trim_frames} 帧 <<<"
        )

        iteration = 0
        try:
            while iteration < max_iterations:
                iteration += 1
                if self.scorer.user_pressed_stop:
                    stop_reason = "user_stop"
                    break

                signal = collect_trimmed_signal()
                if signal is None:
                    emit(
                        f"[眼球上下贴合 {iteration}/{max_iterations}] "
                        f"未收齐每侧 {sample_frames} 个有效帧，本轮不移动"
                    )
                    self.scorer.update_hud([
                        (f"[IrisLid AVG {iteration}/{max_iterations}]", (10, 100), (200, 200, 200)),
                        ("L: no signal", (10, 118), (0, 160, 255)),
                        ("R: no signal", (10, 136), (0, 160, 255)),
                    ])
                    if not wait_after_move():
                        stop_reason = "user_stop"
                        break
                    continue

                actions = {}
                hud_lines = [
                    (f"[IrisLid AVG {iteration}/{max_iterations}]", (10, 100), (200, 200, 200)),
                ]

                for side in vertical_channels:
                    measurement = signal.get(side) or {}
                    offset = measurement.get("contact_offset_px")
                    if offset is not None:
                        final_measurements[side] = measurement
                    if not locked[side] and bool(measurement.get("qualified")):
                        locked[side] = True
                        emit(
                            f"[眼球上下贴合] {side_labels[side]} 已到位并锁定："
                            f"修剪平均 {offset:+.2f}px，后续不再移动该侧"
                        )

                average_log = []
                for side in vertical_channels:
                    measurement = final_measurements[side] or {}
                    offset = measurement.get("contact_offset_px")
                    arc_ratio = measurement.get("contact_arc_ratio")
                    if offset is None or arc_ratio is None:
                        average_log.append(f"{side_labels[side]}=--")
                    else:
                        average_log.append(
                            f"{side_labels[side]}={offset:+.2f}px/"
                            f"{arc_ratio * 100.0:.0f}% {measurement.get('state', '--')}"
                            f"{' LOCK' if locked[side] else ''}"
                        )
                emit(
                    f"[眼球上下贴合 {iteration}/{max_iterations}] 修剪平均："
                    f"{' | '.join(average_log)}"
                )

                if all(locked.values()):
                    stop_reason = "qualified"
                    for row, side in enumerate(vertical_channels, start=1):
                        measurement = final_measurements[side] or {}
                        offset = measurement.get("contact_offset_px")
                        text_offset = "--" if offset is None else f"{offset:+.2f}px"
                        hud_lines.append((
                            f"{side_labels[side]} {text_offset} FIT A{vertical_channels[side]}={current_angles[side]} LOCK",
                            (10, 100 + row * 18),
                            (0, 255, 0),
                        ))
                    self.scorer.update_hud(hud_lines)
                    emit(f"[眼球上下贴合] 左右眼均已到位，共 {iteration} 次迭代")
                    return True, iteration, {
                        "left": final_measurements["left"],
                        "right": final_measurements["right"],
                        "locked": dict(locked),
                        "stop_reason": stop_reason,
                    }

                move_channels = []
                move_angles = []
                move_ranges = []

                for row, side in enumerate(vertical_channels, start=1):
                    channel = vertical_channels[side]
                    measurement = final_measurements[side] or {}
                    offset = measurement.get("contact_offset_px")
                    state = str(measurement.get("state", "--"))
                    color = (0, 255, 0) if locked[side] else (0, 255, 255)

                    if locked[side]:
                        action_text = "LOCK"
                    elif offset is None:
                        action_text = "NO SIGNAL"
                        color = (0, 160, 255)
                    elif offset < 0.0:
                        # 眼球下弧在眼皮上方：眼球需要向下贴近下眼皮。
                        actions[side] = "下"
                        action_text = "DOWN"
                    elif offset > 0.0:
                        # 下眼皮遮住眼球过多：眼球需要向上离开下眼皮。
                        actions[side] = "上"
                        action_text = "UP"
                    else:
                        # 偏差正好为 0 但接触弧仍不足，无法确定安全移动方向。
                        action_text = "NO DIR"
                        color = (0, 160, 255)

                    text_offset = "--" if offset is None else f"{offset:+.2f}px"
                    hud_lines.append((
                        f"{side_labels[side]} {text_offset} {state} A{channel}={current_angles[side]} {action_text}",
                        (10, 100 + row * 18),
                        color,
                    ))

                for side, desired_effect in actions.items():
                    channel = vertical_channels[side]
                    low, high = self._angle_range(channel)
                    delta = self._motion_delta(channel, desired_effect, step)
                    new_angle = max(low, min(high, round(current_angles[side] + delta)))
                    if new_angle == current_angles[side]:
                        emit(
                            f"[眼球上下贴合 {iteration}/{max_iterations}] {side_labels[side]} "
                            f"A{channel} 已到范围边界 {current_angles[side]}，仍未贴合"
                        )
                        continue
                    move_channels.append(channel)
                    move_angles.append(new_angle)
                    move_ranges.append((low, high))

                self.scorer.update_hud(hud_lines)

                if move_channels:
                    sent = self.send_servos(move_channels, move_angles, move_ranges)
                    if not sent:
                        stop_reason = "serial_send_failed"
                        emit(f"[眼球上下贴合 {iteration}/{max_iterations}] 舵机命令发送失败")
                        break
                    for channel, angle in zip(move_channels, move_angles):
                        side = "left" if channel == vertical_channels["left"] else "right"
                        current_angles[side] = angle
                    move_log = " ".join(
                        f"{side_labels[side]} A{vertical_channels[side]} {actions[side]} -> {current_angles[side]}"
                        for side in actions
                        if vertical_channels[side] in move_channels
                    )
                    emit(f"[眼球上下贴合 {iteration}/{max_iterations}] {move_log}")
                elif not all(locked.values()):
                    stop_reason = "no_legal_move"
                    emit("[眼球上下贴合] 未合格侧没有可确定的安全移动方向，停止调整")
                    break

                if not wait_after_move():
                    stop_reason = "user_stop"
                    break

            emit(
                f"[眼球上下贴合] 未通过：{iteration} 次迭代，"
                f"L={'到位' if locked['left'] else '未到位'}，"
                f"R={'到位' if locked['right'] else '未到位'}，原因={stop_reason}"
            )
            return False, iteration, {
                "left": final_measurements["left"],
                "right": final_measurements["right"],
                "locked": dict(locked),
                "stop_reason": stop_reason,
                "warning": "未合格侧已到舵机边界或当前贴合信号无法确定安全方向"
                if stop_reason == "no_legal_move" else "",
            }
        finally:
            # 调整结束后保持检测和摄像头运行；只按调用方要求决定是否关闭显示窗。
            self.scorer.enable_mp = True
            self.scorer.enable_ellseg = True
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
                    desired_effect = "上" if front_delta > UPPER_LIP_ULR_TOLERANCE else "下"
                    new_front = round(front_angle + self._motion_delta(ch_front, desired_effect, step))
                    new_front = max(front_range[0], min(front_range[1], new_front))
                    if new_front != front_angle:
                        front_angle = new_front
                        moved = True
                elif side_y_out:
                    # 正面达标但侧面 y 超差时允许 A16 小步补偿；若下一轮 ULR 出界，会优先按正面纠正。
                    desired_effect = "上" if side_dy > UPPER_LIP_SIDE_Y_TOLERANCE else "下"
                    new_front = round(front_angle + self._motion_delta(ch_front, desired_effect, step))
                    new_front = max(front_range[0], min(front_range[1], new_front))
                    if new_front != front_angle:
                        front_angle = new_front
                        moved = True
                        final_data["side_dy_front_adjust"] = {
                            "side_dy": side_dy,
                            "front_delta": front_delta,
                        }

                if not side_ok and abs(side_dx) > UPPER_LIP_SIDE_X_TOLERANCE:
                    desired_effect = "后" if side_dx > UPPER_LIP_SIDE_X_TOLERANCE else "前"
                    new_side = round(side_angle + self._motion_delta(ch_side, desired_effect, step))
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
        # 眼皮自动调整暂时不看对称度；保留原始数值，合格判断固定放行，便于以后恢复。
        avg["symmetry_qualified"] = True

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
        引导式自动调整 — 眼皮 A8/A9 舵机。

        左右眼皮各自回到自己的 EAR 基线通过区间即通过。
        对称度暂时不参与调整/通过判定。

        流程（每轮迭代）:
          1. 打开 MP（仅 MediaPipe 关键点）
          2. 采集 60 帧 → 去头尾求平均 EAR
          3. 关闭 MP
          4. P1: 若单侧 EAR 偏离基线容差 → 该侧单独开/合
          5. 左右各自达标 → 通过

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

        print(f"{'='*60}")
        print(f"  引导式自动调整 (仅眼皮 A8-A9) — 暂不看对称度")
        print(f"  目标: 各自达标 (|L_delta|、|R_delta| ≤ {tol})")
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

                print(f"  [Eyelid Iter {iteration:3d}] "
                      f"L_EAR={l_ear:.4f} (Δ={l_delta:+.4f}) "
                      f"R_EAR={r_ear:.4f} (Δ={r_delta:+.4f})")

                # ---- update_hud: 只显示调整专有信息，基本参数由 cv2.putText 常驻 ----
                phase_label = "各自达标"

                self.scorer.update_hud([
                    (f"[Eyelid {phase_label} Iter {iteration}/{max_iterations}]", (10, 100), (200, 200, 200)),
                    (f"A8={current_angles[EYELID_CHANNELS[0]]}° A9={current_angles[EYELID_CHANNELS[1]]}°", (10, 118), (200, 200, 200)),
                ])

                # ---- 检查左右各自是否合格；对称度暂时不参与 ----
                l_ok = abs(l_delta) <= tol
                r_ok = abs(r_delta) <= tol
                if l_ok and r_ok:
                    print(f"\n{'='*60}")
                    print(f"  ★★ 眼皮调整通过! 共迭代 {iteration} 次")
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
                # 左右各自达标 — 单侧 EAR 偏离基线就单独开/合
                # ================================================================
                moves = []
                if not l_ok:
                    moves.append(("L", EYELID_CHANNELS[0], EYELID_NAMES[0], l_delta))
                if not r_ok:
                    moves.append(("R", EYELID_CHANNELS[1], EYELID_NAMES[1], r_delta))

                moved_sides = []
                limit_notes = []
                action_parts = []
                for side, ch, name, deviation in moves:
                    # deviation > 0: 这侧比基线更开 → 合
                    # deviation < 0: 这侧比基线更闭 → 开
                    desired_effect = "合" if deviation > 0 else "开"
                    delta_angle = self._motion_delta(ch, desired_effect, ear_step)
                    old_angle = current_angles[ch]
                    lo, hi = self.angle_ranges[EYE_SERVO_CHANNELS.index(ch)]
                    new_angle = round(old_angle + delta_angle)
                    new_angle = max(lo, min(hi, new_angle))

                    if new_angle == old_angle:
                        limit_note = f"{name}到极限({old_angle}度)"
                        limit_notes.append(limit_note)
                        print(f"  [Eyelid Iter {iteration:3d}] {limit_note}")
                        continue

                    self.send_servo(ch, new_angle, (lo, hi))
                    current_angles[ch] = new_angle
                    moved_sides.append(side)
                    action_parts.append(f"{side}{desired_effect} Δ={deviation:+.4f}")
                    print(f"  [Eyelid Iter {iteration:3d}] {side}: {name}={new_angle}度 "
                          f"(delta={deviation:+.4f}, {desired_effect} to own range)")

                if not moved_sides:
                    print(f"  [Eyelid Iter {iteration:3d}] 未移动: 未通过侧已到极限")
                action_text = (
                    " ".join(action_parts)
                    if action_parts else "未通过侧到极限"
                )
                self.scorer.update_hud([
                    (f"[Eyelid {phase_label} Iter {iteration}/{max_iterations}]", (10, 100), (200, 200, 200)),
                    (action_text, (10, 118), (0, 255, 200)),
                    ("; ".join(limit_notes), (10, 136), (0, 180, 255)),
                ])

                # ---- 等待 ----
                _waited = 0
                while _waited < wait_seconds:
                    time.sleep(min(sleep_chunk, wait_seconds - _waited))
                    _waited += sleep_chunk
                    p_label = "各自达标"
                    self.scorer.update_hud([
                        (f"[Eyelid {p_label} Iter {iteration}/{max_iterations}]", (10, 100), (200, 200, 200)),
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
                        deviation = l_dev
                    else:
                        side = "R"
                        ch = EYEBROW_CHANNELS[1]
                        name = EYEBROW_NAMES[1]
                        deviation = r_dev

                    # deviation > 0 → 这侧更低 → 要抬高
                    # deviation < 0 → 这侧更高 → 要降低
                    desired_effect = "上" if deviation > 0 else "下"
                    delta_angle = self._motion_delta(ch, desired_effect, ebhr_step)
                    new_angle = round(current_angles[ch] + delta_angle)
                    lo, hi = self.eyebrow_ranges[ch]
                    new_angle = max(lo, min(hi, new_angle))
                    self.send_servo(ch, new_angle, (lo, hi))
                    current_angles[ch] = new_angle
                    print(f"  [Eyebrow P{phase} Iter {iteration:3d}] {side}: {name}={new_angle}deg "
                          f"(sym dev={deviation:+.4f}, {desired_effect} toward midpoint)")

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
                        # 偏低 → 两边同时抬高
                        desired_effect = "上"
                    elif avg_delta < 0:
                        # 偏高 → 两边同时降低
                        desired_effect = "下"
                    else:
                        desired_effect = None

                    for side, ch, name in [
                        ("L", EYEBROW_CHANNELS[0], EYEBROW_NAMES[0]),
                        ("R", EYEBROW_CHANNELS[1], EYEBROW_NAMES[1]),
                    ]:
                        delta_angle = self._motion_delta(ch, desired_effect, ebhr_step) if desired_effect else 0
                        new_angle = round(current_angles[ch] + delta_angle)
                        lo, hi = self.eyebrow_ranges[ch]
                        new_angle = max(lo, min(hi, new_angle))
                        self.send_servo(ch, new_angle, (lo, hi))
                        current_angles[ch] = new_angle

                    if desired_effect:
                        print(f"  [Eyebrow P{phase} Iter {iteration:3d}] Sync: L={current_angles[EYEBROW_CHANNELS[0]]}deg "
                              f"R={current_angles[EYEBROW_CHANNELS[1]]}deg "
                              f"(avg_delta={avg_delta:+.4f}, effect={desired_effect})")

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
                # delta>0 → MAR太大(嘴太开) → 合; delta<0 → MAR太小(嘴太闭) → 开。
                desired_effect = "合" if delta > tol else "开"
                move = self._motion_delta(ch, desired_effect, mar_step)
                new_angle = round(current_angle + move)
                new_angle = max(chin_range[0], min(chin_range[1], new_angle))
                self.send_servo(ch, new_angle, chin_range)
                current_angle = new_angle
                print(f"  [Mouth Iter {iteration:3d}] 移动 A26={new_angle}° "
                      f"(MAR Δ={delta:+.4f}, effect={desired_effect}, Δangle={move:+.1f})")

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
                    desired_effect = "上" if front_delta > LOWER_LIP_LLR_TOLERANCE else "下"
                    new_front = round(front_angle + self._motion_delta(ch_front, desired_effect, step))
                    new_front = max(front_range[0], min(front_range[1], new_front))
                    if new_front != front_angle:
                        front_angle = new_front
                        moved = True
                elif side_y_out:
                    # 正面达标但侧面 y 超差时允许 A19 小步补偿；若下一轮 LLR 出界，会优先按正面纠正。
                    desired_effect = "上" if side_dy > LOWER_LIP_SIDE_Y_TOLERANCE else "下"
                    new_front = round(front_angle + self._motion_delta(ch_front, desired_effect, step))
                    new_front = max(front_range[0], min(front_range[1], new_front))
                    if new_front != front_angle:
                        front_angle = new_front
                        moved = True
                        final_data["side_dy_front_adjust"] = {
                            "side_dy": side_dy,
                            "front_delta": front_delta,
                        }

                if not side_ok and abs(side_dx) > LOWER_LIP_SIDE_X_TOLERANCE:
                    desired_effect = "后" if side_dx > LOWER_LIP_SIDE_X_TOLERANCE else "前"
                    new_side = round(side_angle + self._motion_delta(ch_side, desired_effect, step))
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

    def collect_mouth_corner_curl_samples(self, n_frames: int = 60, trim: int = 10):
        """
        打开 MP → 采集嘴角相对本侧内口裂点的上翘比例 → 修剪平均。

        Returns:
            dict 或 None: {"left": {curl,curl_px,delta,qualified},
                           "right": {...},
                           "all_qualified": bool}
        """
        mp_was_on = self.scorer.enable_mp
        ellseg_was_on = self.scorer.enable_ellseg
        self.scorer.enable_mp = True
        self.scorer.enable_ellseg = False

        samples = []
        collected = 0
        no_face_count = 0

        try:
            while collected < n_frames:
                if self.scorer.user_pressed_stop:
                    break

                ok, frame = self.scorer.capture()
                if not ok:
                    time.sleep(0.001)
                    continue

                self.scorer.detect(frame)
                signal = self.scorer.get_mouth_corner_curl_signal()
                if signal is None:
                    no_face_count += 1
                    if no_face_count > 10:
                        break
                    time.sleep(0.001)
                    continue
                no_face_count = 0

                score = (
                    abs(signal["left"]["delta"]) + abs(signal["right"]["delta"])
                    + (max(0.0, signal["min_target"] - signal["left"]["curl"]) if signal.get("min_required", True) else 0.0)
                    + (max(0.0, signal["min_target"] - signal["right"]["curl"]) if signal.get("min_required", True) else 0.0)
                    + signal["symmetry_delta"]
                )
                samples.append((score, {
                    "left": dict(signal["left"]),
                    "right": dict(signal["right"]),
                    "mouth_width": signal["mouth_width"],
                    "min_target": signal["min_target"],
                    "min_required": signal.get("min_required", True),
                    "symmetry_delta": signal["symmetry_delta"],
                    "symmetry_tolerance": signal["symmetry_tolerance"],
                }))
                collected += 1

                self.scorer.update_hud([
                    (f"MouthCurl Sampling: {collected}/{n_frames}", (10, 120), (200, 200, 200)),
                ])
        finally:
            self.scorer.enable_mp = mp_was_on
            self.scorer.enable_ellseg = ellseg_was_on

        if len(samples) < 2 * trim + 1:
            print(f"  [MouthCurl Collect] 有效样本不足: {len(samples)}, 需要 >= {2*trim+1}")
            return None

        samples.sort(key=lambda x: x[0])
        trimmed = samples[trim:-trim]
        n = len(trimmed)

        avg = {"left": {}, "right": {}}
        min_required = any(s[1].get("min_required", True) for s in trimmed)
        for side in ("left", "right"):
            for key in ("curl", "curl_px", "delta", "target"):
                avg[side][key] = sum(s[1][side][key] for s in trimmed) / n
            avg[side]["min_qualified"] = (avg[side]["curl"] >= MOUTH_CORNER_CURL_MIN_TARGET) if min_required else True
            if min_required:
                avg[side]["target_qualified"] = abs(avg[side]["delta"]) <= MOUTH_CORNER_CURL_TOLERANCE
            else:
                avg[side]["target_qualified"] = round(avg[side]["curl"], 3) == round(avg[side]["target"], 3)
            avg[side]["qualified"] = avg[side]["min_qualified"] and avg[side]["target_qualified"]

        avg["mouth_width"] = sum(s[1]["mouth_width"] for s in trimmed) / n
        avg["min_target"] = sum(s[1]["min_target"] for s in trimmed) / n
        avg["min_required"] = min_required
        avg["tolerance"] = MOUTH_CORNER_CURL_TOLERANCE
        avg["symmetry_delta"] = abs(avg["left"]["curl"] - avg["right"]["curl"])
        avg["symmetry_tolerance"] = sum(s[1]["symmetry_tolerance"] for s in trimmed) / n
        avg["symmetry_qualified"] = avg["symmetry_delta"] <= MOUTH_CORNER_CURL_SYMMETRY_TOLERANCE
        avg["all_qualified"] = avg["left"]["qualified"] and avg["right"]["qualified"] and avg["symmetry_qualified"]

        all_scores = [s[0] for s in samples]
        print(f"  [MouthCurl Collect] {len(samples)} frames, trim {trim}: "
              f"L={avg['left']['curl']:+.3f} R={avg['right']['curl']:+.3f} "
              f"target=({avg['left']['target']:+.3f},{avg['right']['target']:+.3f}) "
              f"sym={avg['symmetry_delta']:.3f}<={avg['symmetry_tolerance']:.3f} "
              f"(min={all_scores[0]:.3f}, max={all_scores[-1]:.3f})")
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

        current = {ch: self._temp_deg(ch) for ch in [ch_Lh, ch_Rh, ch_Lv, ch_Rv]}

        print(f"{'='*60}")
        print(f"  嘴角两阶段调整 (A22-A25)")
        print(f"  Phase 1: 竖向 Y (A23+A25) ≤ {tol_v}px")
        print(f"  Phase 2: 水平 X (A24+A22) ≤ {tol_h}px")
        print(f"  步长: {step}°/次 | 最大迭代: {max_iterations} | 等待: {wait_seconds}s")
        bl = self.scorer.mouth_corners_baseline
        print(f"  基线: L=({bl['corner_left'][0]},{bl['corner_left'][1]}) "
              f"R=({bl['corner_right'][0]},{bl['corner_right'][1]})")
        print(f"  方向: servo_motion_rules.yaml 目标动作映射")
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
                        desired_effect = "上" if L_dy > 0 else "下"
                        delta_angle = self._motion_delta(ch_Lv, desired_effect, step)
                        self._move_one(ch_Lv, delta_angle, ranges, current, moved, "A23")
                    if not R_dy_ok:
                        desired_effect = "上" if R_dy > 0 else "下"
                        delta_angle = self._motion_delta(ch_Rv, desired_effect, step)
                        self._move_one(ch_Rv, delta_angle, ranges, current, moved, "A25")
                else:
                    # ---- Phase 2: 水平 ----
                    if not L_dx_ok:
                        desired_effect = "前" if L_dx > 0 else "后"
                        delta_angle = self._motion_delta(ch_Lh, desired_effect, step)
                        self._move_one(ch_Lh, delta_angle, ranges, current, moved, "A24")
                    if not R_dx_ok:
                        desired_effect = "后" if R_dx > 0 else "前"
                        delta_angle = self._motion_delta(ch_Rh, desired_effect, step)
                        self._move_one(ch_Rh, delta_angle, ranges, current, moved, "A22")

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

    def auto_adjust_mouth_corner_curl(
        self,
        step: float = 1.0,
        max_iterations: int = None,
        wait_seconds: float = None,
        keep_display: bool = False,
    ) -> tuple:
        """
        嘴角微翘自动调整：不匹配保存的嘴角像素点，只让嘴角相对本侧内口裂点达到目标上翘比例。

        只调整竖向舵机 A23/A25，水平 A24/A22 不参与。
        """
        if max_iterations is None:
            max_iterations = MOUTH_CORNER_MAX_ITERATIONS
        if wait_seconds is None:
            wait_seconds = MOUTH_CORNER_WAIT_SECONDS

        ch_Lv = MOUTH_CORNER_V_CHANNELS[0]   # A23
        ch_Rv = MOUTH_CORNER_V_CHANNELS[1]   # A25
        ranges = {
            ch: (min(self.controller.list_start_deg[ch], self.controller.list_end_deg[ch]),
                 max(self.controller.list_start_deg[ch], self.controller.list_end_deg[ch]))
            for ch in [ch_Lv, ch_Rv]
        }
        current = {ch: self._temp_deg(ch) for ch in [ch_Lv, ch_Rv]}

        print(f"{'='*60}")
        print("  嘴角微翘自动调整 (A23+A25)")
        curl_baseline = getattr(self.scorer, "mouth_corner_curl_baseline", None)
        if isinstance(curl_baseline, dict):
            left_target = float(curl_baseline.get('left_curl', MOUTH_CORNER_CURL_TARGET))
            right_target = float(curl_baseline.get('right_curl', MOUTH_CORNER_CURL_TARGET))
            print(f"  目标: L={left_target:+.3f} R={right_target:+.3f} "
                  f"(当前模板保存值，三位小数相等才通过)")
        else:
            print(f"  目标: curl={MOUTH_CORNER_CURL_TARGET:.3f} ± {MOUTH_CORNER_CURL_TOLERANCE:.3f} (默认)")
            print(f"  未保存模板时最低微翘: {MOUTH_CORNER_CURL_MIN_TARGET:.3f}")
        print(f"  左右对称: <= {MOUTH_CORNER_CURL_SYMMETRY_TOLERANCE:.3f}")
        print(f"  步长: {step}°/次 | 最大迭代: {max_iterations} | 等待: {wait_seconds}s")
        print(f"  初始: A23={current[ch_Lv]}° A25={current[ch_Rv]}°")
        print(f"{'='*60}\n")

        self.scorer.start_display()
        iteration = 0
        sleep_chunk = 0.1

        def _wait():
            waited = 0.0
            while waited < wait_seconds:
                time.sleep(min(sleep_chunk, wait_seconds - waited))
                waited += sleep_chunk
                if self.scorer.user_pressed_stop:
                    break

        try:
            while iteration < max_iterations:
                iteration += 1
                if self.scorer.user_pressed_stop:
                    print("\n[USER] 收到停止信号")
                    break

                self.scorer.update_hud([
                    (f"[MouthCurl {iteration}/{max_iterations}]", (10, 100), (200, 200, 200)),
                ])

                avg = self.collect_mouth_corner_curl_samples(n_frames=60, trim=10)
                if avg is None:
                    print(f"  [MouthCurl {iteration:3d}] 无效帧")
                    _wait()
                    continue

                l_delta = avg["left"]["delta"]
                r_delta = avg["right"]["delta"]
                l_ok = avg["left"]["qualified"]
                r_ok = avg["right"]["qualified"]
                sym_ok = avg["symmetry_qualified"]

                mark = lambda ok: "✓" if ok else "✗"
                print(f"  [MouthCurl {iteration:3d}] "
                      f"L={avg['left']['curl']:+.3f} Δ={l_delta:+.3f}{mark(l_ok)} "
                      f"R={avg['right']['curl']:+.3f} Δ={r_delta:+.3f}{mark(r_ok)} "
                      f"Sym={avg['symmetry_delta']:.3f}{mark(sym_ok)}")

                if l_ok and r_ok and sym_ok:
                    print(f"\n{'='*60}")
                    print(f"  ★★ 嘴角微翘通过! {iteration} 次")
                    print(f"      A23={current[ch_Lv]}° A25={current[ch_Rv]}°")
                    print(f"{'='*60}\n")
                    all_angles = [self._temp_deg(c) for c in EYE_SERVO_CHANNELS]
                    self.save_result(all_angles)
                    self.export_angle_config(
                        all_angles,
                        chin_angle=self._temp_deg(MOUTH_CHIN_CHANNEL),
                        lower_lip_angle=self._temp_deg(LOWER_LIP_CHANNEL),
                        upper_lip_angle=self._temp_deg(UPPER_LIP_CHANNEL),
                        corner_angles={
                            "A23": {"channel": ch_Lv, "angle": current[ch_Lv]},
                            "A25": {"channel": ch_Rv, "angle": current[ch_Rv]},
                        },
                    )
                    self.scorer.update_hud([("★★ MOUTH CURL PASSED ★★", (100, 80), (0, 255, 0))])
                    time.sleep(1)
                    return True, iteration, avg

                moved = []
                l_curl = avg["left"]["curl"]
                r_curl = avg["right"]["curl"]
                if not l_ok or (not sym_ok and l_curl < r_curl):
                    desired_effect = "上" if l_delta < 0 else "下"
                    if not sym_ok and l_curl < r_curl:
                        desired_effect = "上"
                    delta_angle = self._motion_delta(ch_Lv, desired_effect, step)
                    self._move_one(ch_Lv, delta_angle, ranges, current, moved, "A23")
                if not r_ok or (not sym_ok and r_curl < l_curl):
                    desired_effect = "上" if r_delta < 0 else "下"
                    if not sym_ok and r_curl < l_curl:
                        desired_effect = "上"
                    delta_angle = self._motion_delta(ch_Rv, desired_effect, step)
                    self._move_one(ch_Rv, delta_angle, ranges, current, moved, "A25")

                if moved:
                    print(f"  [MouthCurl {iteration:3d}] -> " + " ".join(moved))
                else:
                    print(f"  [MouthCurl {iteration:3d}] 已达极限")
                _wait()

            print(f"\n{'='*60}")
            print("  ✘ 嘴角微翘未通过")
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
