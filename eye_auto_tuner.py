"""
Eye Auto Tuner — 眼睛舵机自动参数优化适配器 (主入口)
======================================================
两阶段优化流程:
  Phase 1: 眼皮 (A8/A9) → 目标开放度 ≥ least_score
           单眼达标即锁定，只调另一只
  Phase 2: 眼球 (A10-A13) → 目标综合得分 ≥ least_score
           眼皮锁定，仅调整眼球

模块结构:
  eye_constants.py   - 常量 + TuningResult 数据类
  eye_scorer.py      - 摄像头采集 + 面部评分 (EyeScoreCalculator)
  eye_strategies.py  - 优化算法 (CoordinateDescentStrategy)
  eye_auto_tuner.py  - 主编排器 + CLI 入口 (本文件)
"""

import os
import time
import json
from typing import List, Optional, Tuple

from Motor import FaceController
from Communication import UARTDevice
from eye_constants import (
    EYE_SERVO_CHANNELS, EYE_SERVO_NAMES,
    EYELID_CHANNELS, EYEBALL_CHANNELS,
    DEFAULT_ANGLE_MIN, DEFAULT_ANGLE_MAX, DEFAULT_ANGLE_STEP,
    TuningResult,
)
from eye_scorer import EyeScoreCalculator
from eye_strategies import CoordinateDescentStrategy
import cv2


class EyeAutoTuner:
    """眼睛舵机自动参数优化适配器（主编排器）"""

    def __init__(
        self,
        yaml_file: str = "29_servo_config(13).yaml",
        port: str = "COM5",
        baudrate: int = 115200,
        strategy: str = "coordinate",
        servo_num: int = 27,
        max_iterations: int = 150,
        angle_min: int = DEFAULT_ANGLE_MIN,
        angle_max: int = DEFAULT_ANGLE_MAX,
        angle_step: int = DEFAULT_ANGLE_STEP,
        stabilize_frames: int = 8,
        settle_time_ms: int = 300,
        show_preview: bool = True,
        auto_save: bool = True,
        target_score: float = 95.0,
    ):
        self.yaml_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), yaml_file)
        self.port = port
        self.baudrate = baudrate
        self.strategy_name = strategy.lower()
        self.servo_num = servo_num
        self.max_iterations = max_iterations
        self.stabilize_frames = stabilize_frames
        self.settle_time_ms = settle_time_ms
        self.show_preview = show_preview
        self.auto_save = auto_save
        self.target_score = target_score

        # 舵机角度范围（initialize 时从 yaml 覆盖）
        self.angle_ranges = [(angle_min, angle_max)] * len(EYE_SERVO_CHANNELS)
        self.angle_step = angle_step

        # 组件引用（延迟初始化）
        self.controller = None
        self.scorer = None

        # 运行标志
        self._running = False
        self._stopped_by_user = False

    # ==================================================================
    # 配置加载
    # ==================================================================

    def _load_servo_defaults(self, config_file: str = "eyelid_tuner_config.json"):
        """从配置文件加载舵机经验默认值 + 总控制开关

        配置结构:
          servo_defaults:
            eyelid: { center_ratio, window_ratio, max_iterations, least_score }
            eyeball: { center_ratio, window_ratio, max_iterations, least_score }
            A10:     { center_ratio, window_ratio }   # 单通道覆盖

        max_iterations:
          -1 = 必须达标才停止 (搜索空间遍历后自动重置)
           0 = 使用全局默认 max_iterations
          >0 = 该阶段最大迭代次数
        """
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_file)
        defaults = {
            "eyelid": {
                "center_ratio": 0.85, "window_ratio": 0.05,
                "max_iterations": 0, "least_score": 99.0,
            },
            "eyeball": {
                "center_ratio": 0.50, "window_ratio": 0.15,
                "max_iterations": 0, "least_score": 95.0,
            },
        }
        auto_adjust = True
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if "servo_defaults" in cfg:
                for key, val in cfg["servo_defaults"].items():
                    if key in defaults:
                        defaults[key].update(val)
                    else:
                        defaults[key] = dict(val)
            if "auto_adjust" in cfg:
                auto_adjust = bool(cfg["auto_adjust"])
            print(f"[INFO] 舵机经验参数已从 {config_file} 加载")
            print(f"[INFO] 自动调整: {'开启' if auto_adjust else '关闭 (仅采集模式)'}")
        except FileNotFoundError:
            print(f"[WARN] 配置文件未找到: {config_path}，使用内置默认值，自动调整=开")
        return defaults, auto_adjust

    @staticmethod
    def _resolve_max_iter(raw_value: int, global_default: int) -> int:
        """解析 max_iterations: -1=不限, 0=用全局默认, >0=指定值"""
        if raw_value == -1:
            return -1
        elif raw_value == 0:
            return global_default
        else:
            return raw_value

    # ==================================================================
    # 初始化
    # ==================================================================

    def initialize(self):
        """初始化所有硬件和模型"""
        print("=" * 60)
        print("  EyeAutoTuner - 眼睛舵机自动优化适配器")
        print("=" * 60)

        self.servo_defaults, self.auto_adjust = self._load_servo_defaults()
        eyelid_ratio = self.servo_defaults["eyelid"]["center_ratio"]
        eyelid_window = self.servo_defaults["eyelid"]["window_ratio"]
        eyeball_ratio = self.servo_defaults["eyeball"]["center_ratio"]
        eyeball_window = self.servo_defaults["eyeball"]["window_ratio"]

        # 1. 初始化舵机控制器
        print("\n[1/3] 初始化舵机控制器...")
        interface = UARTDevice(self.port, self.baudrate)
        self.controller = FaceController(interface, self.servo_num, self.yaml_file)
        self.controller.open()
        self.controller.init_data()

        # 计算每个眼睛舵机的搜索范围
        print("\n    眼睛舵机 (A8-A13) 搜索范围:")
        for i, ch in enumerate(EYE_SERVO_CHANNELS):
            start = self.controller.list_start_deg[ch]
            end = self.controller.list_end_deg[ch]
            full_lo = min(start, end)
            full_hi = max(start, end)
            span = float(full_hi - full_lo)

            ch_key = f"A{ch}"
            if ch_key in self.servo_defaults:
                ch_cfg = self.servo_defaults[ch_key]
                ratio = ch_cfg.get("center_ratio", eyeball_ratio)
                window_ratio = ch_cfg.get("window_ratio", eyeball_window)
                tag = f"{ch_key}@{int(ratio*100)}%±{int(window_ratio*100)}%"
            elif ch in EYELID_CHANNELS:
                ratio = eyelid_ratio
                window_ratio = eyelid_window
                tag = f"眼皮@{int(ratio*100)}%±{int(window_ratio*100)}%"
            else:
                ratio = eyeball_ratio
                window_ratio = eyeball_window
                tag = f"眼球@{int(ratio*100)}%±{int(window_ratio*100)}%"

            center = start + (end - start) * ratio
            half_window = span * window_ratio
            search_lo = round(center - half_window)
            search_hi = round(center + half_window)

            self.angle_ranges[i] = (search_lo, search_hi)
            print(f"    A{ch} ({EYE_SERVO_NAMES[i]}): "
                  f"range=[{start}→{end}] 中心={center:.1f}° "
                  f"搜索[{search_lo:3d},{search_hi:3d}] ({tag})")

        # 全量舵机状态
        self.full_angles = self.controller.list_temp_deg[:self.controller.servo_num].copy()

        # 经验最佳中心值
        self.initial_center = []
        for i, ch in enumerate(EYE_SERVO_CHANNELS):
            start = self.controller.list_start_deg[ch]
            end = self.controller.list_end_deg[ch]
            ch_key = f"A{ch}"
            if ch_key in self.servo_defaults:
                ratio = self.servo_defaults[ch_key].get("center_ratio", eyeball_ratio)
            else:
                ratio = eyelid_ratio if ch in EYELID_CHANNELS else eyeball_ratio
            self.initial_center.append(round(start + (end - start) * ratio))

        print(f"\n    经验起点: {dict(zip(EYE_SERVO_NAMES, self.initial_center))}")

        # 2. 初始化评分器
        print("\n[2/3] 初始化面部检测与评分系统...")
        self.scorer = EyeScoreCalculator()

        # 3. 就绪
        print(f"\n[3/3] 优化策略: 坐标下降法 (步长={self.angle_step})")

        print("\n" + "=" * 60)
        print("  初始化完成！按 [Q] 可随时停止优化")
        print("=" * 60 + "\n")

    # ==================================================================
    # 硬件操作
    # ==================================================================

    def _send_servo(self, channel: int, angle: int,
                    angle_range: Tuple[int, int] = None):
        """发送单个舵机角度，自动 clamp 到范围"""
        if angle_range:
            angle = max(angle_range[0], min(angle_range[1], angle))
        self.controller.set_servo_angle_time_32(
            [angle], [channel], 200, servo_num=self.controller.servo_num
        )

    def _send_servos(self, channels: List[int], angles: List[int],
                     angle_ranges: List[Tuple[int, int]] = None):
        """发送多个舵机角度"""
        for i, (ch, ang) in enumerate(zip(channels, angles)):
            ar = angle_ranges[i] if angle_ranges else None
            self._send_servo(ch, ang, ar)

    def _capture_score(self):
        """等待舵机稳定后采集评分"""
        time.sleep(self.settle_time_ms / 1000.0)
        return self.scorer.capture_and_score(
            stabilize_frames=self.stabilize_frames,
            show_preview=self.show_preview
        )

    def _check_stop_key(self) -> bool:
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            print("\n[USER] 收到停止信号...")
            return True
        return False

    # ==================================================================
    # Phase 1: 眼皮优化 (A8/A9) — 单眼达标即锁定
    # ==================================================================

    def _run_phase1_eyelid(self) -> Tuple[List[int], float, float, int]:
        """
        Phase 1: 眼皮优化

        Returns:
            (best_eyelid_angles, best_l_open, best_r_open, eyelid_iter)
        """
        eyelid_cfg = self.servo_defaults["eyelid"]
        eyelid_least = eyelid_cfg.get("least_score", 99.0)
        eyelid_max_iter = self._resolve_max_iter(
            eyelid_cfg.get("max_iterations", 0), self.max_iterations)

        print(f"{'='*60}")
        print(f"  Phase 1: 眼皮优化 (A8/A9) | 单眼达标即锁定")
        print(f"  目标: L_Openness ≥ {eyelid_least}% 且 R_Openness ≥ {eyelid_least}%")
        if eyelid_max_iter == -1:
            print(f"  迭代: 不限 (必须达标才进入 Phase 2)")
        else:
            print(f"  最大迭代: {eyelid_max_iter}")
        print(f"{'='*60}\n")

        # 发送经验起点
        eyelid_ranges = self.angle_ranges[:2]
        eyelid_angles = list(self.initial_center[:2])
        self._send_servos(EYELID_CHANNELS, eyelid_angles, eyelid_ranges)
        time.sleep(1.5)

        # 追踪各眼最佳分数和角度
        best_l_open = 0.0
        best_r_open = 0.0
        best_l_angle = eyelid_angles[0]
        best_r_angle = eyelid_angles[1]
        l_locked = False
        r_locked = False
        eyelid_iter = 0
        reset_count = 0

        def create_optimizer(active_indices):
            """为活跃通道创建优化器，用各通道历史最佳角度作为起点"""
            chs = [EYELID_CHANNELS[i] for i in active_indices]
            ranges = [eyelid_ranges[i] for i in active_indices]
            center = [best_l_angle if i == 0 else best_r_angle for i in active_indices]
            return CoordinateDescentStrategy(
                chs, ranges, self.angle_step, initial_center=center
            ), chs, ranges

        active = [0, 1]
        optimizer, active_chs, active_ranges = create_optimizer(active)

        while self._running and not (l_locked and r_locked):
            combination = optimizer.get_next_combination()

            if combination is None:
                # 搜索空间遍历完
                all_met = best_l_open >= eyelid_least and best_r_open >= eyelid_least
                if eyelid_max_iter == -1 and not all_met:
                    reset_count += 1
                    not_met = []
                    if best_l_open < eyelid_least:
                        not_met.append(f"L={best_l_open:.1f}%")
                    if best_r_open < eyelid_least:
                        not_met.append(f"R={best_r_open:.1f}%")
                    print(f"\n[INFO] Phase 1 搜索空间遍历 (第{reset_count}轮), "
                          f"未达标 {' '.join(not_met)}, 重置优化器继续")
                    # 恢复到历史最佳角度
                    eyelid_angles[0] = best_l_angle
                    eyelid_angles[1] = best_r_angle
                    for idx, ch in enumerate(EYELID_CHANNELS):
                        if (idx == 0 and not l_locked) or (idx == 1 and not r_locked):
                            self._send_servo(ch, eyelid_angles[idx], eyelid_ranges[idx])
                    optimizer, active_chs, active_ranges = create_optimizer(active)
                    continue
                print("\n[INFO] Phase 1 搜索空间已完全遍历")
                break

            if self._check_stop_key():
                self._stopped_by_user = True
                break

            eyelid_iter += 1

            # 移动活跃通道舵机
            for j, (channel, angle) in enumerate(zip(active_chs, combination)):
                clamped = max(active_ranges[j][0], min(active_ranges[j][1], angle))
                idx = EYELID_CHANNELS.index(channel)
                eyelid_angles[idx] = clamped
                self._send_servo(channel, clamped)

            # 打分
            _, region_scores = self._capture_score()
            l_open = region_scores.get("L_Openness", 0.0)
            r_open = region_scores.get("R_Openness", 0.0)

            # 更新最佳
            l_improved = (not l_locked) and (l_open > best_l_open)
            r_improved = (not r_locked) and (r_open > best_r_open)
            if l_improved:
                best_l_open = l_open
                best_l_angle = eyelid_angles[0]
            if r_improved:
                best_r_open = r_open
                best_r_angle = eyelid_angles[1]

            # 日志
            if l_improved or r_improved:
                lock_str = (" [L已锁]" if l_locked else "") + (" [R已锁]" if r_locked else "")
                print(f"  ★ [P1 Iter {eyelid_iter:3d}] "
                      f"L_Opn={l_open:.1f}% R_Opn={r_open:.1f}% "
                      f"| A8={eyelid_angles[0]}° A9={eyelid_angles[1]}°{lock_str}")
            elif eyelid_iter % 10 == 0:
                print(f"    [P1 Iter {eyelid_iter:3d}] "
                      f"L_Opn={l_open:.1f}% R_Opn={r_open:.1f}% "
                      f"(Best: L={best_l_open:.1f}% R={best_r_open:.1f}%)")

            # 反馈: 只用活跃眼的分数
            if l_locked and not r_locked:
                feedback_score = r_open
            elif r_locked and not l_locked:
                feedback_score = l_open
            else:
                feedback_score = (l_open + r_open) / 2.0
            optimizer.update_feedback(combination, feedback_score)

            # 检查单眼达标 → 锁定
            need_rebuild = False
            if not l_locked and l_open >= eyelid_least:
                l_locked = True
                print(f"  ► 左眼皮达标! L_Opn={l_open:.1f}% ≥ {eyelid_least}% "
                      f"→ A8={eyelid_angles[0]}° 已锁定")
                need_rebuild = True
            if not r_locked and r_open >= eyelid_least:
                r_locked = True
                print(f"  ► 右眼皮达标! R_Opn={r_open:.1f}% ≥ {eyelid_least}% "
                      f"→ A9={eyelid_angles[1]}° 已锁定")
                need_rebuild = True

            if l_locked and r_locked:
                print(f"\n{'='*60}")
                print(f"  ★★ Phase 1 达标! 双眼均已锁定")
                print(f"      L_Opn={best_l_open:.1f}%  R_Opn={best_r_open:.1f}%")
                print(f"      A8={best_l_angle}°  A9={best_r_angle}°")
                print(f"{'='*60}\n")
                break

            # 某只眼锁定后，重建优化器只调另一只
            if need_rebuild:
                active = []
                if not l_locked:
                    active.append(0)
                if not r_locked:
                    active.append(1)
                if active:
                    optimizer, active_chs, active_ranges = create_optimizer(active)
                    active_names = ['A8' if i == 0 else 'A9' for i in active]
                    print(f"  >> 重建优化器, 活跃通道: {active_names}")

            if eyelid_max_iter > 0 and eyelid_iter >= eyelid_max_iter:
                print(f"\n[WARN] Phase 1 达到最大迭代 {eyelid_max_iter}")
                break

        # 用历史最佳角度（而非最后一次尝试的角度）
        best_eyelid_angles = [best_l_angle, best_r_angle]
        self._send_servos(EYELID_CHANNELS, best_eyelid_angles, eyelid_ranges)
        for i, ch in enumerate(EYELID_CHANNELS):
            self.full_angles[ch] = best_eyelid_angles[i]
        print(f"  >> 眼皮锁定: A8={best_eyelid_angles[0]}°  A9={best_eyelid_angles[1]}°  "
              f"(L_Opn={best_l_open:.1f}%  R_Opn={best_r_open:.1f}%)\n")

        return best_eyelid_angles, best_l_open, best_r_open, eyelid_iter

    # ==================================================================
    # Phase 2: 眼球优化 (A10-A13)
    # ==================================================================

    def _run_phase2_eyeball(self, best_eyelid_angles: List[int]) -> Tuple[List[int], float, dict, int]:
        """
        Phase 2: 眼球优化（眼皮已锁定）

        Returns:
            (best_eyeball_angles, best_eye_score, best_region_scores, eyeball_iter)
        """
        eyeball_cfg = self.servo_defaults["eyeball"]
        eyeball_least = eyeball_cfg.get("least_score", 95.0)
        eyeball_max_iter = self._resolve_max_iter(
            eyeball_cfg.get("max_iterations", 0), self.max_iterations)

        print(f"{'='*60}")
        print(f"  Phase 2: 眼球优化 (A10-A13) | 目标综合得分 ≥ {eyeball_least}%")
        print(f"  眼皮锁定: A8={best_eyelid_angles[0]}°  A9={best_eyelid_angles[1]}°")
        if eyeball_max_iter == -1:
            print(f"  迭代: 不限 (必须达标才停止)")
        else:
            print(f"  最大迭代: {eyeball_max_iter}")
        print(f"{'='*60}\n")

        eyeball_ranges = self.angle_ranges[2:]
        eyeball_center = self.initial_center[2:]

        # 发送经验中心到眼球舵机
        self._send_servos(EYEBALL_CHANNELS, eyeball_center, eyeball_ranges)
        time.sleep(1.0)

        eyeball_optimizer = CoordinateDescentStrategy(
            EYEBALL_CHANNELS, eyeball_ranges, self.angle_step,
            initial_center=eyeball_center
        )

        best_eyeball_angles = eyeball_center.copy()
        best_eye_score = -float('inf')
        best_region_scores = {}
        eyeball_iter = 0
        eyeball_reset_count = 0

        while self._running:
            combination = eyeball_optimizer.get_next_combination()
            if combination is None:
                if eyeball_max_iter == -1 and best_eye_score < eyeball_least:
                    eyeball_reset_count += 1
                    print(f"\n[INFO] Phase 2 搜索空间遍历 (第{eyeball_reset_count}轮), "
                          f"未达标 Avg={best_eye_score:.1f}% < {eyeball_least}%, 重置优化器继续")
                    eyeball_optimizer = CoordinateDescentStrategy(
                        EYEBALL_CHANNELS, eyeball_ranges, self.angle_step,
                        initial_center=best_eyeball_angles
                    )
                    continue
                print("\n[INFO] Phase 2 搜索空间已完全遍历")
                break

            if self._check_stop_key():
                self._stopped_by_user = True
                break

            eyeball_iter += 1

            # 仅移动 A10-A13
            for i, (channel, angle) in enumerate(zip(EYEBALL_CHANNELS, combination)):
                clamped = max(eyeball_ranges[i][0], min(eyeball_ranges[i][1], angle))
                self._send_servo(channel, clamped)

            # 打分
            eye_score, region_scores = self._capture_score()

            if eye_score > best_eye_score:
                best_eye_score = eye_score
                best_eyeball_angles = combination.copy()
                best_region_scores = region_scores
                angles_str = " ".join([f"{n}={a}°" for n, a in
                                       zip(["A10", "A11", "A12", "A13"], combination)])
                print(f"  ★ [P2 Iter {eyeball_iter:3d}] NEW BEST! "
                      f"Total={eye_score:.1f}% | {angles_str}")
            elif eyeball_iter % 10 == 0:
                print(f"    [P2 Iter {eyeball_iter:3d}] "
                      f"Total={eye_score:.1f}% (Best: {best_eye_score:.1f}%)")

            eyeball_optimizer.update_feedback(combination, eye_score)

            if best_eye_score >= eyeball_least:
                print(f"\n{'='*60}")
                print(f"  ★★ Phase 2 达标! 综合得分={best_eye_score:.1f}% >= {eyeball_least}%")
                print(f"{'='*60}\n")
                break

            if eyeball_max_iter > 0 and eyeball_iter >= eyeball_max_iter:
                print(f"\n[WARN] Phase 2 达到最大迭代 {eyeball_max_iter}")
                break

        # 应用最佳眼球角度
        self._send_servos(EYEBALL_CHANNELS, best_eyeball_angles, eyeball_ranges)
        print(f"  >> 眼球锁定: "
              + " ".join([f"{n}={a}°" for n, a in
                          zip(["A10", "A11", "A12", "A13"], best_eyeball_angles)])
              + f"  (Score={best_eye_score:.1f}%)\n")

        return best_eyeball_angles, best_eye_score, best_region_scores, eyeball_iter

    # ==================================================================
    # 主运行入口
    # ==================================================================

    def run(self) -> Optional[TuningResult]:
        """
        两阶段自动优化: 先眼皮 → 再眼球

        Phase 1: 仅调整 A8/A9 (眼皮), 目标开放度 ≥ eyelid.least_score
        Phase 2: 锁定眼皮, 仅调整 A10-A13 (眼球), 目标综合得分 ≥ eyeball.least_score
        """
        try:
            self.initialize()
            self._running = True
            self._stopped_by_user = False

            if not self.auto_adjust:
                return self._run_monitor_mode("full")

            # 发送所有经验起点到硬件
            eye_display = ", ".join(
                [f"{n}={a}°" for n, a in zip(EYE_SERVO_NAMES, self.initial_center)])
            print(f"\n  >> [初始化] 发送经验起点到舵机: [{eye_display}]")
            self._send_servos(EYE_SERVO_CHANNELS, self.initial_center, self.angle_ranges)
            print("  >> [初始化] 等待舵机到位 (1.5s)...\n")
            time.sleep(1.5)

            # ---- Phase 1: 眼皮 ----
            best_eyelid_angles, best_l_open, best_r_open, eyelid_iter = \
                self._run_phase1_eyelid()

            eyelid_least = self.servo_defaults["eyelid"].get("least_score", 99.0)

            if self._stopped_by_user:
                avg_score = (best_l_open + best_r_open) / 2.0
                return self._make_partial_result(
                    best_eyelid_angles, [90, 90, 90, 90],
                    avg_score, eyelid_iter,
                    {"L_Openness": best_l_open, "R_Openness": best_r_open}
                )

            # 眼皮未达标 → 不进 Phase 2
            if not (best_l_open >= eyelid_least and best_r_open >= eyelid_least):
                avg_score = (best_l_open + best_r_open) / 2.0
                not_met = []
                if best_l_open < eyelid_least:
                    not_met.append(f"L_Opn={best_l_open:.1f}%")
                if best_r_open < eyelid_least:
                    not_met.append(f"R_Opn={best_r_open:.1f}%")
                print(f"\n{'='*60}")
                print(f"  ✘ Phase 1 未达标! {' '.join(not_met)} < {eyelid_least}%")
                print(f"    不进入 Phase 2 (眼球优化)")
                print(f"{'='*60}\n")

                best_angles_all = best_eyelid_angles + [90, 90, 90, 90]
                result = TuningResult(
                    best_angles=best_angles_all,
                    best_score=avg_score,
                    best_eye_score=avg_score,
                    iteration=eyelid_iter,
                    total_iterations=eyelid_iter,
                    score_history=[],
                    region_scores={"L_Openness": best_l_open, "R_Openness": best_r_open},
                )
                self._save_result(result)
                return result

            # ---- Phase 2: 眼球 ----
            best_eyeball_angles, best_eye_score, best_region_scores, eyeball_iter = \
                self._run_phase2_eyeball(best_eyelid_angles)

            # ---- 合并结果 ----
            best_angles_all = best_eyelid_angles + best_eyeball_angles
            total_iter = eyelid_iter + eyeball_iter
            eyeball_least = self.servo_defaults["eyeball"].get("least_score", 95.0)

            result = TuningResult(
                best_angles=best_angles_all,
                best_score=best_eye_score,
                best_eye_score=best_eye_score,
                iteration=total_iter,
                total_iterations=total_iter,
                score_history=[],
                region_scores=best_region_scores,
            )

            self._print_two_phase_report(
                result,
                eyelid_iter, best_l_open, best_r_open, best_eyelid_angles,
                eyeball_iter, best_eye_score, best_eyeball_angles,
                eyelid_least, eyeball_least,
            )

            self._save_result(result)
            self._export_angle_config(result.best_angles)

            # 应用最终最佳角度
            if self.controller:
                eye_display = ", ".join(
                    [f"{n}={a}°" for n, a in zip(EYE_SERVO_NAMES, best_angles_all)])
                print(f"\n  >> [最终] 写入最佳角度: [{eye_display}]")
                self._send_servos(EYE_SERVO_CHANNELS, best_angles_all, self.angle_ranges)
                print("  >> [最终] 舵机已定位到最佳角度！\n")

            return result

        except KeyboardInterrupt:
            print("\n\n[INTERRUPTED] 用户中断了优化过程")
            return None
        except Exception as e:
            print(f"\n[ERROR] 优化过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            self.cleanup()

    def run_eyelid_phase(self) -> Optional[TuningResult]:
        """仅眼皮调优模式 (A8/A9)"""
        try:
            self.initialize()
            self._running = True
            self._stopped_by_user = False

            # 发送经验起点
            eyelid_ranges = self.angle_ranges[:2]
            eyelid_center = self.initial_center[:2]
            self._send_servos(EYELID_CHANNELS, eyelid_center, eyelid_ranges)
            time.sleep(1.5)

            best_eyelid_angles, best_l_open, best_r_open, eyelid_iter = \
                self._run_phase1_eyelid()

            avg_score = (best_l_open + best_r_open) / 2.0
            result = TuningResult(
                best_angles=best_eyelid_angles,
                best_score=avg_score,
                best_eye_score=avg_score,
                iteration=eyelid_iter,
                total_iterations=eyelid_iter,
                score_history=[],
                region_scores={
                    "L_Openness": best_l_open,
                    "R_Openness": best_r_open,
                    "average": avg_score,
                },
            )

            save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "best_eyelid_config.json")
            result.save(save_path)
            print(f"[INFO] 眼皮调优结果已保存 → {save_path}")
            return result

        except KeyboardInterrupt:
            print("\n\n[INTERRUPTED] 用户中断了眼皮优化过程")
            return None
        except Exception as e:
            print(f"\n[ERROR] 眼皮优化过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            self.cleanup()

   
    # ==================================================================
    # 结果保存与报告
    # ==================================================================

    def _save_result(self, result: TuningResult):
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "best_eye_config.json")
        result.save(save_path)

    def _make_partial_result(self, eyelid_angles, eyeball_angles,
                             score, eyelid_iter, region_scores):
        """构造中途停止的部分结果"""
        result = TuningResult(
            best_angles=eyelid_angles + eyeball_angles,
            best_score=score,
            best_eye_score=score,
            iteration=eyelid_iter,
            total_iterations=eyelid_iter,
            score_history=[],
            region_scores=region_scores,
        )
        self._save_result(result)
        return result

    def _print_two_phase_report(self, result,
                                 eyelid_iter, l_open, r_open, eyelid_angles,
                                 eyeball_iter, eye_score, eyeball_angles,
                                 eyelid_target, eyeball_target):
        """打印两阶段调优最终报告"""
        print("\n" + "=" * 65)
        print("  两阶段优化完成 - 最终报告")
        print("=" * 65)

        print(f"\n  Phase 1: 眼皮 (A8/A9) | 目标开放度 ≥ {eyelid_target}% (单眼达标即锁定)")
        l_mark = "OK" if l_open >= eyelid_target else "X "
        r_mark = "OK" if r_open >= eyelid_target else "X "
        print(f"    迭代: {eyelid_iter}")
        print(f"    L_Openness: {l_open:.1f}% [{l_mark}]  "
              f"R_Openness: {r_open:.1f}% [{r_mark}]  "
              f"Avg: {(l_open + r_open) / 2:.1f}%")
        print(f"    A8={eyelid_angles[0]}°  A9={eyelid_angles[1]}°")

        print(f"\n  Phase 2: 眼球 (A10-A13) | 目标综合得分 ≥ {eyeball_target}%")
        score_mark = "OK" if eye_score >= eyeball_target else "X "
        print(f"    迭代: {eyeball_iter}")
        print(f"    综合得分: {eye_score:.1f}% [{score_mark}]")
        for name, angle in zip(["A10", "A11", "A12", "A13"], eyeball_angles):
            print(f"    {name}: {angle}°")

        if result.region_scores:
            print(f"\n  各区域得分详情:")
            items = []
            for region, score in result.region_scores.items():
                if region.startswith("_"):
                    continue
                try:
                    s = float(score)
                except (TypeError, ValueError):
                    continue
                items.append((region, s))
            for region, score in sorted(items, key=lambda x: -x[1]):
                bar_len = int(score / 4)
                bar = "#" * bar_len + "-" * (25 - bar_len)
                print(f"    {region:<15}: {score:>5.1f}%  |{bar}|")

        if self._stopped_by_user:
            print(f"\n  注意: 优化被用户手动停止")

        print(f"\n  最佳舵机角度 (A8-A13):")
        for name, angle in zip(EYE_SERVO_NAMES, result.best_angles):
            print(f"    {name}: {angle}°")

        print("\n" + "=" * 65 + "\n")

    def _export_angle_config(self, angles: List[int]):
        config = {
            "description": "EyeAutoTuner 最佳眼睛舵机角度配置",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "strategy": self.strategy_name,
            "servo_angles": {
                name: {"channel": ch, "angle": ang}
                for name, ch, ang in zip(EYE_SERVO_NAMES, EYE_SERVO_CHANNELS, angles)
            },
            "quick_apply": {
                "indexes": EYE_SERVO_CHANNELS,
                "angles": angles,
            }
        }
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "eye_angles_best.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"[INFO] 角度配置已导出 → {path}")

    # ==================================================================
    # 清理
    # ==================================================================

    def cleanup(self):
        """清理资源"""
        self._running = False
        if self.scorer:
            try:
                self.scorer.close()
            except Exception:
                pass
        if self.controller:
            try:
                self.controller.close()
            except Exception:
                pass
        cv2.destroyAllWindows()
        print("[INFO] 资源已释放")


# ==================================================================
# CLI 入口
# ==================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="眼睛舵机自动参数优化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python eye_auto_tuner.py --port COM3 --iter 100
  python eye_auto_tuner.py --step 2 --iter 500
        """
    )

    parser.add_argument("--port", type=str, default="COM5",
                       help="串口端口号 (默认: COM5)")
    parser.add_argument("--baud", type=int, default=115200,
                       help="波特率 (默认: 115200)")
    parser.add_argument("--strategy", type=str, default="coordinate",
                       choices=["coordinate"],
                       help="优化策略 (默认: coordinate)")
    parser.add_argument("--iter", type=int, default=100,
                       help="最大迭代次数 (默认: 100)")
    parser.add_argument("--step", type=int, default=5,
                       help="坐标下降步长 (默认: 5)")
    parser.add_argument("--stabilize", type=int, default=8,
                       help="每次采集稳定帧数 (默认: 8)")
    parser.add_argument("--settle", type=int, default=300,
                       help="舵机等待时间ms (默认: 300)")
    parser.add_argument("--no-preview", action="store_true",
                       help="禁用预览窗口")
    parser.add_argument("--phase", type=str, default="two-phase",
                       choices=["two-phase", "eyelid"],
                       help="优化模式: two-phase=先眼皮再眼球(默认), eyelid=仅A8/A9")
    parser.add_argument("--config", type=str, default="eyelid_tuner_config.json",
                       help="配置文件 (默认: eyelid_tuner_config.json)")

    args = parser.parse_args()

    print(f"""
+==============================================================+
|       Eye Auto Tuner - {'两阶段优化' if args.phase == 'two-phase' else '眼皮模式 (A8/A9)':24s}|
+==============================================================+
|  {'Phase 1: 眼皮 (A8/A9) → 开放度 ≥ 99%' if args.phase == 'two-phase' else 'Phase:   眼皮 (A8/A9) → 开放度 ≥ 99%':40s}|
|  {'Phase 2: 眼球 (A10-A13) → 综合得分' if args.phase == 'two-phase' else '':40s}|
|  Keys: [Q/Esc] Stop                                         |
|  Output: best_eye_config.json                                |
+==============================================================+
""")

    tuner = EyeAutoTuner(
        yaml_file="29_servo_config(13).yaml",
        port=args.port,
        baudrate=args.baud,
        strategy=args.strategy,
        max_iterations=args.iter,
        angle_step=args.step,
        stabilize_frames=args.stabilize,
        settle_time_ms=args.settle,
        show_preview=not args.no_preview,
    )

    if args.phase == "eyelid":
        result = tuner.run_eyelid_phase()
    else:
        result = tuner.run()

    if result:
        print("\nDone! Results saved.")
    else:
        print("\nTuning did not complete normally.")


if __name__ == "__main__":
    main()
