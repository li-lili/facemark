"""
Optimization Strategies — 优化策略
====================================
纯算法层，不依赖硬件或摄像头。
提供可插拔的搜索策略接口，用于遍历舵机角度组合空间。
"""

from typing import List, Optional, Tuple


class OptimizationStrategy:
    """优化策略基类"""

    def __init__(self, channels: List[int],
                 angle_ranges: List[Tuple[int, int]],
                 initial_center: Optional[List[int]] = None):
        self.channels = channels
        self.angle_ranges = angle_ranges  # [(min, max), ...]
        self.initial_center = initial_center
        self.iteration = 0
        self._first_used = False

    def get_next_combination(self) -> Optional[List[int]]:
        """返回下一个要测试的角度组合，返回 None 表示结束"""
        raise NotImplementedError

    def update_feedback(self, angles: List[int], score: float):
        """接收上一次的反馈（可选，用于自适应策略）"""
        self.iteration += 1

    def reset(self):
        """重置状态"""
        self.iteration = 0


class CoordinateDescentStrategy(OptimizationStrategy):
    """
    坐标轮换/下降法 - 每次只优化一个舵机的角度

    搜索逻辑:
      1. 从经验起点 initial_center 出发
      2. 每轮依次遍历每个通道
      3. 对当前通道尝试一组偏移量: 0, +step, -step, +2*step, -2*step, ...
      4. 找到更好分数时，更新该通道的当前角度
      5. 一轮内无任何改善则停止，或达到 max_passes
    """

    def __init__(self, channels: List[int],
                 angle_ranges: List[Tuple[int, int]],
                 step: int = 3,
                 max_passes: int = 10,
                 initial_center: Optional[List[int]] = None):
        super().__init__(channels, angle_ranges, initial_center)
        self.step = step
        self.max_passes = max_passes
        # ★ 从经验中心开始（而非中点）
        if initial_center is not None:
            self.current_angles = list(initial_center)
        else:
            self.current_angles = [((r[0] + r[1]) // 2) for r in angle_ranges]

        self.current_channel_idx = 0
        self.current_pass = 0
        self.improved_this_pass = False

        # 偏移序列: +step, -step, +2*step, -2*step, ... (共20个偏移)
        self._offsets = []
        for k in range(1, 11):
            self._offsets.append(k * step)
            self._offsets.append(-k * step)
        self._offset_idx = 0

        # 当前通道的搜索状态
        self._phase = 'base'  # 'base' = 先测基准, 'offsets' = 尝试偏移, 'done' = 结束
        self._channel_base = self.current_angles[0]
        self._channel_best_score = -float('inf')
        self._channel_best_angle = self.current_angles[0]

        print(f"[CoordinateDescent] 步长={step}, 最大轮次={max_passes}")

    def _advance_channel(self):
        """当前通道搜索完毕，移到下一个通道或下一轮"""
        # 将当前通道角度更新为本通道找到的最佳值
        self.current_angles[self.current_channel_idx] = self._channel_best_angle

        self.current_channel_idx += 1

        if self.current_channel_idx >= len(self.channels):
            # 一轮结束
            self.current_channel_idx = 0
            self.current_pass += 1
            if not self.improved_this_pass:
                self._phase = 'done'
                print(f"[CoordinateDescent] 第 {self.current_pass} 轮无改善，停止搜索")
                return
            self.improved_this_pass = False
            if self.current_pass >= self.max_passes:
                self._phase = 'done'
                print(f"[CoordinateDescent] 达到最大轮次 {self.max_passes}")
                return
            print(f"[CoordinateDescent] Pass {self.current_pass}/{self.max_passes} completed")

        # 准备下一个通道
        self._phase = 'base'
        self._offset_idx = 0

    def get_next_combination(self) -> Optional[List[int]]:
        while True:
            if self._phase == 'done':
                return None

            ch_idx = self.current_channel_idx

            if self._phase == 'base':
                # 先测当前基准角度
                self._channel_base = self.current_angles[ch_idx]
                self._channel_best_score = -float('inf')
                self._channel_best_angle = self.current_angles[ch_idx]
                self._phase = 'offsets'
                self._offset_idx = 0
                return self.current_angles.copy()

            if self._phase == 'offsets':
                if self._offset_idx < len(self._offsets):
                    result = self.current_angles.copy()
                    offset = self._offsets[self._offset_idx]
                    new_angle = self._channel_base + offset
                    new_angle = max(self.angle_ranges[ch_idx][0],
                                  min(self.angle_ranges[ch_idx][1], new_angle))
                    result[ch_idx] = new_angle
                    self._offset_idx += 1
                    return result
                else:
                    # 当前通道偏移已用完，移到下一个
                    self._advance_channel()
                    continue

            return None

    def update_feedback(self, angles: List[int], score: float):
        super().update_feedback(angles, score)

        ch_idx = self.current_channel_idx
        if score > self._channel_best_score:
            self._channel_best_score = score
            self._channel_best_angle = angles[ch_idx]
            self.improved_this_pass = True
