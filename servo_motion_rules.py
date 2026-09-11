"""Servo motion rule lookup based on measured mechanical effects."""

import os
from typing import Any

from utility import read_yaml


DEFAULT_SERVO_MOTION_RULES_FILE = "servo_motion_rules.yaml"


class ServoMotionRules:
    """Map desired physical effects to angle +/- using measured rules."""

    def __init__(self, rules_file: str = DEFAULT_SERVO_MOTION_RULES_FILE):
        if os.path.isabs(rules_file):
            self.rules_file = rules_file
        else:
            self.rules_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), rules_file)
        raw = read_yaml(self.rules_file) or {}
        self.effects: dict[str, dict[str, Any]] = raw.get("SERVO_EFFECTS", raw)

    @staticmethod
    def _key(channel: int | str) -> str:
        if isinstance(channel, str):
            return channel if channel.startswith("A") else f"A{channel}"
        return f"A{int(channel)}"

    def angle_delta(self, channel: int | str, desired_effect: str, step: float = 1.0) -> float:
        """Return +step/-step when desired_effect matches angle_plus/angle_minus."""
        key = self._key(channel)
        rule = self.effects.get(key)
        if not rule:
            raise KeyError(f"No servo motion rule for {key}")

        effect = str(desired_effect).strip()
        plus = str(rule.get("angle_plus", "")).strip()
        minus = str(rule.get("angle_minus", "")).strip()
        if effect == plus:
            return float(step)
        if effect == minus:
            return -float(step)
        raise ValueError(
            f"{key} cannot produce effect {effect!r}; "
            f"angle_plus={plus!r}, angle_minus={minus!r}"
        )

    def describe(self, channel: int | str) -> str:
        key = self._key(channel)
        rule = self.effects.get(key, {})
        return f"{key}: +={rule.get('angle_plus', '?')} -={rule.get('angle_minus', '?')}"
