"""Hardware-free A10--A13 step planner. Coordinates are unmodified guide errors.

No camera, serial, model or motor imports. Angle signs come from the supplied
measured-motion-rule callback, never from a fitted inverse motor direction.
"""
from dataclasses import dataclass
import math
from statistics import pstdev


AXES = ((12, 0, "内", "外"), (10, 1, "上", "下"),
        (13, 2, "外", "内"), (11, 3, "上", "下"))
ERROR_KEYS = ("L_dX", "L_dY", "R_dX", "R_dY")
SIGNAL_KEYS = ERROR_KEYS + ("L_dist", "R_dist")


@dataclass(frozen=True)
class AdaptiveConfig:
    probe_deg: float = 1.0
    min_step_deg: float = 0.5  # Also the command increment quantum.
    max_step_deg: float = 3.0
    damping: float = 0.6
    min_response_px_deg: float = 0.2
    max_std_px: float = 1.5
    max_drift_px: float = 1.0
    growth_px: float = 0.25
    max_cross_ratio: float = 0.5
    max_response_ratio: float = 2.0
    max_stalls: int = 3  # Also bounds consecutive invalid measurement batches.

    def __post_init__(self):
        values = tuple(vars(self).values())
        if not all(math.isfinite(x) and x > 0 for x in values):
            raise ValueError("All adaptive parameters must be finite and positive")
        if not self.min_step_deg <= self.probe_deg <= self.max_step_deg <= 3.0:
            raise ValueError("Require min step <= probe <= max step <= 3 degrees")
        if self.probe_deg > 1 or self.damping > 1 or self.max_response_ratio < 1:
            raise ValueError("Unsafe probe, damping or response ratio")


def valid_signal(signal):
    return (isinstance(signal, dict)
            and all(isinstance(signal.get(k), (int, float))
                    and math.isfinite(signal[k]) for k in SIGNAL_KEYS)
            and signal["L_dist"] >= 0 and signal["R_dist"] >= 0)


def sample_quality(ordered_samples):
    """Diagnostics only; never changes the production 60/10/40 reducer."""
    rows = [s[1] for s in ordered_samples]
    count = len(rows)
    if count < 2:
        return {"count": count, "std": math.inf, "drift": math.inf}
    half = count // 2
    return {
        "count": count,
        "std": max(pstdev(r[k] for r in rows) for k in ERROR_KEYS),
        "drift": max(abs(sum(r[k] for r in rows[:half]) / half
                         - sum(r[k] for r in rows[half:]) / (count-half))
                     for k in ERROR_KEYS),
    }


class EyeballStepPlanner:
    def __init__(self, limits, motion_delta, mode="fixed", tolerance=1.0, config=None):
        if mode not in ("fixed", "adaptive"):
            raise ValueError("Eyeball mode must be fixed or adaptive")
        self.config = config or AdaptiveConfig()
        self.mode, self.tolerance = mode, tolerance
        self.limits, self.motion_delta = limits, motion_delta
        self.models = {}  # A channel's measured four-coordinate response / degree.
        self.pending = None
        self.fallback = False  # Sticky for this run; a new run starts new probes.
        self.stalls = 0
        self.best_cost = None
        self.reason = "ready"

    def degrade(self, reason):
        self.fallback = True
        self.pending = None  # Never fit a response across a failed/unstable batch.
        self.models.clear()
        self.reason = "fallback: " + reason

    @staticmethod
    def vector(signal):
        return [signal[k] for k in ERROR_KEYS]

    def observe(self, signal):
        if self.pending is None:
            return
        before, moves = self.pending
        self.pending = None
        cost_before = before["L_dist"] + before["R_dist"]
        if self.best_cost is None:
            self.best_cost = cost_before
        cost_after = signal["L_dist"] + signal["R_dist"]
        if cost_after < self.best_cost - 1e-9:
            self.best_cost, self.stalls = cost_after, 0
        else:
            self.stalls += 1
        if self.mode != "adaptive":
            return
        ch, _, delta, axis = moves[0]
        old, new = self.vector(before), self.vector(signal)
        response = [(b-a)/delta for a, b in zip(old, new)]
        own = response[axis]
        prior = self.models.get(ch)
        reason = None
        own_eye = "L_dist" if axis < 2 else "R_dist"
        other_eye = "R_dist" if axis < 2 else "L_dist"
        if (signal[own_eye] > before[own_eye]
                or signal[other_eye] > before[other_eye] + self.config.growth_px):
            reason = "radial error grew"
        elif old[axis] * new[axis] < 0:
            reason = "error sign reversed"
        elif (new[axis]-old[axis]) * old[axis] >= 0:
            reason = "no response or wrong direction"
        elif abs(own) < self.config.min_response_px_deg:
            reason = "response too small"
        elif math.sqrt(sum(x*x for i, x in enumerate(response) if i != axis)) > abs(own)*self.config.max_cross_ratio:
            reason = "strong cross-axis/eye coupling"
        elif prior and (own * prior[axis] <= 0 or not
                        1/self.config.max_response_ratio <= abs(own/prior[axis]) <= self.config.max_response_ratio):
            reason = "response changed"
        if reason:
            self.degrade(reason)
        elif not self.fallback:
            self.models[ch] = (response if prior is None else
                               [(a+b)/2 for a, b in zip(prior, response)])
            self.reason = f"A{ch} response={own:+.3f}px/deg"

    def committed(self, signal, moves):
        """Call only after successful software send, with actual limited deltas."""
        self.pending = (dict(signal), list(moves))

    def plan(self, signal, angles):
        """Return (channel, absolute float angle, actual delta, coordinate index).

        Fixed mode retains simultaneous 1-degree moves. Adaptive/fallback moves
        one channel, then requires a fresh batch before another decision.
        """
        if not valid_signal(signal):
            raise ValueError("Invalid double-eye measurement")
        if self.stalls >= self.config.max_stalls:
            self.reason = "stopped: no progress"
            return []
        errors = self.vector(signal)
        candidates = []
        for eye, start in (("L", 0), ("R", 2)):
            if signal[eye+"_dist"] <= self.tolerance:
                continue
            selected = [i for i in (start, start+1) if abs(errors[i]) > self.tolerance]
            if not selected:
                # Radial fail with both components inside deadband: move the
                # larger nonzero component. Never permanently lock an axis.
                selected = [max((start, start+1), key=lambda i: abs(errors[i]))]
            candidates.extend(i for i in selected if errors[i] != 0)
        candidates.sort(key=lambda i: abs(errors[i]), reverse=True)
        moves = []
        for axis in candidates:
            ch, _, positive, negative = next(a for a in AXES if a[1] == axis)
            current = float(angles[ch])
            lo, hi = self.limits[ch]
            if not (math.isfinite(current) and math.isfinite(lo) and math.isfinite(hi)
                    and lo <= current <= hi):
                raise ValueError(f"A{ch}: invalid cached angle or limits")
            sign = self.motion_delta(ch, positive if errors[axis] > 0 else negative, 1.0)
            if sign not in (-1.0, 1.0):
                raise ValueError("Measured rule must return +/-1 degree")
            model = self.models.get(ch) if not self.fallback else None
            step = 1.0
            if self.mode == "adaptive" and model:
                desired = self.config.damping * abs(errors[axis]/model[axis])
                step = max(self.config.min_step_deg, min(self.config.max_step_deg,
                           math.floor(desired/self.config.min_step_deg)*self.config.min_step_deg))
            elif self.mode == "adaptive" and not self.fallback:
                step = self.config.probe_deg
            while step > 0:
                angle = max(lo, min(hi, current + sign*step))
                delta = angle-current
                if abs(delta) < 1e-9:
                    break
                if self.mode == "adaptive" and model:
                    predicted = [a+b*delta for a, b in zip(errors, model)]
                    norms = [math.hypot(*errors[i:i+2]) for i in (0, 2)]
                    after = [math.hypot(*predicted[i:i+2]) for i in (0, 2)]
                    other = 1-axis//2
                    safe = (sum(after) < sum(norms)
                            and after[other] <= max(self.tolerance, norms[other])+self.config.growth_px
                            and abs(predicted[axis]) < abs(errors[axis]))
                    if not safe:
                        step = round(step-self.config.min_step_deg, 10)
                        continue
                moves.append((ch, angle, delta, axis))
                break
            if moves and self.mode == "adaptive":
                break
        if not moves:
            self.stalls += 1
            self.reason = "no safe move: limits, resolution, coupling or noise floor"
        return moves


def hud_lines(mode="fixed", config=None):
    c = config or AdaptiveConfig()
    return (
        f"EyeCtl {mode.upper()} probe{c.probe_deg:g} step{c.min_step_deg:g}-{c.max_step_deg:g} "
        f"gain{c.damping:g} resp>={c.min_response_px_deg:g}px/d",
        f"Safe SD{c.max_std_px:g} drift{c.max_drift_px:g} grow{c.growth_px:g} "
        f"cross{c.max_cross_ratio:g} ratio{c.max_response_ratio:g} stall{c.max_stalls} N60t10 try360",
    )
