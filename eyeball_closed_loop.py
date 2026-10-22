"""Eye-only orchestration; injectable tuner interface permits hardware-free tests."""
import time

from eyeball_adaptive import AdaptiveConfig, EyeballStepPlanner, valid_signal


def run_eyeball_loop(tuner, max_iterations=50, wait_seconds=1.0,
                     keep_display=False, mode="fixed", log_func=None):
    config = AdaptiveConfig()
    scorer = tuner.scorer

    def log(text):
        print(text)
        if log_func:
            log_func(text)

    if scorer.baseline is None:
        log("[Eyeball] No baseline; no movement.")
        return False, 0, {}
    channels = (10, 11, 12, 13)
    baseline_signature = repr(scorer.baseline)
    limits = {ch: tuner.angle_ranges[tuner.eye_servo_channels.index(ch)] for ch in channels}
    angles = {ch: float(tuner.controller.list_temp_deg[ch]) for ch in channels}
    planner = EyeballStepPlanner(limits, tuner._motion_delta, mode=mode,
                                 tolerance=tuner.eyeball_tolerance, config=config)
    scorer.eyeball_control_mode = mode
    original_flags = scorer.enable_mp, scorer.enable_ellseg
    scorer.start_display()
    iteration, failures = 0, 0
    log(f"[Eyeball] mode={mode}; full 60/10/40, each eye <= {planner.tolerance}px; max step 3deg")

    def wait():
        deadline = time.monotonic() + max(0, wait_seconds)
        while not scorer.user_pressed_stop:
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(0.1, remaining))

    try:
        for iteration in range(1, max_iterations+1):
            if scorer.user_pressed_stop:
                log("[Eyeball] User stop; no further command.")
                break
            # Reset metadata: an older successful batch must never validate this one.
            tuner._last_guide_sample_info = None
            signal = tuner.collect_guide_samples(n_frames=60, trim=10)
            info = tuner._last_guide_sample_info or {}
            if scorer.user_pressed_stop:
                log("[Eyeball] Stopped during sampling; discarded batch.")
                break
            if repr(scorer.baseline) != baseline_signature:
                log("[Eyeball] Baseline changed during the run; stop without fitting or moving.")
                break
            complete = valid_signal(signal) and info.get("count") == 60
            stable = (info.get("std", float("inf")) <= config.max_std_px
                      and info.get("drift", float("inf")) <= config.max_drift_px)
            if not complete or (mode == "adaptive" and not stable):
                failures += 1
                planner.degrade("invalid/incomplete batch" if not complete else "unstable batch")
                scorer.eyeball_control_mode = "fallback" if mode == "adaptive" else mode
                log(f"[Eyeball {iteration}] {planner.reason}; no movement; quality={info}")
                if failures >= config.max_stalls:
                    log("[Eyeball] Stopped: consecutive unusable batches.")
                    break
                wait()
                continue
            failures = 0
            planner.observe(signal)
            scorer.eyeball_control_mode = "fallback" if mode == "adaptive" and planner.fallback else mode
            log(f"[Eyeball {iteration}] L({signal['L_dX']:+.3f},{signal['L_dY']:+.3f}) "
                f"d={signal['L_dist']:.3f}; R({signal['R_dX']:+.3f},{signal['R_dY']:+.3f}) "
                f"d={signal['R_dist']:.3f}; {planner.reason}")
            scorer.update_hud([
                (f"Eye {mode} {iteration}/{max_iterations}", (10, 100), (200, 200, 200)),
                (planner.reason, (10, 118), (0, 200, 255)),
                (f"L:{signal['L_dist']:.3f} R:{signal['R_dist']:.3f} <= {planner.tolerance:g}px",
                 (10, 136), (200, 200, 200)),
            ])
            # Exactly the original average-of-distances acceptance, never the
            # distance of averaged dx/dy, nor a predicted post-move error.
            if signal["L_dist"] <= planner.tolerance and signal["R_dist"] <= planner.tolerance:
                log(f"[Eyeball] PASSED after {iteration} full measurement batches.")
                scorer.update_hud([("Eye PASSED (60/10/40)", (10, 100), (0, 255, 0))])
                return True, iteration, {}
            if iteration == max_iterations:
                log("[Eyeball] Iteration limit; no unverified final movement.")
                break
            moves = planner.plan(signal, angles)
            if not moves:
                log(f"[Eyeball] {planner.reason} ({planner.stalls}/{config.max_stalls})")
                if planner.stalls >= config.max_stalls:
                    break
                wait()
                continue
            # Last stop check is deliberately adjacent to the sole send point.
            if scorer.user_pressed_stop:
                break
            if not tuner.send_servos([m[0] for m in moves], [m[1] for m in moves],
                                     [limits[m[0]] for m in moves]):
                log("[Eyeball] Send failed; stop without advancing the software model.")
                break
            for ch, angle, _, _ in moves:
                angles[ch] = angle
            planner.committed(signal, moves)
            log("[Eyeball] " + " ".join(f"A{ch}={a:.3f} (delta {d:+.3f}deg)" for ch, a, d, _ in moves))
            wait()
        log("[Eyeball] NOT PASSED; stopped without homing.")
        return False, iteration, {}
    finally:
        scorer.enable_mp, scorer.enable_ellseg = original_flags
        if not keep_display:
            scorer.stop_display()
