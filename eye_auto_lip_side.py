"""EyeAutoTuner 的侧面嘴唇采样与基线保存逻辑。"""

import time

import cv2

from ellseg_scorer import (
    load_side_profile_slope_baseline,
    save_lower_lip_baseline,
    save_upper_lip_baseline,
)
from eye_constants import (
    LOWER_LIP_SIDE_A_DELTA,
    LOWER_LIP_SIDE_FACE_DIRECTION,
    LOWER_LIP_SIDE_FLIP_VERTICAL,
    LOWER_LIP_SIDE_MIN_AREA,
    LOWER_LIP_SIDE_MIN_SAT,
    LOWER_LIP_SIDE_ROI,
    LOWER_LIP_SIDE_SCORE_PCT,
    LOWER_LIP_SIDE_SPLIT_PCT,
    LOWER_LIP_SIDE_X_TOLERANCE,
    LOWER_LIP_SIDE_Y_TOLERANCE,
    UPPER_LIP_SIDE_A_DELTA,
    UPPER_LIP_SIDE_CAMERA_INDEX,
    UPPER_LIP_SIDE_FACE_DIRECTION,
    UPPER_LIP_SIDE_FLIP_VERTICAL,
    UPPER_LIP_SIDE_MIN_AREA,
    UPPER_LIP_SIDE_MIN_SAT,
    UPPER_LIP_SIDE_ROI,
    UPPER_LIP_SIDE_SCORE_PCT,
    UPPER_LIP_SIDE_SPLIT_PCT,
    UPPER_LIP_SIDE_X_TOLERANCE,
    UPPER_LIP_SIDE_Y_TOLERANCE,
    LOWER_LIP_SIDE_CHANNEL,
    MOUTH_CHIN_CHANNEL,
    SIDE_PROFILE_BACKGROUND_DELTA,
    SIDE_PROFILE_CHIN_LINE_TOLERANCE_PX,
    SIDE_PROFILE_CROP_SIZE,
    SIDE_PROFILE_FACE_DIRECTION,
    SIDE_PROFILE_LIP_CHIN_TARGET_SLOPE,
    SIDE_PROFILE_LIP_MIN_AREA,
    SIDE_PROFILE_LOWER_LINE_TOLERANCE_PX,
    SIDE_PROFILE_NOSE_UPPER_TARGET_SLOPE,
    SIDE_PROFILE_NOSE_EDGE_CONTRAST_MIN,
    SIDE_PROFILE_NOSE_X_RANGE,
    SIDE_PROFILE_NOSE_Y_RANGE,
    SIDE_PROFILE_SAMPLE_FRAMES,
    SIDE_PROFILE_SLOPE_TOLERANCE,
    SIDE_PROFILE_TRIM_FRAMES,
    SIDE_PROFILE_WAIT_SECONDS,
    UPPER_LIP_CHANNEL,
)
from lip_front_detector import (
    center_square_rect,
    draw_side_profile_geometry,
    find_lip_front_points,
    find_side_profile_points,
    measure_side_profile_geometry,
)


SIDE_CAMERA_WIDTH = 1920
SIDE_CAMERA_HEIGHT = 1080
SIDE_READ_RETRY_SECONDS = 0.01
SIDE_NO_TIP_RETRY_SECONDS = 0.005
SIDE_SAMPLE_FRAMES = 30
SIDE_SAMPLE_TRIM = 5
SIDE_HUD_PANEL_TOP_LEFT = (6, 8)
SIDE_HUD_PANEL_BOTTOM_RIGHT = (620, 176)
SIDE_HUD_ALPHA = 0.58


class LipSideSamplingMixin:
    def _side_camera_index(self):
        return int(getattr(self, "side_camera_index", UPPER_LIP_SIDE_CAMERA_INDEX))

    def _ensure_side_cap(self):
        if self.side_cap is not None and self.side_cap.isOpened():
            return self.side_cap
        camera_index = self._side_camera_index()
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"[UpperLip Side] Cannot open camera {camera_index}")
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, SIDE_CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SIDE_CAMERA_HEIGHT)
        self.side_cap = cap
        self._owns_side_cap = True
        print(f"[UpperLip Side] Camera {camera_index} opened")
        return self.side_cap

    def collect_upper_lip_side_samples(self, n_frames: int = SIDE_SAMPLE_FRAMES, trim: int = SIDE_SAMPLE_TRIM):
        cap = self._ensure_side_cap()
        if cap is None:
            return None

        baseline = self.scorer.upper_lip_baseline or {}
        roi = baseline.get("side_roi", UPPER_LIP_SIDE_ROI)
        side_window = f"Side Camera {self._side_camera_index()} - Upper Lip"
        samples = []
        collected = 0
        while collected < n_frames:
            if self.scorer.user_pressed_stop:
                break
            ok, frame = cap.read()
            if not ok:
                time.sleep(SIDE_READ_RETRY_SECONDS)
                continue
            if UPPER_LIP_SIDE_FLIP_VERTICAL:
                frame = cv2.flip(frame, 0)
            result = find_lip_front_points(
                frame,
                tuple(roi),
                score_pct=UPPER_LIP_SIDE_SCORE_PCT,
                a_delta=UPPER_LIP_SIDE_A_DELTA,
                min_sat=UPPER_LIP_SIDE_MIN_SAT,
                split_pct=UPPER_LIP_SIDE_SPLIT_PCT,
                min_area=UPPER_LIP_SIDE_MIN_AREA,
                face_direction=UPPER_LIP_SIDE_FACE_DIRECTION,
            )
            tip = result["upper"]
            self._draw_upper_lip_side_sample_hud(frame, roi, tip, baseline)
            cv2.imshow(side_window, frame)
            self._pump_cv_events()
            if tip is None:
                time.sleep(SIDE_NO_TIP_RETRY_SECONDS)
                continue
            samples.append(tip)
            collected += 1
            self.scorer.update_hud([
                (f"UpperLip Side Sampling: {collected}/{n_frames}", (10, 136), (200, 200, 200)),
            ])

        if len(samples) < 2 * trim + 1:
            print(f"  [UpperLip Side] valid samples not enough: {len(samples)}, need >= {2*trim+1}")
            return None

        xs = sorted(point[0] for point in samples)[trim:-trim]
        ys = sorted(point[1] for point in samples)[trim:-trim]
        tip = (sum(xs) / len(xs), sum(ys) / len(ys))
        ref = baseline.get("side_upper_tip")
        out = {
            "tip": tip,
            "roi": list(roi),
            "qualified": False,
        }
        if ref is not None:
            dx = tip[0] - float(ref[0])
            dy = tip[1] - float(ref[1])
            out.update({
                "ref": ref,
                "dx": dx,
                "dy": dy,
                "qualified": abs(dx) <= UPPER_LIP_SIDE_X_TOLERANCE and abs(dy) <= UPPER_LIP_SIDE_Y_TOLERANCE,
            })
        print(f"  [UpperLip Side] tip=({tip[0]:.1f},{tip[1]:.1f}) roi={roi}")
        return out

    def _draw_upper_lip_side_sample_hud(self, frame, roi, tip, baseline):
        x1, y1, x2, y2 = [int(value) for value in roi]
        self._draw_hud_panel(frame, SIDE_HUD_PANEL_TOP_LEFT, SIDE_HUD_PANEL_BOTTOM_RIGHT)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 80, 255), 2)
        cv2.putText(frame, f"Side cam {self._side_camera_index()} UpperLip ROI",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"ROI=[{x1},{y1},{x2},{y2}]  Dir={UPPER_LIP_SIDE_FACE_DIRECTION}  VFlip={UPPER_LIP_SIDE_FLIP_VERTICAL}",
                    (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Tol: X<={UPPER_LIP_SIDE_X_TOLERANCE:.1f}px Y<={UPPER_LIP_SIDE_Y_TOLERANCE:.1f}px",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 255), 1, cv2.LINE_AA)
        if tip is not None:
            cv2.circle(frame, tip, 7, (0, 0, 255), -1)
            cv2.circle(frame, tip, 12, (255, 255, 255), 2)
            cv2.putText(frame, f"Upper tip: x={tip[0]} y={tip[1]}",
                        (10, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Upper tip: not found",
                        (10, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)
        if isinstance(baseline, dict) and baseline.get("side_upper_tip") is not None:
            ref = [int(value) for value in baseline["side_upper_tip"]]
            cv2.circle(frame, tuple(ref), 6, (0, 255, 255), -1)
            cv2.putText(frame, f"Ref: x={ref[0]} y={ref[1]}",
                        (10, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
            if tip is not None:
                dx = tip[0] - ref[0]
                dy = tip[1] - ref[1]
                ok = abs(dx) <= UPPER_LIP_SIDE_X_TOLERANCE and abs(dy) <= UPPER_LIP_SIDE_Y_TOLERANCE
                color = (0, 255, 0) if ok else (0, 0, 255)
                cv2.putText(frame, f"dX={dx:+.1f} dY={dy:+.1f} {'OK' if ok else 'BAD'}",
                            (10, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
                cv2.line(frame, tip, tuple(ref), color, 1)
        else:
            cv2.putText(frame, "Ref: missing, click Save upper lip (front+side)",
                        (10, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 1, cv2.LINE_AA)

    @staticmethod
    def _draw_hud_panel(frame, top_left, bottom_right, alpha=SIDE_HUD_ALPHA):
        x1, y1 = top_left
        x2, y2 = bottom_right
        h, w = frame.shape[:2]
        x1 = max(0, min(w - 1, int(x1)))
        y1 = max(0, min(h - 1, int(y1)))
        x2 = max(0, min(w - 1, int(x2)))
        y2 = max(0, min(h - 1, int(y2)))
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (18, 18, 18), -1)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), 1)

    @staticmethod
    def _pump_cv_events():
        try:
            cv2.pollKey()
        except AttributeError:
            cv2.waitKey(1)

    def save_current_upper_lip_side_baseline(self):
        if self.scorer.upper_lip_baseline is None or "ulr" not in self.scorer.upper_lip_baseline:
            print("[UpperLip Side] Save front ULR baseline first.")
            return False
        side = self.collect_upper_lip_side_samples(n_frames=SIDE_SAMPLE_FRAMES, trim=SIDE_SAMPLE_TRIM)
        if side is None:
            return False
        tip = [int(round(side["tip"][0])), int(round(side["tip"][1]))]
        roi = side["roi"]
        ulr = self.scorer.upper_lip_baseline["ulr"]
        self.scorer.upper_lip_baseline = {
            **self.scorer.upper_lip_baseline,
            "side_upper_tip": tip,
            "side_roi": roi,
        }
        save_upper_lip_baseline(ulr=ulr, side_upper_tip=tip, side_roi=roi)
        print(f"[UpperLip Side] Saved side upper tip={tip}, roi={roi}")
        return True

    def collect_lower_lip_side_samples(self, n_frames: int = SIDE_SAMPLE_FRAMES, trim: int = SIDE_SAMPLE_TRIM):
        cap = self._ensure_side_cap()
        if cap is None:
            return None

        baseline = self.scorer.lower_lip_baseline or {}
        roi = baseline.get("side_roi", LOWER_LIP_SIDE_ROI)
        side_window = f"Side Camera {self._side_camera_index()} - Lips"
        samples = []
        collected = 0
        while collected < n_frames:
            if self.scorer.user_pressed_stop:
                break
            ok, frame = cap.read()
            if not ok:
                time.sleep(SIDE_READ_RETRY_SECONDS)
                continue
            if LOWER_LIP_SIDE_FLIP_VERTICAL:
                frame = cv2.flip(frame, 0)
            result = find_lip_front_points(
                frame,
                tuple(roi),
                score_pct=LOWER_LIP_SIDE_SCORE_PCT,
                a_delta=LOWER_LIP_SIDE_A_DELTA,
                min_sat=LOWER_LIP_SIDE_MIN_SAT,
                split_pct=LOWER_LIP_SIDE_SPLIT_PCT,
                min_area=LOWER_LIP_SIDE_MIN_AREA,
                face_direction=LOWER_LIP_SIDE_FACE_DIRECTION,
            )
            tip = result["lower"]
            self._draw_lower_lip_side_sample_hud(frame, roi, tip, baseline)
            cv2.imshow(side_window, frame)
            self._pump_cv_events()
            if tip is None:
                time.sleep(SIDE_NO_TIP_RETRY_SECONDS)
                continue
            samples.append(tip)
            collected += 1
            self.scorer.update_hud([
                (f"LowerLip Side Sampling: {collected}/{n_frames}", (10, 136), (200, 200, 200)),
            ])

        if len(samples) < 2 * trim + 1:
            print(f"  [LowerLip Side] valid samples not enough: {len(samples)}, need >= {2*trim+1}")
            return None

        xs = sorted(point[0] for point in samples)[trim:-trim]
        ys = sorted(point[1] for point in samples)[trim:-trim]
        tip = (sum(xs) / len(xs), sum(ys) / len(ys))
        ref = baseline.get("side_lower_tip")
        out = {
            "tip": tip,
            "roi": list(roi),
            "qualified": False,
        }
        if ref is not None:
            dx = tip[0] - float(ref[0])
            dy = tip[1] - float(ref[1])
            out.update({
                "ref": ref,
                "dx": dx,
                "dy": dy,
                "qualified": abs(dx) <= LOWER_LIP_SIDE_X_TOLERANCE and abs(dy) <= LOWER_LIP_SIDE_Y_TOLERANCE,
            })
        print(f"  [LowerLip Side] tip=({tip[0]:.1f},{tip[1]:.1f}) roi={roi}")
        return out

    def _draw_lower_lip_side_sample_hud(self, frame, roi, tip, baseline):
        x1, y1, x2, y2 = [int(value) for value in roi]
        self._draw_hud_panel(frame, SIDE_HUD_PANEL_TOP_LEFT, SIDE_HUD_PANEL_BOTTOM_RIGHT)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 80, 80), 2)
        cv2.putText(frame, f"Side cam {self._side_camera_index()} LowerLip ROI",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"ROI=[{x1},{y1},{x2},{y2}]  Dir={LOWER_LIP_SIDE_FACE_DIRECTION}  VFlip={LOWER_LIP_SIDE_FLIP_VERTICAL}",
                    (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Tol: X<={LOWER_LIP_SIDE_X_TOLERANCE:.1f}px Y<={LOWER_LIP_SIDE_Y_TOLERANCE:.1f}px",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 255), 1, cv2.LINE_AA)
        if tip is not None:
            cv2.circle(frame, tip, 7, (255, 0, 180), -1)
            cv2.circle(frame, tip, 12, (255, 255, 255), 2)
            cv2.putText(frame, f"Lower tip: x={tip[0]} y={tip[1]}",
                        (10, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Lower tip: not found",
                        (10, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)
        if isinstance(baseline, dict) and baseline.get("side_lower_tip") is not None:
            ref = [int(value) for value in baseline["side_lower_tip"]]
            cv2.circle(frame, tuple(ref), 6, (0, 255, 255), -1)
            cv2.putText(frame, f"Ref: x={ref[0]} y={ref[1]}",
                        (10, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
            if tip is not None:
                dx = tip[0] - ref[0]
                dy = tip[1] - ref[1]
                ok = abs(dx) <= LOWER_LIP_SIDE_X_TOLERANCE and abs(dy) <= LOWER_LIP_SIDE_Y_TOLERANCE
                color = (0, 255, 0) if ok else (0, 0, 255)
                cv2.putText(frame, f"dX={dx:+.1f} dY={dy:+.1f} {'OK' if ok else 'BAD'}",
                            (10, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
                cv2.line(frame, tip, tuple(ref), color, 1)
        else:
            cv2.putText(frame, "Ref: missing, click Save lower lip (front+side)",
                        (10, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 1, cv2.LINE_AA)

    def save_current_lower_lip_side_baseline(self):
        if self.scorer.lower_lip_baseline is None or "llr" not in self.scorer.lower_lip_baseline:
            print("[LowerLip Side] Save front LLR baseline first.")
            return False
        side = self.collect_lower_lip_side_samples(n_frames=SIDE_SAMPLE_FRAMES, trim=SIDE_SAMPLE_TRIM)
        if side is None:
            return False
        tip = [int(round(side["tip"][0])), int(round(side["tip"][1]))]
        roi = side["roi"]
        llr = self.scorer.lower_lip_baseline["llr"]
        self.scorer.lower_lip_baseline = {
            **self.scorer.lower_lip_baseline,
            "side_lower_tip": tip,
            "side_roi": roi,
        }
        save_lower_lip_baseline(llr=llr, side_lower_tip=tip, side_roi=roi)
        print(f"[LowerLip Side] Saved side lower tip={tip}, roi={roi}")
        return True

    def check_upper_lip_front_side(self):
        front = self.collect_upper_lip_samples(n_frames=SIDE_SAMPLE_FRAMES, trim=SIDE_SAMPLE_TRIM)
        side = self.collect_upper_lip_side_samples(n_frames=SIDE_SAMPLE_FRAMES, trim=SIDE_SAMPLE_TRIM)
        if front is None or side is None:
            return {"qualified": False, "front": front, "side": side}
        ok = bool(front["qualified"] and side.get("qualified"))
        return {"qualified": ok, "front": front, "side": side}

    def _side_profile_target_slopes(self):
        if not hasattr(self, "_side_profile_slope_baseline_cache"):
            self._side_profile_slope_baseline_cache = load_side_profile_slope_baseline()
        baseline = self._side_profile_slope_baseline_cache
        if not isinstance(baseline, dict):
            baseline = {}
        return (
            float(baseline.get(
                "nose_upper_slope", SIDE_PROFILE_NOSE_UPPER_TARGET_SLOPE)),
            float(baseline.get(
                "lip_chin_slope", SIDE_PROFILE_LIP_CHIN_TARGET_SLOPE)),
            "saved" if baseline else "default",
        )

    def _measure_side_profile(self, points):
        nose_upper_target, lip_chin_target, _mode = self._side_profile_target_slopes()
        return measure_side_profile_geometry(
            points,
            face_direction=SIDE_PROFILE_FACE_DIRECTION,
            nose_upper_target_slope=nose_upper_target,
            lip_chin_target_slope=lip_chin_target,
            slope_tolerance=SIDE_PROFILE_SLOPE_TOLERANCE,
            lower_line_tolerance_px=SIDE_PROFILE_LOWER_LINE_TOLERANCE_PX,
            chin_line_tolerance_px=SIDE_PROFILE_CHIN_LINE_TOLERANCE_PX,
        )

    def _draw_side_profile_sample_hud(self, frame, signal, phase_text="sampling"):
        h, w = frame.shape[:2]
        x = max(10, w - 650)
        self._draw_hud_panel(frame, (x - 8, 8), (w - 8, 164))
        cv2.putText(
            frame,
            f"Side Profile {SIDE_PROFILE_CROP_SIZE}x{SIDE_PROFILE_CROP_SIZE} {phase_text}",
            (x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"Center crop / no saved ROI / dir={SIDE_PROFILE_FACE_DIRECTION}",
            (x, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 220, 255), 1, cv2.LINE_AA)
        if not signal or not signal.get("valid"):
            reason = (signal or {}).get("reason", "points not found")
            cv2.putText(
                frame, f"N/U/L/C: INVALID ({reason})", (x, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 1, cv2.LINE_AA)
            return
        p1_color = (0, 255, 0) if signal["phase1_qualified"] else (0, 0, 255)
        line_ok = signal["lower_qualified"] and signal["chin_qualified"] and signal["slope_qualified"]
        line_color = (0, 255, 0) if line_ok else (0, 0, 255)
        cv2.putText(
            frame,
            f"N-U slope={signal['nose_upper_slope']:.2f}/"
            f"{signal['nose_upper_target_slope']:.2f} "
            f"tol={SIDE_PROFILE_SLOPE_TOLERANCE:.2f}",
            (x, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.48, p1_color, 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"U-C slope={signal['lip_chin_slope']:.2f}/"
            f"{signal['lip_chin_target_slope']:.2f} "
            f"tol={SIDE_PROFILE_SLOPE_TOLERANCE:.2f}",
            (x, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.48, line_color, 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"Line err Lower={signal['lower_line_error_px']:+.1f}px "
            f"Chin={signal['chin_line_error_px']:+.1f}px (+forward)",
            (x, 134), cv2.FONT_HERSHEY_SIMPLEX, 0.48, line_color, 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"Tol Lower<={SIDE_PROFILE_LOWER_LINE_TOLERANCE_PX:.1f}px "
            f"Chin<={SIDE_PROFILE_CHIN_LINE_TOLERANCE_PX:.1f}px",
            (x, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 220, 255), 1, cv2.LINE_AA)

    def collect_side_profile_samples(
        self,
        n_frames: int = SIDE_PROFILE_SAMPLE_FRAMES,
        trim: int = SIDE_PROFILE_TRIM_FRAMES,
        phase_text: str = "sampling",
    ):
        """采集画面中心 400x400 内的鼻/上唇/下唇/下巴端点，不读取旧 ROI。"""
        cap = self._ensure_side_cap()
        if cap is None:
            return None
        samples = []
        attempts = 0
        max_attempts = max(n_frames * 10, n_frames + 10)
        side_window = f"Side Camera {self._side_camera_index()} - Lips"
        while len(samples) < n_frames and attempts < max_attempts:
            if self.scorer.user_pressed_stop:
                break
            attempts += 1
            ok, frame = cap.read()
            if not ok:
                time.sleep(SIDE_READ_RETRY_SECONDS)
                continue
            if UPPER_LIP_SIDE_FLIP_VERTICAL:
                frame = cv2.flip(frame, 0)
            detected = find_side_profile_points(
                frame,
                crop_size=SIDE_PROFILE_CROP_SIZE,
                face_direction=SIDE_PROFILE_FACE_DIRECTION,
                background_delta=SIDE_PROFILE_BACKGROUND_DELTA,
                lip_min_area=SIDE_PROFILE_LIP_MIN_AREA,
                nose_roi=getattr(self.scorer, "side_nose_roi", None),
                require_nose_roi=True,
                upper_lip_roi=(self.scorer.upper_lip_baseline or {}).get("side_roi"),
                lower_lip_roi=(self.scorer.lower_lip_baseline or {}).get("side_roi"),
                separate_lip_rois=True,
                chin_roi=getattr(self.scorer, "side_chin_roi", None),
                require_feature_rois=True,
                nose_edge_contrast_min=SIDE_PROFILE_NOSE_EDGE_CONTRAST_MIN,
            )
            signal = self._measure_side_profile(detected.get("points") or {})
            signal["rect"] = detected.get("rect")
            signal["nose_search_rect"] = detected.get("nose_search_rect")
            signal["mouth_search_rect"] = detected.get("mouth_search_rect")
            signal["upper_search_rect"] = detected.get("upper_search_rect")
            signal["lower_search_rect"] = detected.get("lower_search_rect")
            signal["chin_search_rect"] = detected.get("chin_search_rect")
            draw_side_profile_geometry(
                frame, signal,
                show_face_overlays=bool(getattr(self.scorer, "show_face_overlays", True)))
            self._draw_side_profile_sample_hud(frame, signal, phase_text)
            cv2.imshow(side_window, frame)
            self._pump_cv_events()
            if not signal.get("valid"):
                time.sleep(SIDE_NO_TIP_RETRY_SECONDS)
                continue
            samples.append(signal["points"])
            self.scorer.update_hud([
                (f"SideProfile {phase_text}: {len(samples)}/{n_frames}",
                 (10, 136), (200, 200, 200)),
            ])

        if len(samples) < 2 * trim + 1:
            print(
                f"  [SideProfile] valid samples not enough: {len(samples)}, "
                f"need >= {2 * trim + 1} (attempts={attempts})")
            return None

        averaged_points = {}
        for name in ("nose", "upper_lip", "lower_lip", "chin"):
            averaged = []
            for axis in (0, 1):
                values = sorted(float(sample[name][axis]) for sample in samples)
                kept = values[trim:-trim] if trim > 0 else values
                averaged.append(sum(kept) / len(kept))
            averaged_points[name] = tuple(averaged)
        signal = self._measure_side_profile(averaged_points)
        # 绘制时仍使用当前画面中心框，不保存任何 ROI。
        signal["rect"] = center_square_rect(frame, SIDE_PROFILE_CROP_SIZE)
        signal["sample_frames"] = len(samples)
        signal["trim_frames"] = trim
        return signal

    def auto_adjust_side_profile_alignment(
        self,
        step: float = 1.0,
        max_iterations: int = 100,
        wait_seconds: float = SIDE_PROFILE_WAIT_SECONDS,
        keep_display: bool = False,
        log_func=None,
    ) -> tuple:
        """
        独立的侧面轮廓路线：
        P1 只调 A16，使鼻尖—上唇斜率到位后锁定；
        P2 只调 A18/A26，使下唇和下巴落在上唇起始的目标斜率线上。
        """
        emit = log_func or print
        channels = {
            "upper": UPPER_LIP_CHANNEL,
            "lower": LOWER_LIP_SIDE_CHANNEL,
            "chin": MOUTH_CHIN_CHANNEL,
        }
        ranges = {name: self._angle_range(channel) for name, channel in channels.items()}
        angles = {
            name: self._temp_deg(channel, default=round(sum(ranges[name]) / 2))
            for name, channel in channels.items()
        }
        phase = 1
        final_signal = {}
        missing_rounds = 0
        self._side_profile_slope_baseline_cache = load_side_profile_slope_baseline()
        nose_upper_target, lip_chin_target, target_mode = self._side_profile_target_slopes()
        self.scorer.start_display()
        emit(
            f"[侧面轮廓] 中心 {SIDE_PROFILE_CROP_SIZE}x{SIDE_PROFILE_CROP_SIZE}，"
            f"免 ROI；P1 A16 目标斜率={nose_upper_target:.2f}，"
            f"P2 A18/A26 目标斜率={lip_chin_target:.2f} ({target_mode})")
        try:
            for iteration in range(1, max_iterations + 1):
                if self.scorer.user_pressed_stop:
                    return False, iteration - 1, {
                        **final_signal, "stop_reason": "user_stop"}
                phase_text = "P1 nose-upper" if phase == 1 else "P2 lip-chin line"
                self.scorer.update_hud([
                    (f"[SideProfile {phase_text} {iteration}/{max_iterations}]",
                     (10, 100), (200, 200, 200)),
                    (f"A16={angles['upper']} A18={angles['lower']} A26={angles['chin']}",
                     (10, 118), (0, 255, 255)),
                ])
                signal = self.collect_side_profile_samples(
                    n_frames=SIDE_PROFILE_SAMPLE_FRAMES,
                    trim=SIDE_PROFILE_TRIM_FRAMES,
                    phase_text=phase_text,
                )
                if signal is None:
                    missing_rounds += 1
                    emit(f"[侧面轮廓 {iteration}] 有效端点帧不足，不移动舵机")
                    if missing_rounds >= 3:
                        warning = "连续 3 轮无法在中心 400x400 内稳定找到鼻/上唇/下唇/下巴"
                        emit(f"[侧面轮廓] {warning}")
                        return False, iteration, {
                            "stop_reason": "profile_not_found", "warning": warning}
                    time.sleep(wait_seconds)
                    continue
                missing_rounds = 0
                final_signal = signal

                if phase == 1:
                    error = signal["nose_upper_slope_error"]
                    emit(
                        f"[侧面轮廓 P1 {iteration}] N-U slope="
                        f"{signal['nose_upper_slope']:.3f}/"
                        f"{signal['nose_upper_target_slope']:.3f} "
                        f"d={error:+.3f} A16={angles['upper']}")
                    if signal["phase1_qualified"]:
                        phase = 2
                        emit(f"[侧面轮廓 P1] 上唇到位，锁定 A16={angles['upper']}")
                        continue
                    desired = "上" if error > 0.0 else "下"
                    new_angle = round(
                        angles["upper"]
                        + self._motion_delta(channels["upper"], desired, step))
                    new_angle = max(ranges["upper"][0], min(ranges["upper"][1], new_angle))
                    if new_angle == angles["upper"]:
                        warning = f"A16 已到边界，仍无法达到鼻尖—上唇目标斜率"
                        emit(f"[侧面轮廓] {warning}")
                        return False, iteration, {
                            **signal, "stop_reason": "no_legal_move", "warning": warning}
                    angles["upper"] = new_angle
                    self.send_servo(channels["upper"], new_angle, ranges["upper"])
                    emit(f"[侧面轮廓 P1 {iteration}] A16 {desired} -> {new_angle}")
                else:
                    lower_error = signal["lower_line_error_px"]
                    chin_error = signal["chin_line_error_px"]
                    line_ok = (
                        signal["lower_qualified"]
                        and signal["chin_qualified"]
                        and signal["slope_qualified"])
                    emit(
                        f"[侧面轮廓 P2 {iteration}] U-C slope="
                        f"{signal['lip_chin_slope']:.3f}/"
                        f"{signal['lip_chin_target_slope']:.3f} "
                        f"Lerr={lower_error:+.2f}px Cerr={chin_error:+.2f}px")
                    if line_ok:
                        self.scorer.update_hud([
                            ("SIDE PROFILE ALIGNMENT PASSED", (100, 80), (0, 255, 0)),
                            (f"A16={angles['upper']} LOCK A18={angles['lower']} A26={angles['chin']}",
                             (10, 118), (0, 255, 0)),
                        ])
                        emit(
                            f"[侧面轮廓] 通过：A16={angles['upper']}(锁定) "
                            f"A18={angles['lower']} A26={angles['chin']}")
                        return True, iteration, signal

                    moves = []
                    if not signal["lower_qualified"]:
                        effect = "后" if lower_error > 0.0 else "前"
                        moves.append(("lower", effect))
                    if not signal["chin_qualified"] or not signal["slope_qualified"]:
                        effect = "合" if chin_error > 0.0 else "开"
                        moves.append(("chin", effect))

                    move_channels = []
                    move_angles = []
                    move_ranges = []
                    action_text = []
                    for name, effect in moves:
                        channel = channels[name]
                        new_angle = round(
                            angles[name] + self._motion_delta(channel, effect, step))
                        new_angle = max(ranges[name][0], min(ranges[name][1], new_angle))
                        if new_angle == angles[name]:
                            continue
                        angles[name] = new_angle
                        move_channels.append(channel)
                        move_angles.append(new_angle)
                        move_ranges.append(ranges[name])
                        action_text.append(f"A{channel}{effect}->{new_angle}")
                    if not move_channels:
                        warning = "A18/A26 已到边界，下唇/下巴仍未落在目标线"
                        emit(f"[侧面轮廓] {warning}")
                        return False, iteration, {
                            **signal, "stop_reason": "no_legal_move", "warning": warning}
                    self.send_servos(move_channels, move_angles, move_ranges)
                    emit(f"[侧面轮廓 P2 {iteration}] {' '.join(action_text)}")

                waited = 0.0
                while waited < wait_seconds and not self.scorer.user_pressed_stop:
                    chunk = min(0.1, wait_seconds - waited)
                    time.sleep(chunk)
                    waited += chunk

            warning = f"达到最大迭代 {max_iterations} 次仍未通过"
            emit(f"[侧面轮廓] {warning}")
            return False, max_iterations, {
                **final_signal, "stop_reason": "max_iterations", "warning": warning}
        finally:
            if not keep_display:
                self.scorer.stop_display()
