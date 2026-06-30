"""EyeAutoTuner 的侧面嘴唇采样与基线保存逻辑。"""

import time

import cv2

from ellseg_scorer import save_lower_lip_baseline, save_upper_lip_baseline
from eye_constants import (
    LOWER_LIP_SIDE_A_DELTA,
    LOWER_LIP_SIDE_CAMERA_INDEX,
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
)
from lip_front_detector import find_lip_front_points


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
    def _ensure_side_cap(self):
        if self.side_cap is not None and self.side_cap.isOpened():
            return self.side_cap
        cap = cv2.VideoCapture(UPPER_LIP_SIDE_CAMERA_INDEX, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"[UpperLip Side] Cannot open camera {UPPER_LIP_SIDE_CAMERA_INDEX}")
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, SIDE_CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SIDE_CAMERA_HEIGHT)
        self.side_cap = cap
        self._owns_side_cap = True
        print(f"[UpperLip Side] Camera {UPPER_LIP_SIDE_CAMERA_INDEX} opened")
        return self.side_cap

    def collect_upper_lip_side_samples(self, n_frames: int = SIDE_SAMPLE_FRAMES, trim: int = SIDE_SAMPLE_TRIM):
        cap = self._ensure_side_cap()
        if cap is None:
            return None

        baseline = self.scorer.upper_lip_baseline or {}
        roi = baseline.get("side_roi", UPPER_LIP_SIDE_ROI)
        side_window = f"Side Camera {UPPER_LIP_SIDE_CAMERA_INDEX} - Upper Lip"
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
        cv2.putText(frame, f"Side cam {UPPER_LIP_SIDE_CAMERA_INDEX} UpperLip ROI",
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
        side_window = f"Side Camera {LOWER_LIP_SIDE_CAMERA_INDEX} - Lips"
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
        cv2.putText(frame, f"Side cam {LOWER_LIP_SIDE_CAMERA_INDEX} LowerLip ROI",
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
