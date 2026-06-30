import os
import queue
import sys
import time
from typing import Any, Callable, Optional

import cv2
import numpy as np

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

try:
    from PySide6.QtCore import QThread, Qt, Signal
    from PySide6.QtWidgets import (
        QApplication,
        QComboBox,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QInputDialog,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )
except ModuleNotFoundError:
    from PyQt5.QtCore import QThread, Qt, pyqtSignal as Signal
    from PyQt5.QtWidgets import (
        QApplication,
        QComboBox,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QInputDialog,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )

from ellseg_scorer import (
    EllSegDetector,
    get_active_template,
    get_template_names,
    load_eyebrow_baseline,
    load_eyelid_baseline,
    load_head_position_baseline,
    load_lower_lip_baseline,
    load_mouth_baseline,
    load_mouth_corners_baseline,
    load_upper_lip_baseline,
    save_lower_lip_side_roi,
    save_upper_lip_side_roi,
    save_current_as_template,
    set_active_template,
    template_section_exists,
)

from eye_auto_tuner import EyeAutoTuner
from utility import read_yaml, write_yaml


DEFAULT_SERVO_CONFIG = "29_servo_config(13).yaml"
DEFAULT_ROUTE_MAX_ITERATIONS = 100
FALLBACK_ROUTE_MAX_ITERATIONS = 50
MAX_ROUTE_ITERATIONS = 500
CAMERA_FRAME_WIDTH = 1920
CAMERA_FRAME_HEIGHT = 1080
SIDE_WINDOW_WIDTH = 960
SIDE_WINDOW_HEIGHT = 540
SIDE_READ_FAIL_FILL = 180
MAIN_LOOP_SLEEP_SECONDS = 0.001
ROUTE_LOOP_SLEEP_SECONDS = 0.01
EYELID_MANUAL_STEP_DEGREE = 1
SIDE_ROI_MIN_SIZE = 12
SIDE_PANEL_LEFT_TOP = (6, 8)
SIDE_PANEL_RIGHT_BOTTOM = (620, 256)
SIDE_PANEL_SELECT_BOTTOM = (620, 286)
SIDE_PANEL_ALPHA = 0.58
SIDE_TITLE_POS = (10, 28)
SIDE_META_POS = (10, 54)
SIDE_ROI_SELECT_POS = (10, 274)
SIDE_UPPER_STATUS_Y = 28
SIDE_LOWER_STATUS_Y = 146
BASELINE_BUTTON_COLUMNS = 2

LIP_TARGET_UPPER = "upper"
LIP_TARGET_LOWER = "lower"


def clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, int(value)))


def int_list(values: Any) -> list[int]:
    return [int(value) for value in values]


def apply_template(template_name: str) -> tuple[bool, str]:
    set_active_template(template_name)
    return True, f"Active template -> {template_name} (face_point.json)"


class DetectorThread(QThread):
    log = Signal(str)
    status = Signal(str)
    stopped = Signal()
    route_finished = Signal(str, bool)
    item_result = Signal(str, str)

    def __init__(self, template_getter: Callable[[], str], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.template_getter = template_getter
        self.commands: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._running = True
        self._route_running = False
        self._route_abort_requested = False
        self.detector = None
        self._last_servo_temp_degrees = None
        self.side_cap = None
        self.side_window_name = f"Side Camera {UPPER_LIP_SIDE_CAMERA_INDEX} - Lips"
        self._upper_side_runtime_roi = None
        self._lower_side_runtime_roi = None
        self._side_roi_select_enabled = False
        self._side_roi_select_target = "upper"
        self._side_roi_dragging = False
        self._side_roi_start = None
        self._side_roi_current = None

    def _create_tuner(self, use_servo: bool) -> Optional[EyeAutoTuner]:
        """按需创建 tuner；只有 use_servo=True 时才打开串口。"""
        tuner = None
        try:
            tuner = EyeAutoTuner(
                yaml_file=DEFAULT_SERVO_CONFIG,
                detector=self.detector,
            )
            tuner.side_cap = self.side_cap
            tuner._owns_side_cap = False
            if use_servo:
                tuner.initialize()
                self._restore_tuner_temp_degrees(tuner)
                self.log.emit("Servo controller opened for current operation.")
            else:
                tuner.scorer = self.detector
                tuner._owns_scorer = False
            return tuner
        except Exception as exc:
            self.log.emit(f"[ERROR] Cannot init servo controller: {exc}")
            if tuner is not None:
                tuner.cleanup()
            return None

    def _restore_tuner_temp_degrees(self, tuner: EyeAutoTuner) -> None:
        if not self._last_servo_temp_degrees or tuner.controller is None:
            return
        count = min(len(self._last_servo_temp_degrees), len(tuner.controller.list_temp_deg))
        tuner.controller.list_temp_deg[:count] = self._last_servo_temp_degrees[:count]

    def _remember_tuner_temp_degrees(self, tuner: Optional[EyeAutoTuner]) -> None:
        if tuner is None or tuner.controller is None:
            return
        self._last_servo_temp_degrees = list(tuner.controller.list_temp_deg)

    def _cleanup_tuner(self, tuner: Optional[EyeAutoTuner]) -> None:
        if tuner is None:
            return
        had_controller = tuner.controller is not None
        self._remember_tuner_temp_degrees(tuner)
        tuner.cleanup()
        if had_controller:
            self.log.emit("Servo controller released.")

    def _reload_template_baselines(self) -> None:
        if self.detector is None:
            return
        self.detector.eyelid_baseline = load_eyelid_baseline()
        self.detector.eyebrow_baseline = load_eyebrow_baseline()
        self.detector.mouth_baseline = load_mouth_baseline()
        self.detector.lower_lip_baseline = load_lower_lip_baseline()
        self.detector.upper_lip_baseline = load_upper_lip_baseline()
        self.detector.mouth_corners_baseline = load_mouth_corners_baseline()
        self.detector.head_position_baseline = load_head_position_baseline()

    def send(self, command: tuple[str, Any]) -> None:
        self.commands.put(command)

    def stop(self) -> None:
        self._running = False
        self.commands.put(("stop", None))

    def request_abort_route(self) -> bool:
        if not self._route_running or self.detector is None:
            return False
        self._route_abort_requested = True
        self.detector.request_user_stop("abort_route")
        self.log.emit("Abort requested: current route will stop at the next safe check. Camera remains open.")
        return True

    def run(self) -> None:
        try:
            self._start_detector()
            self._open_side_camera()
            while self._running:
                if self._route_running:
                    self._idle_while_route_running()
                    continue
                if self._process_detector_frame():
                    self.log.emit("OpenCV window requested stop.")
                    break
                time.sleep(MAIN_LOOP_SLEEP_SECONDS)
        except Exception as exc:
            self.log.emit(f"[ERROR] Detector failed: {exc}")
        finally:
            self._shutdown_detector()

    def _start_detector(self) -> None:
        self.detector = EllSegDetector()
        actual_w = int(self.detector.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.detector.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.log.emit(f"Camera opened: {actual_w}x{actual_h}")
        self.detector.enable_mp = True
        self.detector.enable_ellseg = True
        self.detector.start_display()
        self.log.emit("Calibration camera started. [MP+EllSeg ON by default]")

    def _process_detector_frame(self) -> bool:
        if self.detector is None:
            return False
        ok, frame = self.detector.capture()
        if ok:
            self.detector.detect(frame)
        self._update_side_camera()
        self._drain_commands()
        return bool(self.detector.user_pressed_stop)

    def _idle_while_route_running(self) -> None:
        # 自动路线会复用同一个 detector，主循环暂停采集避免重复读摄像头。
        self._drain_commands()
        time.sleep(ROUTE_LOOP_SLEEP_SECONDS)

    def _shutdown_detector(self) -> None:
        try:
            if self.detector is not None:
                self.detector.stop_display()
                self.detector.close()
            self._close_side_camera()
        finally:
            self.detector = None
            self.status.emit("Camera stopped")
            self.stopped.emit()

    def _open_side_camera(self) -> bool:
        if self.side_cap is not None and self.side_cap.isOpened():
            return True
        cap = cv2.VideoCapture(UPPER_LIP_SIDE_CAMERA_INDEX, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self.log.emit(f"[WARN] Side camera {UPPER_LIP_SIDE_CAMERA_INDEX} failed to open.")
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_HEIGHT)
        self.side_cap = cap
        cv2.namedWindow(self.side_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.side_window_name, SIDE_WINDOW_WIDTH, SIDE_WINDOW_HEIGHT)
        cv2.setMouseCallback(self.side_window_name, self._on_side_mouse)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.log.emit(f"Side camera opened: cam{UPPER_LIP_SIDE_CAMERA_INDEX} {w}x{h}")
        return True

    def _close_side_camera(self) -> None:
        if self.side_cap is not None:
            try:
                self.side_cap.release()
            except Exception:
                pass
            self.side_cap = None
        try:
            cv2.destroyWindow(self.side_window_name)
        except Exception:
            pass

    def _update_side_camera(self) -> None:
        if self.side_cap is None or not self.side_cap.isOpened():
            return
        ok, frame = self.side_cap.read()
        if not ok:
            frame = np.full(
                (SIDE_WINDOW_HEIGHT, SIDE_WINDOW_WIDTH, 3),
                SIDE_READ_FAIL_FILL,
                dtype=np.uint8,
            )
            cv2.putText(frame, f"Side camera {UPPER_LIP_SIDE_CAMERA_INDEX}: read failed",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow(self.side_window_name, frame)
            self._pump_cv_events()
            return
        if UPPER_LIP_SIDE_FLIP_VERTICAL:
            frame = cv2.flip(frame, 0)
        self._draw_side_hud(frame)
        cv2.imshow(self.side_window_name, frame)
        self._pump_cv_events()

    def _pump_cv_events(self) -> None:
        try:
            cv2.pollKey()
        except AttributeError:
            cv2.waitKey(1)

    def _draw_side_hud(self, frame: np.ndarray) -> None:
        upper_baseline = self.detector.upper_lip_baseline if self.detector is not None else None
        lower_baseline = self.detector.lower_lip_baseline if self.detector is not None else None
        upper_roi = self._current_side_roi("upper")
        lower_roi = self._current_side_roi("lower")

        upper_tip = self._find_lip_front_tip(frame, upper_roi, LIP_TARGET_UPPER)
        lower_tip = self._find_lip_front_tip(frame, lower_roi, LIP_TARGET_LOWER)

        panel_bottom = SIDE_PANEL_SELECT_BOTTOM if self._side_roi_select_enabled else SIDE_PANEL_RIGHT_BOTTOM
        self._draw_hud_panel(frame, SIDE_PANEL_LEFT_TOP, panel_bottom)
        self._draw_lip_side_status(
            frame, "Upper", upper_roi, upper_tip, upper_baseline,
            "side_upper_tip", UPPER_LIP_SIDE_X_TOLERANCE, UPPER_LIP_SIDE_Y_TOLERANCE,
            (0, 80, 255), (0, 0, 255), SIDE_UPPER_STATUS_Y,
        )
        self._draw_lip_side_status(
            frame, "Lower", lower_roi, lower_tip, lower_baseline,
            "side_lower_tip", LOWER_LIP_SIDE_X_TOLERANCE, LOWER_LIP_SIDE_Y_TOLERANCE,
            (255, 80, 80), (255, 0, 180), SIDE_LOWER_STATUS_Y,
        )
        cv2.putText(frame, f"Side cam {UPPER_LIP_SIDE_CAMERA_INDEX} Lips",
                    SIDE_TITLE_POS, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Dir={UPPER_LIP_SIDE_FACE_DIRECTION}  VFlip={UPPER_LIP_SIDE_FLIP_VERTICAL}",
                    SIDE_META_POS, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 255), 1, cv2.LINE_AA)

        if self._side_roi_select_enabled:
            cv2.putText(frame, f"ROI SELECT {self._side_roi_select_target}: drag lip area",
                        SIDE_ROI_SELECT_POS, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        if self._side_roi_dragging and self._side_roi_start and self._side_roi_current:
            sx, sy = self._side_roi_start
            cx, cy = self._side_roi_current
            x_min, x_max = sorted((sx, cx))
            y_min, y_max = sorted((sy, cy))
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)

    def _find_lip_front_tip(self, frame: np.ndarray, roi: list[int], target: str) -> Optional[tuple[int, int]]:
        if target == LIP_TARGET_LOWER:
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
            return result.get("lower")

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
        return result.get("upper")

    @staticmethod
    def _draw_lip_side_status(frame, label, roi, tip, baseline, ref_key,
                              tol_x, tol_y, roi_color, tip_color, y):
        x1, y1, x2, y2 = int_list(roi)
        cv2.rectangle(frame, (x1, y1), (x2, y2), roi_color, 2)
        cv2.putText(frame, f"{label} ROI=[{x1},{y1},{x2},{y2}] Tol X<={tol_x:.1f} Y<={tol_y:.1f}",
                    (10, y + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 220, 255), 1, cv2.LINE_AA)
        if tip is not None:
            cv2.circle(frame, tip, 7, tip_color, -1)
            cv2.circle(frame, tip, 12, (255, 255, 255), 2)
            cv2.putText(frame, f"{label} tip: x={tip[0]} y={tip[1]}",
                        (10, y + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, f"{label} tip: not found",
                        (10, y + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        if isinstance(baseline, dict) and baseline.get(ref_key) is not None:
            ref = int_list(baseline[ref_key])
            cv2.circle(frame, tuple(ref), 6, (0, 255, 255), -1)
            if tip is not None:
                dx = tip[0] - ref[0]
                dy = tip[1] - ref[1]
                ok = abs(dx) <= tol_x and abs(dy) <= tol_y
                color = (0, 255, 0) if ok else (0, 0, 255)
                cv2.putText(frame, f"{label} dX={dx:+.1f} dY={dy:+.1f} {'OK' if ok else 'BAD'}",
                            (10, y + 104), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                cv2.line(frame, tip, tuple(ref), color, 1)
            else:
                cv2.putText(frame, f"{label} ref: x={ref[0]} y={ref[1]}",
                            (10, y + 104), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, f"{label} ref: missing",
                        (10, y + 104), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1, cv2.LINE_AA)

    @staticmethod
    def _draw_hud_panel(frame, top_left, bottom_right, alpha=SIDE_PANEL_ALPHA):
        x1, y1 = top_left
        x2, y2 = bottom_right
        h, w = frame.shape[:2]
        x1 = clamp(x1, 0, w - 1)
        y1 = clamp(y1, 0, h - 1)
        x2 = clamp(x2, 0, w - 1)
        y2 = clamp(y2, 0, h - 1)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (18, 18, 18), -1)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), 1)

    def _current_side_roi(self, target: str = LIP_TARGET_UPPER) -> list[int]:
        runtime_roi = self._lower_side_runtime_roi if target == LIP_TARGET_LOWER else self._upper_side_runtime_roi
        if runtime_roi is not None:
            return int_list(runtime_roi)

        baseline = None
        default_roi = UPPER_LIP_SIDE_ROI
        if self.detector is not None and target == LIP_TARGET_LOWER:
            baseline = self.detector.lower_lip_baseline
            default_roi = LOWER_LIP_SIDE_ROI
        elif self.detector is not None:
            baseline = self.detector.upper_lip_baseline

        if isinstance(baseline, dict) and baseline.get("side_roi") is not None:
            return int_list(baseline["side_roi"])
        return int_list(default_roi)

    def _on_side_mouse(self, event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if not self._side_roi_select_enabled:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self._side_roi_dragging = True
            self._side_roi_start = (x, y)
            self._side_roi_current = (x, y)
            return
        if event == cv2.EVENT_MOUSEMOVE and self._side_roi_dragging:
            self._side_roi_current = (x, y)
            return
        if event == cv2.EVENT_LBUTTONUP and self._side_roi_dragging:
            self._finish_side_roi_drag(x, y)

    def _finish_side_roi_drag(self, x: int, y: int) -> None:
        self._side_roi_dragging = False
        self._side_roi_current = (x, y)
        rect = self._normalize_side_roi(self._side_roi_start, self._side_roi_current)
        self._side_roi_start = None
        self._side_roi_current = None
        if rect is None:
            self.log.emit("侧面嘴唇 ROI 过小，未保存。")
            return
        target = self._side_roi_select_target
        self._set_runtime_side_roi(target, rect)
        self._side_roi_select_enabled = False
        self._save_side_roi_to_current_template(rect, target)

    def _set_runtime_side_roi(self, target: str, roi: list[int]) -> None:
        if target == LIP_TARGET_LOWER:
            self._lower_side_runtime_roi = roi
            return
        self._upper_side_runtime_roi = roi

    def _normalize_side_roi(self, p1: Optional[tuple[int, int]], p2: Optional[tuple[int, int]]) -> Optional[list[int]]:
        if p1 is None or p2 is None:
            return None
        x1, y1 = p1
        x2, y2 = p2
        x1, x2 = sorted((int(x1), int(x2)))
        y1, y2 = sorted((int(y1), int(y2)))
        if x2 - x1 < SIDE_ROI_MIN_SIZE or y2 - y1 < SIDE_ROI_MIN_SIZE:
            return None
        return [x1, y1, x2, y2]

    def _save_side_roi_to_current_template(self, roi: list[int], target: str = LIP_TARGET_UPPER) -> None:
        if self.detector is None:
            return
        if target == LIP_TARGET_LOWER:
            baseline = self.detector.lower_lip_baseline
            side_tip = baseline.get("side_lower_tip") if isinstance(baseline, dict) else None
            save_lower_lip_side_roi(side_roi=roi, side_lower_tip=side_tip)
            self.detector.lower_lip_baseline = load_lower_lip_baseline()
            self.log.emit(f"侧面下唇 ROI 已保存到当前模板 JSON: {roi}")
            return
        baseline = self.detector.upper_lip_baseline
        side_tip = baseline.get("side_upper_tip") if isinstance(baseline, dict) else None
        save_upper_lip_side_roi(side_roi=roi, side_upper_tip=side_tip)
        self.detector.upper_lip_baseline = load_upper_lip_baseline()
        self.log.emit(f"侧面嘴唇 ROI 已保存到当前模板 JSON: {roi}")

    def _drain_commands(self) -> None:
        while True:
            try:
                command, payload = self.commands.get_nowait()
            except queue.Empty:
                break
            self._handle_command(command, payload)

    def _handle_command(self, command: str, payload: Any) -> None:
        if command == "stop":
            self._running = False
            return
        if self.detector is None:
            return

        try:
            handler = self._command_handlers().get(command)
            if handler is not None:
                handler(payload)
        except Exception as exc:
            self.log.emit(f"[ERROR] {command}: {exc}")

    def _command_handlers(self) -> dict[str, Callable[[Any], None]]:
        return {
            "template": self._handle_template_command,
            "toggle_mp": lambda _payload: self._toggle_detector_flag("enable_mp", "MediaPipe"),
            "toggle_ellseg": lambda _payload: self._toggle_detector_flag("enable_ellseg", "EllSeg"),
            "toggle_landmarks": lambda _payload: self._toggle_detector_flag("show_all_landmarks", "Landmarks"),
            "toggle_overlay": lambda _payload: self._toggle_detector_flag("show_baseline_overlay", "Baseline overlay"),
            "toggle_eyeline": lambda _payload: self._toggle_detector_flag("show_eye_line_offset", "EyeLine HUD"),
            "eyelid_open": lambda _payload: self._handle_manual_eyelid("eyelid_open"),
            "eyelid_close": lambda _payload: self._handle_manual_eyelid("eyelid_close"),
            "check_upper_lip_front_side": lambda _payload: self._check_upper_lip_front_side(),
            "select_upper_lip_side_roi": lambda _payload: self._start_side_roi_selection(LIP_TARGET_UPPER),
            "select_lower_lip_side_roi": lambda _payload: self._start_side_roi_selection(LIP_TARGET_LOWER),
            "save_upper_lip": lambda _payload: self._save_upper_lip_front_side(),
            "save_lower_lip": lambda _payload: self._save_lower_lip_front_side(),
            "save_eye": lambda _payload: self._run_named_save("save_eye", self.detector.save_current_baseline),
            "save_eyelid": lambda _payload: self._run_named_save("save_eyelid", self.detector.save_current_eyelid_baseline),
            "save_eyebrow": lambda _payload: self._run_named_save("save_eyebrow", self.detector.save_current_eyebrow_baseline),
            "save_mouth": lambda _payload: self._run_named_save("save_mouth", self.detector.save_current_mouth_baseline),
            "save_corners": lambda _payload: self._run_named_save("save_corners", self.detector.save_current_mouth_corners_baseline),
            "save_head": lambda _payload: self._run_named_save("save_head", self.detector.save_current_head_position_baseline),
            "adjust_eyeline": lambda _payload: self._run_named_save("adjust_eyeline", self.detector.adjust_baseline_by_vertical_offset),
            "save_all": lambda _payload: self._save_all_baselines(),
            "export_adjusted_yaml": lambda _payload: self._export_adjusted_yaml(),
            "route": self._run_route,
        }

    def _handle_template_command(self, template_name: str) -> None:
        _ok, msg = apply_template(template_name)
        self.log.emit(msg)
        self._reload_template_baselines()

    def _toggle_detector_flag(self, attribute_name: str, label: str) -> None:
        current_value = bool(getattr(self.detector, attribute_name))
        setattr(self.detector, attribute_name, not current_value)
        self.log.emit(f"{label}: {'ON' if getattr(self.detector, attribute_name) else 'OFF'}")

    def _handle_manual_eyelid(self, command: str) -> None:
        tuner = self._create_tuner(use_servo=True)
        if tuner is None:
            self.log.emit("[ERROR] Servo controller not available.")
            return

        try:
            ctrl = tuner.controller
            a8_low, a8_high = self._servo_limits(ctrl, 8)
            a9_low, a9_high = self._servo_limits(ctrl, 9)
            a8_angle = tuner._temp_deg(8, default=(a8_low + a8_high) // 2)
            a9_angle = tuner._temp_deg(9, default=(a9_low + a9_high) // 2)

            # A8/A9 机械方向相反，手动开合必须分别按配置边界裁剪。
            if command == "eyelid_open":
                a8_angle = clamp(a8_angle - EYELID_MANUAL_STEP_DEGREE, a8_low, a8_high)
                a9_angle = clamp(a9_angle + EYELID_MANUAL_STEP_DEGREE, a9_low, a9_high)
                action_label = "开"
            else:
                a8_angle = clamp(a8_angle + EYELID_MANUAL_STEP_DEGREE, a8_low, a8_high)
                a9_angle = clamp(a9_angle - EYELID_MANUAL_STEP_DEGREE, a9_low, a9_high)
                action_label = "合"

            tuner.send_servo(8, a8_angle)
            tuner.send_servo(9, a9_angle)
            self.log.emit(f"眼皮{action_label}: A8={a8_angle}° A9={a9_angle}°")
        finally:
            self._cleanup_tuner(tuner)

    @staticmethod
    def _servo_limits(controller: Any, servo_index: int) -> tuple[int, int]:
        start = int(controller.list_start_deg[servo_index])
        end = int(controller.list_end_deg[servo_index])
        return min(start, end), max(start, end)

    def _check_upper_lip_front_side(self) -> None:
        tuner = self._create_tuner(use_servo=False)
        if tuner is None:
            self.log.emit("[ERROR] Camera controller not available.")
            return
        try:
            result = tuner.check_upper_lip_front_side()
            front = result.get("front") or {}
            side = result.get("side") or {}
            passed = bool(result.get("qualified"))
            status_text = "通过" if passed else "失败"
            self.item_result.emit("上唇", status_text)
            self.log.emit(
                "上唇正侧检查: "
                f"{status_text} | "
                f"front delta={front.get('delta', None)} | "
                f"side dx={side.get('dx', None)} dy={side.get('dy', None)}"
            )
        finally:
            self._cleanup_tuner(tuner)


    def _start_side_roi_selection(self, target: str) -> None:
        self._side_roi_select_enabled = True
        self._side_roi_select_target = target
        self._side_roi_dragging = False
        self._side_roi_start = None
        self._side_roi_current = None
        if target == LIP_TARGET_LOWER:
            self.log.emit("侧面下唇 ROI 框选模式已开启：请在 Side Camera 窗口拖框下唇区域。")
            return
        self.log.emit("侧面嘴唇 ROI 框选模式已开启：请在 Side Camera 窗口拖框嘴唇区域。")

    def _run_named_save(self, command_name: str, save_func: Callable[[], bool]) -> None:
        ok = save_func()
        self.log.emit(f"{command_name}: {'OK' if ok else 'FAILED'}")

    def _save_all_baselines(self) -> None:
        calls = [
            ("eye", self.detector.save_current_baseline),
            ("eyelid", self.detector.save_current_eyelid_baseline),
            ("eyebrow", self.detector.save_current_eyebrow_baseline),
            ("mouth", self.detector.save_current_mouth_baseline),
            ("lower_lip", self.detector.save_current_lower_lip_baseline),
            ("upper_lip", self.detector.save_current_upper_lip_baseline),
            ("corners", self.detector.save_current_mouth_corners_baseline),
            ("head", self.detector.save_current_head_position_baseline),
        ]
        for name, func in calls:
            self.log.emit(f"save {name}: {'OK' if func() else 'FAILED'}")
        self.log.emit(f"save all: written to template {self.template_getter()} in face_point.json")

    def _export_adjusted_yaml(self) -> None:
        output_path = self._export_adjusted_yaml_from_snapshot()
        self.log.emit(f"导出调整后 YAML: {output_path}")

    def _export_adjusted_yaml_from_snapshot(self) -> str:
        yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DEFAULT_SERVO_CONFIG)
        config = read_yaml(yaml_path)
        servo_info = config.get("SERVO_INFO", config)
        temp_degrees = self._last_servo_temp_degrees

        for _servo_name, info in servo_info.items():
            channel_idx = info.get("channel_idx")
            if channel_idx is None:
                continue
            if temp_degrees is None or not (0 <= channel_idx < len(temp_degrees)):
                current_angle = int(info.get("temp_deg", 0))
            else:
                current_angle = int(round(temp_degrees[channel_idx]))
            low = min(int(info["start_deg"]), int(info["end_deg"]))
            high = max(int(info["start_deg"]), int(info["end_deg"]))
            info["temp_deg"] = max(low, min(high, current_angle))

        base, ext = os.path.splitext(yaml_path)
        output_path = f"{base}_adjusted{ext or '.yaml'}"
        write_yaml(config, output_path)
        return output_path

    def _save_upper_lip_front_side(self) -> None:
        ok_front = self.detector.save_current_upper_lip_baseline()
        if not ok_front:
            self.log.emit("保存上唇(正面 ULR): FAILED")
            return
        self._save_lip_side_baseline("上唇", self._save_upper_lip_side_with_tuner)

    def _save_lower_lip_front_side(self) -> None:
        ok_front = self.detector.save_current_lower_lip_baseline()
        if not ok_front:
            self.log.emit("保存下唇(正面 LLR): FAILED")
            return
        self._save_lip_side_baseline("下唇", self._save_lower_lip_side_with_tuner)

    def _save_lip_side_baseline(self, label: str, save_side: Callable[[EyeAutoTuner], bool]) -> None:
        tuner = self._create_tuner(use_servo=False)
        if tuner is None:
            self._reload_template_baselines()
            self.log.emit(f"保存{label}: 正面 OK，侧面 FAILED (camera controller not available)")
            return

        try:
            ok_side = save_side(tuner)
            self._reload_template_baselines()
            self.log.emit(f"保存{label}(正侧): {'OK' if ok_side else 'PARTIAL'} "
                          f"(front=OK side={'OK' if ok_side else 'FAILED'})")
        finally:
            self._cleanup_tuner(tuner)

    @staticmethod
    def _save_upper_lip_side_with_tuner(tuner: EyeAutoTuner) -> bool:
        return tuner.save_current_upper_lip_side_baseline()

    @staticmethod
    def _save_lower_lip_side_with_tuner(tuner: EyeAutoTuner) -> bool:
        return tuner.save_current_lower_lip_side_baseline()

    def _run_route(self, payload: Any) -> None:
        route, max_iterations = self._parse_route_payload(payload)
        if self._route_running:
            self.log.emit("Route is already running.")
            return

        self._prepare_route(route)
        if not self._can_run_route(route):
            return

        self._route_running = True
        self._route_abort_requested = False
        self.log.emit(f"Starting route: {route}, max_iterations={max_iterations}")
        self.status.emit(f"Route running: {route}")

        try:
            tuner = self._create_tuner(use_servo=True)
            if tuner is None:
                self.log.emit("[ERROR] Servo controller not available.")
                return
            self.log.emit("Servo controller is occupied by current route.")
            self._run_route_steps(route, tuner, max_iterations)
        except Exception as exc:
            self.log.emit(f"[Route Error] {exc}")
            import traceback
            self.log.emit(traceback.format_exc())
        finally:
            aborted_by_button = self._route_abort_requested
            if aborted_by_button and self.detector is not None:
                self.detector.clear_user_stop()
            self._route_running = False
            self._route_abort_requested = False
            self._cleanup_tuner(tuner if "tuner" in locals() else None)
            self.route_finished.emit(route, aborted_by_button)
            self.status.emit("Ready")
            self.log.emit("Route aborted. Camera remains open." if aborted_by_button else "Route finished. Camera remains open.")

    @staticmethod
    def _parse_route_payload(payload: Any) -> tuple[str, int]:
        if isinstance(payload, dict):
            route = payload.get("route")
            max_iterations = int(payload.get("max_iterations", DEFAULT_ROUTE_MAX_ITERATIONS))
            return route, max_iterations
        return payload, FALLBACK_ROUTE_MAX_ITERATIONS

    def _prepare_route(self, route: str) -> None:
        _ok, msg = apply_template(self.template_getter())
        self.log.emit(msg)
        self._reload_template_baselines()

    def _can_run_route(self, route: str) -> bool:
        if route in ("full", "eye_only", "eyebrow") and self.detector.eyebrow_baseline is None:
            self.log.emit("[WARN] Current template has no eyebrow baseline; cannot run this route.")
            return False
        return True

    def _run_route_steps(self, route: str, tuner: EyeAutoTuner, max_iterations: int) -> None:
        route_steps = {
            "eyeball": (self._run_eyeball,),
            "eye_only": (self._run_eyeball, self._run_eyebrow, self._run_eyelid),
            "full": (
                self._run_eyeball,
                self._run_upper_lip,
                self._run_mouth,
                self._run_corners,
                self._run_lower_lip,
                self._run_eyebrow,
                self._run_eyelid,
            ),
            "eyebrow": (self._run_eyebrow,),
            "eyelid": (self._run_eyelid,),
            "upper_lip": (self._run_upper_lip,),
            "mouth": (self._run_mouth,),
            "corners": (self._run_corners,),
            "lower_lip": (self._run_lower_lip,),
        }
        for step in route_steps.get(route, ()):
            if self._route_abort_requested or self.detector.user_pressed_stop:
                self.log.emit("Route aborted before next item.")
                break
            step(tuner, max_iterations)

    def _emit_item_result(self, item: str, passed: bool, iterations: int) -> None:
        text = f"{'通过' if passed else '失败'} ({iterations}次迭代)"
        self.log.emit(f"{item}: {text}")
        self.item_result.emit(item, text)

    def _emit_tuning_warning(self, item: str, result_data: dict[str, Any]) -> None:
        if isinstance(result_data, dict) and result_data.get("stop_reason") == "no_legal_move":
            self.log.emit(f"[WARN] {item}: {result_data.get('warning', 'no legal move available')}")

    def _emit_item_running(self, item: str) -> None:
        self.item_result.emit(item, "进行中")

    def _run_eyeball(self, tuner: EyeAutoTuner, max_iterations: int) -> None:
        self.log.emit(">>> 调整: 眼球 A10-A13 <<<")
        self._emit_item_running("眼球")
        passed, iterations, _ = tuner.auto_adjust(
            pixel_to_degree=1.0,
            max_iterations=max_iterations,
            wait_seconds=1.0,
            keep_display=True,
        )
        self._emit_item_result("眼球", passed, iterations)

    def _run_eyebrow(self, tuner: EyeAutoTuner, max_iterations: int) -> None:
        self.log.emit(">>> 调整: 眉毛 A0-A1 <<<")
        self._emit_item_running("眉毛")
        passed, iterations, _ = tuner.auto_adjust_eyebrow(
            ebhr_step=1.0,
            max_iterations=max_iterations,
            wait_seconds=1.0,
            keep_display=True,
            log_func=self.log.emit,
        )
        self._emit_item_result("眉毛", passed, iterations)

    def _run_eyelid(self, tuner: EyeAutoTuner, max_iterations: int) -> None:
        self.log.emit(">>> 调整: 眼皮 A8-A9 <<<")
        self._emit_item_running("眼皮")
        passed, iterations, _ = tuner.auto_adjust_eyelid(
            ear_step=1.0,
            max_iterations=max_iterations,
            wait_seconds=1.0,
            keep_display=True,
        )
        self._emit_item_result("眼皮", passed, iterations)

    def _run_upper_lip(self, tuner: EyeAutoTuner, max_iterations: int) -> None:
        self.log.emit(">>> Adjust: UpperLip A16(front)+A17(side) <<<")
        self._emit_item_running("上唇")
        if tuner.scorer.upper_lip_baseline is None:
            self.log.emit("上唇: 跳过 (无基线)")
            self.item_result.emit("上唇", "跳过 (无基线)")
            return
        if "ulr" not in tuner.scorer.upper_lip_baseline:
            self.log.emit("上唇: 跳过 (无正面 ULR 基线，请先点“保存上唇 U”)")
            self.item_result.emit("上唇", "跳过 (无正面基线)")
            return
        if "side_upper_tip" not in tuner.scorer.upper_lip_baseline:
            self.log.emit("上唇: 跳过 (无侧面基线，请先点“保存上唇(正侧) U”)")
            self.item_result.emit("上唇", "跳过 (无侧面基线)")
            return
        passed, iterations, result_data = tuner.auto_adjust_upper_lip_front_side(
            step=1.0,
            max_iterations=max_iterations,
            wait_seconds=1.0,
            keep_display=True,
        )
        self._emit_tuning_warning("上唇", result_data)
        self._emit_item_result("上唇", passed, iterations)

    def _run_mouth(self, tuner: EyeAutoTuner, max_iterations: int) -> None:
        self._emit_item_running("下巴")
        if tuner.scorer.mouth_baseline is None:
            self.log.emit("下巴: 跳过 (无基线)")
            self.item_result.emit("下巴", "跳过 (无基线)")
            return
        passed, iterations, _ = tuner.auto_adjust_mouth_chin(
            mar_step=1.0,
            max_iterations=max_iterations,
            wait_seconds=1.0,
            keep_display=True,
        )
        self._emit_item_result("下巴", passed, iterations)

    def _run_corners(self, tuner: EyeAutoTuner, max_iterations: int) -> None:
        self._emit_item_running("嘴角")
        if tuner.scorer.mouth_corners_baseline is None:
            self.log.emit("嘴角: 跳过 (无基线)")
            self.item_result.emit("嘴角", "跳过 (无基线)")
            return
        passed, iterations, _ = tuner.auto_adjust_mouth_corners(
            step=1.0,
            max_iterations=max_iterations,
            wait_seconds=1.0,
            keep_display=True,
        )
        self._emit_item_result("嘴角", passed, iterations)

    def _run_lower_lip(self, tuner: EyeAutoTuner, max_iterations: int) -> None:
        self._emit_item_running("下唇")
        if tuner.scorer.lower_lip_baseline is None:
            self.log.emit("下唇: 跳过 (无基线)")
            self.item_result.emit("下唇", "跳过 (无基线)")
            return
        if "llr" not in tuner.scorer.lower_lip_baseline:
            self.log.emit("下唇: 跳过 (无正面 LLR 基线，请先点“保存下唇(正侧) L”)")
            self.item_result.emit("下唇", "跳过 (无正面基线)")
            return
        if "side_lower_tip" not in tuner.scorer.lower_lip_baseline:
            self.log.emit("下唇: 跳过 (无侧面基线，请先点“保存下唇(正侧) L”)")
            self.item_result.emit("下唇", "跳过 (无侧面基线)")
            return
        passed, iterations, result_data = tuner.auto_adjust_lower_lip_front_side(
            step=1.0,
            max_iterations=max_iterations,
            wait_seconds=1.0,
            keep_display=True,
        )
        self._emit_tuning_warning("下唇", result_data)
        self._emit_item_result("下唇", passed, iterations)

class CalibrationPanel(QMainWindow):
    BASELINE_BUTTONS = [
        ("保存眼球 S", "save_eye"),
        ("保存眼皮 E", "save_eyelid"),
        ("保存眉毛 B", "save_eyebrow"),
        ("保存嘴部 M", "save_mouth"),
        ("保存下唇(正侧) L", "save_lower_lip"),
        ("保存上唇(正侧) U", "save_upper_lip"),
        ("框选侧面嘴ROI", "select_upper_lip_side_roi"),
        ("保存嘴角 C", "save_corners"),
        ("保存头位 D", "save_head"),
        ("眼线修正 V", "adjust_eyeline"),
        ("保存全部 A", "save_all"),
        ("切换 MP 1", "toggle_mp"),
        ("切换 EllSeg 2", "toggle_ellseg"),
        ("切换关键点 3", "toggle_landmarks"),
        ("基线叠加 G", "toggle_overlay"),
        ("EyeLine HUD Y", "toggle_eyeline"),
        ("框选侧面下唇ROI", "select_lower_lip_side_roi"),
    ]
    ROUTE_BUTTONS = [
        ("完整默认路线", "full"),
        ("眼球 + 眉毛 + 眼皮", "eye_only"),
        ("只调眼球", "eyeball"),
        ("只调眉毛", "eyebrow"),
        ("只调眼皮", "eyelid"),
        ("调整上唇(正侧)", "upper_lip"),
        ("调整下巴", "mouth"),
        ("调整左右嘴角", "corners"),
        ("调整下唇(正侧)", "lower_lip"),
    ]
    RESULT_ITEMS = [
        ("eyeball", "眼球"),
        ("eyebrow", "眉毛"),
        ("eyelid", "眼皮"),
        ("upper_lip", "上唇"),
        ("mouth", "下巴"),
        ("corners", "嘴角"),
        ("lower_lip", "下唇"),
    ]
    ROUTE_ITEMS = {
        "eyeball": ["眼球"],
        "eye_only": ["眼球", "眉毛", "眼皮"],
        "full": ["眼球", "上唇", "下巴", "嘴角", "下唇", "眉毛", "眼皮"],
        "eyebrow": ["眉毛"],
        "eyelid": ["眼皮"],
        "upper_lip": ["上唇"],
        "mouth": ["下巴"],
        "corners": ["嘴角"],
        "lower_lip": ["下唇"],
    }
    ITEM_KEYS = {
        "眼球": "eyeball",
        "眉毛": "eyebrow",
        "眼皮": "eyelid",
        "上唇": "upper_lip",
        "下巴": "mouth",
        "嘴角": "corners",
        "下唇": "lower_lip",
    }
    STATUS_STYLES = {
        "未开始": "color: #6b7280;",
        "进行中": "color: #d97706; font-weight: 600;",
        "通过": "color: #15803d; font-weight: 600;",
        "失败": "color: #b91c1c; font-weight: 600;",
        "跳过": "color: #6b7280;",
    }

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Face Calibration Panel")
        self.resize(920, 640)
        self.detector_thread = None

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        top = QHBoxLayout()
        layout.addLayout(top)

        self.template_combo = QComboBox()
        self.template_combo.currentIndexChanged.connect(self.on_template_changed)
        top.addWidget(QLabel("当前模板"))
        top.addWidget(self.template_combo)
        save_template_btn = QPushButton("保存当前模板")
        save_template_btn.clicked.connect(self.save_current_template)
        top.addWidget(save_template_btn)

        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        top.addWidget(self.status_label, 1)

        body = QHBoxLayout()
        layout.addLayout(body, 1)

        body.addWidget(self._build_baseline_group(), 1)
        body.addWidget(self._build_manual_group(), 1)
        body.addWidget(self._build_route_group(), 1)
        body.addWidget(self._build_result_group(), 1)

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        layout.addWidget(self.log_box, 1)

        self.refresh_template_combo()
        self.refresh_template_status()
        self.start_detector()  # 摄像头自动打开

    def current_template(self) -> str:
        return self.template_combo.currentText()

    def append_log(self, text: str) -> None:
        self.log_box.appendPlainText(text)

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _build_baseline_group(self) -> QGroupBox:
        group = QGroupBox("基线录制")
        grid = QGridLayout(group)

        for idx, (label, command) in enumerate(self.BASELINE_BUTTONS):
            btn = self._command_button(label, command)
            grid.addWidget(btn, idx // BASELINE_BUTTON_COLUMNS, idx % BASELINE_BUTTON_COLUMNS)
        return group

    def _build_manual_group(self) -> QGroupBox:
        group = QGroupBox("手动控制")
        layout = QVBoxLayout(group)
        layout.addWidget(self._command_button("眼皮开", "eyelid_open"))
        layout.addWidget(self._command_button("眼皮合", "eyelid_close"))
        layout.addStretch(1)
        return group

    def _build_route_group(self) -> QGroupBox:
        group = QGroupBox("检测 / 自动调整路线")
        layout = QVBoxLayout(group)

        iter_row = QHBoxLayout()
        iter_row.addWidget(QLabel("最大调整次数"))
        self.max_iterations_spin = QSpinBox()
        self.max_iterations_spin.setRange(1, MAX_ROUTE_ITERATIONS)
        self.max_iterations_spin.setValue(DEFAULT_ROUTE_MAX_ITERATIONS)
        self.max_iterations_spin.setSuffix(" 次")
        iter_row.addWidget(self.max_iterations_spin)
        layout.addLayout(iter_row)

        for label, route in self.ROUTE_BUTTONS:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _checked=False, route=route: self.start_route(route))
            layout.addWidget(btn)
        abort_btn = QPushButton("中断当前调整")
        abort_btn.clicked.connect(self.abort_current_route)
        abort_btn.setStyleSheet("color: #b91c1c; font-weight: 600;")
        layout.addWidget(abort_btn)
        export_btn = QPushButton("导出调整后YAML")
        export_btn.clicked.connect(lambda _checked=False: self.send_detector("export_adjusted_yaml"))
        layout.addWidget(export_btn)
        layout.addStretch(1)
        return group

    def _build_result_group(self) -> QGroupBox:
        group = QGroupBox("通过状态")
        grid = QGridLayout(group)
        self.result_labels = {}
        for row, (key, label) in enumerate(self.RESULT_ITEMS):
            grid.addWidget(QLabel(label), row, 0)
            value = QLabel("未开始")
            value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            value.setStyleSheet(self.STATUS_STYLES["未开始"])
            self.result_labels[key] = value
            grid.addWidget(value, row, 1)
        return group

    def _command_button(self, label: str, command: str) -> QPushButton:
        button = QPushButton(label)
        button.clicked.connect(lambda _checked=False, command=command: self.send_detector(command))
        return button

    def set_result_status(self, item: str, text: str) -> None:
        key = self.ITEM_KEYS.get(item, item)
        label = self.result_labels.get(key)
        if label is None:
            return
        label.setText(text)
        state = text.split(" ", 1)[0]
        label.setStyleSheet(self.STATUS_STYLES.get(state, ""))

    def reset_route_statuses(self, route: str) -> None:
        for item, key in self.ITEM_KEYS.items():
            self.set_result_status(item, "未开始")

    def refresh_template_combo(self) -> None:
        active = get_active_template()
        names = get_template_names()
        if active not in names:
            names.insert(0, active)
        self.template_combo.blockSignals(True)
        self.template_combo.clear()
        self.template_combo.addItems(names)
        idx = self.template_combo.findText(active)
        if idx >= 0:
            self.template_combo.setCurrentIndex(idx)
        self.template_combo.blockSignals(False)

    def refresh_template_status(self) -> None:
        template_name = self.current_template()
        brow = "FOUND" if template_section_exists(template_name, "eyebrow") else "MISSING"
        eyelid = "FOUND" if template_section_exists(template_name, "eyelid") else "MISSING"
        self.append_log(f"{template_name} in face_point.json: eyebrow={brow}, eyelid={eyelid}")

    def on_template_changed(self) -> None:
        template_name = self.current_template()
        _ok, msg = apply_template(template_name)
        self.append_log(msg)
        self.refresh_template_status()
        if self.detector_thread is not None:
            self.detector_thread.send(("template", template_name))

    def save_current_template(self) -> None:
        current = self.current_template() or get_active_template()
        name, ok = QInputDialog.getText(
            self,
            "保存当前模板",
            "模板名称:",
            text=current,
        )
        if not ok:
            return
        name = name.strip()
        if not name:
            QMessageBox.warning(self, "提示", "模板名称不能为空。")
            return
        try:
            save_current_as_template(name)
            self.refresh_template_combo()
            self.append_log(f"Saved current template as: {name}")
            if self.detector_thread is not None:
                self.detector_thread.send(("template", name))
        except Exception as exc:
            QMessageBox.critical(self, "错误", str(exc))

    def start_detector(self) -> None:
        if self.detector_thread is not None:
            self.append_log("Calibration camera is already running.")
            return
        _ok, msg = apply_template(self.current_template())
        self.append_log(msg)
        self.detector_thread = DetectorThread(self.current_template)
        self.detector_thread.log.connect(self.append_log)
        self.detector_thread.status.connect(self.set_status)
        self.detector_thread.stopped.connect(self.on_detector_stopped)
        self.detector_thread.route_finished.connect(self.on_route_finished)
        self.detector_thread.item_result.connect(self.on_item_result)
        self.detector_thread.start()
        self.set_status("Camera running")

    def stop_detector(self) -> None:
        if self.detector_thread is not None:
            self.detector_thread.stop()
            self.detector_thread.wait(3000)

    def on_detector_stopped(self) -> None:
        self.detector_thread = None

    def send_detector(self, command: str) -> None:
        if self.detector_thread is None:
            return
        self.detector_thread.send((command, None))

    def start_route(self, route: str) -> None:
        if self.detector_thread is None:
            return
        if self.detector_thread._route_running:
            QMessageBox.information(self, "提示", "已有路线正在运行。")
            return
        payload = {
            "route": route,
            "max_iterations": self.max_iterations_spin.value(),
        }
        self.reset_route_statuses(route)
        self.append_log(
            f"Route {route} requested, max_iterations={payload['max_iterations']}"
        )
        self.detector_thread.send(("route", payload))

    def abort_current_route(self) -> None:
        if self.detector_thread is None:
            self.append_log("No detector thread is running.")
            return
        if not self.detector_thread.request_abort_route():
            self.append_log("No route is currently running.")

    def on_route_finished(self, route: str, aborted: bool = False) -> None:
        status = "aborted" if aborted else "finished"
        self.append_log(f"Route {route} {status}.")

    def on_item_result(self, item: str, text: str) -> None:
        self.set_result_status(item, text)
        self.append_log(f"[STATUS] {item}: {text}")

    def closeEvent(self, event: Any) -> None:
        self.stop_detector()
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    panel = CalibrationPanel()
    panel.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
