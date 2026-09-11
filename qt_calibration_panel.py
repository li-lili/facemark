import os
import queue
import sys
import time
from collections import deque
from typing import Any, Callable, Optional

import cv2
import numpy as np

from eye_constants import (
    CAMERA_INDEX,
    DEFAULT_SERVO_CONFIG_FILE,
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
    SIDE_PROFILE_PREVIEW_FILTER_FRAMES,
    SIDE_PROFILE_PREVIEW_MAX_JUMP_PX,
    SIDE_PROFILE_SLOPE_TOLERANCE,
)
from lip_front_detector import (
    center_square_rect,
    draw_side_profile_geometry,
    find_lip_front_points,
    find_side_profile_points,
    StableFeaturePoint,
    measure_side_profile_geometry,
)

try:
    from PySide6.QtCore import QThread, Qt, Signal
    from PySide6.QtGui import QIntValidator
    from PySide6.QtWidgets import (
        QApplication,
        QComboBox,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QInputDialog,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QScrollArea,
        QSlider,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )
except ModuleNotFoundError:
    from PyQt5.QtCore import QThread, Qt, pyqtSignal as Signal
    from PyQt5.QtGui import QIntValidator
    from PyQt5.QtWidgets import (
        QApplication,
        QComboBox,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QInputDialog,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QScrollArea,
        QSlider,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )

from ellseg_scorer import (
    EllSegDetector,
    get_active_template,
    get_template_names,
    load_head_adapter_baseline,
    load_eyebrow_baseline,
    load_eyelid_baseline,
    load_head_position_baseline,
    load_lower_lip_baseline,
    load_mouth_baseline,
    load_mouth_corner_curl_baseline,
    load_mouth_corners_baseline,
    load_side_profile_slope_baseline,
    load_side_nose_roi,
    load_side_chin_roi,
    save_side_chin_roi,
    save_side_nose_roi,
    load_upper_lip_baseline,
    save_lower_lip_side_roi,
    save_upper_lip_side_roi,
    save_current_as_template,
    save_side_profile_slope_baseline,
    set_active_template,
    template_section_exists,
)

from eye_auto_tuner import EyeAutoTuner
from utility import read_yaml, write_yaml


DEFAULT_SERVO_CONFIG = DEFAULT_SERVO_CONFIG_FILE
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
CAMERA_CHOICES = list(range(10))

LIP_TARGET_UPPER = "upper"
LIP_TARGET_LOWER = "lower"
LIP_TARGET_BOTH = "both"


def clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, int(value)))


def int_list(values: Any) -> list[int]:
    return [int(value) for value in values]


def apply_template(template_name: str) -> tuple[bool, str]:
    set_active_template(template_name)
    return True, f"Active template -> {template_name} (face_point.json)"


def load_manual_servo_channels() -> frozenset[int]:
    """以实测动作规则中的通道作为当前项目允许手动控制的舵机。"""
    rules_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "servo_motion_rules.yaml")
    config = read_yaml(rules_path)
    effects = config.get("SERVO_EFFECTS", {})
    channels = []
    for servo_name in effects:
        if str(servo_name).startswith("A") and str(servo_name)[1:].isdigit():
            channels.append(int(str(servo_name)[1:]))
    return frozenset(channels)


class DetectorThread(QThread):
    log = Signal(str)
    status = Signal(str)
    stopped = Signal()
    route_finished = Signal(str, bool)
    item_result = Signal(str, str)
    servo_values = Signal(object)

    def __init__(
        self,
        template_getter: Callable[[], str],
        front_camera_index: int = CAMERA_INDEX,
        side_camera_index: int = UPPER_LIP_SIDE_CAMERA_INDEX,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.template_getter = template_getter
        self.front_camera_index = int(front_camera_index)
        self.side_camera_index = int(side_camera_index)
        self.commands: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._running = True
        self._route_running = False
        self._route_abort_requested = False
        self.detector = None
        self._last_servo_temp_degrees = None
        self.side_cap = None
        self.side_window_name = self._side_window_name()
        self._upper_side_runtime_roi = None
        self._lower_side_runtime_roi = None
        self._side_roi_select_enabled = False
        self._side_roi_select_target = "upper"
        self._side_roi_dragging = False
        self._side_roi_start = None
        self._side_roi_current = None

        # 仅控制正面眼球计算线显示；切换摄像机时沿用当前开关状态。
        self._iris_cover_overlay_enabled = False
        # 正、侧面贴在人物脸上的点/线/框总开关；HUD 始终显示。
        self._face_overlays_enabled = True
        # 新侧面轮廓路线启动后，在闲置侧面画面中保留结果叠加。
        self._side_profile_overlay_enabled = False
        self._side_profile_show_lines = False
        self._side_profile_points_preview_enabled = False
        self._side_profile_expected_slope_enabled = False
        self._side_profile_coordinate_grid_enabled = False
        self._side_profile_point_history = {
            name: deque(maxlen=SIDE_PROFILE_PREVIEW_FILTER_FRAMES)
            for name in ("nose", "upper_lip", "lower_lip", "chin")
        }
        self._side_profile_reject_counts = {
            name: 0 for name in self._side_profile_point_history
        }
        self._last_side_profile_signal = None
        self._last_side_profile_signal_time = 0.0
        self._side_profile_slope_baseline = load_side_profile_slope_baseline()

    def _create_tuner(self, use_servo: bool) -> Optional[EyeAutoTuner]:
        """按需创建 tuner；use_servo=True 时打开串口并保持到 cleanup。"""
        tuner = None
        try:
            tuner = EyeAutoTuner(
                yaml_file=DEFAULT_SERVO_CONFIG,
                detector=self.detector,
            )
            tuner.servo_values_callback = self._publish_servo_values
            tuner.side_cap = self.side_cap
            tuner._owns_side_cap = False
            tuner.side_camera_index = self.side_camera_index
            if use_servo:
                tuner.initialize()
                self._restore_tuner_temp_degrees(tuner)
                if not tuner.open_serial():
                    self.log.emit(f"[ERROR] Cannot open servo serial port: {tuner.port}")
                    self._remember_tuner_temp_degrees(tuner)
                    tuner.cleanup()
                    return None
                self.log.emit(f"Servo serial opened: {tuner.port}.")
            else:
                tuner.scorer = self.detector
                tuner._owns_scorer = False
            return tuner
        except Exception as exc:
            self.log.emit(f"[ERROR] Cannot init servo controller: {exc}")
            if tuner is not None:
                self._remember_tuner_temp_degrees(tuner)
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
        self._publish_servo_values(tuner.controller.list_temp_deg)

    def _publish_servo_values(self, values: Any) -> None:
        self._last_servo_temp_degrees = list(values)
        self.servo_values.emit(list(self._last_servo_temp_degrees))

    def _cleanup_tuner(self, tuner: Optional[EyeAutoTuner]) -> None:
        if tuner is None:
            return
        had_controller = tuner.controller is not None
        self._remember_tuner_temp_degrees(tuner)
        tuner.cleanup()
        if had_controller:
            self.log.emit("Servo controller cleaned up. Serial port closed.")

    def _reload_template_baselines(self) -> None:
        if self.detector is None:
            return
        self.detector.eyelid_baseline = load_eyelid_baseline()
        self.detector.eyebrow_baseline = load_eyebrow_baseline()
        self.detector.mouth_baseline = load_mouth_baseline()
        self.detector.lower_lip_baseline = load_lower_lip_baseline()
        self.detector.upper_lip_baseline = load_upper_lip_baseline()
        self.detector.mouth_corners_baseline = load_mouth_corners_baseline()
        self.detector.mouth_corner_curl_baseline = load_mouth_corner_curl_baseline()
        self.detector.head_position_baseline = load_head_position_baseline()
        self.detector.head_adapter_baseline = load_head_adapter_baseline()
        self._side_profile_slope_baseline = load_side_profile_slope_baseline()
        self.detector.side_nose_roi = load_side_nose_roi()
        self.detector.side_chin_roi = load_side_chin_roi()
        self._reset_side_profile_point_filter()
        self.detector.side_profile_slope_baseline = self._side_profile_slope_baseline

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
        self.detector = EllSegDetector(camera_index=self.front_camera_index)
        actual_w = int(self.detector.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.detector.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.log.emit(f"Front camera opened: cam{self.front_camera_index} {actual_w}x{actual_h}")
        self.detector.enable_mp = True
        self.detector.enable_ellseg = True
        self.detector.show_iris_cover_overlay = self._iris_cover_overlay_enabled
        self.detector.show_face_overlays = self._face_overlays_enabled
        self.detector.show_side_profile_preview = self._side_profile_points_preview_enabled
        self.detector.show_side_profile_expected_slope = self._side_profile_expected_slope_enabled
        self.detector.show_side_profile_coordinate_grid = self._side_profile_coordinate_grid_enabled
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
        self.side_window_name = self._side_window_name()
        cap = cv2.VideoCapture(self.side_camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self.log.emit(f"[WARN] Side camera {self.side_camera_index} failed to open.")
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_HEIGHT)
        self.side_cap = cap
        cv2.namedWindow(self.side_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.side_window_name, SIDE_WINDOW_WIDTH, SIDE_WINDOW_HEIGHT)
        cv2.setMouseCallback(self.side_window_name, self._on_side_mouse)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.log.emit(f"Side camera opened: cam{self.side_camera_index} {w}x{h}")
        return True

    def _side_window_name(self) -> str:
        return f"Side Camera {self.side_camera_index} - Lips"

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
            cv2.putText(frame, f"Side camera {self.side_camera_index}: read failed",
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
        if self._side_profile_coordinate_grid_enabled:
            self._draw_side_profile_coordinate_grid(frame)
            return

        # 纯四端点观察时不绘制旧 ROI/旧唇点，避免与新点位混淆。
        if self._side_profile_overlay_enabled and not self._side_profile_show_lines:
            self._draw_side_profile_overlay(frame, points_only=True)
            return

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
            self._face_overlays_enabled,
        )
        self._draw_lip_side_status(
            frame, "Lower", lower_roi, lower_tip, lower_baseline,
            "side_lower_tip", LOWER_LIP_SIDE_X_TOLERANCE, LOWER_LIP_SIDE_Y_TOLERANCE,
            (255, 80, 80), (255, 0, 180), SIDE_LOWER_STATUS_Y,
            self._face_overlays_enabled,
        )
        cv2.putText(frame, f"Side cam {self.side_camera_index} Lips",
                    SIDE_TITLE_POS, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Dir={UPPER_LIP_SIDE_FACE_DIRECTION}  VFlip={UPPER_LIP_SIDE_FLIP_VERTICAL}",
                    SIDE_META_POS, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 255), 1, cv2.LINE_AA)

        if self._side_roi_select_enabled:
            target_label = "lips" if self._side_roi_select_target == LIP_TARGET_BOTH else self._side_roi_select_target
            cv2.putText(frame, f"ROI SELECT {target_label}: drag lip area",
                        SIDE_ROI_SELECT_POS, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        if (self._face_overlays_enabled and self._side_roi_dragging
                and self._side_roi_start and self._side_roi_current):
            sx, sy = self._side_roi_start
            cx, cy = self._side_roi_current
            x_min, x_max = sorted((sx, cx))
            y_min, y_max = sorted((sy, cy))
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)

        if self._side_profile_overlay_enabled:
            self._draw_side_profile_overlay(frame, points_only=False)

    def _draw_side_profile_coordinate_grid(self, frame: np.ndarray) -> None:
        x1, y1, x2, y2 = center_square_rect(frame, SIDE_PROFILE_CROP_SIZE)
        color = (255, 255, 255)
        grid_color = (170, 170, 170)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        nose_x_min, nose_x_max = sorted(SIDE_PROFILE_NOSE_X_RANGE)
        nose_y_min, nose_y_max = sorted(SIDE_PROFILE_NOSE_Y_RANGE)
        cv2.rectangle(
            frame,
            (x1 + nose_x_min, y1 + nose_y_min),
            (x1 + nose_x_max, y1 + nose_y_max),
            (0, 255, 255), 3, cv2.LINE_AA,
        )
        for local_value in range(0, SIDE_PROFILE_CROP_SIZE + 1, 100):
            x = x1 + local_value
            y = y1 + local_value
            if 0 < local_value < SIDE_PROFILE_CROP_SIZE:
                cv2.line(frame, (x, y1), (x, y2), grid_color, 1, cv2.LINE_AA)
                cv2.line(frame, (x1, y), (x2, y), grid_color, 1, cv2.LINE_AA)
            cv2.putText(
                frame, str(local_value), (x + 3, y1 + 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(
                frame, str(local_value), (x1 + 4, min(y2 - 4, y + 17)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
        panel_x = max(10, frame.shape[1] - 650)
        self._draw_hud_panel(frame, (panel_x - 8, 8), (frame.shape[1] - 8, 88))
        cv2.putText(
            frame, f"Side Profile local coordinates 0..{SIDE_PROFILE_CROP_SIZE}",
            (panel_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.56,
            (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            frame, f"origin=({x1},{y1})  X=yellow top  Y=cyan left",
            (panel_x, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
            (200, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(
            frame, "Report nose range as local X=min..max, Y=min..max",
            (panel_x, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (200, 220, 255), 1, cv2.LINE_AA)

    def _draw_side_profile_overlay(
        self,
        frame: np.ndarray,
        points_only: bool,
    ) -> None:
        detected = find_side_profile_points(
            frame,
            crop_size=SIDE_PROFILE_CROP_SIZE,
            face_direction=SIDE_PROFILE_FACE_DIRECTION,
            background_delta=SIDE_PROFILE_BACKGROUND_DELTA,
            lip_min_area=SIDE_PROFILE_LIP_MIN_AREA,
            nose_roi=getattr(self.detector, "side_nose_roi", None),
            require_nose_roi=True,
            upper_lip_roi=(self.detector.upper_lip_baseline or {}).get("side_roi"),
            lower_lip_roi=(self.detector.lower_lip_baseline or {}).get("side_roi"),
            separate_lip_rois=True,
            chin_roi=getattr(self.detector, "side_chin_roi", None),
            require_feature_rois=True,
            nose_edge_contrast_min=SIDE_PROFILE_NOSE_EDGE_CONTRAST_MIN,
        )
        detected = self._stabilize_side_profile_detection(detected)
        nose_upper_target, lip_chin_target, _target_mode = self._side_profile_targets()
        signal = measure_side_profile_geometry(
            detected.get("points") or {},
            face_direction=SIDE_PROFILE_FACE_DIRECTION,
            nose_upper_target_slope=nose_upper_target,
            lip_chin_target_slope=lip_chin_target,
            slope_tolerance=SIDE_PROFILE_SLOPE_TOLERANCE,
            lower_line_tolerance_px=SIDE_PROFILE_LOWER_LINE_TOLERANCE_PX,
            chin_line_tolerance_px=SIDE_PROFILE_CHIN_LINE_TOLERANCE_PX,
        )
        signal["rect"] = detected.get("rect")
        signal["nose_search_rect"] = detected.get("nose_search_rect")
        signal["mouth_search_rect"] = detected.get("mouth_search_rect")
        signal["upper_search_rect"] = detected.get("upper_search_rect")
        signal["lower_search_rect"] = detected.get("lower_search_rect")
        signal["chin_search_rect"] = detected.get("chin_search_rect")
        self._last_side_profile_signal = signal
        self._last_side_profile_signal_time = time.monotonic()
        draw_side_profile_geometry(
            frame,
            signal,
            show_face_overlays=self._face_overlays_enabled,
            draw_lines=not points_only,
            draw_expected_dashed=self._side_profile_expected_slope_enabled,
        )
        self._draw_side_profile_status(frame, signal, points_only=points_only)

    def _side_profile_targets(self) -> tuple[float, float, str]:
        baseline = (
            self._side_profile_slope_baseline
            if isinstance(self._side_profile_slope_baseline, dict) else {})
        return (
            float(baseline.get(
                "nose_upper_slope", SIDE_PROFILE_NOSE_UPPER_TARGET_SLOPE)),
            float(baseline.get(
                "lip_chin_slope", SIDE_PROFILE_LIP_CHIN_TARGET_SLOPE)),
            "saved" if baseline else "default",
        )

    def _reset_side_profile_point_filter(self) -> None:
        self._side_feature_pending = {}
        self._side_feature_latches = {}
        for history in self._side_profile_point_history.values():
            history.clear()
        for name in self._side_profile_reject_counts:
            self._side_profile_reject_counts[name] = 0

    def _stabilize_side_profile_detection(self, detected: dict) -> dict:
        points = dict(detected.get("points") or {})
        stable_points = {}
        reacquire_frames = max(3, SIDE_PROFILE_PREVIEW_FILTER_FRAMES // 2 + 1)
        if not hasattr(self, "_side_feature_pending"):
            self._side_feature_pending = {}

        for name, history in self._side_profile_point_history.items():
            point = points.get(name)
            if name != "nose":
                if not hasattr(self, "_side_feature_latches"):
                    self._side_feature_latches = {}
                from eye_constants import LIP_PREVIEW_HOLD_PX, LIP_PREVIEW_CONFIRM_FRAMES, LIP_PREVIEW_CLUSTER_PX
                if name not in self._side_feature_latches:
                    self._side_feature_latches[name] = (
                        StableFeaturePoint(LIP_PREVIEW_HOLD_PX, LIP_PREVIEW_CONFIRM_FRAMES,
                                           LIP_PREVIEW_CLUSTER_PX)
                        if name in ("upper_lip", "lower_lip") else StableFeaturePoint())
                latch = self._side_feature_latches[name]
                stable_points[name] = latch.update(point)
                continue
            if point is None:
                if name != "nose":
                    self._side_feature_pending.pop(name, None)
                    self._side_profile_reject_counts[name] += 1
                    if self._side_profile_reject_counts[name] >= SIDE_PROFILE_PREVIEW_FILTER_FRAMES:
                        history.clear()
                    stable_points[name] = None
                    continue
                history.clear()
                self._side_profile_reject_counts[name] = 0
                stable_points[name] = None
                continue
            if point is not None:
                candidate = np.asarray(point, dtype=np.float64)
                if history:
                    previous = np.median(np.asarray(history), axis=0)
                    jump = float(np.linalg.norm(candidate - previous))
                    if jump > SIDE_PROFILE_PREVIEW_MAX_JUMP_PX:
                        if name != "nose":
                            pending = self._side_feature_pending.get(name, [])
                            if pending and np.linalg.norm(candidate - np.median(pending, axis=0)) > SIDE_PROFILE_PREVIEW_MAX_JUMP_PX:
                                pending = []
                            pending.append(tuple(candidate))
                            self._side_feature_pending[name] = pending
                            self._side_profile_reject_counts[name] = len(pending) - 1
                        self._side_profile_reject_counts[name] += 1
                        if self._side_profile_reject_counts[name] >= reacquire_frames:
                            self._side_feature_pending.pop(name, None)
                            history.clear()
                            history.append(tuple(candidate))
                            self._side_profile_reject_counts[name] = 0
                        else:
                            stable_points[name] = None
                            continue
                    else:
                        self._side_feature_pending.pop(name, None)
                        history.append(tuple(candidate))
                        self._side_profile_reject_counts[name] = 0
                else:
                    history.append(tuple(candidate))
                    self._side_profile_reject_counts[name] = 0

            if history:
                median_point = np.median(np.asarray(history), axis=0)
                stable_points[name] = (
                    int(round(float(median_point[0]))),
                    int(round(float(median_point[1]))),
                )
            else:
                stable_points[name] = None

        result = dict(detected)
        result["points"] = stable_points
        result["valid"] = all(stable_points.get(name) is not None for name in stable_points)
        result["reason"] = "" if result["valid"] else detected.get("reason", "profile points unstable")
        return result

    def _draw_side_profile_status(
        self,
        frame: np.ndarray,
        signal: dict,
        points_only: bool = False,
    ) -> None:
        h, w = frame.shape[:2]
        x = max(10, w - 650)
        self._draw_hud_panel(frame, (x - 8, 8), (w - 8, 164))
        cv2.putText(
            frame,
            f"Side Profile {SIDE_PROFILE_CROP_SIZE}x{SIDE_PROFILE_CROP_SIZE} / manual nose ROI",
            (x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            frame, f"Dir={SIDE_PROFILE_FACE_DIRECTION} F{SIDE_PROFILE_PREVIEW_FILTER_FRAMES} "
            f"J{SIDE_PROFILE_PREVIEW_MAX_JUMP_PX:.0f}px (+error = too forward)",
            (x, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 220, 255), 1, cv2.LINE_AA)
        if points_only:
            points = signal.get("points") or {}

            def point_text(name):
                point = points.get(name)
                if point is None:
                    return f"{name}=--"
                return f"{name}=({point[0]:.0f},{point[1]:.0f})"

            valid_color = (0, 255, 0) if signal.get("valid") else (0, 0, 255)
            cv2.putText(
                frame,
                "4-POINT PREVIEW / SERVOS OFF / "
                f"DASH={'ON' if self._side_profile_expected_slope_enabled else 'OFF'}",
                (x, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(
                frame, f"{point_text('nose')}  {point_text('upper_lip')}", (x, 108),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, valid_color, 1, cv2.LINE_AA)
            cv2.putText(
                frame, f"{point_text('lower_lip')}  {point_text('chin')}", (x, 134),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, valid_color, 1, cv2.LINE_AA)
            if signal.get("valid"):
                _nose_target, _chin_target, target_mode = self._side_profile_targets()
                cv2.putText(
                    frame,
                    f"Slope N-U={signal['nose_upper_slope']:.3f} "
                    f"U-C={signal['lip_chin_slope']:.3f} target={target_mode}",
                    (x, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    (200, 220, 255), 1, cv2.LINE_AA)
            return
        if not signal.get("valid"):
            cv2.putText(
                frame, f"N/U/L/C invalid: {signal.get('reason', 'not found')}",
                (x, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 1, cv2.LINE_AA)
            return
        phase1_color = (0, 255, 0) if signal["phase1_qualified"] else (0, 0, 255)
        line_ok = signal["lower_qualified"] and signal["chin_qualified"] and signal["slope_qualified"]
        line_color = (0, 255, 0) if line_ok else (0, 0, 255)
        _nose_target, _chin_target, target_mode = self._side_profile_targets()
        cv2.putText(
            frame,
            f"N-U slope={signal['nose_upper_slope']:.2f}/"
            f"{signal['nose_upper_target_slope']:.2f} "
            f"tol={SIDE_PROFILE_SLOPE_TOLERANCE:.2f} {target_mode}",
            (x, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.48, phase1_color, 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"U-C slope={signal['lip_chin_slope']:.2f}/"
            f"{signal['lip_chin_target_slope']:.2f} "
            f"tol={SIDE_PROFILE_SLOPE_TOLERANCE:.2f}",
            (x, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.48, line_color, 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"Line L={signal['lower_line_error_px']:+.1f}/+/-{SIDE_PROFILE_LOWER_LINE_TOLERANCE_PX:.1f}px "
            f"C={signal['chin_line_error_px']:+.1f}/+/-{SIDE_PROFILE_CHIN_LINE_TOLERANCE_PX:.1f}px",
            (x, 134), cv2.FONT_HERSHEY_SIMPLEX, 0.48, line_color, 1, cv2.LINE_AA)

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
                              tol_x, tol_y, roi_color, tip_color, y,
                              show_face_overlays=True):
        x1, y1, x2, y2 = int_list(roi)
        if show_face_overlays:
            cv2.rectangle(frame, (x1, y1), (x2, y2), roi_color, 2)
        cv2.putText(frame, f"{label} ROI=[{x1},{y1},{x2},{y2}] Tol X<={tol_x:.1f} Y<={tol_y:.1f}",
                    (10, y + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 220, 255), 1, cv2.LINE_AA)
        if tip is not None:
            if show_face_overlays:
                cv2.circle(frame, tip, 7, tip_color, -1)
                cv2.circle(frame, tip, 12, (255, 255, 255), 2)
            cv2.putText(frame, f"{label} tip: x={tip[0]} y={tip[1]}",
                        (10, y + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, f"{label} tip: not found",
                        (10, y + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        if isinstance(baseline, dict) and baseline.get(ref_key) is not None:
            ref = int_list(baseline[ref_key])
            if show_face_overlays:
                cv2.circle(frame, tuple(ref), 6, (0, 255, 255), -1)
            if tip is not None:
                dx = tip[0] - ref[0]
                dy = tip[1] - ref[1]
                ok = abs(dx) <= tol_x and abs(dy) <= tol_y
                color = (0, 255, 0) if ok else (0, 0, 255)
                cv2.putText(frame, f"{label} dX={dx:+.1f} dY={dy:+.1f} {'OK' if ok else 'BAD'}",
                            (10, y + 104), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                if show_face_overlays:
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
        if target in ("nose", "chin"):
            if target == "nose":
                save_side_nose_roi(rect)
                self.detector.side_nose_roi = rect
            else:
                save_side_chin_roi(rect)
                self.detector.side_chin_roi = rect
            self._last_side_profile_signal = None
            self._side_roi_select_enabled = False
            self._reset_side_profile_point_filter()
            self._side_profile_points_preview_enabled = True
            self._side_profile_overlay_enabled = True
            self._side_profile_show_lines = False
            self.detector.show_side_profile_preview = True
            self.log.emit(f"{target} ROI 已保存：{rect}，已开启检测预览。")
            return
        self._set_runtime_side_roi(target, rect)
        self._side_roi_select_enabled = False
        self._save_side_roi_to_current_template(rect, target)
        self._last_side_profile_signal = None
        self._reset_side_profile_point_filter()
        self._side_profile_overlay_enabled = True
        self._side_profile_show_lines = False
        self._side_profile_points_preview_enabled = True
        self.detector.show_side_profile_preview = True

    def _set_runtime_side_roi(self, target: str, roi: list[int]) -> None:
        if target == LIP_TARGET_BOTH:
            self._upper_side_runtime_roi = roi
            self._lower_side_runtime_roi = roi
            return
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
        if target == LIP_TARGET_BOTH:
            upper_baseline = self.detector.upper_lip_baseline
            upper_tip = upper_baseline.get("side_upper_tip") if isinstance(upper_baseline, dict) else None
            lower_baseline = self.detector.lower_lip_baseline
            lower_tip = lower_baseline.get("side_lower_tip") if isinstance(lower_baseline, dict) else None
            save_upper_lip_side_roi(side_roi=roi, side_upper_tip=upper_tip)
            save_lower_lip_side_roi(side_roi=roi, side_lower_tip=lower_tip)
            self.detector.upper_lip_baseline = load_upper_lip_baseline()
            self.detector.lower_lip_baseline = load_lower_lip_baseline()
            self.log.emit(f"侧面嘴唇共享 ROI 已保存到上唇/下唇当前模板 JSON: {roi}")
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
            "set_face_overlays": self._handle_face_overlays_command,
            "set_side_profile_preview": self._handle_side_profile_preview_command,
            "set_side_profile_expected_slope": self._handle_side_profile_expected_slope_command,
            "set_side_profile_coordinate_grid": self._handle_side_profile_coordinate_grid_command,
            "save_side_profile_slopes": lambda _payload: self._save_side_profile_slopes(),
            "set_front_camera": self._handle_front_camera_command,
            "set_side_camera": self._handle_side_camera_command,
            "eyelid_open": lambda _payload: self._handle_manual_eyelid("eyelid_open"),
            "eyelid_close": lambda _payload: self._handle_manual_eyelid("eyelid_close"),
            "manual_servo": self._handle_manual_servo,
            "check_upper_lip_front_side": lambda _payload: self._check_upper_lip_front_side(),
            "select_side_chin_roi": lambda _payload: self._start_side_roi_selection("chin"),
            "select_side_nose_roi": lambda _payload: self._start_side_roi_selection("nose"),
            "select_upper_lip_side_roi": lambda _payload: self._start_side_roi_selection(LIP_TARGET_UPPER),
            "select_lower_lip_side_roi": lambda _payload: self._start_side_roi_selection(LIP_TARGET_LOWER),
            "save_upper_lip": lambda _payload: self._save_upper_lip_front_side(),
            "save_lower_lip": lambda _payload: self._save_lower_lip_front_side(),
            "save_eye": lambda _payload: self._run_named_save("save_eye", self.detector.save_current_baseline),
            "save_eyelid": lambda _payload: self._run_named_save("save_eyelid", self.detector.save_current_eyelid_baseline),
            "save_eyebrow": lambda _payload: self._run_named_save("save_eyebrow", self.detector.save_current_eyebrow_baseline),
            "save_mouth": lambda _payload: self._run_named_save("save_mouth", self.detector.save_current_mouth_baseline),
            "save_corners": lambda _payload: self._run_named_save("save_corners", self.detector.save_current_mouth_corners_baseline),
            "save_corner_curl": lambda _payload: self._run_named_save("save_corner_curl", self.detector.save_current_mouth_corner_curl_baseline),
            "save_head": lambda _payload: self._run_named_save("save_head", self.detector.save_current_head_position_baseline),
            "save_head_adapter": lambda _payload: self._run_named_save("save_head_adapter", self.detector.save_current_head_adapter_baseline),
            "adjust_eyeline": lambda _payload: self._run_named_save("adjust_eyeline", self.detector.adjust_baseline_by_vertical_offset),
            "save_all": lambda _payload: self._save_all_baselines(),
            "export_adjusted_yaml": lambda _payload: self._export_adjusted_yaml(),
            "route": self._run_route,
        }

    def _handle_template_command(self, template_name: str) -> None:
        _ok, msg = apply_template(template_name)
        self.log.emit(msg)
        self._reload_template_baselines()

    def _handle_front_camera_command(self, camera_index: int) -> None:
        camera_index = int(camera_index)
        if camera_index == self.front_camera_index:
            return
        if self._route_running:
            self.log.emit("[WARN] Front camera switch ignored while route is running.")
            return
        old_detector = self.detector
        self.detector = None
        if old_detector is not None:
            old_detector.stop_display()
            old_detector.close()
        self.front_camera_index = camera_index
        self._start_detector()
        self._reload_template_baselines()
        self.log.emit(f"Front camera switched to cam{self.front_camera_index}")

    def _handle_side_camera_command(self, camera_index: int) -> None:
        camera_index = int(camera_index)
        if camera_index == self.side_camera_index:
            return
        if self._route_running:
            self.log.emit("[WARN] Side camera switch ignored while route is running.")
            return
        self._close_side_camera()
        self.side_camera_index = camera_index
        self.side_window_name = self._side_window_name()
        if self._open_side_camera():
            self.log.emit(f"Side camera switched to cam{self.side_camera_index}")

    def _toggle_detector_flag(self, attribute_name: str, label: str) -> None:
        current_value = bool(getattr(self.detector, attribute_name))
        setattr(self.detector, attribute_name, not current_value)
        self.log.emit(f"{label}: {'ON' if getattr(self.detector, attribute_name) else 'OFF'}")

    def _handle_face_overlays_command(self, enabled: Any) -> None:
        self._face_overlays_enabled = bool(enabled)
        self.detector.show_face_overlays = self._face_overlays_enabled
        state = "显示" if self._face_overlays_enabled else "隐藏"
        self.log.emit(f"人物脸部绘制已{state}，HUD 保持显示。")

    def _handle_side_profile_preview_command(self, enabled: Any) -> None:
        self._side_profile_points_preview_enabled = bool(enabled)
        self._side_profile_overlay_enabled = bool(
            self._side_profile_points_preview_enabled
            or self._side_profile_expected_slope_enabled)
        self._side_profile_show_lines = False
        self._reset_side_profile_point_filter()
        self.detector.show_side_profile_preview = self._side_profile_points_preview_enabled
        state = "开启" if self._side_profile_points_preview_enabled else "停止"
        self.log.emit(
            f"侧面鼻尖/上唇/下唇/下巴四端点检测已{state}；"
            "仅显示端点，不打开串口，不移动舵机。")

    def _handle_side_profile_expected_slope_command(self, enabled: Any) -> None:
        self._side_profile_expected_slope_enabled = bool(enabled)
        self._side_profile_overlay_enabled = bool(
            self._side_profile_points_preview_enabled
            or self._side_profile_expected_slope_enabled)
        self._side_profile_show_lines = False
        self._reset_side_profile_point_filter()
        self.detector.show_side_profile_expected_slope = self._side_profile_expected_slope_enabled
        state = "显示" if self._side_profile_expected_slope_enabled else "隐藏"
        self.log.emit(
            f"侧面期望斜率虚线已{state}；仅绘制参考线，不打开串口，不移动舵机。")

    def _handle_side_profile_coordinate_grid_command(self, enabled: Any) -> None:
        self._side_profile_coordinate_grid_enabled = bool(enabled)
        self.detector.show_side_profile_coordinate_grid = (
            self._side_profile_coordinate_grid_enabled)
        state = "显示" if self._side_profile_coordinate_grid_enabled else "隐藏"
        self.log.emit(
            f"侧面 {SIDE_PROFILE_CROP_SIZE}x{SIDE_PROFILE_CROP_SIZE} 局部坐标网格已{state}；"
            "左上角为 (0,0)，只显示网格，不操作串口或舵机。")

    def _save_side_profile_slopes(self) -> None:
        signal = self._last_side_profile_signal
        signal_age = time.monotonic() - self._last_side_profile_signal_time
        if not signal or not signal.get("valid") or signal_age > 2.0:
            self.log.emit(
                "[WARN] 保存侧面标准斜率失败：请先开启“检测侧面4端点”，"
                "确认四点稳定后再保存。")
            return
        nose_upper_slope = float(signal["nose_upper_slope"])
        lip_chin_slope = float(signal["lip_chin_slope"])
        save_side_profile_slope_baseline(
            nose_upper_slope=nose_upper_slope,
            lip_chin_slope=lip_chin_slope,
        )
        self._side_profile_slope_baseline = load_side_profile_slope_baseline()
        self.detector.side_nose_roi = load_side_nose_roi()
        self.detector.side_chin_roi = load_side_chin_roi()
        self._reset_side_profile_point_filter()
        self.detector.side_profile_slope_baseline = self._side_profile_slope_baseline
        self.log.emit(
            f"已保存当前模板 {self.template_getter()} 的侧面标准斜率："
            f"鼻尖-上唇={nose_upper_slope:.4f}，上唇-下巴={lip_chin_slope:.4f}。"
            "期望斜率虚线和自动调整已切换到保存值。")

    def _handle_manual_eyelid(self, command: str) -> None:
        if self._route_running:
            self.log.emit("[WARN] 自动调整进行中，忽略手动眼皮命令。")
            return
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

    def _handle_manual_servo(self, payload: Any) -> None:
        """单次手动发送：临时打开串口，发送完成后立即关闭。"""
        if self._route_running:
            self.log.emit("[WARN] 自动调整进行中，忽略手动舵机命令。")
            if self._last_servo_temp_degrees is not None:
                self.servo_values.emit(list(self._last_servo_temp_degrees))
            return
        if not isinstance(payload, dict):
            self.log.emit("[ERROR] 手动舵机命令格式无效。")
            return

        try:
            channel = int(payload["channel"])
            requested_angle = int(payload["angle"])
        except (KeyError, TypeError, ValueError):
            self.log.emit("[ERROR] 手动舵机通道或角度无效。")
            return

        if channel not in load_manual_servo_channels():
            self.log.emit(f"[WARN] A{channel} 当前项目未使用，忽略手动舵机命令。")
            if self._last_servo_temp_degrees is not None:
                self.servo_values.emit(list(self._last_servo_temp_degrees))
            return

        tuner = self._create_tuner(use_servo=True)
        if tuner is None:
            self.log.emit("[ERROR] Servo controller not available.")
            if self._last_servo_temp_degrees is not None:
                self.servo_values.emit(list(self._last_servo_temp_degrees))
            return

        try:
            if not (0 <= channel < len(tuner.controller.list_temp_deg)):
                self.log.emit(f"[ERROR] 手动舵机通道越界: A{channel}")
                return
            low, high = self._servo_limits(tuner.controller, channel)
            angle = clamp(requested_angle, low, high)
            if tuner.send_servo(channel, angle, (low, high)):
                self.log.emit(
                    f"手动调节: A{channel}={angle}°，范围={low}~{high}°；串口发送后关闭。"
                )
            else:
                self.log.emit(f"[ERROR] 手动调节发送失败: A{channel}={angle}°")
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
        self._side_profile_coordinate_grid_enabled = False
        self._side_profile_overlay_enabled = False
        self._side_roi_select_enabled = True
        self._side_roi_select_target = target
        self._side_roi_dragging = False
        self._side_roi_start = None
        self._side_roi_current = None
        if target == LIP_TARGET_BOTH:
            self.log.emit("侧面嘴唇共享 ROI 框选模式已开启：请在 Side Camera 窗口拖框包含上唇和下唇的嘴唇区域。")
            return
        if target == LIP_TARGET_LOWER:
            self.log.emit("侧面下唇 ROI 框选模式已开启：请在 Side Camera 窗口拖框下唇区域。")
            return
        if target == "chin":
            self.log.emit("请拖框下巴前缘，包含最前突点上下两侧的回收轮廓，松开保存。")
            return
        if target == "nose":
            self.log.emit("请在 Side Camera 窗口拖框鼻部，包含鼻梁、鼻尖和鼻底；松开鼠标保存。")
            return
        self.log.emit("侧面上唇 ROI 框选模式已开启：请只框上唇前缘，不包含下唇。")

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
            ("corner_curl", self.detector.save_current_mouth_corner_curl_baseline),
            ("head", self.detector.save_current_head_position_baseline),
            ("head_adapter", self.detector.save_current_head_adapter_baseline),
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

        try:
            self._prepare_route(route)
            can_run = self._can_run_route(route)
        except Exception as exc:
            self.log.emit(f"[Route Prepare Error] {exc}")
            can_run = False
        if not can_run:
            self.route_finished.emit(route, False)
            self.status.emit("Ready")
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
            self.log.emit("Servo route is running; serial port stays open for this route.")
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
            "iris_lid_contact": (self._run_iris_lid_contact,),
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
            "corner_curl": (self._run_corner_curl,),
            "lower_lip": (self._run_lower_lip,),
            "side_profile": (self._run_side_profile,),
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

    def _run_iris_lid_contact(self, tuner: EyeAutoTuner, max_iterations: int) -> None:
        self.log.emit(">>> 调整: 眼球上下贴合 L=A10 / R=A11（左右独立） <<<")
        self._emit_item_running("眼球")
        # 点击调整按钮后自动显示当前虹膜下弧、下眼皮曲线和贴合区。
        self._iris_cover_overlay_enabled = True
        self.detector.show_iris_cover_overlay = True
        passed, iterations, result_data = tuner.auto_adjust_iris_lid_contact(
            step=1.0,
            max_iterations=max_iterations,
            wait_seconds=1.0,
            keep_display=True,
            log_func=self.log.emit,
        )
        self._emit_tuning_warning("眼球", result_data)
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

    def _run_corner_curl(self, tuner: EyeAutoTuner, max_iterations: int) -> None:
        self._emit_item_running("嘴角微翘")
        passed, iterations, _ = tuner.auto_adjust_mouth_corner_curl(
            step=1.0,
            max_iterations=max_iterations,
            wait_seconds=1.0,
            keep_display=True,
        )
        self._emit_item_result("嘴角微翘", passed, iterations)

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

    def _run_side_profile(self, tuner: EyeAutoTuner, max_iterations: int) -> None:
        self.log.emit(">>> 侧面嘴唇下巴对线（嘴部中心400x400，鼻部手动ROI）: A16 -> A18+A26 <<<")
        self._emit_item_running("侧唇下巴")
        self._side_profile_overlay_enabled = True
        self._side_profile_show_lines = True
        self._reset_side_profile_point_filter()
        self.detector.show_side_profile_preview = True
        passed, iterations, result_data = tuner.auto_adjust_side_profile_alignment(
            step=1.0,
            max_iterations=max_iterations,
            wait_seconds=1.0,
            keep_display=True,
            log_func=self.log.emit,
        )
        self._emit_tuning_warning("侧唇下巴", result_data)
        self._emit_item_result("侧唇下巴", passed, iterations)

class CalibrationPanel(QMainWindow):
    BASELINE_BUTTONS = [
        ("保存眼球 S", "save_eye"),
        ("保存眼皮 E", "save_eyelid"),
        ("保存眉毛 B", "save_eyebrow"),
        ("保存嘴部 M", "save_mouth"),
        ("保存下唇(正侧) L", "save_lower_lip"),
        ("保存上唇(正侧) U", "save_upper_lip"),
        ("框选侧面下巴ROI", "select_side_chin_roi"),
        ("框选侧面鼻部ROI", "select_side_nose_roi"),
        ("框选侧面上唇ROI", "select_upper_lip_side_roi"),
        ("框选侧面下唇ROI", "select_lower_lip_side_roi"),
        ("保存嘴角 C", "save_corners"),
        ("保存嘴角微翘", "save_corner_curl"),
        ("保存头位 D", "save_head"),
        ("记录头偏差", "save_head_adapter"),
        ("眼线修正 V", "adjust_eyeline"),
        ("保存全部 A", "save_all"),
        ("切换 MP 1", "toggle_mp"),
        ("切换 EllSeg 2", "toggle_ellseg"),
        ("切换关键点 3", "toggle_landmarks"),
        ("基线叠加 G", "toggle_overlay"),
        ("EyeLine HUD Y", "toggle_eyeline"),
        ("隐藏脸部绘制", "set_face_overlays"),
        ("检测侧面4端点", "set_side_profile_preview"),
        ("显示期望斜率虚线", "set_side_profile_expected_slope"),
        ("显示400x400坐标区间", "set_side_profile_coordinate_grid"),
        ("保存侧面标准斜率", "save_side_profile_slopes"),
        ("调整眼球上下贴合", "run_iris_lid_contact"),
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
        ("调整嘴角微翘", "corner_curl"),
        ("调整下唇(正侧)", "lower_lip"),
        ("侧面嘴唇下巴对线(鼻部ROI)", "side_profile"),
    ]
    RESULT_ITEMS = [
        ("eyeball", "眼球"),
        ("eyebrow", "眉毛"),
        ("eyelid", "眼皮"),
        ("upper_lip", "上唇"),
        ("mouth", "下巴"),
        ("corners", "嘴角"),
        ("corner_curl", "嘴角微翘"),
        ("lower_lip", "下唇"),
        ("side_profile", "侧唇下巴"),
    ]
    ROUTE_ITEMS = {
        "iris_lid_contact": ["眼球"],
        "eyeball": ["眼球"],
        "eye_only": ["眼球", "眉毛", "眼皮"],
        "full": ["眼球", "上唇", "下巴", "嘴角", "下唇", "眉毛", "眼皮"],
        "eyebrow": ["眉毛"],
        "eyelid": ["眼皮"],
        "upper_lip": ["上唇"],
        "mouth": ["下巴"],
        "corners": ["嘴角"],
        "corner_curl": ["嘴角微翘"],
        "lower_lip": ["下唇"],
        "side_profile": ["侧唇下巴"],
    }
    ITEM_KEYS = {
        "眼球": "eyeball",
        "眉毛": "eyebrow",
        "眼皮": "eyelid",
        "上唇": "upper_lip",
        "下巴": "mouth",
        "嘴角": "corners",
        "嘴角微翘": "corner_curl",
        "下唇": "lower_lip",
        "侧唇下巴": "side_profile",
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
        self.resize(1180, 900)
        self.detector_thread = None
        self.front_camera_index = CAMERA_INDEX
        self.side_camera_index = UPPER_LIP_SIDE_CAMERA_INDEX
        self.servo_specs = self._load_servo_specs()
        self.servo_controls = {}
        self.manual_controls_enabled = True
        self.servo_control_group = None
        self.face_overlays_button = None
        self.side_profile_preview_button = None
        self.side_profile_expected_slope_button = None
        self.side_profile_coordinate_grid_button = None
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

        top.addWidget(QLabel("正面摄像机"))
        self.front_camera_combo = self._camera_combo(self.front_camera_index)
        self.front_camera_combo.currentIndexChanged.connect(self.on_front_camera_changed)
        top.addWidget(self.front_camera_combo)

        top.addWidget(QLabel("侧面摄像机"))
        self.side_camera_combo = self._camera_combo(self.side_camera_index)
        self.side_camera_combo.currentIndexChanged.connect(self.on_side_camera_changed)
        top.addWidget(self.side_camera_combo)

        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        top.addWidget(self.status_label, 1)

        body = QHBoxLayout()
        layout.addLayout(body, 1)

        body.addWidget(self._build_baseline_group(), 1)
        body.addWidget(self._build_route_group(), 1)
        body.addWidget(self._build_result_group(), 1)

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        layout.addWidget(self.log_box, 1)

        self.servo_control_group = self._build_servo_control_group()
        layout.addWidget(self.servo_control_group)

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
            if command == "run_iris_lid_contact":
                btn = QPushButton(label)
                btn.clicked.connect(
                    lambda _checked=False: self.start_route("iris_lid_contact"))
            elif command == "set_face_overlays":
                btn = QPushButton(label)
                btn.setCheckable(True)
                btn.toggled.connect(self.on_face_overlays_toggled)
                self.face_overlays_button = btn
            elif command == "set_side_profile_preview":
                btn = QPushButton(label)
                btn.setCheckable(True)
                btn.toggled.connect(self.on_side_profile_preview_toggled)
                self.side_profile_preview_button = btn
            elif command == "set_side_profile_expected_slope":
                btn = QPushButton(label)
                btn.setCheckable(True)
                btn.toggled.connect(self.on_side_profile_expected_slope_toggled)
                self.side_profile_expected_slope_button = btn
            elif command == "set_side_profile_coordinate_grid":
                btn = QPushButton(label)
                btn.setCheckable(True)
                btn.toggled.connect(self.on_side_profile_coordinate_grid_toggled)
                self.side_profile_coordinate_grid_button = btn
            else:
                btn = self._command_button(label, command)
            grid.addWidget(btn, idx // BASELINE_BUTTON_COLUMNS, idx % BASELINE_BUTTON_COLUMNS)
        return group

    @staticmethod
    def _load_servo_specs() -> list[dict[str, Any]]:
        yaml_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), DEFAULT_SERVO_CONFIG)
        config = read_yaml(yaml_path)
        servo_info = config.get("SERVO_INFO", config)
        enabled_channels = load_manual_servo_channels()
        specs = []
        for servo_name, info in servo_info.items():
            channel = int(info["channel_idx"])
            start = int(info["start_deg"])
            end = int(info["end_deg"])
            low = min(start, end)
            high = max(start, end)
            value = clamp(int(info.get("temp_deg", low)), low, high)
            description, minus_effect, plus_effect = (
                CalibrationPanel._parse_servo_description_and_effects(
                    str(info.get("info", "")), start, end))
            specs.append({
                "name": servo_name,
                "channel": channel,
                "low": low,
                "high": high,
                "value": value,
                "info": str(info.get("info", "")),
                "description": description,
                "minus_effect": minus_effect,
                "plus_effect": plus_effect,
                "enabled": channel in enabled_channels,
            })
        return sorted(specs, key=lambda item: item["channel"])

    @staticmethod
    def _parse_servo_description_and_effects(
        info_text: str,
        start_angle: int,
        end_angle: int,
    ) -> tuple[str, str, str]:
        """从 ``中文名称 起点动作——终点动作`` 推导角度 -/+1 的动作。"""
        parts = str(info_text).strip().split()
        description = parts[-2] if len(parts) >= 2 else "舵机"
        direction_text = parts[-1] if parts else ""
        endpoint_effects = None
        for separator in ("——", "—", "->", "-"):
            if separator in direction_text:
                endpoint_effects = direction_text.split(separator, 1)
                break
        if not endpoint_effects or len(endpoint_effects) != 2:
            return description, "减", "增"

        start_effect = endpoint_effects[0].strip() or "减"
        end_effect = endpoint_effects[1].strip() or "增"
        if start_angle < end_angle:
            # 角度减小靠近 start，角度增加靠近 end。
            return description, start_effect, end_effect
        # start 是较大角度：角度增加靠近 start，减小靠近 end。
        return description, end_effect, start_effect

    def _build_servo_control_group(self) -> QGroupBox:
        group = QGroupBox(
            "舵机手动调节（灰色=当前未使用；滑条松手 / 文本回车 / 动作按钮每次 1°）")
        outer = QVBoxLayout(group)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(280)
        content = QWidget()
        grid = QGridLayout(content)
        grid.setContentsMargins(6, 4, 6, 4)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(3)

        rows_per_column = max(1, (len(self.servo_specs) + 2) // 3)
        for index, spec in enumerate(self.servo_specs):
            channel = spec["channel"]
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)

            label = QLabel(f"A{channel} {spec['description']}")
            label.setFixedWidth(96)
            label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            tooltip = (
                f"{spec['name']}  {spec['info']}\n"
                f"允许范围: {spec['low']}° ~ {spec['high']}°"
            )
            if not spec["enabled"]:
                tooltip += "\n当前项目未使用，手动控制已禁用。"
            label.setToolTip(tooltip)
            row_layout.addWidget(label)

            slider = QSlider(Qt.Horizontal)
            slider.setRange(spec["low"], spec["high"])
            slider.setValue(spec["value"])
            slider.setMinimumWidth(95)
            slider.setToolTip(tooltip)
            row_layout.addWidget(slider, 1)

            editor = QLineEdit(str(spec["value"]))
            editor.setValidator(QIntValidator(spec["low"], spec["high"], editor))
            editor.setFixedWidth(46)
            editor.setAlignment(Qt.AlignCenter)
            editor.setToolTip(tooltip)
            row_layout.addWidget(editor)

            minus_button = QPushButton(spec["minus_effect"])
            minus_button.setFixedWidth(28)
            minus_button.setToolTip(
                f"{spec['minus_effect']}：角度减小 1° 并立即发送")
            row_layout.addWidget(minus_button)

            plus_button = QPushButton(spec["plus_effect"])
            plus_button.setFixedWidth(28)
            plus_button.setToolTip(
                f"{spec['plus_effect']}：角度增加 1° 并立即发送")
            row_layout.addWidget(plus_button)

            slider.valueChanged.connect(
                lambda value, channel=channel: self._on_servo_slider_changed(channel, value))
            slider.sliderReleased.connect(
                lambda channel=channel: self._commit_servo_slider(channel))
            editor.returnPressed.connect(
                lambda channel=channel: self._commit_servo_editor(channel))
            minus_button.clicked.connect(
                lambda _checked=False, channel=channel: self._step_manual_servo(channel, -1))
            plus_button.clicked.connect(
                lambda _checked=False, channel=channel: self._step_manual_servo(channel, 1))

            self.servo_controls[channel] = {
                "row": row_widget,
                "slider": slider,
                "editor": editor,
                "minus": minus_button,
                "plus": plus_button,
                "spec": spec,
            }
            row_widget.setEnabled(spec["enabled"])
            grid.addWidget(
                row_widget,
                index % rows_per_column,
                index // rows_per_column,
            )

        scroll.setWidget(content)
        outer.addWidget(scroll)
        return group

    def _on_servo_slider_changed(self, channel: int, value: int) -> None:
        controls = self.servo_controls.get(channel)
        if controls is None:
            return
        editor = controls["editor"]
        editor.blockSignals(True)
        editor.setText(str(int(value)))
        editor.blockSignals(False)

    def _commit_servo_slider(self, channel: int) -> None:
        controls = self.servo_controls.get(channel)
        if controls is not None:
            self._request_manual_servo(channel, controls["slider"].value())

    def _commit_servo_editor(self, channel: int) -> None:
        controls = self.servo_controls.get(channel)
        if controls is None:
            return
        try:
            value = int(controls["editor"].text())
        except ValueError:
            self._set_servo_control_value(channel, controls["slider"].value())
            return
        self._request_manual_servo(channel, value)

    def _step_manual_servo(self, channel: int, delta: int) -> None:
        controls = self.servo_controls.get(channel)
        if controls is None:
            return
        try:
            current = int(controls["editor"].text())
        except ValueError:
            current = controls["slider"].value()
        self._request_manual_servo(channel, current + int(delta))

    def _request_manual_servo(self, channel: int, angle: int) -> None:
        if not self.manual_controls_enabled or self.detector_thread is None:
            return
        controls = self.servo_controls.get(channel)
        if controls is None:
            return
        spec = controls["spec"]
        if not spec["enabled"]:
            self.append_log(f"A{channel} 当前项目未使用，不能手动调节。")
            return
        angle = clamp(angle, spec["low"], spec["high"])
        self._set_servo_control_value(channel, angle)
        self.detector_thread.send((
            "manual_servo", {"channel": channel, "angle": angle}))

    def _set_servo_control_value(self, channel: int, value: Any) -> None:
        controls = self.servo_controls.get(channel)
        if controls is None:
            return
        spec = controls["spec"]
        value = clamp(int(round(float(value))), spec["low"], spec["high"])
        slider = controls["slider"]
        editor = controls["editor"]
        slider.blockSignals(True)
        editor.blockSignals(True)
        slider.setValue(value)
        editor.setText(str(value))
        slider.blockSignals(False)
        editor.blockSignals(False)

    def update_servo_values(self, values: Any) -> None:
        if not isinstance(values, (list, tuple)):
            return
        for channel, value in enumerate(values):
            if channel in self.servo_controls:
                self._set_servo_control_value(channel, value)

    def set_manual_controls_enabled(self, enabled: bool) -> None:
        self.manual_controls_enabled = bool(enabled)
        if self.servo_control_group is not None:
            self.servo_control_group.setEnabled(self.manual_controls_enabled)

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

    def _camera_combo(self, current_index: int) -> QComboBox:
        combo = QComboBox()
        for camera_index in CAMERA_CHOICES:
            combo.addItem(f"cam{camera_index}", camera_index)
        idx = combo.findData(int(current_index))
        if idx >= 0:
            combo.setCurrentIndex(idx)
        return combo

    def on_front_camera_changed(self) -> None:
        camera_index = int(self.front_camera_combo.currentData())
        self.front_camera_index = camera_index
        self.append_log(f"Front camera selected: cam{camera_index}")
        if self.detector_thread is not None:
            self.detector_thread.send(("set_front_camera", camera_index))

    def on_side_camera_changed(self) -> None:
        camera_index = int(self.side_camera_combo.currentData())
        self.side_camera_index = camera_index
        self.append_log(f"Side camera selected: cam{camera_index}")
        if self.detector_thread is not None:
            self.detector_thread.send(("set_side_camera", camera_index))

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
        self.detector_thread = DetectorThread(
            self.current_template,
            front_camera_index=self.front_camera_index,
            side_camera_index=self.side_camera_index,
        )
        self.detector_thread.log.connect(self.append_log)
        self.detector_thread.status.connect(self.set_status)
        self.detector_thread.stopped.connect(self.on_detector_stopped)
        self.detector_thread.route_finished.connect(self.on_route_finished)
        self.detector_thread.item_result.connect(self.on_item_result)
        self.detector_thread.servo_values.connect(self.update_servo_values)
        self.detector_thread.start()
        self.set_status("Camera running")

    def stop_detector(self) -> None:
        if self.detector_thread is not None:
            self.detector_thread.stop()
            self.detector_thread.wait(3000)

    def on_detector_stopped(self) -> None:
        self.detector_thread = None
        self.set_manual_controls_enabled(True)

    def send_detector(self, command: str) -> None:
        if self.detector_thread is None:
            return
        self.detector_thread.send((command, None))

    def on_face_overlays_toggled(self, hidden: bool) -> None:
        if self.face_overlays_button is not None:
            self.face_overlays_button.setText(
                "显示脸部绘制" if hidden else "隐藏脸部绘制")
        if self.detector_thread is not None:
            self.detector_thread.send(("set_face_overlays", not hidden))

    def on_side_profile_preview_toggled(self, enabled: bool) -> None:
        if self.side_profile_preview_button is not None:
            self.side_profile_preview_button.setText(
                "停止检测侧面4端点" if enabled else "检测侧面4端点")
        if self.detector_thread is not None:
            self.detector_thread.send(("set_side_profile_preview", enabled))

    def on_side_profile_expected_slope_toggled(self, enabled: bool) -> None:
        if self.side_profile_expected_slope_button is not None:
            self.side_profile_expected_slope_button.setText(
                "隐藏期望斜率虚线" if enabled else "显示期望斜率虚线")
        if self.detector_thread is not None:
            self.detector_thread.send(("set_side_profile_expected_slope", enabled))

    def on_side_profile_coordinate_grid_toggled(self, enabled: bool) -> None:
        if self.side_profile_coordinate_grid_button is not None:
            self.side_profile_coordinate_grid_button.setText(
                "隐藏400x400坐标区间" if enabled else "显示400x400坐标区间")
        if self.detector_thread is not None:
            self.detector_thread.send(("set_side_profile_coordinate_grid", enabled))

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
        self.set_manual_controls_enabled(False)
        self.detector_thread.send(("route", payload))

    def abort_current_route(self) -> None:
        if self.detector_thread is None:
            self.append_log("No detector thread is running.")
            return
        if not self.detector_thread.request_abort_route():
            self.append_log("No route is currently running.")

    def on_route_finished(self, route: str, aborted: bool = False) -> None:
        self.set_manual_controls_enabled(True)
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
