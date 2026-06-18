import queue
import sys
import time

import cv2

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
    save_current_as_template,
    set_active_template,
    template_section_exists,
)

from eye_auto_tuner import EyeAutoTuner


def apply_template(template_name):
    set_active_template(template_name)
    return True, f"Active template -> {template_name} (face_point.json)"


class DetectorThread(QThread):
    log = Signal(str)
    status = Signal(str)
    stopped = Signal()
    route_finished = Signal(str)
    item_result = Signal(str, str)

    def __init__(self, template_getter, parent=None):
        super().__init__(parent)
        self.template_getter = template_getter
        self.commands = queue.Queue()
        self._running = True
        self._route_running = False
        self.detector = None
        self._tuner = None  # 缓存 tuner，供手动控制使用

    def _ensure_tuner(self):
        """延迟初始化调优实例（舵机控制器），供手动控制命令复用"""
        if self._tuner is None:
            try:
                t = EyeAutoTuner(
                    yaml_file="29_servo_config(13).yaml",
                    detector=self.detector,
                )
                t.initialize()
                self._tuner = t
                self.log.emit("Servo controller initialized for manual control.")
            except Exception as exc:
                self.log.emit(f"[ERROR] Cannot init servo controller: {exc}")
                self._tuner = False  # 标记失败，不再重试
        return self._tuner if self._tuner is not False else None

    def _reload_template_baselines(self):
        self.detector.eyelid_baseline = load_eyelid_baseline()
        self.detector.eyebrow_baseline = load_eyebrow_baseline()
        self.detector.mouth_baseline = load_mouth_baseline()
        self.detector.lower_lip_baseline = load_lower_lip_baseline()
        self.detector.upper_lip_baseline = load_upper_lip_baseline()
        self.detector.mouth_corners_baseline = load_mouth_corners_baseline()
        self.detector.head_position_baseline = load_head_position_baseline()

    def send(self, command):
        self.commands.put(command)

    def stop(self):
        self._running = False
        self.commands.put(("stop", None))

    def run(self):
        try:
            self.detector = EllSegDetector()
            actual_w = int(self.detector.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.detector.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.log.emit(f"Camera opened: {actual_w}x{actual_h}")
            self.detector.enable_mp = True
            self.detector.enable_ellseg = True
            self.detector.start_display()
            self.log.emit("Calibration camera started. [MP+EllSeg ON by default]")
            while self._running:
                if self._route_running:
                    # 路线运行中，只 drain 命令，不做 capture/detect
                    self._drain_commands()
                    time.sleep(0.01)
                    continue
                ok, frame = self.detector.capture()
                if ok:
                    self.detector.detect(frame)
                self._drain_commands()
                if self.detector.user_pressed_stop:
                    self.log.emit("OpenCV window requested stop.")
                    break
                time.sleep(0.001)
        except Exception as exc:
            self.log.emit(f"[ERROR] Detector failed: {exc}")
        finally:
            try:
                if self.detector is not None:
                    self.detector.stop_display()
                    self.detector.close()
            finally:
                self.detector = None
                self.status.emit("Camera stopped")
                self.stopped.emit()

    def _drain_commands(self):
        while True:
            try:
                command, payload = self.commands.get_nowait()
            except queue.Empty:
                break
            self._handle_command(command, payload)

    def _handle_command(self, command, payload):
        if command == "stop":
            self._running = False
            return
        if self.detector is None:
            return

        try:
            if command == "template":
                template_name = payload
                ok, msg = apply_template(template_name)
                self.log.emit(msg)
                self._reload_template_baselines()
                return

            if command == "toggle_mp":
                self.detector.enable_mp = not self.detector.enable_mp
                self.log.emit(f"MediaPipe: {'ON' if self.detector.enable_mp else 'OFF'}")
                return
            if command == "toggle_ellseg":
                self.detector.enable_ellseg = not self.detector.enable_ellseg
                self.log.emit(f"EllSeg: {'ON' if self.detector.enable_ellseg else 'OFF'}")
                return
            if command == "toggle_landmarks":
                self.detector.show_all_landmarks = not self.detector.show_all_landmarks
                self.log.emit(f"Landmarks: {'ON' if self.detector.show_all_landmarks else 'OFF'}")
                return
            if command == "toggle_overlay":
                self.detector.show_baseline_overlay = not self.detector.show_baseline_overlay
                self.log.emit(f"Baseline overlay: {'ON' if self.detector.show_baseline_overlay else 'OFF'}")
                return
            if command == "toggle_eyeline":
                self.detector.show_eye_line_offset = not self.detector.show_eye_line_offset
                self.log.emit(f"EyeLine HUD: {'ON' if self.detector.show_eye_line_offset else 'OFF'}")
                return

            # ---- 手动控制：眼皮开/合 ----
            if command in ("eyelid_open", "eyelid_close"):
                tuner = self._ensure_tuner()
                if tuner is None:
                    self.log.emit("[ERROR] Servo controller not available.")
                    return
                ctrl = tuner.controller
                # 从 YAML 配置读取范围边界
                a8_lo = min(ctrl.list_start_deg[8], ctrl.list_end_deg[8])
                a8_hi = max(ctrl.list_start_deg[8], ctrl.list_end_deg[8])
                a9_lo = min(ctrl.list_start_deg[9], ctrl.list_end_deg[9])
                a9_hi = max(ctrl.list_start_deg[9], ctrl.list_end_deg[9])
                # 读取当前角度
                a8 = tuner._temp_deg(8, default=(a8_lo + a8_hi) // 2)
                a9 = tuner._temp_deg(9, default=(a9_lo + a9_hi) // 2)
                # A8(左眼皮): 开=减角度, 合=加角度
                # A9(右眼皮): 开=加角度, 合=减角度
                if command == "eyelid_open":
                    a8 = max(a8_lo, a8 - 1)
                    a9 = min(a9_hi, a9 + 1)
                    label = "开"
                else:
                    a8 = min(a8_hi, a8 + 1)
                    a9 = max(a9_lo, a9 - 1)
                    label = "合"
                tuner.send_servo(8, a8)
                tuner.send_servo(9, a9)
                self.log.emit(f"眼皮{label}: A8={a8}° A9={a9}°")
                return



            save_map = {
                "save_eye": self.detector.save_current_baseline,
                "save_eyelid": self.detector.save_current_eyelid_baseline,
                "save_eyebrow": self.detector.save_current_eyebrow_baseline,
                "save_mouth": self.detector.save_current_mouth_baseline,
                "save_lower_lip": self.detector.save_current_lower_lip_baseline,
                "save_upper_lip": self.detector.save_current_upper_lip_baseline,
                "save_corners": self.detector.save_current_mouth_corners_baseline,
                "save_head": self.detector.save_current_head_position_baseline,
                "adjust_eyeline": self.detector.adjust_baseline_by_vertical_offset,
            }
            if command in save_map:
                ok = save_map[command]()
                self.log.emit(f"{command}: {'OK' if ok else 'FAILED'}")
                return

            if command == "save_all":
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

            if command == "route":
                self._run_route(payload)

        except Exception as exc:
            self.log.emit(f"[ERROR] {command}: {exc}")

    def _run_route(self, payload):
        if isinstance(payload, dict):
            route = payload.get("route")
            max_iterations = int(payload.get("max_iterations", 50))
        else:
            route = payload
            max_iterations = 50
        if self._route_running:
            self.log.emit("Route is already running.")
            return

        ok, msg = apply_template(self.template_getter())
        self.log.emit(msg)
        self._reload_template_baselines()
        if route in ("full", "eye_only", "eyebrow") and self.detector.eyebrow_baseline is None:
            self.log.emit("[WARN] Current template has no eyebrow baseline; cannot run this route.")
            return

        self._route_running = True
        self.log.emit(f"Starting route: {route}, max_iterations={max_iterations}")
        self.status.emit(f"Route running: {route}")

        try:
            tuner = self._ensure_tuner()
            if tuner is None:
                self.log.emit("[ERROR] Servo controller not available.")
                return
            self.log.emit("Using cached servo controller state for route.")

            if route == "eyeball":
                self._run_eyeball(tuner, max_iterations)
            elif route == "eye_only":
                self._run_eyeball(tuner, max_iterations)
                self._run_eyebrow(tuner, max_iterations)
                self._run_eyelid(tuner, max_iterations)
            elif route == "full":
                self._run_eyeball(tuner, max_iterations)
                self._run_upper_lip(tuner, max_iterations)
                self._run_mouth(tuner, max_iterations)
                self._run_corners(tuner, max_iterations)
                self._run_lower_lip(tuner, max_iterations)
                self._run_eyebrow(tuner, max_iterations)
                self._run_eyelid(tuner, max_iterations)
            elif route == "eyebrow":
                self._run_eyebrow(tuner, max_iterations)
            elif route == "eyelid":
                self._run_eyelid(tuner, max_iterations)
        except Exception as exc:
            self.log.emit(f"[Route Error] {exc}")
            import traceback
            self.log.emit(traceback.format_exc())
        finally:
            self._route_running = False
            self.route_finished.emit(route)
            self.status.emit("Ready")
            self.log.emit("Route finished. Camera remains open.")

    def _emit_item_result(self, item, passed, iterations):
        text = f"{'通过' if passed else '失败'} ({iterations}次迭代)"
        self.log.emit(f"{item}: {text}")
        self.item_result.emit(item, text)

    def _emit_item_running(self, item):
        self.item_result.emit(item, "进行中")

    def _run_eyeball(self, tuner, max_iterations):
        self.log.emit(">>> 调整: 眼球 A10-A13 <<<")
        self._emit_item_running("眼球")
        passed, iterations, _ = tuner.auto_adjust(
            pixel_to_degree=1.0,
            max_iterations=max_iterations,
            wait_seconds=1.0,
            keep_display=True,
        )
        self._emit_item_result("眼球", passed, iterations)

    def _run_eyebrow(self, tuner, max_iterations):
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

    def _run_eyelid(self, tuner, max_iterations):
        self.log.emit(">>> 调整: 眼皮 A8-A9 <<<")
        self._emit_item_running("眼皮")
        passed, iterations, _ = tuner.auto_adjust_eyelid(
            ear_step=1.0,
            max_iterations=max_iterations,
            wait_seconds=1.0,
            keep_display=True,
        )
        self._emit_item_result("眼皮", passed, iterations)

    def _run_upper_lip(self, tuner, max_iterations):
        self._emit_item_running("上唇")
        if tuner.scorer.upper_lip_baseline is None:
            self.log.emit("上唇: 跳过 (无基线)")
            self.item_result.emit("上唇", "跳过 (无基线)")
            return
        passed, iterations, _ = tuner.auto_adjust_upper_lip(
            ulr_step=1.0,
            max_iterations=max_iterations,
            wait_seconds=1.0,
            keep_display=True,
        )
        self._emit_item_result("上唇", passed, iterations)

    def _run_mouth(self, tuner, max_iterations):
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

    def _run_corners(self, tuner, max_iterations):
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

    def _run_lower_lip(self, tuner, max_iterations):
        self._emit_item_running("下唇")
        if tuner.scorer.lower_lip_baseline is None:
            self.log.emit("下唇: 跳过 (无基线)")
            self.item_result.emit("下唇", "跳过 (无基线)")
            return
        passed, iterations, _ = tuner.auto_adjust_lower_lip(
            step=1.0,
            max_iterations=max_iterations,
            wait_seconds=1.0,
            keep_display=True,
        )
        self._emit_item_result("下唇", passed, iterations)

class CalibrationPanel(QMainWindow):
    ROUTE_ITEMS = {
        "eyeball": ["眼球"],
        "eye_only": ["眼球", "眉毛", "眼皮"],
        "full": ["眼球", "上唇", "下巴", "嘴角", "下唇", "眉毛", "眼皮"],
        "eyebrow": ["眉毛"],
        "eyelid": ["眼皮"],
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

    def __init__(self):
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

    def current_template(self):
        return self.template_combo.currentText()

    def append_log(self, text):
        self.log_box.appendPlainText(text)

    def set_status(self, text):
        self.status_label.setText(text)

    def _build_baseline_group(self):
        group = QGroupBox("基线录制")
        grid = QGridLayout(group)

        buttons = [
            ("保存眼球 S", lambda: self.send_detector("save_eye")),
            ("保存眼皮 E", lambda: self.send_detector("save_eyelid")),
            ("保存眉毛 B", lambda: self.send_detector("save_eyebrow")),
            ("保存嘴部 M", lambda: self.send_detector("save_mouth")),
            ("保存下唇 L", lambda: self.send_detector("save_lower_lip")),
            ("保存上唇 U", lambda: self.send_detector("save_upper_lip")),
            ("保存嘴角 C", lambda: self.send_detector("save_corners")),
            ("保存头位 D", lambda: self.send_detector("save_head")),
            ("眼线修正 V", lambda: self.send_detector("adjust_eyeline")),
            ("保存全部 A", lambda: self.send_detector("save_all")),
            ("切换 MP 1", lambda: self.send_detector("toggle_mp")),
            ("切换 EllSeg 2", lambda: self.send_detector("toggle_ellseg")),
            ("切换关键点 3", lambda: self.send_detector("toggle_landmarks")),
            ("基线叠加 G", lambda: self.send_detector("toggle_overlay")),
            ("EyeLine HUD Y", lambda: self.send_detector("toggle_eyeline")),
        ]
        for idx, (label, slot) in enumerate(buttons):
            btn = QPushButton(label)
            btn.clicked.connect(slot)
            grid.addWidget(btn, idx // 2, idx % 2)
        return group

    def _build_manual_group(self):
        group = QGroupBox("手动控制")
        layout = QVBoxLayout(group)
        btn_open = QPushButton("眼皮开")
        btn_open.clicked.connect(lambda: self.send_detector("eyelid_open"))
        btn_close = QPushButton("眼皮合")
        btn_close.clicked.connect(lambda: self.send_detector("eyelid_close"))
        layout.addWidget(btn_open)
        layout.addWidget(btn_close)
        layout.addStretch(1)
        return group

    def _build_route_group(self):
        group = QGroupBox("检测 / 自动调整路线")
        layout = QVBoxLayout(group)

        iter_row = QHBoxLayout()
        iter_row.addWidget(QLabel("最大调整次数"))
        self.max_iterations_spin = QSpinBox()
        self.max_iterations_spin.setRange(1, 500)
        self.max_iterations_spin.setValue(50)
        self.max_iterations_spin.setSuffix(" 次")
        iter_row.addWidget(self.max_iterations_spin)
        layout.addLayout(iter_row)

        routes = [
            ("只调眼球", lambda: self.start_route("eyeball")),
            ("眼球 + 眉毛 + 眼皮", lambda: self.start_route("eye_only")),
            ("完整默认路线", lambda: self.start_route("full")),
            ("只调眉毛", lambda: self.start_route("eyebrow")),
            ("只调眼皮", lambda: self.start_route("eyelid")),
        ]
        for label, slot in routes:
            btn = QPushButton(label)
            btn.clicked.connect(slot)
            layout.addWidget(btn)
        layout.addStretch(1)
        return group

    def _build_result_group(self):
        group = QGroupBox("通过状态")
        grid = QGridLayout(group)
        self.result_labels = {}
        items = [
            ("eyeball", "眼球"),
            ("eyebrow", "眉毛"),
            ("eyelid", "眼皮"),
            ("upper_lip", "上唇"),
            ("mouth", "下巴"),
            ("corners", "嘴角"),
            ("lower_lip", "下唇"),
        ]
        for row, (key, label) in enumerate(items):
            grid.addWidget(QLabel(label), row, 0)
            value = QLabel("未开始")
            value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            value.setStyleSheet(self.STATUS_STYLES["未开始"])
            self.result_labels[key] = value
            grid.addWidget(value, row, 1)
        return group

    def set_result_status(self, item, text):
        key = self.ITEM_KEYS.get(item, item)
        label = self.result_labels.get(key)
        if label is None:
            return
        label.setText(text)
        state = text.split(" ", 1)[0]
        label.setStyleSheet(self.STATUS_STYLES.get(state, ""))

    def reset_route_statuses(self, route):
        active_items = set(self.ROUTE_ITEMS.get(route, []))
        for item, key in self.ITEM_KEYS.items():
            if item in active_items:
                self.set_result_status(item, "未开始")
            else:
                self.set_result_status(item, "未开始")

    def refresh_template_combo(self):
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

    def refresh_template_status(self):
        template_name = self.current_template()
        brow = "FOUND" if template_section_exists(template_name, "eyebrow") else "MISSING"
        eyelid = "FOUND" if template_section_exists(template_name, "eyelid") else "MISSING"
        self.append_log(f"{template_name} in face_point.json: eyebrow={brow}, eyelid={eyelid}")

    def on_template_changed(self):
        template_name = self.current_template()
        ok, msg = apply_template(template_name)
        self.append_log(msg)
        self.refresh_template_status()
        if self.detector_thread is not None:
            self.detector_thread.send(("template", template_name))

    def save_current_template(self):
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

    def start_detector(self):
        if self.detector_thread is not None:
            self.append_log("Calibration camera is already running.")
            return
        ok, msg = apply_template(self.current_template())
        self.append_log(msg)
        self.detector_thread = DetectorThread(self.current_template)
        self.detector_thread.log.connect(self.append_log)
        self.detector_thread.status.connect(self.set_status)
        self.detector_thread.stopped.connect(self.on_detector_stopped)
        self.detector_thread.route_finished.connect(self.on_route_finished)
        self.detector_thread.item_result.connect(self.on_item_result)
        self.detector_thread.start()
        self.set_status("Camera running")

    def stop_detector(self):
        if self.detector_thread is not None:
            self.detector_thread.stop()
            self.detector_thread.wait(3000)

    def on_detector_stopped(self):
        self.detector_thread = None

    def send_detector(self, command):
        if self.detector_thread is None:
            return
        self.detector_thread.send((command, None))

    def start_route(self, route):
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

    def on_route_finished(self, route):
        self.append_log(f"Route {route} finished.")

    def on_item_result(self, item, text):
        self.set_result_status(item, text)
        self.append_log(f"[STATUS] {item}: {text}")

    def closeEvent(self, event):
        self.stop_detector()
        event.accept()


def main():
    app = QApplication(sys.argv)
    panel = CalibrationPanel()
    panel.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
