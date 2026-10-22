"""Qt camera parameter search using the unchanged EllSeg detector.

python camera_auto_calibrate.py
python camera_auto_calibrate.py --exposures=-8,-7,-6,-5 --fps=30,60
python camera_auto_calibrate.py --gains=0,16,32 --focus=80,100,120
"""
import argparse
import hashlib
import json
import itertools
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QMainWindow,
                              QPushButton, QTextEdit, QVBoxLayout, QWidget)

from camera_tuning import (SAMPLE_SECONDS, SETTLE_SECONDS, apply_checked,
                           draw_camera_hud, measurement, restore, save_report,
                           snapshot, summarize)


class Cancelled(Exception):
    pass


class Worker(QThread):
    preview = Signal(QImage)
    log = Signal(str)
    busy = Signal(bool)
    ready = Signal()

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.shutdown = threading.Event()
        self.cancel = threading.Event()
        self.start_search = threading.Event()
        self.open_settings = threading.Event()
        self.last_preview = 0.0
        self.settings = {}
        self.detector = None

    def check_cancel(self):
        if self.cancel.is_set() or self.shutdown.is_set():
            raise Cancelled()

    def capture(self):
        ok, frame = self.detector.capture()
        if not ok or frame is None:
            time.sleep(0.01)
            return None, None
        if frame.shape[:2] != (1080, 1920):
            raise RuntimeError(f'实际尺寸 {frame.shape[1]}x{frame.shape[0]}，需要1920x1080以保持像素比较与HUD布局一致')
        det = self.detector.detect(frame)
        if time.monotonic() - self.last_preview >= 0.1:
            display = frame.copy()
            for side, label in (('left', 'L'), ('right', 'R')):
                center = det.get(f'{side}_iris')
                if center is not None:
                    cv2.circle(display, tuple(map(int, center)), 3, (0, 255, 0), 1)
                offset = det.get('offset', {}).get(side)
                text = f'{label} raw dX={offset[0]:+.2f} dY={offset[1]:+.2f}' if offset else f'{label} raw dX/dY=-- (no baseline/detection)'
                cv2.putText(display, text, (1490, 42 if side == 'left' else 62),
                            0, 0.4, (0, 255, 255), 1)
            draw_camera_hud(display, self.settings, self.args.seconds)
            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            img = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0],
                         QImage.Format.Format_RGB888).copy()
            self.preview.emit(img)
            self.last_preview = time.monotonic()
        return det, frame

    def collect(self, label, reference=None):
        self.log.emit(f'采样 {label}：保持头部、相机、灯光静止')
        for smoother in (self.detector.lm_smoother, self.detector.ell_smooth_L,
                         self.detector.ell_smooth_R):
            smoother.reset()
        # Drain and run the same detector while settling; do not reuse old results.
        deadline = time.monotonic() + SETTLE_SECONDS
        while time.monotonic() < deadline:
            self.check_cancel()
            self.capture()
        samples, attempts = [], 0
        start = time.monotonic()
        while time.monotonic() - start < self.args.seconds:
            self.check_cancel()
            det, frame = self.capture()
            attempts += 1
            sample = measurement(det, frame) if det is not None else None
            if sample is not None:
                sample['frame_digest'] = hashlib.blake2b(frame, digest_size=8).hexdigest()
            samples.append(sample)
        metrics = summarize(samples, attempts, time.monotonic() - start, reference)
        self.log.emit(f'{label}: {"通过" if metrics["eligible"] else "失败"} '
                      f'valid={metrics["valid_rate"]:.1%}, '
                      f'jitter={metrics["jitter_px"]}px, '
                      f'processing={metrics["process_fps"]:.1f}fps '
                      + ' / '.join(metrics['reasons']))
        return dict(label=label, settings=dict(self.settings), metrics=metrics, samples=samples)

    def restore_original(self, original):
        failed = restore(self.detector.cap, original)
        self.settings = snapshot(self.detector.cap)
        if failed:
            self.log.emit('恢复参数未获驱动确认：' + ', '.join(failed))
        return failed

    def search(self):
        self.busy.emit(True)
        original = snapshot(self.detector.cap)
        self.settings = dict(original)
        report = dict(version=1, camera=self.args.camera, backend='DSHOW',
                      resolution=[1920, 1080], original=original, trials=[],
                      search=dict(fps=self.args.fps, exposures=self.args.exposures,
                                  gains=self.args.gains, focus=self.args.focus,
                                  seconds=self.args.seconds, manual_exposure=self.args.manual_exposure),
                      status='running', best=None,
                      note='当前环境及候选范围内比较；处理FPS不是传感器FPS；使用现有检测输出，无额外平滑。')
        path = Path(self.args.output) / ('camera_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.json')
        accepted = False
        try:
            if not all(np.isfinite(list(original.values()))):
                raise RuntimeError('驱动参数读回非有限数值，不能可靠恢复')
            baseline = self.collect('当前设置')
            report['trials'].append(baseline)
            if not baseline['metrics']['eligible']:
                raise RuntimeError('初始参考未通过，请先用驱动设置窗口改善对焦、光照；没有有效参考不推荐参数')
            reference = baseline['metrics']['sharpness']
            # Freeze optional automatic controls only when the driver confirms them.
            self.settings = dict(original)
            fixed = {}
            for auto, manual in (('autofocus', 'focus'), ('auto_wb', 'wb')):
                if original[auto] != 0:
                    _, errors = apply_checked(self.detector.cap, {auto: 0})
                    if errors:
                        self.log.emit(f'无法锁定 {auto}：' + '; '.join(errors))
                        self.restore_original(original)
                        raise RuntimeError('请在驱动窗口手动锁定自动参数后再测试')
                    fixed[auto] = 0
                    fixed[manual] = float(self.detector.cap.get(
                        cv2.CAP_PROP_FOCUS if manual == 'focus' else cv2.CAP_PROP_WB_TEMPERATURE))
            candidates = itertools.product(self.args.fps, self.args.exposures,
                                           self.args.gains or [None], self.args.focus or [None])
            seen, shortlist = set(), [baseline]
            for i, (fps, exposure, gain, focus) in enumerate(candidates, 1):
                self.check_cancel()
                if self.restore_original(original):
                    raise RuntimeError('无法恢复参考设置，停止比较以避免参数串扰')
                requested = dict(fixed, fps=fps, auto_exposure=self.args.manual_exposure,
                                 exposure=exposure)
                if gain is not None:
                    requested['gain'] = gain
                if focus is not None:
                    requested.update(autofocus=0, focus=focus)
                self.settings, errors = apply_checked(self.detector.cap, requested)
                key = tuple(sorted(self.settings.items()))
                if errors:
                    self.log.emit(f'跳过组合{i}：' + '; '.join(errors))
                    report['trials'].append(dict(requested=requested, readback=dict(self.settings), errors=errors))
                    continue
                if key in seen:
                    continue
                seen.add(key)
                row = self.collect(f'组合{i} {requested}', reference)
                row['requested'] = requested
                report['trials'].append(row)
                if row['metrics']['eligible']:
                    shortlist.append(row)
                save_report(path, report)
            if len(shortlist) == 1:
                self.log.emit('没有通过筛选的新组合，本次仅能复测原设置，不能据此认为已找到更优参数')
            # Independent repeat of top three including original when it ranks there.
            shortlist.sort(key=lambda r: r['metrics']['jitter_px'])
            finalists = []
            # Always repeat the original; apparent improvement must survive comparison.
            repeats = [baseline] + [r for r in shortlist if r is not baseline][:2]
            for row in repeats:
                self.check_cancel()
                if self.restore_original(row['settings']):
                    continue
                repeat = self.collect('复测 ' + row['label'], reference)
                report['trials'].append(repeat)
                if repeat['metrics']['eligible']:
                    finalists.append((max(row['metrics']['jitter_px'], repeat['metrics']['jitter_px']), row))
            if not finalists:
                raise RuntimeError('候选复测均未通过，保留原设置')
            score, winner = min(finalists, key=lambda item: item[0])
            self.check_cancel()
            if self.restore_original(winner['settings']):
                raise RuntimeError('最佳设置应用失败')
            report.update(status='complete', best=dict(settings=winner['settings'],
                          worst_repeat_jitter_px=score, source=winner['label']))
            save_report(path, report)
            accepted = True
            self.log.emit(f'完成：已应用本次最佳设置，复测最差波动={score:.3f}px。报告：{path}')
        except Cancelled:
            report['status'] = 'cancelled'
            self.log.emit('已取消，恢复开始前设置；相机继续预览')
        except Exception as exc:
            report.update(status='failed', error=str(exc))
            self.log.emit(f'标定失败：{exc}')
        finally:
            if not accepted:
                self.restore_original(original)
            try:
                # Non-finite initial driver readings cannot be serialized as strict JSON.
                if all(np.isfinite(list(original.values()))):
                    save_report(path, report)
            except Exception as exc:
                self.log.emit(f'保存失败：{exc}')
            self.busy.emit(False)

    def run(self):
        try:
            self.log.emit('加载检测模型并打开摄像头；不会创建舵机控制器')
            from ellseg_scorer import EllSegDetector
            self.detector = EllSegDetector(camera_index=self.args.camera)
            self.detector.enable_mp = True
            self.detector.enable_ellseg = True
            self.settings = snapshot(self.detector.cap)
            if self.args.apply_report:
                with Path(self.args.apply_report).open(encoding='utf-8') as f:
                    saved = json.load(f)
                if (saved.get('status') != 'complete' or saved.get('camera') != self.args.camera
                        or saved.get('resolution') != [1920, 1080]
                        or saved.get('backend') != 'DSHOW'):
                    raise ValueError('报告未完成或相机编号/尺寸/后端不匹配')
                target = saved['best']['settings']
                if set(target) != set(self.settings) or not np.isfinite(list(target.values())).all():
                    raise ValueError('报告中的相机参数不完整或无效')
                original = dict(self.settings)
                if self.restore_original(target):
                    self.restore_original(original)
                    self.log.emit('保存配置应用失败，已尝试恢复启动设置')
                else:
                    self.log.emit('已应用保存报告的推荐参数，请在当前光照下检查或重新标定')
            self.ready.emit()
            failures = 0
            while not self.shutdown.is_set():
                if self.open_settings.is_set():
                    self.open_settings.clear()
                    self.detector.cap.set(cv2.CAP_PROP_SETTINGS, 0)
                    self.settings = snapshot(self.detector.cap)
                if self.start_search.is_set():
                    self.start_search.clear()
                    self.search()
                det, _ = self.capture()
                failures = failures + 1 if det is None else 0
                if failures == 30:
                    self.log.emit('连续读帧失败，请检查相机连接或占用')
        except Exception as exc:
            self.log.emit(f'相机程序错误：{exc}')
        finally:
            # Only explicit exit or unrecoverable initialization/preview error closes it.
            if self.detector is not None:
                self.detector.close()


class Window(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.setWindowTitle('虹膜相机参数标定（无舵机动作）')
        self.resize(1200, 900)
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.addWidget(QLabel('固定头部、相机与光照。关闭其他占用该摄像头的程序。完成后继续预览；关闭窗口才退出。'))
        self.video = QLabel('正在加载模型…')
        self.video.setMinimumSize(960, 540)
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.video, 1)
        buttons = QHBoxLayout()
        self.start = QPushButton('开始搜索最佳参数')
        self.stop = QPushButton('取消并恢复原参数')
        self.settings_button = QPushButton('相机驱动设置（对焦/曝光）')
        for button in (self.start, self.stop, self.settings_button):
            buttons.addWidget(button)
            button.setEnabled(False)
        layout.addLayout(buttons)
        self.logs = QTextEdit()
        self.logs.setReadOnly(True)
        self.logs.setMaximumHeight(220)
        layout.addWidget(self.logs)
        self.setCentralWidget(root)
        self.worker = Worker(args)
        self.worker.preview.connect(self.show_frame)
        self.worker.log.connect(self.logs.append)
        self.worker.ready.connect(lambda: self.set_busy(False))
        self.worker.busy.connect(self.set_busy)
        self.worker.finished.connect(self.on_finished)
        self.start.clicked.connect(self.begin)
        self.stop.clicked.connect(self.worker.cancel.set)
        self.settings_button.clicked.connect(self.worker.open_settings.set)
        self.worker.start()

    def begin(self):
        self.worker.cancel.clear()
        self.set_busy(True)
        self.worker.start_search.set()

    def set_busy(self, busy):
        self.start.setEnabled(not busy)
        self.settings_button.setEnabled(not busy)
        self.stop.setEnabled(busy)

    def show_frame(self, image):
        self.video.setPixmap(QPixmap.fromImage(image).scaled(
            self.video.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def on_finished(self):
        for button in (self.start, self.stop, self.settings_button):
            button.setEnabled(False)
        if self.worker.shutdown.is_set():
            self.close()

    def closeEvent(self, event):
        if self.worker.isRunning():
            self.worker.shutdown.set()
            self.worker.cancel.set()
            self.logs.append('正在停止采集并退出…')
            event.ignore()
        else:
            event.accept()


def numbers(value):
    try:
        result = list(dict.fromkeys(float(x) for x in value.split(',')))
        if not result or not np.isfinite(result).all():
            raise ValueError()
        return result
    except ValueError:
        raise argparse.ArgumentTypeError('需要逗号分隔的有限数值')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--exposures', type=numbers, default=numbers('-8,-7,-6,-5,-4'))
    parser.add_argument('--fps', type=numbers, default=numbers('30,60'))
    parser.add_argument('--gains', type=numbers, default=None)
    parser.add_argument('--focus', type=numbers, default=None)
    parser.add_argument('--manual-exposure', type=float, default=0,
                        help='DirectShow手动曝光模式值，驱动不支持时跳过候选')
    parser.add_argument('--seconds', type=float, default=SAMPLE_SECONDS)
    parser.add_argument('--output', default=str(Path(__file__).resolve().parent / 'camera_calibration_results'))
    parser.add_argument('--apply-report', help='显式应用上次完成的报告，然后保持预览')
    args = parser.parse_args()
    if not np.isfinite([args.seconds, args.manual_exposure]).all() or args.seconds <= 0 or min(args.fps) <= 0 or args.camera < 0:
        parser.error('时长/FPS必须为正数，摄像头编号非负，参数必须有限')
    app = QApplication([])
    window = Window(args)
    window.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
