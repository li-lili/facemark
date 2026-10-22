"""Offline scheduling checks: execute panel methods without opening cameras/models."""
import ast
from datetime import datetime, timedelta
from pathlib import Path
import threading
import unittest
from unittest.mock import Mock


class HourlySnapshotTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(Path(__file__).with_name('qt_calibration_panel.py').read_text(encoding='utf-8'))
        names = {'_schedule_next_hourly_snapshot', '_toggle_hourly_snapshot',
                 '_check_hourly_snapshot', '_save_hourly_snapshot'}
        methods = [node for cls in tree.body if isinstance(cls, ast.ClassDef)
                   for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
        self.clock = Mock()
        self.clock.now.return_value = datetime(2026, 10, 16, 9, 15)
        namespace = dict(datetime=self.clock, timedelta=timedelta, threading=threading,
                         timestamp=lambda: 'test-time')
        exec(compile(ast.Module(body=methods, type_ignores=[]), '<panel methods>', 'exec'), namespace)
        harness = type('PanelHarness', (), {name: namespace[name] for name in names})
        self.panel = harness()
        self.panel._hourly_cancelled = threading.Event()
        self.panel.next_snapshot_label = Mock()
        self.panel.hourly_snapshot_timer = Mock()
        self.panel.append_log = Mock()
        self.panel.detector_thread = Mock()
        self.panel.log = Mock()
        self.panel.detector = Mock()

    def test_wait_until_hour_and_no_duplicate(self):
        p = self.panel
        p._toggle_hourly_snapshot(True)
        self.assertEqual(p._next_hourly_snapshot, datetime(2026, 10, 16, 10))
        p._check_hourly_snapshot()
        p.detector_thread.send.assert_not_called()
        self.clock.now.return_value = datetime(2026, 10, 16, 10)
        p._check_hourly_snapshot()
        p._check_hourly_snapshot()
        p.detector_thread.send.assert_called_once()
        self.assertEqual(p._next_hourly_snapshot, datetime(2026, 10, 16, 11))

    def test_midnight_and_resume_skip_backlog(self):
        p = self.panel
        self.clock.now.return_value = datetime(2026, 10, 16, 23, 59)
        p._toggle_hourly_snapshot(True)
        self.assertEqual(p._next_hourly_snapshot, datetime(2026, 10, 17))
        self.clock.now.return_value = datetime(2026, 10, 19, 12, 30)
        p._check_hourly_snapshot()
        p.detector_thread.send.assert_called_once()
        self.assertEqual(p._next_hourly_snapshot, datetime(2026, 10, 19, 13))

    def test_disable_cancels_queued_request(self):
        p = self.panel
        p._toggle_hourly_snapshot(True)
        token = p._hourly_cancelled
        p._toggle_hourly_snapshot(False)
        p._save_hourly_snapshot(('scheduled', token))
        p._check_hourly_snapshot()
        p.detector.save_landmark_snapshot.assert_not_called()
        p.detector_thread.send.assert_not_called()
        p._toggle_hourly_snapshot(True)
        self.assertTrue(token.is_set())
        self.assertFalse(p._hourly_cancelled.is_set())

    def test_unavailable_and_exception_keep_next_hour(self):
        p = self.panel
        p._toggle_hourly_snapshot(True)
        p.detector_thread.isRunning.return_value = False
        self.clock.now.return_value = datetime(2026, 10, 16, 10)
        p._check_hourly_snapshot()
        self.assertIn('[ERROR]', p.append_log.call_args.args[0])
        self.assertEqual(p._next_hourly_snapshot, datetime(2026, 10, 16, 11))
        p.detector.save_landmark_snapshot.side_effect = RuntimeError('disconnected')
        p._save_hourly_snapshot(('10:00', p._hourly_cancelled))
        self.assertIn('disconnected', p.log.emit.call_args.args[0])
        p.detector.save_landmark_snapshot.side_effect = None
        p._save_hourly_snapshot(('11:00', p._hourly_cancelled))
        self.assertEqual(p.detector.save_landmark_snapshot.call_count, 2)


if __name__ == '__main__':
    unittest.main()
