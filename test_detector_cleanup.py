"""Cleanup regression tests without importing detector/models or opening hardware."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class CleanupTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(Path('ellseg_scorer.py').read_text(encoding='utf-8-sig'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'EllSegDetector')
        methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                   and n.name in ('__init__', 'stop_display', 'close', '__del__')]
        self.cv = SimpleNamespace(destroyAllWindows=Mock())
        self.ns = {'cv2': self.cv, 'LandmarkSnapshots': lambda: SimpleNamespace(invalidate=Mock())}
        shell = ast.ClassDef(name='Detector', bases=[], keywords=[], body=methods, decorator_list=[])
        exec(compile(ast.fix_missing_locations(ast.Module(body=[shell], type_ignores=[])),
                     '<cleanup-only>', 'exec'), self.ns)
        self.cls = self.ns['Detector']

    def test_partial_object_has_no_display_state(self):
        obj = self.cls.__new__(self.cls)
        obj.close()
        self.assertIsNone(obj.cap)
        self.assertIsNone(obj.landmarker)

    def test_resources_closed_once(self):
        obj = self.cls.__new__(self.cls)
        model, cap = Mock(), Mock()
        obj.landmarker, obj.cap = model, cap
        obj.close()
        obj.close()
        obj.__del__()
        model.close.assert_called_once()
        cap.release.assert_called_once()

    def test_model_failure_still_releases_camera(self):
        obj = self.cls.__new__(self.cls)
        model, cap = Mock(), Mock()
        model.close.side_effect = RuntimeError('model cleanup error')
        obj.landmarker, obj.cap = model, cap
        with self.assertRaisesRegex(RuntimeError, 'model cleanup error'):
            obj.close()
        cap.release.assert_called_once()
        obj.__del__()
        model.close.assert_called_once()

    def test_existing_display_thread_stops(self):
        obj = self.cls.__new__(self.cls)
        obj._stop_event, obj._display_thread = Mock(), Mock()
        obj._display_thread.is_alive.return_value = True
        obj.stop_display()
        obj._stop_event.set.assert_called_once()
        obj._display_thread.join.assert_called_once_with(timeout=2)

    def test_real_init_mode_failure_preserves_primary_error(self):
        model, cap = Mock(), Mock()
        cap.isOpened.return_value = True
        cap.get.return_value = 0
        self.cv.CAP_DSHOW = 'DSHOW'
        for name in ('FOURCC', 'FPS', 'FRAME_WIDTH', 'FRAME_HEIGHT', 'BUFFERSIZE'):
            setattr(self.cv, 'CAP_PROP_'+name, name)
        self.cv.VideoCapture = Mock(return_value=cap)
        self.cv.VideoWriter_fourcc = Mock(return_value=123)
        self.ns['_init_mediapipe'] = lambda: model
        obj = self.cls.__new__(self.cls)
        with patch('builtins.print'):
            with self.assertRaisesRegex(RuntimeError, 'Front camera mode mismatch'):
                obj.__init__(camera_index=0)
        self.assertFalse(hasattr(obj, '_stop_event'))
        obj.__del__()
        model.close.assert_called_once()
        cap.release.assert_called_once()


if __name__ == '__main__':
    unittest.main()
