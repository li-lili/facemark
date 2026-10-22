"""Offline tests: no production imports, models, camera, Qt window or serial."""
import ast
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from calibration_result_snapshot import save_calibration_result


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.original = b'{"active_template":"template_2","common":{"untouched":true}}'
        (self.root / "face_point.json").write_bytes(self.original)
        self.yaml = self.root / "servo_config_adjusted.yaml"
        self.yaml.write_bytes(b"SERVO_INFO: {A10: {temp_deg: 145}}\n")
        # Compile only the real target-coordinate helpers, without importing models.
        tree = ast.parse(Path("ellseg_scorer.py").read_text(encoding="utf-8-sig"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "EllSegDetector")
        names = {"_eyeball_head_adapter_offset", "_eyeball_target_point",
                 "_calc_offset", "_calc_target_offset", "_eyeball_target_mode", "_is_qualified"}
        methods = ast.Module(body=[n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
        namespace = {"np": np, "TOLERANCE": 1.0}
        exec(compile(methods, "<target-helpers-only>", "exec"), namespace)
        class Detector:
            pass
        for name in names:
            setattr(Detector, name, namespace[name])
        self.detector = Detector()
        self.detector.enable_mp = self.detector.enable_ellseg = True
        self.detector.baseline = {"iris_left": [40, 20], "iris_right": [20, 20]}
        self.detector.head_adapter_baseline = {
            "eye_right_offset": [0.768, 0.473], "eye_left_offset": [0.178, 0.934]}
        self.detector.last_result = {
            "frame": np.zeros((60, 80, 3), np.uint8), "left_iris": (41, 21), "right_iris": (20, 21),
            "left_iris_ellipse": ((41.3, 21.9), (6, 7), 0),
            "right_iris_ellipse": ((20.4, 21.1), (6, 7), 0),
            "target_offset": {}, "offset": {"left": (1, 1, 2**.5), "right": (0, 1, 1)},
            "qualified": True,
        }
        for side in ("left", "right"):
            target = self.detector._eyeball_target_point(side)
            center = self.detector.last_result[f"{side}_iris"]
            dx, dy = center[0] - target[0], center[1] - target[1]
            self.detector.last_result["target_offset"][side] = (dx, dy, float(np.hypot(dx, dy)))
        self.args = dict(root=self.root, detector=self.detector, template_name="template_2",
                         camera_info={"fourcc": "MJPG", "width": 80, "height": 60,
                                      "front_camera_index": 1, "rotate_180": True,
                                      "flip_horizontal": False, "flip_vertical": False},
                         servo_cache=None, head_id="Head_01", qt_log="offline log",
                         detection_metadata={"status": "detected", "detected_at": "test-time"},
                         prior_yaml=self.yaml)

    def test_complete_export_and_no_mutation(self):
        before = copy.deepcopy(self.detector.last_result)
        path = save_calibration_result(**self.args)
        result = json.loads((path / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["eyes"]["L"]["effective_target"], [40.0, 20.0])
        self.assertEqual(result["eyes"]["R"]["effective_target"], [20.0, 20.0])
        self.assertEqual(result["eyes"]["L"]["integer_center"], [41, 21])
        self.assertEqual(result["eyes"]["L"]["smoothed_float_center"], [41.3, 21.9])
        self.assertIsNone(result["last_calibration_60"])
        self.assertIsNone(result["calibration_iterations"])
        self.assertIsNone(result["servo_cache"]["values_by_channel"])
        self.assertIn("pre_restart", result["prior_servo_export"]["source"])
        self.assertEqual((path / "pre_restart_servo_commands.yaml").read_bytes(), self.yaml.read_bytes())
        self.assertEqual((self.root / "face_point.json").read_bytes(), self.original)
        self.assertEqual((path / "face_point.json").read_bytes(), self.original)
        np.testing.assert_array_equal(cv2.imread(str(path / "frame.png")), before["frame"])
        np.testing.assert_array_equal(self.detector.last_result["frame"], before["frame"])
        image = cv2.imread(str(path / "detection.png"))
        self.assertEqual(image[21, 41].tolist(), [0, 255, 255])
        self.assertEqual(self.detector.last_result["target_offset"], before["target_offset"])
        from reference_head_overlay import load_reference
        reference = load_reference(path / 'result.json', self.args['camera_info'])
        self.assertEqual(reference['centers']['L'], (41,21))

    def test_unique_paths_and_cached_commands(self):
        self.args["servo_cache"] = list(range(27))
        a = save_calibration_result(**self.args)
        b = save_calibration_result(**self.args)
        self.assertNotEqual(a, b)
        result = json.loads((b / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["servo_cache"]["values_by_channel"][10], 10)

    def test_nonzero_adapter_does_not_correct_target_or_errors(self):
        self.detector.baseline['nose'] = [30, 35]
        self.detector.last_result['nose'] = [30, 35]
        adapter = copy.deepcopy(self.detector.head_adapter_baseline)
        self.assertEqual(self.detector._eyeball_target_mode(), 'raw')
        self.assertEqual(self.detector._eyeball_head_adapter_offset('left'), (0,0))
        self.assertEqual(self.detector._eyeball_head_adapter_offset('right'), (0,0))
        raw = self.detector._calc_offset(self.detector.last_result)
        target = self.detector._calc_target_offset(self.detector.last_result)
        self.assertEqual(raw, target)
        self.assertEqual(target['left'], (1,1,float(np.sqrt(2))))
        self.assertFalse(self.detector._is_qualified(target))
        self.assertEqual(self.detector.head_adapter_baseline, adapter)
        self.detector.last_result['left_iris'] = (40,20)
        target = self.detector._calc_target_offset(self.detector.last_result)
        self.assertTrue(self.detector._is_qualified(target))

    def test_snapshot_stays_same_frame_if_live_result_advances(self):
        encode = cv2.imencode
        def advance(*args):
            self.detector.last_result['left_iris'] = (60, 40)
            self.detector.last_result['frame'][:] = 200
            return encode(*args)
        with patch.object(cv2, 'imencode', side_effect=advance):
            path = save_calibration_result(**self.args)
        result = json.loads((path / 'result.json').read_text(encoding='utf-8'))
        self.assertEqual(result['eyes']['L']['integer_center'], [41,21])
        self.assertFalse(cv2.imread(str(path / 'frame.png')).any())
        image = cv2.imread(str(path / 'detection.png'))
        self.assertEqual(image[21,41].tolist(), [0,255,255])

    def test_missing_eye_refuses(self):
        self.detector.last_result["left_iris"] = None
        with self.assertRaises(ValueError):
            save_calibration_result(**self.args)
        self.assertFalse((self.root / "calibration_results").exists())

    def test_target_changed_refuses(self):
        self.detector.baseline["iris_left"][0] += 1
        with self.assertRaises(ValueError):
            save_calibration_result(**self.args)

    def test_disabled_or_stale_refuses(self):
        self.detector.enable_mp = False
        with self.assertRaises(ValueError):
            save_calibration_result(**self.args)
        self.detector.enable_mp = True
        self.args["detection_metadata"] = {"status": "stale"}
        with self.assertRaises(ValueError):
            save_calibration_result(**self.args)

    def test_invalid_id_and_missing_yaml(self):
        self.args["head_id"] = "../Head_01"
        with self.assertRaises(ValueError):
            save_calibration_result(**self.args)
        self.args["head_id"] = "Head_01"
        self.args["prior_yaml"] = self.root / "missing.yaml"
        with self.assertRaises(FileNotFoundError):
            save_calibration_result(**self.args)

    def test_failed_image_has_no_completion_record(self):
        with patch.object(cv2, "imencode", return_value=(False, None)):
            with self.assertRaises(RuntimeError):
                save_calibration_result(**self.args)
        self.assertEqual(list((self.root / "calibration_results").glob("*/result.json")), [])

    def test_qt_save_path_has_no_hardware_actions(self):
        tree = ast.parse(Path("qt_calibration_panel.py").read_text(encoding="utf-8-sig"))
        methods = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        source = ast.unparse(methods["_save_calibration_result"])
        for forbidden in ("_create_tuner", "capture(", ".detect(", "open_serial", "send_servo", ".set("):
            self.assertNotIn(forbidden, source)
        # Exercise the real handler with fakes: camera properties are reads only.
        ns = {"time": SimpleNamespace(monotonic=lambda: 10), "copy": copy,
              "cv2": cv2, "os": __import__("os"), "__file__": str(self.root / "panel.py")}
        calls = []
        ns["save_calibration_result"] = lambda **kwargs: calls.append(kwargs) or self.root
        exec(compile(ast.Module(body=[methods["_save_calibration_result"]], type_ignores=[]), "<handler-only>", "exec"), ns)
        import threading
        self.detector.landmark_snapshots = SimpleNamespace(
            _lock=threading.Lock(), MAX_AGE_SECONDS=2,
            _sources={"front": (9.9, {"status": "detected"})})
        values = {cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*"MJPG"),
                  cv2.CAP_PROP_FRAME_WIDTH: 80, cv2.CAP_PROP_FRAME_HEIGHT: 60, cv2.CAP_PROP_FPS: 30}
        self.detector.cap = SimpleNamespace(get=values.__getitem__)
        self.detector.rotate_180 = True
        thread = SimpleNamespace(_route_running=False, detector=self.detector, log=SimpleNamespace(emit=lambda text: None),
                                 front_camera_index=1, side_camera_index=0, _snapshot_template_name="template_2",
                                 _last_servo_temp_degrees=None)
        payload = {"head_id": "Head_01", "prior_yaml": str(self.yaml), "qt_log": "test"}
        ns["_save_calibration_result"](thread, payload)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["camera_info"]["fourcc"], "MJPG")
        thread._route_running = True
        ns["_save_calibration_result"](thread, payload)
        self.assertEqual(len(calls), 1)
        thread._route_running = False
        self.detector.landmark_snapshots._sources["front"] = (1, {"status": "detected"})
        ns["_save_calibration_result"](thread, payload)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
