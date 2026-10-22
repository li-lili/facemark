"""Synthetic-image reference overlay tests, without models, Qt or hardware."""
import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
from reference_head_overlay import COORDINATE_SYSTEM, load_reference, draw_reference


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'result.json'
        self.camera = dict(front_camera_index=1, width=160, height=100, fourcc='MJPG',
                           rotate_180=True, flip_horizontal=False, flip_vertical=False)
        self.data = dict(schema_version=1, head_id='Head_01', camera=self.camera.copy(),
                         coordinate_system=COORDINATE_SYSTEM,
                         eyes={'L': {'integer_center': [100, 40], 'effective_target': [140, 80]},
                               'R': {'integer_center': [40, 40], 'effective_target': [10, 10]}})

    def load(self):
        self.path.write_text(json.dumps(self.data), encoding='utf-8')
        return load_reference(self.path, self.camera)

    def test_real_points_not_targets_and_signed_distances(self):
        ref = self.load()
        self.assertEqual(ref['centers']['L'], (100, 40))
        det = {'left_iris': (103, 44), 'right_iris': (37, 36), 'qualified': False}
        before = copy.deepcopy(det)
        image = np.zeros((100, 160, 3), dtype=np.uint8)
        lines = draw_reference(image, det, (ref, True), self.camera)
        self.assertIn('dX=+3.0 dY=+4.0 d=5.00px', lines[1])
        self.assertIn('dX=-3.0 dY=-4.0 d=5.00px', lines[2])
        self.assertEqual(image[44, 103].tolist(), [0, 255, 255])
        self.assertGreater(int(image[40, 110, 1]), 0)
        self.assertEqual(det, before)
        self.assertEqual(ref['centers']['L'], (100, 40))

    def test_off_does_not_change_image_or_points(self):
        image = np.zeros((100, 160, 3), dtype=np.uint8)
        cv2.circle(image, (103, 44), 4, (0, 255, 255), -1)
        before = image.copy()
        lines = draw_reference(image, {'left_iris': (103, 44)}, (self.load(), False), self.camera)
        np.testing.assert_array_equal(before, image)
        self.assertIn('OFF', lines[0])

    def test_all_metadata_mismatches_and_missing_fields_block(self):
        for key, value in dict(front_camera_index=0, width=320, height=200, fourcc='YUY2',
                               rotate_180=False, flip_horizontal=True, flip_vertical=True).items():
            with self.subTest(key=key):
                self.data['camera'] = self.camera.copy()
                self.data['camera'][key] = value
                with self.assertRaisesRegex(ValueError, key):
                    self.load()
                self.data['camera'] = self.camera.copy()
                del self.data['camera'][key]
                with self.assertRaisesRegex(ValueError, key):
                    self.load()
        self.data['camera'] = self.camera.copy()
        self.data['coordinate_system'] = 'ROI-local'
        with self.assertRaisesRegex(ValueError, 'coordinate_system'):
            self.load()

    def test_live_camera_change_and_frame_size_block_drawing(self):
        ref = self.load()
        for camera, shape in ((dict(self.camera, front_camera_index=0), (100,160,3)),
                              (self.camera, (90,160,3))):
            image = np.zeros(shape, dtype=np.uint8)
            lines = draw_reference(image, {}, (ref, True), camera)
            self.assertEqual(lines[0], 'RefHead: BLOCKED')
            self.assertFalse(image.any())

    def test_invalid_missing_and_outside_points_block(self):
        for point in (None, [1], [True, 2], [1.5, 2], [float('nan'), 2], [-1, 2], [160, 2]):
            with self.subTest(point=point):
                self.data['eyes']['L']['integer_center'] = point
                with self.assertRaises(ValueError):
                    self.load()

    def test_missing_current_eye_still_draws_reference(self):
        image = np.zeros((100,160,3), dtype=np.uint8)
        lines = draw_reference(image, {}, (self.load(), True), self.camera)
        self.assertEqual(lines[1], 'Ref L: --')
        self.assertGreater(int(image[40,110,1]), 0)

    def test_display_and_snapshot_use_same_result_fields(self):
        tree = ast.parse(Path('ellseg_scorer.py').read_text(encoding='utf-8-sig'))
        display = next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_display_loop')
        text = ast.unparse(display)
        self.assertIn("det = self.last_result", text)
        self.assertIn("img = det['frame'].copy()", text)
        for side in ('left', 'right'):
            self.assertIn(f"cv2.circle(img, det['{side}_iris'], 4, (0, 255, 255), -1)", text)
        # No reference reads may enter any detection, target or servo algorithm.
        for filename in ('ellseg_scorer.py', 'eye_auto_tuner.py'):
            tree = ast.parse(Path(filename).read_text(encoding='utf-8-sig'))
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name != '_display_loop':
                    if node.name == '__init__':
                        continue
                    self.assertNotIn('_reference_head_display', ast.unparse(node), node.name)


if __name__ == '__main__':
    unittest.main()
