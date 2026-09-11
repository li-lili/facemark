"""Offline nose regression checks; no cameras or servos are opened."""
import ast
from pathlib import Path
import unittest

import numpy as np

from lip_front_detector import _contrast_profile_boundary, _nose_from_boundary


class NoseContourTests(unittest.TestCase):
    def test_lip_latch_rejects_alternating_edges(self):
        from lip_front_detector import StableFeaturePoint
        from eye_constants import LIP_PREVIEW_HOLD_PX, LIP_PREVIEW_CONFIRM_FRAMES, LIP_PREVIEW_CLUSTER_PX
        latch = StableFeaturePoint(LIP_PREVIEW_HOLD_PX, LIP_PREVIEW_CONFIRM_FRAMES, LIP_PREVIEW_CLUSTER_PX)
        for _ in range(8):
            latch.update((100,100))
        for point in [(106,101),(99,107)] * 10:
            self.assertEqual(latch.update(point), (100,100))
        for point in [(120,100),(125,100)] * 10:
            self.assertEqual(latch.update(point), (100,100))
        for _ in range(8):
            result = latch.update((120,100))
        self.assertEqual(result, (120,100))

    def test_flat_lower_lip_darker_than_background(self):
        from lip_front_detector import find_side_profile_points
        frame = np.full((120,160,3), 140, np.uint8)
        for y in range(120):
            x = round(55+0.015*max(0, abs(y-60)-12)**2)
            frame[y,x:] = (85,100,125)
        result = find_side_profile_points(frame, face_direction="left",
                    separate_lip_rois=True, lower_lip_roi=[20,10,130,110])
        point = result["points"]["lower_lip"]
        self.assertIsNotNone(point)
        self.assertLessEqual(abs(point[0]-55),2)
        self.assertLessEqual(abs(point[1]-60),3)

    def test_independent_lip_rois_and_missing_lower(self):
        from lip_front_detector import find_side_profile_points
        frame = np.full((220,220,3), 50, np.uint8)
        for y in range(220):
            x = round(90-25*np.exp(-((y-65)/18)**2)-20*np.exp(-((y-145)/20)**2))
            frame[y,x:] = 190
        args = dict(face_direction="left", separate_lip_rois=True,
                    upper_lip_roi=[20,25,160,105], lower_lip_roi=[20,110,160,190])
        result = find_side_profile_points(frame, **args)
        self.assertLessEqual(abs(result["points"]["upper_lip"][1]-65),2)
        self.assertLessEqual(abs(result["points"]["lower_lip"][1]-145),2)
        args["lower_lip_roi"] = None
        result = find_side_profile_points(frame, **args)
        self.assertIsNotNone(result["points"]["upper_lip"])
        self.assertIsNone(result["points"]["lower_lip"])

    def test_display_latch_holds_jitter_but_follows_sustained_motion(self):
        from lip_front_detector import StableFeaturePoint
        latch = StableFeaturePoint()
        for _ in range(5):
            latch.update((100,100))
        for point in [(102,99),(97,102),(103,103)] * 10:
            self.assertEqual(latch.update(point), (100,100))
        for _ in range(5):
            moved = latch.update((115,110))
        self.assertEqual(moved, (115,110))
        self.assertIsNone(latch.update(None))

    def test_chin_is_lower_turn_not_foremost_point(self):
        from lip_front_detector import _manual_profile_tips
        frame = np.full((160,240,3), 40, np.uint8)
        for y in range(160):
            x = round(50 + 0.01*(y-40)**2)
            frame[y,x:] = 190
        tips = _manual_profile_tips(frame,[10,10,230,145],"left",12,chin_turn=True)
        self.assertEqual(len(tips),1)
        self.assertLessEqual(abs(tips[0][1]-90),5)
        self.assertGreater(tips[0][1],65)

    def test_broad_chin_under_pixel_noise(self):
        from lip_front_detector import _manual_profile_tips
        rng = np.random.default_rng(17)
        points = []
        for _ in range(15):
            frame = np.full((150, 160, 3), 50, np.uint8)
            for y in range(150):
                x = round(60 + 0.01 * (y-75)**2)
                frame[y, x:] = 190
            noisy = np.clip(frame.astype(float) + rng.normal(0, 1.5, frame.shape), 0, 255).astype(np.uint8)
            tips = _manual_profile_tips(noisy, [10,10,150,140], "left", 12)
            self.assertEqual(len(tips), 1)
            points.append(tips[0])
        self.assertLessEqual(np.ptp(np.array(points)[:, 0]), 2)
        self.assertLessEqual(np.ptp(np.array(points)[:, 1]), 4)
        self.assertLessEqual(abs(np.median(np.array(points)[:, 1])-75), 3)

    def test_shared_lips_and_chin_front(self):
        from lip_front_detector import _manual_profile_tips, find_side_profile_points
        frame = np.full((220, 220, 3), 50, np.uint8)
        for y in range(220):
            x = round(90 - 25*np.exp(-((y-65)/18)**2)
                      - 20*np.exp(-((y-145)/20)**2))
            frame[y, x:] = 190
        lips = _manual_profile_tips(frame, [20,20,160,190], "left", 12, True)
        self.assertEqual(len(lips), 2)
        for point, expected in zip(lips, [(65,65),(70,145)]):
            self.assertLessEqual(np.linalg.norm(np.array(point)-expected), 2)
        self.assertEqual(_manual_profile_tips(frame, [20,110,160,190], "left", 12), [(70,145)])
        self.assertEqual(_manual_profile_tips(frame, [20,110,160,190], "left", 12, True), [])
        mirrored = _manual_profile_tips(frame[:, ::-1].copy(), [59,20,199,190], "right", 12, True)
        self.assertEqual(len(mirrored), 2)
        self.assertLessEqual(abs(mirrored[0][0] - 154), 2)
        result = find_side_profile_points(frame, require_feature_rois=True)
        self.assertIsNone(result["points"]["upper_lip"])
        self.assertIsNone(result["points"]["lower_lip"])
        self.assertIsNone(result["points"]["chin"])

    def test_manual_roi_with_support_in_front_of_nose(self):
        from lip_front_detector import find_side_profile_points
        frame = np.full((300, 400, 3), 90, np.uint8)
        frame[:, 35:55] = 15  # Strong vertical support edge before the nose.
        for y in range(50, 200):
            x = round(115 + 0.025 * (y - 125) ** 2)
            frame[y, x:] = 180
        result = find_side_profile_points(
            frame, face_direction="left", nose_roi=[20, 70, 230, 185],
            require_nose_roi=True)
        tip = result["points"]["nose"]
        self.assertIsNotNone(tip)
        self.assertLessEqual(abs(tip[0] - 115), 2)
        self.assertLessEqual(abs(tip[1] - 125), 2)

    def detect(self, boundary, direction="left"):
        return _nose_from_boundary(boundary, 400, direction, (80, 155), (105, 180))

    def test_rounded_nose_and_mirror(self):
        for direction, sign in (("left", 1), ("right", -1)):
            boundary = {y: 115 + sign * round(0.04 * (y - 143) ** 2)
                        for y in range(115, 173)}
            point = self.detect(boundary, direction)
            self.assertIsNotNone(point)
            self.assertLessEqual(abs(point[0] - 115), 1)
            self.assertLessEqual(abs(point[1] - 143), 1)

    def test_reject_missing_straight_and_isolated_edges(self):
        for boundary in ({}, {y: 100 for y in range(106, 180)},
                         {140: 100, 141: 99, 142: 100},
                         {y: 90 + 0.3 * (y - 105) for y in range(106, 180)}):
            self.assertIsNone(self.detect(boundary))

    def test_stronger_interior_edge_does_not_replace_silhouette(self):
        frame = np.full((400, 400, 3), 40, np.uint8)
        for y in range(105, 181):
            x = round(100 + 0.03 * (y - 143) ** 2)
            frame[y, x:] = 130
            frame[y, 150:] = 250
        boundary = _contrast_profile_boundary(frame, "left", (80, 155), (105, 180), 12)
        point = self.detect(boundary)
        self.assertIsNotNone(point)
        self.assertLessEqual(abs(point[0] - 100), 2)
        self.assertLessEqual(abs(point[1] - 143), 2)

    def test_missing_nose_is_not_replaced_by_history(self):
        # Exercise the actual Qt method without importing Qt/model/hardware code.
        from collections import deque
        from types import SimpleNamespace
        tree = ast.parse(Path("qt_calibration_panel.py").read_text(encoding="utf-8-sig"))
        method = next(node for node in ast.walk(tree)
                      if isinstance(node, ast.FunctionDef)
                      and node.name == "_stabilize_side_profile_detection")
        namespace = {"np": np, "SIDE_PROFILE_PREVIEW_FILTER_FRAMES": 5,
                     "SIDE_PROFILE_PREVIEW_MAX_JUMP_PX": 10}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "filter", "exec"), namespace)
        owner = SimpleNamespace(_side_profile_point_history={"nose": deque([(100, 140)], maxlen=5)},
                                _side_profile_reject_counts={"nose": 0})
        update = namespace[method.name]
        result = update(owner, {"points": {"nose": (140, 140)}})
        self.assertIsNone(result["points"]["nose"])
        result = update(owner, {"points": {"nose": None}})
        self.assertFalse(result["valid"])
        self.assertIsNone(result["points"]["nose"])
        self.assertFalse(owner._side_profile_point_history["nose"])


if __name__ == "__main__":
    unittest.main()
