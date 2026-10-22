"""Offline export checks. Does not import detector models or open hardware."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from landmark_snapshot import LandmarkSnapshots, mediapipe_points, timestamp


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.logs = []
        self.store = LandmarkSnapshots(self.temp.name, self.logs.append)

    def publish(self):
        points = mediapipe_points(
            [SimpleNamespace(x=0.25, y=0.5, z=-0.03, visibility=None, presence=0.8)],
            [[51, 49]], 200, 100)
        data = {"status": "detected", "point_count": 1, "landmarks": points}
        self.store.publish("front", data)
        return data

    def save(self):
        self.assertTrue(self.store.request_save())
        self.store._writer.join(5)
        self.assertFalse(self.store._writer.is_alive())

    def test_raw_coordinates_and_snapshot_isolation(self):
        data = self.publish()
        data["landmarks"][0]["raw_normalized"]["x"] = 99
        self.save()
        saved = json.loads(next(Path(self.temp.name).glob("*.json")).read_text(encoding="utf-8"))
        point = saved["sources"]["front"]["landmarks"][0]
        self.assertEqual(point["id"], 0)
        self.assertEqual(point["raw_pixel"], {"x": 50, "y": 50})
        self.assertEqual(point["raw_normalized"]["x"], 0.25)
        self.assertEqual(point["smoothed_pixel"], {"x": 51, "y": 49})
        self.assertNotIn("visibility", point)
        self.assertEqual(point["presence"], 0.8)
        self.assertEqual(saved["sources"]["side"]["point_count"], 0)

    def test_missing_invalid_and_stale_never_create_files(self):
        self.assertFalse(self.store.request_save())
        self.publish()
        self.store.invalidate("front", "no_face")
        self.assertFalse(self.store.request_save())
        self.publish()
        with patch("landmark_snapshot.time.monotonic", return_value=10**12):
            self.assertFalse(self.store.request_save())
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])

    def test_identical_timestamps_do_not_overwrite(self):
        self.publish()
        with patch("landmark_snapshot.timestamp", return_value=timestamp()):
            self.save()
            self.save()
        self.assertEqual(len(list(Path(self.temp.name).glob("*.json"))), 2)

    def test_all_478_indices_are_preserved(self):
        face = [SimpleNamespace(x=i / 478, y=0.5, z=-i / 1000) for i in range(478)]
        points = mediapipe_points(face, [[i, i + 1] for i in range(478)], 1920, 1080)
        self.store.publish("front", {"status": "detected", "point_count": len(points),
                                     "landmarks": points})
        self.save()
        saved = json.loads(next(Path(self.temp.name).glob("*.json")).read_text(encoding="utf-8"))
        exported = saved["sources"]["front"]["landmarks"]
        self.assertEqual([point["id"] for point in exported], list(range(478)))
        self.assertEqual(exported[-1]["raw_normalized"]["z"], face[-1].z)

    def test_side_only_and_partial_detection(self):
        self.store.invalidate("front", "no_face")
        self.store.publish("side", {"status": "detected", "point_count": 1,
                                   "points": {"upper_lip": [30, 40], "lower_lip": None}})
        self.save()
        self.assertIn("特征点数量=1", self.logs[-1])

    def test_write_failure_and_nonfinite_values(self):
        self.publish()
        with patch("landmark_snapshot.Path.mkdir", side_effect=PermissionError("denied")):
            self.save()
        self.assertIn("保存失败", self.logs[-1])
        self.store.publish("front", {"status": "detected", "point_count": 1, "x": float("nan")})
        self.save()
        self.assertIn("保存失败", self.logs[-1])
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])

    def test_request_logger_isolated_from_manual_save(self):
        automatic_logs = []
        self.publish()
        self.assertTrue(self.store.request_save(logger=automatic_logs.append))
        self.store._writer.join(5)
        self.assertIn('[OK]', automatic_logs[-1])
        self.assertIn('front=1 (detected)', automatic_logs[-1])
        self.assertIn('side=0', automatic_logs[-1])
        self.assertEqual(self.logs, [])
        self.save()
        self.assertIn('[OK]', self.logs[-1])
        with patch('landmark_snapshot.Path.mkdir', side_effect=PermissionError('denied')):
            self.assertTrue(self.store.request_save(logger=automatic_logs.append))
            self.store._writer.join(5)
        self.assertIn('[ERROR]', automatic_logs[-1])
        self.assertIn('front=1 (detected)', automatic_logs[-1])
        self.assertIn('side=0', automatic_logs[-1])


if __name__ == "__main__":
    unittest.main()
