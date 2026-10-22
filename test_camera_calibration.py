"""Offline regression checks; no camera, model, serial port or motor is opened."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from camera_tuning import (PROPERTIES, apply_checked, measurement, restore,
                           save_report, snapshot, summarize)
from camera_auto_calibrate import Cancelled, Worker


class FakeCamera:
    def __init__(self):
        self.values = {prop: 0.0 for prop in PROPERTIES.values()}
        self.values[PROPERTIES['fps']] = 30.0
        self.values[PROPERTIES['exposure']] = -6.0
        self.reject = set()
        self.ignore = set()

    def get(self, prop):
        return self.values[prop]

    def set(self, prop, value):
        if prop in self.reject:
            return False
        if prop not in self.ignore:
            self.values[prop] = value
        return True


def samples(scale=1.0):
    return [dict(centers=[scale*(i % 2), 0, 0, scale*(i % 2)],
                 sharpness=100, brightness=[100, 100], clipped=0,
                 frame_digest=str(i)) for i in range(40)]


class MetricsTests(unittest.TestCase):
    def test_lower_jitter_ranks_better(self):
        low = summarize(samples(0.1), 40, 8, 100)
        high = summarize(samples(2), 40, 8, 100)
        self.assertTrue(low['eligible'])
        self.assertLess(low['jitter_px'], high['jitter_px'])

    def test_missing_frames_count_against_success(self):
        result = summarize(samples() + [None]*20, 60, 8)
        self.assertFalse(result['eligible'])
        self.assertAlmostEqual(result['valid_rate'], 2/3)

    def test_blur_and_dark_do_not_win_with_zero_jitter(self):
        rows = samples(0)
        for row in rows:
            row.update(sharpness=1, brightness=[5, 100])
        result = summarize(rows, 40, 8, 100)
        self.assertFalse(result['eligible'])
        self.assertEqual(len(result['reasons']), 2)

    def test_frozen_image_rejected(self):
        rows = samples(0)
        for row in rows:
            row['frame_digest'] = 'same'
        self.assertFalse(summarize(rows, 40, 8)['eligible'])

    def test_empty_or_slow_samples_rejected(self):
        self.assertFalse(summarize([], 10, 8)['eligible'])
        self.assertFalse(summarize(samples(), 40, 100)['eligible'])

    def test_measurement_preserves_raw_offsets_and_rejects_missing_eye(self):
        frame = np.full((100, 200, 3), 100, dtype=np.uint8)
        det = dict(left_iris=(40, 50), right_iris=(140, 50),
                   left_iris_ellipse=((40, 50), (20, 20), 0),
                   right_iris_ellipse=((140, 50), (20, 20), 0),
                   offset={'left': (2.5, -4.5, 5.1), 'right': (-1, 3, 3.2)})
        result = measurement(det, frame)
        self.assertEqual(result['raw_offsets']['left'], [2.5, -4.5, 5.1])
        self.assertEqual(result['centers'], [40, 50, 140, 50])
        det['right_iris'] = None
        self.assertIsNone(measurement(det, frame))


class PropertyTests(unittest.TestCase):
    def test_rejected_and_ignored_properties_reported(self):
        cap = FakeCamera()
        cap.reject.add(PROPERTIES['gain'])
        cap.ignore.add(PROPERTIES['exposure'])
        _, errors = apply_checked(cap, {'gain': 10, 'exposure': -8})
        self.assertEqual(len(errors), 2)

    def test_restore_manual_and_auto(self):
        cap = FakeCamera()
        original = snapshot(cap)
        apply_checked(cap, {'fps': 60, 'exposure': -8, 'gain': 10})
        self.assertEqual(restore(cap, original), [])
        self.assertEqual(snapshot(cap), original)
        original['auto_exposure'] = 3
        original['exposure'] = -5
        self.assertEqual(restore(cap, original), [])
        self.assertEqual(cap.get(PROPERTIES['auto_exposure']), 3)

    def test_atomic_write_keeps_previous_on_invalid_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            save_report(path, {'best': 1})
            with self.assertRaises(ValueError):
                save_report(path, {'best': float('nan')})
            self.assertEqual(json.loads(path.read_text()), {'best': 1})
            self.assertFalse(path.with_suffix('.json.tmp').exists())


class SearchTests(unittest.TestCase):
    def worker(self, directory):
        args = argparse.Namespace(camera=0, fps=[30], exposures=[-7], gains=None,
                                  focus=None, seconds=8, manual_exposure=0, output=directory)
        worker = Worker(args)
        worker.detector = argparse.Namespace(cap=FakeCamera())
        return worker

    def test_search_retests_and_applies_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = self.worker(directory)
            calls = []
            def collect(label, reference=None):
                calls.append(label)
                scale = 0.2 if worker.settings['exposure'] == -7 else 1
                return dict(label=label, settings=dict(worker.settings), samples=[],
                            metrics=summarize(samples(scale), 40, 8, reference))
            worker.collect = collect
            worker.search()
            self.assertEqual(len(calls), 4)  # baseline, candidate, both repeated
            self.assertEqual(worker.detector.cap.get(PROPERTIES['exposure']), -7)
            report = json.loads(next(Path(directory).glob('*.json')).read_text(encoding='utf-8'))
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(report['best']['settings']['exposure'], -7)

    def test_cancel_restores_and_saves_no_best(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = self.worker(directory)
            original = snapshot(worker.detector.cap)
            def collect(*args):
                worker.detector.cap.set(PROPERTIES['gain'], 20)
                raise Cancelled()
            worker.collect = collect
            worker.search()
            self.assertEqual(snapshot(worker.detector.cap), original)
            report = json.loads(next(Path(directory).glob('*.json')).read_text(encoding='utf-8'))
            self.assertEqual(report['status'], 'cancelled')
            self.assertIsNone(report['best'])


if __name__ == '__main__':
    unittest.main()
