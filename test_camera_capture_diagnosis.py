"""Offline checks for diagnosis measurements; no capture/models/GUI."""
from pathlib import Path
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import diagnose_camera_capture as diag


class DiagnosticTests(unittest.TestCase):
    def test_production_requests_match_current_source(self):
        self.assertEqual(diag.production_requests(cv2), [
            ('FRAME_WIDTH',1920),('FRAME_HEIGHT',1080),('FPS',30),
            ('FOURCC',cv2.VideoWriter_fourcc(*'MJPG')),('BUFFERSIZE',1)])

    def test_black_and_nonblack(self):
        black=np.zeros((108,192,3),np.uint8)
        result=diag.metrics(black,cv2,np)
        self.assertTrue(result['all_zero'])
        self.assertTrue(result['near_black_candidate'])
        self.assertFalse(result['horizontal_seam_candidate'])
        result=diag.metrics(np.full_like(black,100),cv2,np)
        self.assertFalse(result['all_zero'])
        self.assertFalse(result['near_black_candidate'])

    def test_synthetic_horizontal_shift_screen(self):
        gray=np.tile(np.arange(192,dtype=np.uint8),(108,1))
        gray[54:]=np.roll(gray[54:],16,axis=1)
        result=diag.metrics(cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR),cv2,np)
        # Strong cyclic shift is a screening observation, not proof of corruption.
        self.assertGreater(result['cyclic_shift_improvement'],.5)

    def test_offscreen_renderer_intercepts_gui_and_copies_output(self):
        image=np.zeros((20,30,3),np.uint8)
        obj=SimpleNamespace(_stop_event=threading.Event())
        def display():
            cv2.imshow('test',image)
            cv2.waitKey(1)
        obj._display_loop=display
        with patch.object(cv2,'imshow',side_effect=AssertionError('real GUI forbidden')):
            result=diag.render_offscreen(obj,cv2)
        self.assertTrue(obj._stop_event.is_set())
        image[:]=255
        self.assertFalse(result.any())

    def test_cli_requires_confirmation_before_hardware(self):
        with patch('sys.argv',['diagnose_camera_capture.py','--run-camera']), \
             patch.object(diag.bench,'preflight',return_value={}):
            with patch.object(cv2,'VideoCapture',side_effect=AssertionError('hardware forbidden')):
                with self.assertRaises(SystemExit) as error:
                    diag.main()
                self.assertEqual(error.exception.code,2)


if __name__=='__main__':
    unittest.main()
