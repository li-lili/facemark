"""Offline simulations only. Never imports Qt panel, detector, Motor or serial."""
import ast
import contextlib
import io
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import yaml
from eyeball_adaptive import (AdaptiveConfig, EyeballStepPlanner, ERROR_KEYS,
                              sample_quality, valid_signal, hud_lines)
from eyeball_closed_loop import run_eyeball_loop


def extract_methods(filename, names, namespace=None):
    tree = ast.parse(Path(filename).read_text(encoding="utf-8-sig"))
    methods = [n for c in tree.body if isinstance(c, ast.ClassDef)
               for n in c.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = {} if namespace is None else namespace
    module = ast.Module(body=methods, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), filename, "exec"), ns)
    return ns


RULES = yaml.safe_load(Path("servo_motion_rules.yaml").read_text(encoding="utf-8"))
RULES = RULES.get("SERVO_EFFECTS", RULES)


def motion(ch, effect, step=1.0):
    rule = RULES[f"A{ch}"]
    if rule["angle_plus"] == effect:
        return step
    if rule["angle_minus"] == effect:
        return -step
    raise AssertionError((ch, effect))


def signal(vector, radial=None):
    s = dict(zip(ERROR_KEYS, vector))
    s["L_dist"], s["R_dist"] = radial or (math.hypot(*vector[:2]), math.hypot(*vector[2:]))
    return s


class Simulation:
    eye_servo_channels = (8, 9, 10, 11, 12, 13)
    eyeball_tolerance = 1.0
    _motion_delta = staticmethod(motion)

    def __init__(self, initial=(20., 0., 0., 0.), gain=1., columns=None):
        self.initial = list(initial)
        self.columns = columns or {12: (-gain, 0, 0, 0), 10: (0, -gain, 0, 0),
                                   13: (0, 0, -gain, 0), 11: (0, 0, 0, gain)}
        self.angle_ranges = [(80., 120.)]*6
        self.controller = SimpleNamespace(list_temp_deg=[100.]*27)
        self.scorer = SimpleNamespace(baseline={}, user_pressed_stop=False,
                                      enable_mp=True, enable_ellseg=True,
                                      start_display=Mock(), stop_display=Mock(),
                                      update_hud=Mock())
        self.commands, self.measurements = [], []
        self.fail_send = False
        self.stop_after_collect = False
        self.count = 60
        self.std, self.drift = 0., 0.
        self.invalid = False

    def collect_guide_samples(self, n_frames, trim):
        assert (n_frames, trim) == (60, 10)
        e = list(self.initial)
        for ch, column in self.columns.items():
            for i in range(4):
                e[i] += column[i]*(self.controller.list_temp_deg[ch]-100)
        s = None if self.invalid else signal(e)
        self.measurements.append(s)
        self._last_guide_sample_info = dict(count=self.count, std=self.std, drift=self.drift)
        if self.stop_after_collect:
            self.scorer.user_pressed_stop = True
        return s

    def send_servos(self, channels, angles, ranges):
        assert not self.scorer.user_pressed_stop
        if self.fail_send:
            return False
        deltas = []
        for ch, angle, (lo, hi) in zip(channels, angles, ranges):
            assert lo <= angle <= hi
            deltas.append(angle-self.controller.list_temp_deg[ch])
            self.controller.list_temp_deg[ch] = angle
        self.commands.append((channels, angles, deltas))
        return True

    def run(self, mode="adaptive", limit=50):
        with contextlib.redirect_stdout(io.StringIO()):
            return run_eyeball_loop(self, limit, 0., True, mode)


class PlannerTests(unittest.TestCase):
    def planner(self, mode="adaptive", limits=None):
        return EyeballStepPlanner(limits or {ch: (80, 120) for ch in (10,11,12,13)}, motion, mode)

    def test_first_probe_and_adaptive_convergence_faster_single_axis(self):
        fixed, adaptive = Simulation(), Simulation()
        self.assertTrue(fixed.run("fixed")[0])
        self.assertTrue(adaptive.run()[0])
        deltas = [abs(d) for _, _, ds in adaptive.commands for d in ds]
        self.assertEqual(deltas[0], 1)
        self.assertGreater(max(deltas), 1)
        self.assertLessEqual(max(deltas), 3)
        self.assertLess(len(adaptive.measurements), len(fixed.measurements))
        self.assertTrue(all(len(ch) == 1 for ch, _, _ in adaptive.commands))

    def test_half_degree_reaches_real_command_argument(self):
        sim = Simulation((2.8, 0, 0, 0), gain=1.2)
        self.assertTrue(sim.run()[0])
        self.assertIn(.5, [abs(d) for _, _, ds in sim.commands for d in ds])

    def test_measured_yaml_direction_for_all_axes_and_signs(self):
        angles = dict.fromkeys((10,11,12,13), 100.)
        expected_positive = {0: (12, 1), 1: (10, 1), 2: (13, 1), 3: (11, -1)}
        for axis, (ch, sign) in expected_positive.items():
            for direction in (-1, 1):
                v = [0.]*4; v[axis] = direction*5
                move = self.planner().plan(signal(v), angles)
                self.assertEqual((move[0][0], move[0][2]), (ch, sign*direction))

    def test_limit_clamp_uses_actual_probe_delta(self):
        p = self.planner(limits={ch: (99.5, 100.5) for ch in (10,11,12,13)})
        before = signal((5,0,0,0)); angles = dict.fromkeys((10,11,12,13), 100.)
        moves = p.plan(before, angles)
        self.assertEqual(moves[0][2], .5)
        p.committed(before, moves); p.observe(signal((4,0,0,0)))
        self.assertEqual(p.models[12][0], -2.)

    def test_limits_stop_without_repeated_noop_commands(self):
        sim = Simulation(); sim.angle_ranges = [(100,100)]*6
        self.assertFalse(sim.run()[0]); self.assertEqual(sim.commands, [])
        self.assertEqual(len(sim.measurements), 3)

    def test_radial_deadband_stall_is_avoided_both_modes(self):
        for mode in ("fixed", "adaptive"):
            sim = Simulation((.8,.8,0,0))
            self.assertTrue(sim.run(mode)[0]); self.assertTrue(sim.commands)

    def test_previously_qualified_axis_can_be_reactivated(self):
        p = self.planner("fixed"); a = dict.fromkeys((10,11,12,13), 100.)
        p.plan(signal((.2,4,0,0)), a)
        moves = p.plan(signal((3,0,0,0)), a)
        self.assertEqual(moves[0][0], 12)

    def test_reversal_worsening_wrong_direction_and_coupling_degrade(self):
        for after in ((-2,0,0,0), (7,0,0,0), (5,0,0,0), (4,2,0,0), (4,0,2,0)):
            p = self.planner(); before = signal((5,0,0,0))
            moves = p.plan(before, dict.fromkeys((10,11,12,13),100.))
            p.committed(before,moves); p.observe(signal(after))
            self.assertTrue(p.fallback, after)
            self.assertEqual(p.models, {})
            next_moves=p.plan(signal(after), dict.fromkeys((10,11,12,13),100.))
            self.assertTrue(all(abs(m[2]) <= 1 for m in next_moves))

    def test_response_gain_change_degrades(self):
        p=self.planner(); a=dict.fromkeys((10,11,12,13),100.)
        before=signal((10,0,0,0)); moves=p.plan(before,a)
        p.committed(before,moves); p.observe(signal((9,0,0,0)))
        p.committed(signal((9,0,0,0)),[(12,101,1,0)])
        p.observe(signal((5,0,0,0)))
        self.assertTrue(p.fallback)

    def test_small_unsent_eye_noise_does_not_invalidate_probe(self):
        p=self.planner();before=signal((5,0,0,0),radial=(5.,.4))
        moves=p.plan(before,dict.fromkeys((10,11,12,13),100.))
        p.committed(before,moves);p.observe(signal((4,0,.05,0),radial=(4.,.45)))
        self.assertFalse(p.fallback)
        self.assertIn(12,p.models)

    def test_coupled_model_rejects_damage_to_other_eye(self):
        p=self.planner(); p.models[12]=[-1,0,.4,0]
        moves=p.plan(signal((20,0,.9,0)),dict.fromkeys((10,11,12,13),100.))
        self.assertEqual(len(moves),1)
        self.assertEqual(moves[0][2],.5)  # 3 deg would displace the passing eye.

    def test_repeated_no_response_and_oscillation_stop(self):
        for gain in (0., 3.):
            sim=Simulation((1.2,0,0,0),gain=gain)
            self.assertFalse(sim.run()[0])
            self.assertLessEqual(len(sim.commands),4)

    def test_zero_mean_but_radial_noise_is_not_a_pass(self):
        p=self.planner(); s=signal((0,0,0,0),(1.3,1.3))
        for _ in range(3):self.assertEqual(p.plan(s,dict.fromkeys((10,11,12,13),100.)),[])
        self.assertEqual(p.stalls,3)

    def test_fractional_protocol_without_creating_motor(self):
        ns=extract_methods("Motor.py",{"data_restart","data2cmd_32"})
        obj=SimpleNamespace(seq=0,lock=1)
        obj.data_restart=lambda *args:ns["data_restart"](obj,*args)
        p1=ns["data2cmd_32"](obj,[130.],[12],200,27)
        p2=ns["data2cmd_32"](obj,[130.5],[12],200,27)
        def pulse(packet):
            pos=9+12*3
            return (packet[pos+1]<<8)|packet[pos+2]
        self.assertGreater(pulse(p2),pulse(p1))

    def test_production_sender_preserves_fractional_command_and_cache(self):
        ns=extract_methods('eye_auto_tuner.py',{'send_servos'},dict(List=list,Tuple=tuple))
        low_level=Mock(return_value=True)
        obj=SimpleNamespace(controller=SimpleNamespace(
            set_servo_angle_time_32=low_level,servo_num=27,list_temp_deg=[100.]*27),
            _notify_servo_values=Mock(),port='OFFLINE-MOCK')
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(ns['send_servos'](obj,[12],[100.5],[(80,120)]))
        self.assertEqual(low_level.call_args.args[0],[100.5])
        self.assertEqual(obj.controller.list_temp_deg[12],100.5)

    def test_panel_passes_selected_mode_only_to_eye_pixel_route(self):
        ns=extract_methods('qt_calibration_panel.py',{'_run_eyeball'},dict(EyeAutoTuner=object))
        obj=SimpleNamespace(log=SimpleNamespace(emit=Mock()),
            _emit_item_running=Mock(),_emit_item_result=Mock(),_eyeball_control_mode='adaptive')
        tuner=SimpleNamespace(auto_adjust=Mock(return_value=(True,3,{})))
        ns['_run_eyeball'](obj,tuner,10)
        self.assertEqual(tuner.auto_adjust.call_args.kwargs['mode'],'adaptive')

    def test_invalid_parameters_and_measurements(self):
        for args in (dict(max_step_deg=10),dict(probe_deg=2),dict(damping=2),dict(max_std_px=math.nan)):
            with self.assertRaises(ValueError):AdaptiveConfig(**args)
        for s in ({},None,signal((math.nan,0,0,0))):self.assertFalse(valid_signal(s))


class LoopTests(unittest.TestCase):
    def test_default_mode_is_fixed_and_parallel(self):
        sim=Simulation((5,5,5,5))
        with contextlib.redirect_stdout(io.StringIO()):
            result=run_eyeball_loop(sim,20,0,True)
        self.assertTrue(result[0]); self.assertEqual(sim.scorer.eyeball_control_mode,"fixed")
        self.assertEqual(len(sim.commands[0][0]),4)
        self.assertTrue(all(abs(d)==1 for _,_,ds in sim.commands for d in ds))

    def test_stopped_before_and_during_sampling_never_sends(self):
        for before in (False,True):
            sim=Simulation(); sim.scorer.user_pressed_stop=before;sim.stop_after_collect=True
            self.assertFalse(sim.run()[0]);self.assertEqual(sim.commands,[])

    def test_send_failure_stops_and_does_not_update_cache(self):
        sim=Simulation();sim.fail_send=True
        self.assertFalse(sim.run()[0]);self.assertEqual(len(sim.measurements),1)
        self.assertEqual(sim.controller.list_temp_deg,[100.]*27)

    def test_unstable_incomplete_or_failed_batches_never_move_or_pass(self):
        for kwargs in (dict(std=2),dict(drift=2),dict(count=59),dict(invalid=True)):
            sim=Simulation((0,0,0,0))
            for key,val in kwargs.items():setattr(sim,key,val)
            self.assertFalse(sim.run()[0]);self.assertEqual(sim.commands,[])
            self.assertEqual(len(sim.measurements),3)

    def test_recovery_after_failed_batch_is_sticky_fixed_step(self):
        sim=Simulation(); original=sim.collect_guide_samples;counter=[0]
        def collect(**kwargs):
            counter[0]+=1;sim.invalid=counter[0]==2
            return original(**kwargs)
        sim.collect_guide_samples=collect
        self.assertTrue(sim.run()[0])
        self.assertEqual(sim.scorer.eyeball_control_mode,"fallback")
        self.assertTrue(all(abs(d)<=1 for _,_,ds in sim.commands for d in ds))

    def test_baseline_change_does_not_look_like_motor_response(self):
        sim=Simulation();original=sim.collect_guide_samples
        def collect(**kwargs):
            data=original(**kwargs)
            sim.scorer.baseline={'iris_left':[100,100]}
            return data
        sim.collect_guide_samples=collect
        self.assertFalse(sim.run()[0]);self.assertEqual(sim.commands,[])

    def test_unstable_batch_after_probe_discards_response_history(self):
        sim=Simulation();original=sim.collect_guide_samples;counter=[0]
        def collect(**kwargs):
            counter[0]+=1
            sim.drift=2 if counter[0]==2 else 0
            return original(**kwargs)
        sim.collect_guide_samples=collect
        self.assertTrue(sim.run()[0])
        self.assertEqual(sim.scorer.eyeball_control_mode,'fallback')
        self.assertTrue(all(abs(d)<=1 for _,_,ds in sim.commands for d in ds))

    def test_final_iteration_never_moves_without_verification(self):
        sim=Simulation();self.assertFalse(sim.run(limit=1)[0]);self.assertEqual(sim.commands,[])

    def test_keeps_camera_and_restores_visual_flags(self):
        sim=Simulation();sim.run()
        sim.scorer.stop_display.assert_not_called()
        self.assertTrue(sim.scorer.enable_mp);self.assertTrue(sim.scorer.enable_ellseg)


class CollectionTests(unittest.TestCase):
    def collect(self, rows, stop_at=None, read_failure=False):
        cursor=[-1]
        scorer=SimpleNamespace(user_pressed_stop=False,update_hud=Mock(),detect=Mock())
        def capture():
            cursor[0]+=1
            if stop_at is not None and cursor[0]>=stop_at:scorer.user_pressed_stop=True
            return (not read_failure,None)
        scorer.capture=capture
        scorer.get_guide_signal=lambda:rows[min(cursor[0],len(rows)-1)]
        tuner=SimpleNamespace(scorer=scorer)
        ns=extract_methods("eye_auto_tuner.py",{"collect_guide_samples"},
                           dict(valid_signal=valid_signal,sample_quality=sample_quality,time=SimpleNamespace(sleep=Mock())))
        with contextlib.redirect_stdout(io.StringIO()):result=ns["collect_guide_samples"](tuner)
        return result,tuner._last_guide_sample_info,cursor[0]+1

    def test_exact_original_sixty_frame_reducer_including_outliers(self):
        rows=[signal((float((i*13)%19-9),float(i%7-3),float(i%9-4),float(i%5-2))) for i in range(60)]
        rows[0]=signal((200,200,100,100))
        expected=sorted(rows,key=lambda s:s['L_dist']+s['R_dist'])[10:-10]
        result,info,count=self.collect(rows)
        for key in ERROR_KEYS+('L_dist','R_dist'):
            self.assertEqual(result[key],sum(s[key] for s in expected)/40)
        self.assertEqual(info['count'],60);self.assertEqual(count,60)
        self.assertNotAlmostEqual(result['L_dist'],math.hypot(result['L_dX'],result['L_dY']))

    def test_partial_batch_on_stop_and_missing_one_eye_rejected(self):
        self.assertIsNone(self.collect([signal((3,0,0,0))],stop_at=30)[0])
        result,info,count=self.collect([{'L_dX':0,'L_dY':0,'L_dist':0}])
        self.assertIsNone(result);self.assertEqual(info['count'],0)
        self.assertEqual(count,11)

    def test_read_failure_is_bounded(self):
        result,info,count=self.collect([None],read_failure=True)
        self.assertIsNone(result);self.assertEqual(count,360)

    def test_quality_drift_preserves_temporal_order(self):
        rows=[(0,signal((v,0,0,0))) for v in [0]*30+[2]*30]
        info=sample_quality(rows)
        self.assertEqual(info,dict(count=60,std=1.,drift=2.))

    def test_hud_fits_reserved_rows(self):
        import cv2  # Image/text operations only, no camera or detector creation.
        for line in hud_lines('adaptive'):
            (width,height),baseline=cv2.getTextSize(line,0,.29,1)
            self.assertLess(width,430);self.assertLess(height+baseline,16)


if __name__ == '__main__':
    unittest.main()
