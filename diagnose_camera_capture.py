"""Isolated capture diagnosis. Default: static check only, no camera/models.

--run-camera --apps-closed-confirmed starts ONE fresh child process for one
backend/codec/stage. Production stage is DSHOW/MJPG/raw by default. No controls,
serial, calibration, configuration writes, or second camera. Native hangs are
bounded by the parent timeout. Outputs are confined to a fresh benchmark folder.
"""
import argparse
import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
import benchmark_front_camera as bench

ROOT = Path(__file__).resolve().parent


def production_requests(cv2):
    tree = ast.parse((ROOT/'ellseg_scorer.py').read_text(encoding='utf-8-sig'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'EllSegDetector')
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    calls = [n.value for n in init.body if isinstance(n, ast.Expr)
             and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == 'self.cap.set']
    names = [ast.unparse(c.args[0]).split('CAP_PROP_')[-1] for c in calls]
    if names != ['FRAME_WIDTH','FRAME_HEIGHT','FPS','FOURCC','BUFFERSIZE']:
        raise RuntimeError(f'Production setter order changed: {names}; review first.')
    env = {'cv2': cv2, 'width': 1920, 'height': 1080, '__builtins__': {}}
    return [(name, eval(compile(ast.Expression(c.args[1]), '<production-setting>', 'eval'), env))
            for name,c in zip(names,calls)]


def metrics(frame, cv2, np):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (192,108), interpolation=cv2.INTER_AREA).astype(np.float32)
    jumps = np.abs(small[1:]-small[:-1]).mean(axis=1)
    row = int(jumps.argmax())+1
    direct = float(jumps[row-1])
    # Screen for line discontinuities; scene edges can produce the same signature.
    errors = [(float(np.abs(small[row]-np.roll(small[row-1], s)).mean()),s)
              for s in (-16,-8,-4,4,8,16)]
    best,shift = min(errors)
    mean,std=cv2.meanStdDev(gray)
    dark_fraction=cv2.countNonZero(cv2.inRange(gray,0,5))/gray.size
    return {'all_zero': bool(not frame.any()), 'gray_mean': float(mean[0,0]),
            'gray_std': float(std[0,0]), 'dark_fraction_le_5': dark_fraction,
            'near_black_candidate': bool(dark_fraction>.995),
            'row_jump_max': direct, 'row_jump_y_fraction': row/108,
            'best_cyclic_shift_thumbnail_px': shift,
            'cyclic_shift_improvement': (direct-best)/max(direct,1e-6),
            'horizontal_seam_candidate': bool(direct>20 and best<direct*.4),
            'thumbnail_sha256': hashlib.sha256(small.astype(np.uint8).tobytes()).hexdigest()}


def protect_writes(folder):
    folder = folder.resolve()
    def check(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, Path)):
            return
        mode,flags = args[1],args[2]
        write = (isinstance(mode,str) and any(c in mode for c in 'wax+')) or (
            isinstance(flags,int) and flags & (1|2|64|512|1024))
        if write and not Path(args[0]).resolve().is_relative_to(folder):
            raise PermissionError(f'Diagnostic refuses write outside output folder: {args[0]}')
    sys.addaudithook(check)


def init_hud(detector, data, cap):
    import threading
    template = data['templates'][data['active_template']]
    for section in ('eyelid','eyebrow','mouth','lower_lip','upper_lip','mouth_corners',
                    'mouth_corner_curl','head_position','head_adapter','side_profile_slope'):
        detector.__dict__[section+'_baseline'] = template.get(
            'side_profile_slopes' if section=='side_profile_slope' else section) or None
    for flag in ('show_all_landmarks','show_baseline_overlay','show_eye_line_offset',
                 'show_iris_cover_overlay','show_side_profile_preview',
                 'show_side_profile_expected_slope','show_side_profile_coordinate_grid'):
        setattr(detector, flag, False)
    detector.show_face_overlays = True
    detector._disp_extra_texts = []
    detector._reference_head_display = (None,False)
    detector.side_nose_roi = template.get('side_nose',{}).get('roi')
    detector.side_chin_roi = template.get('side_chin',{}).get('roi')
    detector._capture_fps = 0
    detector._user_key = detector._last_key = None
    detector._stop_event = threading.Event()
    detector.cap = cap


def render_offscreen(detector, cv2):
    # Exercise the ACTUAL production drawing loop, intercept only GUI presentation.
    # This checks pixel drawing, not HighGUI/Qt multi-thread event dispatch.
    from unittest.mock import patch
    frames = []
    def show(name, image):
        frames.append(image.copy())
        detector._stop_event.set()
    detector._stop_event.clear()
    with patch.object(cv2,'imshow',side_effect=show), patch.object(cv2,'waitKey',return_value=255):
        detector._display_loop()
    return frames[0]


def worker(args, folder):
    import cv2
    import numpy as np
    import tempfile
    # Model dependencies may probe a writable temporary directory during import.
    # Keep those writes inside the diagnostic artifact, not production/system paths.
    temp_dir = folder/'temporary'
    temp_dir.mkdir(exist_ok=False)
    tempfile.tempdir = str(temp_dir.resolve())
    protect_writes(folder)
    before = bench.protected_hashes()
    data = bench.preflight()
    report = {'status':'running','started_at':datetime.now().astimezone().isoformat(),
              'backend_requested':args.backend,'codec_requested':args.codec,'stage':args.stage,
              'camera_index':1,'opencv':cv2.__version__,'python':sys.executable,
              'notes':['Throughput includes frame screening; not an isolated FPS benchmark.',
                       'Black/seam flags are screening hints, not a corruption diagnosis.',
                       'HUD stage uses actual drawing with GUI presentation intercepted.',
                       'MSMF numeric device order may differ from DSHOW; verify the recorded scene.',
                       'No exposure/focus/gain/white-balance writes.'],
              'rows':[],'control_samples':[],'images':[],'requests':[]}
    (folder/'opencv_build.txt').write_text(cv2.getBuildInformation(),encoding='utf-8')
    cap = detector = side_cap = None
    pictures = {}
    held = []
    def persist():
        (folder/'data.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,
                                                 allow_nan=False),encoding='utf-8')
    try:
        backend = cv2.CAP_DSHOW if args.backend=='DSHOW' else cv2.CAP_MSMF
        started = time.perf_counter()
        cap = cv2.VideoCapture(1, backend)
        report['open_seconds'] = time.perf_counter()-started
        if not cap.isOpened():
            raise RuntimeError('Could not open camera index 1; no other index attempted.')
        report['initial_settings'] = bench.camera_settings(cap,cv2)
        requests = production_requests(cv2)
        if args.codec=='YUY2':
            requests = [(k,cv2.VideoWriter_fourcc(*'YUY2') if k=='FOURCC' else v) for k,v in requests]
        for name,value in requests:
            accepted = bool(cap.set(getattr(cv2,'CAP_PROP_'+name), value))
            report['requests'].append({'property':name,'value':value,'accepted':accepted,
                                       'readback':bench.camera_settings(cap,cv2)})
        report['negotiated_settings'] = bench.camera_settings(cap,cv2)
        report['mode_matches'] = all(report['negotiated_settings'][k]==v for k,v in
                                    (('width',1920),('height',1080),('fourcc',args.codec)))
        if args.with_side:
            if args.backend!='DSHOW' or args.codec!='MJPG':
                raise ValueError('Paired Qt read-order probe requires production DSHOW/MJPG.')
            side_index=int(data['common']['camera_config']['side_camera_index'])
            if side_index==1:
                raise ValueError('Refusing to open the same numeric camera twice.')
            side_cap=cv2.VideoCapture(side_index,cv2.CAP_DSHOW)
            if not side_cap.isOpened():
                raise RuntimeError(f'Side camera {side_index} failed to open.')
            # EXACT side-camera initialization in Qt; no FPS/format/control writes.
            side_cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
            side_cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
            report['side_camera_index']=side_index
            report['side_settings']=bench.camera_settings(side_cap,cv2)
        if args.stage in ('vision','hud'):
            _,detector = bench.load_visual_pipeline(data)
            if args.stage=='hud':
                init_hud(detector,data,cap)
        report['setup_complete_at'] = datetime.now().astimezone().isoformat()
        persist()
        started = time.perf_counter()
        next_control = next_progress = 0
        while len(report['rows'])<args.max_frames and time.perf_counter()-started<args.seconds:
            t0 = time.perf_counter()
            if t0-started>=next_control:
                report['control_samples'].append({'t':t0-started,'settings':bench.camera_settings(cap,cv2)})
                next_control += 1
            read_start = time.perf_counter()
            ok, frame = cap.read()
            t1 = time.perf_counter()
            row = {'attempt':len(report['rows'])+1,'t':t1-started,
                   'read_ms':(t1-read_start)*1000,'ok':bool(ok and frame is not None and frame.size)}
            report['rows'].append(row)
            if row['ok']:
                row['shape'] = list(frame.shape)
                if row['attempt']%30==0:
                    for probe in held:
                        if hashlib.sha256(probe[0].tobytes()).hexdigest()!=probe[1]:
                            probe[2]=True
                if len(held)<3:
                    held.append([frame,hashlib.sha256(frame.tobytes()).hexdigest(),False])
                if frame.ndim!=3 or frame.shape[2]!=3:
                    raise RuntimeError(f'Unexpected decoded layout: {frame.shape}')
                row['raw'] = metrics(frame,cv2,np)
                index = row['attempt']
                save = index in (1,30,60,120) or (
                    (row['raw']['near_black_candidate'] or row['raw']['horizontal_seam_candidate'])
                    and len(pictures)<12)
                if save:
                    pictures[f'frame{index:04d}_raw'] = frame.copy()
                if args.stage!='raw':
                    oriented = cv2.rotate(frame,cv2.ROTATE_180)
                    row['rotation_ms'] = (time.perf_counter()-t1)*1000
                    row['rotation_roundtrip_exact'] = bool(np.array_equal(
                        cv2.rotate(oriented,cv2.ROTATE_180),frame))
                    if detector is not None:
                        input_hash = hashlib.sha256(oriented.tobytes()).hexdigest()
                        tv = time.perf_counter()
                        result = detector.detect(oriented)
                        row['detect_ms'] = (time.perf_counter()-tv)*1000
                        row['both_eyes'] = result.get('left_iris') is not None and result.get('right_iris') is not None
                        row['input_changed_by_detection'] = hashlib.sha256(oriented.tobytes()).hexdigest()!=input_hash
                        row['timing'] = result.get('_timing')
                        if args.stage=='hud':
                            tv=time.perf_counter()
                            rendered=render_offscreen(detector,cv2)
                            row['hud_ms']=(time.perf_counter()-tv)*1000
                            row['input_changed_by_hud']=hashlib.sha256(oriented.tobytes()).hexdigest()!=input_hash
                            if save:
                                pictures[f'frame{index:04d}_hud']=rendered
                    if save:
                        pictures[f'frame{index:04d}_rotated']=oriented.copy()
            else:
                time.sleep(.001)
            if side_cap is not None:
                side_start=time.perf_counter()
                side_ok,side_frame=side_cap.read()
                row['side_read_ms']=(time.perf_counter()-side_start)*1000
                row['side_ok']=bool(side_ok and side_frame is not None and side_frame.size)
                if row['side_ok']:
                    row['side_raw']=metrics(side_frame,cv2,np)
                    if row['attempt'] in (1,30,60):
                        pictures[f'frame{row["attempt"]:04d}_side_raw']=side_frame.copy()
            if time.perf_counter()-started>=next_progress:
                print(f'{args.backend}/{args.codec}/{args.stage}: attempts={len(report["rows"])} t={time.perf_counter()-started:.1f}s',flush=True)
                next_progress += 10
        report['elapsed_seconds']=time.perf_counter()-started
        for probe in held:
            if hashlib.sha256(probe[0].tobytes()).hexdigest()!=probe[1]:
                probe[2]=True
        if held and frame is not None and frame.ndim==3:
            pictures['last_raw']=frame.copy()
        report['held_buffers_changed_after_later_reads']=[p[2] for p in held]
        report['final_settings']=bench.camera_settings(cap,cv2)
        valid=[r for r in report['rows'] if r['ok']]
        report['summary']={'attempts':len(report['rows']),'successes':len(valid),
            'read_success_rate':len(valid)/max(len(report['rows']),1),
            'fps_with_diagnostics':len(valid)/report['elapsed_seconds'],
            'all_zero_frames':sum(r['raw']['all_zero'] for r in valid),
            'near_black_candidates':sum(r['raw']['near_black_candidate'] for r in valid),
            'horizontal_seam_candidates':sum(r['raw']['horizontal_seam_candidate'] for r in valid),
            'mean_read_ms':statistics.mean(r['read_ms'] for r in report['rows']) if report['rows'] else None}
        if side_cap is not None:
            report['side_final_settings']=bench.camera_settings(side_cap,cv2)
            report['summary']['mean_side_read_ms']=statistics.mean(r['side_read_ms'] for r in report['rows'])
            report['summary']['side_successes']=sum(r['side_ok'] for r in report['rows'])
        report['status']='complete'
    except BaseException as exc:
        report.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
    finally:
        if cap is not None:
            cap.release()
        if side_cap is not None:
            side_cap.release()
        if detector is not None:
            detector.close()
        for name,frame in pictures.items():
            ok,encoded=cv2.imencode('.png',frame)
            if ok:
                (folder/(name+'.png')).write_bytes(encoded.tobytes())
                report['images'].append(name+'.png')
        after=bench.protected_hashes()
        report['protected_files_changed']=[n for n,h in before.items() if after.get(n)!=h]
        report['forbidden_modules_loaded']=[n for n in sys.modules if n.split('.')[0] in
                                           ('serial','Motor','Communication','eye_auto_tuner','qt_calibration_panel')]
        report['finished_at']=datetime.now().astimezone().isoformat()
        persist()
    print(json.dumps({'folder':str(folder),'status':report['status'],'summary':report.get('summary'),
                      'error':report.get('error'),'protected_files_changed':report['protected_files_changed']}),flush=True)
    return 0 if report['status']=='complete' else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-camera',action='store_true')
    parser.add_argument('--apps-closed-confirmed',action='store_true')
    parser.add_argument('--backend',choices=('DSHOW','MSMF'),default='DSHOW')
    parser.add_argument('--codec',choices=('MJPG','YUY2'),default='MJPG')
    parser.add_argument('--stage',choices=('raw','rotate','vision','hud'),default='raw')
    parser.add_argument('--seconds',type=float,default=30)
    parser.add_argument('--max-frames',type=int,default=900)
    parser.add_argument('--with-side',action='store_true',help='Read the configured side camera after each front read, matching Qt.')
    parser.add_argument('--worker-dir',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if not 1<=args.seconds<=120 or not 1<=args.max_frames<=3600:
        parser.error('Use 1..120 seconds and 1..3600 frames.')
    bench.preflight()
    import cv2
    requests=production_requests(cv2)
    if not args.run_camera:
        print(json.dumps({'static_check':'PASS','camera_opened':False,'opencv':cv2.__version__,
                          'production_order':requests,'model_initialized':False}))
        return 0
    if not args.apps_closed_confirmed:
        parser.error('Close camera-using apps and supply --apps-closed-confirmed.')
    if args.worker_dir:
        if args.worker_dir.resolve().parent != (ROOT/'benchmark_results').resolve():
            parser.error('Worker output must be a direct benchmark_results child.')
        return worker(args,args.worker_dir)
    folder=ROOT/'benchmark_results'/('capture_diagnosis_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    folder.mkdir(exist_ok=False)
    command=[sys.executable,'-B',str(Path(__file__).resolve()),'--run-camera','--apps-closed-confirmed',
             '--backend',args.backend,'--codec',args.codec,'--stage',args.stage,
             '--seconds',str(args.seconds),'--max-frames',str(args.max_frames),'--worker-dir',str(folder)]
    if args.with_side:
        command.append('--with-side')
    try:
        with (folder/'worker.log').open('w',encoding='utf-8') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=args.seconds+120)
        print((folder/'worker.log').read_text(encoding='utf-8'))
        return result.returncode
    except subprocess.TimeoutExpired:
        (folder/'timeout.json').write_text(json.dumps({'status':'native_call_or_setup_timeout',
            'seconds':args.seconds+120,'note':'Only diagnostic child terminated; no other app stopped.'}),encoding='utf-8')
        print(f'Diagnostic child timed out: {folder}',flush=True)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
