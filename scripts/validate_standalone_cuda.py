"""Exercise the standalone Rust binary on local video and real TensorRT.

Python is only the test driver/reference.
The camera subprocess does not embed it.
Never uses a production catalog, camera, Redis, or notification endpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]


def process_pss(pid):
    total = 0
    try:
        parent = psutil.Process(pid)
        processes = [parent, *parent.children(recursive=True)]
    except psutil.Error:
        return 0
    for process in processes:
        try:
            total += process.memory_full_info().pss
        except (psutil.Error, FileNotFoundError, ProcessLookupError):
            continue
    return total


def run_trial(args, cameras):
    root = args.output / f'cameras-{cameras}'
    root.mkdir(parents=True, exist_ok=True)
    config = {
        'state_dir': str(root / 'state'), 'lossless': True,
        'record_detections': True, 'max_frames': args.frames,
        'batch_size': 1 if cameras == 1 else 8,
        'preprocess_workers': args.preprocess_workers,
        'exact_cpu_resize': not args.no_cpu_exact,
        'gpu': {
            'asynchronous': args.asynchronous,
            'exact_resize': args.exact_resize,
            'pipeline_depth': 2 if args.asynchronous else 1,
        },
        'models': [{
            'key': 'fixture', 'artifact': str(args.engine),
            'sha256': hashlib.sha256(args.engine.read_bytes()).hexdigest(),
            'version': 'isolated-cuda-fixture',
        }],
        'streams': [{
            'id': f'camera-{index}', 'site_id': 1, 'site': 'isolated-fixture',
            'name': f'camera-{index}', 'source': str(args.video),
            'model': 'fixture', 'width': 640, 'height': 352,
            'work_start_hour': 0, 'work_end_hour': 24,
            'output': {
                'clean': str(root / f'clean-{index}.mp4'),
                'annotated': str(root / f'overlay-{index}.mp4'),
            },
        } for index in range(cameras)],
    }
    config_file = root / 'config.json'
    config_file.write_text(json.dumps(config, indent=2))
    started = time.perf_counter()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    pss = 0
    with (root / 'stderr.log').open('w') as stderr, (
            root / 'stdout.log'
    ).open('w') as stdout:
        process = subprocess.Popen(
            [
                str(args.binary), 'run', '--config', str(config_file),
            ], stdout=stdout, stderr=stderr,
        )
        try:
            while process.poll() is None:
                pss = max(pss, process_pss(process.pid))
                if time.perf_counter() - started > 180:
                    raise TimeoutError(
                        'Standalone pipeline exceeded 180 seconds',
                    )
                time.sleep(.02)
            if process.returncode:
                raise RuntimeError(
                    f'Pipeline failed; see {root / "stderr.log"}',
                )
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
    elapsed = time.perf_counter() - started
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    report = json.loads((root / 'state/summary.json').read_text())
    assert report['python_runtime'] is False and report['replay'] is False
    assert len(report['cameras']) == cameras
    for camera in report['cameras']:
        assert camera['captured'] == camera['processed'] == args.frames
        assert camera['skipped'] == camera['stale'] == camera['failures'] == 0
        assert camera['encoded'] == args.frames * 2
    for index in range(cameras):
        for kind in ['clean', 'overlay']:
            probe = subprocess.run(
                [
                    'ffprobe', '-v', 'error', '-count_frames',
                    '-select_streams', 'v:0', '-show_entries',
                    'stream=nb_read_frames', '-of', 'json',
                    str(root / f'{kind}-{index}.mp4'),
                ], check=True, capture_output=True,
            )
            data = json.loads(probe.stdout)
            assert int(data['streams'][0]['nb_read_frames']) == args.frames
    return {
        'cameras': cameras, 'frames_each': args.frames,
        'cold_wall_seconds': elapsed, 'peak_process_tree_pss_bytes': pss,
        'cpu_seconds_including_children': (
            after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime
        ),
        'report': report,
    }


def reference_parity(args):
    import numpy as np
    from ultralytics import YOLO

    from src.danger_detector import DangerDetector
    from src.yolo_detector import YoloDetector

    raw = subprocess.run(
        [
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin',
            '-threads', '1', '-filter_threads', '1', '-i', str(args.video),
            '-an', '-sn', '-dn', '-vf', 'scale=640:352:flags=bilinear',
            '-frames:v', str(args.frames), '-pix_fmt', 'bgr24',
            '-f', 'rawvideo',
            'pipe:1',
        ], check=True, capture_output=True,
    ).stdout
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 352, 640, 3)
    rows = [
        json.loads(line) for line in (
            args.output / 'cameras-1/state/detections-camera-0.jsonl'
        ).read_text().splitlines()
    ]
    model = YOLO(str(args.engine), task='detect')
    tracker = YoloDetector(remote_tracker='centroid')
    hazard = DangerDetector({
        'detect_no_safety_vest_or_helmet': True,
        'detect_near_machinery_or_vehicle': True,
        'detect_in_restricted_area': True,
        'detect_in_utility_pole_restricted_area': False,
        'detect_machinery_close_to_pole': False,
    })
    maximum = 0.
    detections = 0
    for frame, row in zip(frames, rows, strict=True):
        prediction = model.predict(
            frame, device='cuda:0', imgsz=640, rect=False, verbose=False,
        )[0].boxes.data.cpu().numpy()
        expected = np.asarray(tracker.track_detections(prediction.tolist()))
        actual = np.asarray(row['tracks'])
        assert expected.shape == actual.shape
        if expected.size:
            np.testing.assert_array_equal(expected[:, 4:], actual[:, 4:])
            np.testing.assert_allclose(
                expected[:, :4], actual[:, :4], rtol=0, atol=.002,
            )
            maximum = max(
                maximum, float(
                    np.abs(
                        expected[:, :4] - actual[:, :4],
                    ).max(),
                ),
            )
        warnings, _, _ = hazard.detect_danger(expected.tolist())
        assert warnings == row['warnings']
        detections += len(actual)
    return {
        'frames': len(rows), 'detections': detections,
        'maximum_coordinate_error_pixels': maximum,
        'confidence_class_track_ids_exact': True, 'warnings_exact': True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument(
        '--video', type=Path,
        default=ROOT / 'tests/videos/test.mp4',
    )
    parser.add_argument('--preprocess-workers', type=int, default=1)
    parser.add_argument('--no-cpu-exact', action='store_true')
    parser.add_argument('--exact-resize', action='store_true')
    parser.add_argument('--asynchronous', action='store_true')
    parser.add_argument('--frames', type=int, default=16)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cameras', type=int, nargs='+', default=[1, 32])
    args = parser.parse_args()
    assert args.frames > 0 and all(1 <= count <= 32 for count in args.cameras)
    for name in ['binary', 'engine', 'video', 'output']:
        setattr(args, name, getattr(args, name).resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ['PYTHON_DOTENV_DISABLED'] = '1'
    os.environ['POSTPROCESS_BACKEND'] = 'python'
    result = {
        'scope': 'Finite local-video CUDA validation; includes cold startup, '
                 'CPU FFmpeg decode, annotations, H264 encoding and outbox. '
                 'Not live-camera capacity or a Python speed comparison.',
        'binary_sha256': hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        'engine_sha256': hashlib.sha256(args.engine.read_bytes()).hexdigest(),
        'trials': [run_trial(args, cameras) for cameras in args.cameras],
    }
    if 1 in args.cameras:
        result['python_reference'] = reference_parity(args)
    (args.output / 'results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
