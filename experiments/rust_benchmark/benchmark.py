"""Controlled Rust/Python tests using local, recorded inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import resource
import statistics
import subprocess
import sys
import time
from functools import cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / 'data'
MODEL = ROOT / 'models/pt/best_yolo26n.onnx'
RUST = HERE / 'target/release/chd-rust-benchmark'
FLAGS = {
    'detect_no_safety_vest_or_helmet': True,
    'detect_near_machinery_or_vehicle': True,
    'detect_in_restricted_area': False,
    'detect_in_utility_pole_restricted_area': False,
    'detect_machinery_close_to_pole': False,
}


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@cache
def tracker_class():
    # The source module imports PEP 810 lazy dependencies, unavailable in 3.14.
    # Execute the unchanged production class rather than reimplementing it.
    source = (ROOT / 'src/yolo_detector.py').read_text()
    start = source.index('class YoloDetector:')
    definition = source[start:]
    namespace = {'np': np}
    exec(
        compile(
            'from __future__ import annotations\n' + definition,
            str(ROOT / 'src/yolo_detector.py'),
            'exec',
        ),
        namespace,
    )
    return namespace['YoloDetector']


class PythonProcessor:
    def __init__(self):
        sys.path.insert(0, str(ROOT))
        from src.danger_detector import DangerDetector

        self.tracker = tracker_class()()
        self.danger = DangerDetector(FLAGS)

    def process(self, detections):
        tracks = self.tracker.track_detections(detections)
        warnings, cone_polys, pole_polys = self.danger.detect_danger(tracks)
        assert not cone_polys and not pole_polys
        return {'tracks': tracks, 'warnings': warnings}


def make_session(backend):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.log_severity_level = 3
    providers = ['CPUExecutionProvider']
    if backend == 'cuda':
        providers = [
            (
                'CUDAExecutionProvider',
                {
                    'gpu_mem_limit': 1 << 30,
                    'cudnn_conv_algo_search': 'HEURISTIC',
                    'cudnn_conv_use_max_workspace': '0',
                },
            ),
        ]
    session = ort.InferenceSession(
        str(MODEL), sess_options=options, providers=providers,
    )
    if backend == 'cuda':
        assert session.get_providers()[0] == 'CUDAExecutionProvider'
    return session


def preprocess(image):
    import cv2

    height, width = image.shape[:2]
    ratio = min(640 / height, 640 / width)
    size = (round(width * ratio), round(height * ratio))
    resized = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    dw, dh = (640 - size[0]) / 2, (640 - size[1]) / 2
    padded = cv2.copyMakeBorder(
        resized,
        round(dh - 0.1),
        round(dh + 0.1),
        round(dw - 0.1),
        round(dw + 0.1),
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return (
        np.ascontiguousarray(
            padded[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32,
        )
        / 255.0
    )


def synthetic_frames(count):
    frames = []
    rng = np.random.default_rng(20261116 + count)
    classes = [5, 5, 5, 5, 8, 10, 2, 4]
    for frame in range(64):
        rows = []
        for index in rng.permutation(count).tolist():
            if frame % 17 == 0 and index % 7 == 0:
                continue
            cls = classes[index % len(classes)]
            cx = (index % 16) * 64 + 30 + math.sin(frame / 8) * 8
            cy = (index // 16) * 100 + 80 + math.cos(frame / 9) * 5
            width, height = (180, 180) if cls in (8, 10) else (18, 60)
            rows.append(
                [
                    cx - width / 2,
                    cy - height / 2,
                    cx + width / 2,
                    cy + height / 2,
                    0.9,
                    cls,
                ],
            )
        frames.append(rows)
    return frames


def prepare():
    import cv2

    DATA.mkdir(parents=True, exist_ok=True)
    session = make_session('cpu')
    video = ROOT / 'tests/videos/test.mp4'
    capture = cv2.VideoCapture(str(video))
    assert capture.isOpened()
    images = []
    sources = []
    for number in range(0, 864, 18):
        capture.set(cv2.CAP_PROP_POS_FRAMES, number)
        ok, image = capture.read()
        if ok:
            images.append(image)
            sources.append({'video_frame': number})
    capture.release()
    paths = sorted((ROOT / 'tests/cv_dataset/images').glob('*'))
    for path in paths:
        image = cv2.imread(str(path))
        if image is not None:
            images.append(image)
            sources.append(
                {'image': str(path.relative_to(ROOT)), 'sha256': sha(path)},
            )
    real = []
    inputs = []
    for index, image in enumerate(images):
        tensor = preprocess(image)
        output = session.run(None, {'images': tensor[None]})[0][0]
        real.append(output[output[:, 4] >= 0.25].astype(float).tolist())
        if index < 8 or index >= len(images) - 8:
            inputs.append(tensor)
    np.stack(inputs).astype('<f4').tofile(DATA / 'inputs.f32')
    edge = [
        [],
        [[0, 0, 10, 20, 0.9, 5]],
        [[1, 0, 11, 20, 0.9, 5], [1, 0, 11, 20, 0.9, 5]],
        [[200, 200, 100, 100, 0.8, 8], [80, 80, 100, 140, 0.9, 5]],
        *([[]] * 20),
        [[0, 0, 10, 20, 0.9, 5]],
    ]
    corpus = {
        'cases': [
            {
                'name': 'actual',
                'kind': (
                    '48 video frames + local test images; '
                    'real model detections'
                ),
                'frames': real,
            },
            {
                'name': 'edge_cases',
                'kind': 'declared synthetic corner cases',
                'frames': edge,
            },
            *[
                {
                    'name': f'synthetic_{count}',
                    'kind': (
                        'synthetic density stress, not observed camera load'
                    ),
                    'frames': synthetic_frames(count),
                }
                for count in (16, 64, 256)
            ],
        ],
    }
    dump(DATA / 'corpus.json', corpus)
    dump(
        DATA / 'manifest.json',
        {
            'model': str(MODEL.relative_to(ROOT)),
            'model_sha256': sha(MODEL),
            'video': str(video.relative_to(ROOT)),
            'video_sha256': sha(video),
            'sources': sources,
            'real_detection_counts': [len(rows) for rows in real],
            'input_count': len(inputs),
            'input_shape': [1, 3, 640, 640],
            'confidence': 0.25,
            'seed': 20261116,
            'source_hashes': {
                name: sha(ROOT / name)
                for name in (
                    'src/yolo_detector.py',
                    'src/danger_detector.py',
                    'src/geometry.py',
                )
            },
            'rustc': subprocess.check_output(
                ['rustc', '--version'], text=True,
            ).strip(),
            'host': platform.platform(),
        },
    )
    print(
        json.dumps(
            {
                'prepared_real_frames': len(real),
                'input_count': len(inputs),
                'median_detections': statistics.median(map(len, real)),
                'max_detections': max(map(len, real)),
            },
        ),
        flush=True,
    )


def replay(name, repetitions, trace):
    corpus = json.loads((DATA / 'corpus.json').read_text())
    case = next(case for case in corpus['cases'] if case['name'] == name)
    warm = PythonProcessor()
    for frame in case['frames']:
        warm.process(frame)
    times = []
    elapsed = 0.0
    for _ in range(repetitions):
        processor = PythonProcessor()
        start_pass = time.perf_counter()
        for frame in case['frames']:
            start = time.perf_counter()
            processor.process(frame)
            times.append((time.perf_counter() - start) * 1e6)
        elapsed += time.perf_counter() - start_pass
    if trace:
        processor = PythonProcessor()
        dump(
            Path(trace), [
                processor.process(frame)
                for frame in case['frames']
            ],
        )
    return {
        'language': 'python',
        'case': name,
        'kind': case['kind'],
        'elapsed_seconds': elapsed,
        'frames': len(times),
        'latencies_us': times,
        'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'python': sys.version,
    }


def infer(backend, runs, warmup, post, trace):
    session = make_session(backend)
    inputs = np.fromfile(DATA / 'inputs.f32', dtype='<f4').reshape(
        -1, 1, 3, 640, 640,
    )
    processor = PythonProcessor() if post else None
    for index in range(warmup):
        session.run(None, {'images': inputs[index % len(inputs)]})
    times = []
    start_all = time.perf_counter()
    for index in range(runs):
        start = time.perf_counter()
        output = session.run(None, {'images': inputs[index % len(inputs)]})[0]
        if processor is not None:
            rows = output[0]
            detections = rows[rows[:, 4] >= 0.25].astype(float).tolist()
            processor.process(detections)
        else:
            float(output.reshape(-1)[4])
        times.append((time.perf_counter() - start) * 1e6)
    elapsed = time.perf_counter() - start_all
    if trace:
        output = np.stack(
            [session.run(None, {'images': image})[0] for image in inputs],
        )
        output.astype('<f4').tofile(trace)
    return {
        'language': 'python',
        'backend': backend,
        'post': post,
        'elapsed_seconds': elapsed,
        'frames': runs,
        'latencies_us': times,
        'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'python': sys.version,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'replay', 'infer'])
    parser.add_argument('--case', default='actual')
    parser.add_argument('--repetitions', type=int, default=1)
    parser.add_argument('--backend', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--runs', type=int, default=80)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--post', action='store_true')
    parser.add_argument('--trace')
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
        return
    if args.mode == 'replay':
        result = replay(args.case, args.repetitions, args.trace)
    else:
        result = infer(
            args.backend, args.runs, args.warmup, args.post, args.trace,
        )
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
