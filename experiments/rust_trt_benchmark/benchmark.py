"""Compare Rust with the repository's real TensorRT prediction path."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / 'data'


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare():
    import cv2
    import torch

    DATA.mkdir(exist_ok=True)
    capture = cv2.VideoCapture(str(ROOT / 'tests/videos/test.mp4'))
    if not capture.isOpened():
        raise RuntimeError('Test video unavailable')
    frames, sources = [], []
    for number in range(0, 864, 18):
        capture.set(cv2.CAP_PROP_POS_FRAMES, number)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f'Cannot read video frame {number}')
        frames.append(frame)
        sources.append({'video_frame': number})
    capture.release()
    for path in sorted((ROOT / 'tests/cv_dataset/images').glob('*')):
        frame = cv2.imread(str(path))
        if frame is not None:
            frames.append(frame)
            sources.append(
                {'image': str(path.relative_to(ROOT)), 'sha256': sha(path)},
            )
    metadata = []
    for index, (frame, source) in enumerate(zip(frames, sources)):
        name = f'frame.{index}.bgr'
        frame.tofile(DATA / name)
        metadata.append({
            'file': name, 'height': frame.shape[0], 'width': frame.shape[1],
            'sha256': sha(DATA / name), 'source': source,
        })
    # Match the exact reference CUDA normalization for every possible uint8
    # value.
    (
        torch.arange(256, device='cuda', dtype=torch.float32) /
        255
    ).cpu().numpy().tofile(DATA / 'normalize.f32')
    dump(
        DATA / 'manifest.json', {
            'frames': metadata, 'imgsz': 640, 'stride': 32,
            'video_sha256': sha(ROOT / 'tests/videos/test.mp4'),
        },
    )


class PythonWorker:
    def __init__(self, manifest, engine, batch):
        import cv2
        import torch
        from ultralytics import YOLO

        cv2.setNumThreads(1)
        torch.set_num_threads(1)
        self.torch = torch
        self.data = manifest.parent
        metadata = json.loads(manifest.read_text())
        self.frames = [
            np.fromfile(self.data / frame['file'], dtype=np.uint8).reshape(
                frame['height'], frame['width'], 3,
            ) for frame in metadata['frames']
        ]
        self.batches = [
            self.frames[i:i + batch]
            for i in range(0, len(self.frames), batch)
        ]
        self.model = YOLO(str(engine), task='detect')
        for _ in range(8):
            self.predict(self.batches[0])
        self.backend = self.model.predictor.model
        with torch.inference_mode():
            self.first_tensor = self.model.predictor.preprocess(
                self.batches[0],
            )

    def predict(self, frames):
        # Same prediction kwargs and result conversion as
        # YoloWorker._handle_batch.
        results = self.model.predict(
            source=frames, verbose=False, device='cuda:0', imgsz=640,
            batch=len(frames), quantize=8,
        )
        detections = []
        for result in results:
            data = result.boxes.data.cpu().numpy()
            rows = data[:, [0, 1, 2, 3, 4, 5]].astype(
                float, copy=False,
            ).tolist()
            for row in rows:
                row[5] = int(row[5])
            detections.append(rows)
        return detections

    def execute(self, command):
        mode = command['mode']
        repeats = command.get('repeats', 1)
        timings, checksum = [], 0.0
        with self.torch.inference_mode():
            if mode == 'verify':
                detections, shapes = [], []
                for index, frames in enumerate(self.batches):
                    tensor = self.model.predictor.preprocess(frames)
                    tensor.cpu().numpy().tofile(
                        self.data / f'python.input.{index}.f32',
                    )
                    shapes.append(list(tensor.shape))
                    raw = self.backend(tensor)
                    if isinstance(raw, (tuple, list)):
                        raw = raw[0]
                    raw.cpu().numpy().tofile(
                        self.data / f'python.output.{index}.f32',
                    )
                    detections.extend(self.predict(frames))
                return {'detections': detections, 'shapes': shapes}
            if mode == 'infer':
                for _ in range(20):
                    self.backend(self.first_tensor)
                self.torch.cuda.synchronize()
                for _ in range(repeats):
                    started = time.perf_counter_ns()
                    self.backend(self.first_tensor)
                    timings.append((time.perf_counter_ns() - started) / 1e6)
                images = repeats * len(self.batches[0])
            elif mode == 'detect':
                for _ in range(repeats):
                    for frames in self.batches:
                        started = time.perf_counter_ns()
                        result = self.predict(frames)
                        checksum += sum(
                            row[4]
                            for frame in result for row in frame
                        )
                        timings.append(
                            (time.perf_counter_ns() - started) / 1e6,
                        )
                images = repeats * len(self.frames)
            else:
                raise ValueError(mode)
        return {'times_ms': timings, 'images': images, 'checksum': checksum}


def python_worker(args):
    protocol = sys.stdout
    sys.stdout = sys.stderr
    worker = PythonWorker(DATA / 'manifest.json', args.engine, args.batch)
    protocol.write(json.dumps({'ready': True}) + '\n')
    protocol.flush()
    for line in sys.stdin:
        try:
            result = worker.execute(json.loads(line))
        except Exception as error:
            result = {'error': repr(error)}
        protocol.write(json.dumps(result) + '\n')
        protocol.flush()


class Child:
    def __init__(self, command, name):
        self.log = (DATA / f'{name}.log').open('w')
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.log, text=True, bufsize=1,
        )
        ready = self.read()
        if ready.get('ready') is not True:
            raise RuntimeError(f'Worker not ready: {ready}')

    def read(self):
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError(f'Worker exited with {self.process.poll()}')
            if line.startswith('{'):
                result = json.loads(line)
                break
            # TensorRT's native logger can write directly to the stdout FD.
            self.log.write(line)
            self.log.flush()
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result

    def call(self, mode, repeats=1):
        self.process.stdin.write(
            json.dumps(
                {'mode': mode, 'repeats': repeats},
            ) + '\n',
        )
        self.process.stdin.flush()
        return self.read()

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=10)
        self.log.close()

    def resources(self):
        status = Path(f'/proc/{self.process.pid}/status').read_text()
        result = {}
        for line in status.splitlines():
            key, _, value = line.partition(':')
            if key in {'VmHWM', 'VmRSS', 'Threads'}:
                result[key] = value.strip()
        return result


def verify(reference, native):
    expected = reference.call('verify')
    actual = native.call('verify')
    if len(expected['detections']) != len(actual['detections']):
        raise AssertionError('Frame count differs')
    max_coordinate_error, max_score_error = 0.0, 0.0
    for index, (left, right) in enumerate(
            zip(expected['detections'], actual['detections']),
    ):
        a, b = np.asarray(
            left, dtype=np.float32,
        ), np.asarray(
            right, dtype=np.float32,
        )
        if a.shape != b.shape:
            raise AssertionError(
                f'Detection count differs in frame {index}: {
                    a.shape
                } vs {
                    b.shape
                }',
            )
        if a.size:
            if not np.array_equal(a[:, 5], b[:, 5]):
                raise AssertionError(f'Class/order differs at frame {index}')
            max_coordinate_error = max(
                max_coordinate_error, float(
                    np.abs(a[:, :4] - b[:, :4]).max(),
                ),
            )
            max_score_error = max(
                max_score_error, float(
                    np.abs(a[:, 4] - b[:, 4]).max(),
                ),
            )
    input_exact, raw_exact, max_raw_error = True, True, 0.0
    for index in range(len(expected['shapes'])):
        a = np.fromfile(DATA / f'python.input.{index}.f32', dtype=np.float32)
        b = np.fromfile(DATA / f'rust.input.{index}.f32', dtype=np.float32)
        if not np.array_equal(a, b):
            raise AssertionError(
                f'Preprocessing differs at batch {index}: {
                    np.max(
                        np.abs(
                            a - b
                        )
                    )
                }',
            )
        a = np.fromfile(DATA / f'python.output.{index}.f32', dtype=np.float32)
        b = np.fromfile(DATA / f'rust.output.{index}.f32', dtype=np.float32)
        raw_exact = raw_exact and np.array_equal(a, b)
        max_raw_error = max(max_raw_error, float(np.abs(a - b).max()))
    if max_coordinate_error > 0.002 or max_score_error > 0.00001:
        raise AssertionError(
            f'Detection output differs: {max_coordinate_error}, '
            f'{max_score_error}',
        )
    return {
        'frames': len(
            expected['detections'],
        ),
        'detections': sum(
            len(frame) for frame in expected['detections']
        ),
        'preprocessed_tensors_bitwise_equal': input_exact,
        'raw_outputs_bitwise_equal': raw_exact,
        'max_raw_abs_error': max_raw_error,
        'max_coordinate_abs_error': max_coordinate_error,
        'max_confidence_abs_error': max_score_error,
        'input_shapes': expected['shapes'],
    }


def stats(result):
    times = result['times_ms']
    elapsed = sum(times)
    return {
        'mean_ms_per_image': elapsed / result['images'],
        'mean_ms_per_batch': statistics.mean(times),
        'p95_ms_per_batch': float(np.percentile(times, 95)),
        'images_per_second': result['images'] * 1000 / elapsed,
        'images': result['images'], 'batches': len(times),
    }


def gpu_state():
    fields = (
        'name,driver_version,memory.used,memory.total,utilization.gpu,'
        'temperature.gpu,clocks.sm,power.draw'
    )
    return subprocess.check_output(
        [
            'nvidia-smi', f'--query-gpu={fields}',
            '--format=csv,noheader',
        ], text=True,
    ).strip()


def run_case(engine, batch, rounds, target_ms):
    name = f'{engine.stem}.batch{batch}'
    workers = {}
    before = gpu_state()
    try:
        workers['python'] = Child(
            [
                sys.executable, str(Path(__file__)), '--python-worker',
                '--engine', str(engine), '--batch', str(batch),
            ], name + '.python',
        )
        workers['rust'] = Child(
            [
                str(HERE / 'target/release/chd-rust-trt-benchmark'),
                str(DATA / 'manifest.json'), str(engine), str(batch),
            ], name + '.rust',
        )
        parity = verify(workers['python'], workers['rust'])
        print(f'{name}: parity passed ({parity["frames"]} frames)', flush=True)
        metrics = {}
        for mode in ('infer', 'detect'):
            calibration = workers['python'].call(
                mode, 1 if mode == 'detect' else 40,
            )
            repeats = max(
                1, math.ceil(
                    target_ms / sum(calibration['times_ms']),
                ),
            )
            if mode == 'infer':
                repeats *= 40
            for child in workers.values():
                child.call(mode, min(repeats, 40))
            samples = []
            for index in range(rounds):
                order = (
                    'python',
                    'rust',
                ) if index % 2 == 0 else (
                    'rust',
                    'python',
                )
                row = {'gpu_before': gpu_state()}
                for backend in order:
                    raw = workers[backend].call(mode, repeats)
                    dump(DATA / f'{name}.{mode}.{index}.{backend}.json', raw)
                    row[backend] = stats(raw)
                row['gpu_after'] = gpu_state()
                samples.append(row)
            ratios = [
                row['python']['mean_ms_per_image'] /
                row['rust']['mean_ms_per_image'] for row in samples
            ]
            metrics[mode] = {
                'python_ms_per_image': statistics.median(
                    row['python']['mean_ms_per_image'] for row in samples
                ),
                'rust_ms_per_image': statistics.median(
                    row['rust']['mean_ms_per_image'] for row in samples
                ),
                'median_paired_speedup': statistics.median(ratios),
                'paired_speedup_range': [
                    min(ratios),
                    max(ratios),
                ],
                'rounds': samples,
            }
            print(
                name, mode, json.dumps(
                    {k: v for k, v in metrics[mode].items() if k != 'rounds'},
                ), flush=True,
            )
        return {
            'engine': str(
                engine.relative_to(ROOT),
            ),
            'engine_sha256': sha(engine),
            'batch': batch,
            'parity': parity,
            'metrics': metrics,
            'gpu_before': before,
            'gpu_after': gpu_state(),
            'process_resources': {
                name: child.resources() for name, child in workers.items()
            },
        }
    finally:
        for child in workers.values():
            child.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--python-worker', action='store_true')
    parser.add_argument('--engine', type=Path)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--models', nargs='+', default=['yolo26n', 'yolo26s'])
    parser.add_argument('--batches', nargs='+', type=int, default=[1, 8])
    parser.add_argument('--rounds', type=int, default=7)
    parser.add_argument('--target-ms', type=float, default=600)
    parser.add_argument('--core', type=int, default=4)
    args = parser.parse_args()
    if args.python_worker:
        python_worker(args)
        return
    if args.prepare:
        prepare()
        return
    os.sched_setaffinity(0, {args.core})
    import cv2
    import torch
    import tensorrt
    import ultralytics

    sources = [
        *HERE.glob('*.py'),
        *HERE.glob('*.cpp'),
        *HERE.glob('*.cu'),
        HERE / 'Cargo.toml',
        HERE / 'Cargo.lock',
        HERE / 'build.rs',
        HERE / 'src/main.rs',
        ROOT / 'src/yolo_worker.py',
    ]
    result = {
        'python': sys.version, 'platform': platform.platform(),
        'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
        'tensorrt': tensorrt.__version__,
        'ultralytics': ultralytics.__version__,
        'python_opencv': cv2.__version__, 'native_opencv': '4.10.0',
        'rustc': subprocess.check_output(
            ['rustc', '--version'], text=True,
        ).strip(),
        'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'manifest_sha256': sha(DATA / 'manifest.json'),
        'runtime_sha256': {
            'libnvinfer.so.11': sha(
                Path(os.environ['TRT_LIB']) / 'libnvinfer.so.11',
            ),
            'libnvinfer_plugin.so.11': sha(
                Path(os.environ['TRT_LIB']) / 'libnvinfer_plugin.so.11',
            ),
            'libcudart.so': sha(
                Path(os.environ['CUDA_LIB']) / 'libcudart.so',
            ),
        },
        'source_sha256': {
            str(path.relative_to(ROOT)): sha(path) for path in sources
        },
        'scope': (
            'same existing INT8 engine; GPU-resident backend call '
            'and BGR-to-worker-detections'
        ),
        'excludes': (
            'capture, decode, worker IPC/queue wait, tracking, drawing, '
            'encoding, storage, notification'
        ),
        'cases': [],
    }
    for model in args.models:
        engine = ROOT / 'models/int8_engine' / f'best_{model}.engine'
        for batch in args.batches:
            result['cases'].append(
                run_case(
                    engine,
                    batch,
                    args.rounds,
                    args.target_ms,
                ),
            )
            dump(HERE / 'results.json', result)


if __name__ == '__main__':
    main()
