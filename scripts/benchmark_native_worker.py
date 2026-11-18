"""Measure the installed Rust adapter against the actual square INT8 worker."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from experiments.rust_trt_benchmark import benchmark as protocol
from src.native_inference import create_rust_model
from src.settings.inference import WorkerSettings
from src.yolo_worker import YoloWorker
from src.yolo_worker_protocol import _worker_precision_config

ROOT = Path(__file__).resolve().parents[1]


class WorkerBenchmark:
    def __init__(self, args):
        metadata = json.loads(args.manifest.read_text())
        frames = [
            np.fromfile(args.manifest.parent / row['file'], dtype=np.uint8)
            .reshape(row['height'], row['width'], 3)
            for row in metadata['frames']
        ]
        self.batches = [
            frames[i:i + args.batch]
            for i in range(0, len(frames), args.batch)
        ]
        self.worker = YoloWorker(None)
        self.worker.precision_args = _worker_precision_config('int8')[2]
        self.worker.imgsz = 640
        if args.worker == 'rust':
            self.model = create_rust_model(
                args.engine,
                WorkerSettings(
                    backend='rust',
                    batch_size=args.batch,
                ),
                device='cuda:0',
                rect=False,
            )
        else:
            import torch
            from ultralytics import YOLO
            torch.set_num_threads(1)
            self.model = YOLO(str(args.engine), task='detect')
        for _ in range(8):
            self.worker._predict_batch(self.model, self.batches[0])

    def execute(self, command):
        mode = command['mode']
        if mode == 'verify':
            output = []
            for batch in self.batches:
                output.extend(self.worker._predict_batch(self.model, batch))
            return {'detections': output}
        if mode == 'info':
            native = getattr(self.model, 'engine', None)
            activation = (
                native.activation_bytes if native is not None
                else getattr(
                    self.model.predictor.model.model,
                    'device_memory_size_v2', None,
                )
            )
            return {
                'torch_loaded': 'torch' in sys.modules,
                'activation_bytes': activation,
            }
        times, checksum = [], 0.0
        for _ in range(command.get('repeats', 1)):
            for batch in self.batches:
                started = time.perf_counter_ns()
                rows = self.worker._predict_batch(self.model, batch)
                checksum += sum(row[4] for image in rows for row in image)
                times.append((time.perf_counter_ns() - started) / 1e6)
        return {
            'times_ms': times, 'checksum': checksum, 'images': command.get(
                'repeats', 1,
            ) * sum(map(len, self.batches)),
        }


def child_main(args):
    stream = sys.stdout
    sys.stdout = sys.stderr
    worker = WorkerBenchmark(args)
    try:
        stream.write(json.dumps({'ready': True}) + '\n')
        stream.flush()
        for line in sys.stdin:
            try:
                result = worker.execute(json.loads(line))
            except Exception as error:
                result = {'error': repr(error)}
            stream.write(json.dumps(result) + '\n')
            stream.flush()
    finally:
        if args.worker == 'rust':
            worker.model.close()


def compare_outputs(left, right):
    assert len(left) == len(right)
    coordinate_error = 0.0
    for a, b in zip(left, right, strict=True):
        expected = np.asarray(a, dtype=np.float32).reshape(-1, 6)
        actual = np.asarray(b, dtype=np.float32).reshape(-1, 6)
        assert expected.shape == actual.shape
        np.testing.assert_array_equal(expected[:, 4:], actual[:, 4:])
        np.testing.assert_allclose(
            expected[:, :4], actual[:, :4], rtol=0, atol=.002,
        )
        if expected.size:
            coordinate_error = max(
                coordinate_error, float(
                    abs(expected[:, :4] - actual[:, :4]).max(),
                ),
            )
    return {
        'frames': len(left), 'confidence_and_labels_equal': True,
        'max_coordinate_error': coordinate_error,
    }


def run_case(args, model, batch):
    engine = ROOT / 'models/int8_engine' / f'best_{model}.engine'
    children = {}
    try:
        for backend in ('python', 'rust'):
            children[backend] = protocol.Child(
                [
                    sys.executable, '-m', 'scripts.benchmark_native_worker',
                    '--worker', backend, '--engine', str(
                        engine,
                    ), '--batch', str(batch),
                    '--manifest', str(args.manifest),
                ], f'{model}.b{batch}.{backend}',
            )
        parity = compare_outputs(
            children['python'].call('verify')['detections'],
            children['rust'].call('verify')['detections'],
        )
        print(model, batch, 'parity', parity, flush=True)
        warm = children['python'].call('detect', 1)
        repeats = max(1, math.ceil(args.target_ms / sum(warm['times_ms'])))
        children['rust'].call('detect', repeats)
        rounds = []
        for index in range(args.rounds):
            row = {'gpu_before': protocol.gpu_state()}
            order = (
                'python',
                'rust',
            ) if index % 2 == 0 else (
                'rust',
                'python',
            )
            for backend in order:
                raw = children[backend].call('detect', repeats)
                protocol.dump(
                    args.data /
                    f'{model}.b{batch}.{index}.{backend}.json',
                    raw,
                )
                row[backend] = protocol.stats(raw)
            row['gpu_after'] = protocol.gpu_state()
            rounds.append(row)
        ratios = [
            r['python']['mean_ms_per_image'] /
            r['rust']['mean_ms_per_image'] for r in rounds
        ]
        result = {
            'model': model,
            'batch': batch,
            'engine_sha256': protocol.sha(engine),
            'parity': parity,
            'python_ms': statistics.median(
                r['python']['mean_ms_per_image'] for r in rounds
            ),
            'rust_ms': statistics.median(
                r['rust']['mean_ms_per_image'] for r in rounds
            ),
            'paired_speedup': statistics.median(ratios),
            'speedup_range': [
                min(ratios),
                max(ratios),
            ],
            'resources': {
                k: v.resources() for k,
                v in children.items()
            },
            'backend_info': {
                k: v.call('info') for k,
                v in children.items()
            },
            'rounds': rounds,
        }
        print(model, batch, 'speedup', result['paired_speedup'], flush=True)
        return result
    finally:
        for child in children.values():
            child.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=('python', 'rust'))
    parser.add_argument('--engine', type=Path)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument(
        '--manifest',
        type=Path,
        default=ROOT /
        'experiments/rust_trt_benchmark/data/manifest.json',
    )
    parser.add_argument(
        '--data',
        type=Path,
        default=ROOT /
        'outputs/native_inference_benchmark',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=ROOT /
        'native/inference/benchmark_results.json',
    )
    parser.add_argument('--models', nargs='+', default=['yolo26n', 'yolo26s'])
    parser.add_argument('--batches', nargs='+', type=int, default=[1, 8])
    parser.add_argument('--rounds', type=int, default=7)
    parser.add_argument('--target-ms', type=float, default=600)
    parser.add_argument('--core', type=int, default=4)
    args = parser.parse_args()
    if args.worker:
        child_main(args)
        return
    os.sched_setaffinity(0, {args.core})
    args.data.mkdir(parents=True, exist_ok=True)
    protocol.DATA = args.data
    paths = [
        Path(__file__),
        ROOT / 'src/native_inference.py',
        ROOT / 'src/yolo_worker.py',
        *sorted((ROOT / 'native/inference/src').glob('*.rs')),
        ROOT / 'native/inference/bridge.cpp',
        ROOT / 'native/inference/preprocess.cu',
    ]
    result = {
        'rect': False,
        'imgsz': 640,
        'scope': 'installed adapter + YoloWorker._predict_batch',
        'python': sys.version,
        'ultralytics': importlib.metadata.version('ultralytics'),
        'torch': importlib.metadata.version('torch'),
        'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'manifest_sha256': protocol.sha(args.manifest),
        'source_sha256': {
            str(p.relative_to(ROOT)): protocol.sha(p) for p in paths
        },
        'cases': [],
    }
    for model in args.models:
        for batch in args.batches:
            result['cases'].append(run_case(args, model, batch))
            protocol.dump(args.output, result)


if __name__ == '__main__':
    main()
