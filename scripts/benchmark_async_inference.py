"""Compare synchronous and two-slot asynchronous inference on leased frames."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

from experiments.rust_trt_benchmark import benchmark as protocol
from scripts.benchmark_frame_buffers import BufferBenchmark
from src.native_inference import create_rust_model
from src.settings.inference import WorkerSettings

ROOT = Path(__file__).resolve().parents[1]


class AsyncBenchmark(BufferBenchmark):
    def __init__(self, args):
        super().__init__(args)
        self.model.close()
        self.model = create_rust_model(
            args.engine, WorkerSettings(
                backend='rust', batch_size=args.batch,
                native_async=args.asynchronous,
            ), device='cuda:0', rect=False,
        )
        self.inputs = self.frames * 2
        for _ in range(20):
            self.model.predict_detections(self.inputs)

    def execute(self, command):
        if not hasattr(self, 'inputs'):
            return super().execute(command)
        if command['mode'] == 'verify':
            rows = self.model.predict_detections(self.inputs)
            return {
                'sha256': hashlib.sha256(
                    json.dumps(rows).encode(),
                ).hexdigest(),
            }
        times = []
        started_cpu = time.process_time_ns()
        for _ in range(command['repeats']):
            started = time.perf_counter_ns()
            self.model.predict_detections(self.inputs)
            times.append((time.perf_counter_ns() - started) / 1e6)
        return {
            'times_ms': times,
            'cpu_ms': (time.process_time_ns() - started_cpu) / 1e6,
            'pinned_bytes': self.model.engine.pinned_bytes,
            'workspace_bytes': self.model.engine.workspace_bytes,
            'async_stats': self.model.engine.async_stats,
        }


def worker_main(args):
    stream = sys.stdout
    sys.stdout = sys.stderr
    worker = AsyncBenchmark(args)
    try:
        stream.write(json.dumps({'ready': True}) + '\n')
        stream.flush()
        for line in sys.stdin:
            result = worker.execute(json.loads(line))
            stream.write(json.dumps(result) + '\n')
            stream.flush()
    finally:
        worker.close()


def run_case(args, batch):
    rounds = []
    signatures = {}
    for index in range(args.rounds):
        row = {}
        for mode in (
            ('sync', 'async') if index % 2 == 0 else ('async', 'sync')
        ):
            command = [
                sys.executable, '-m', 'scripts.benchmark_async_inference',
                '--worker', '--batch', str(batch),
                '--height', str(args.height), '--engine', str(args.engine),
                '--manifest', str(args.manifest),
            ]
            if mode == 'async':
                command.append('--asynchronous')
            child = protocol.Child(command, f'async.b{batch}.{index}.{mode}')
            try:
                signatures[mode] = child.call('verify')
                raw = child.call('detect', args.repeats)
                samples = sorted(raw.pop('times_ms'))
                row[mode] = {
                    **raw, 'median_ms_per_frame':
                    statistics.median(samples) / 16,
                    'p95_ms_per_16_frames': samples[
                        min(len(samples) - 1, int(len(samples) * .95))
                    ], 'samples_ms_per_16_frames': samples,
                    'resources': child.resources(),
                }
            finally:
                child.close()
        assert signatures['sync'] == signatures['async'], signatures
        rounds.append(row)
    return {
        'batch': batch, 'input_frames': 16, 'rounds': rounds,
        'exact_detection_parity': True, 'detection_sha256': signatures['sync'],
        'sync_ms_per_frame': statistics.median(
            row['sync']['median_ms_per_frame'] for row in rounds
        ), 'async_ms_per_frame': statistics.median(
            row['async']['median_ms_per_frame'] for row in rounds
        ), 'paired_speedup_median': statistics.median(
            row['sync']['median_ms_per_frame'] /
            row['async']['median_ms_per_frame'] for row in rounds
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--asynchronous', action='store_true')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--batches', type=int, nargs='+', default=[1, 8])
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=50)
    parser.add_argument('--core', type=int, default=4)
    parser.add_argument(
        '--engine', type=Path,
        default=ROOT / 'models/int8_engine/best_yolo26n.engine',
    )
    parser.add_argument(
        '--manifest', type=Path, default=ROOT /
        'experiments/rust_trt_benchmark/data/manifest.json',
    )
    parser.add_argument(
        '--output', type=Path,
        default=ROOT / 'native/inference/async_results.json',
    )
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.core})
    if args.worker:
        worker_main(args)
        return
    result = {
        'engine_sha256': protocol.sha(args.engine),
        'manifest_sha256': protocol.sha(args.manifest),
        'source_sha256': {
            str(path.relative_to(ROOT)): protocol.sha(path)
            for path in [
                Path(__file__).resolve(), ROOT / 'src/native_inference.py',
                ROOT / 'native/inference/bridge.cpp',
                ROOT / 'native/inference/src/engine.rs',
            ]
        }, 'policy': (
            '1080p leased shared frames, 16 per request, same engine and '
            'microbatch size; CPU resize, exact output gate. Sequential '
            'fresh workers alternate mode; 20 warmups excluded. Existing '
            'GPU workloads remain active. Excludes decode, IPC and encoding.'
        ), 'gpu_before': protocol.gpu_state(), 'core': args.core,
        'cases': [run_case(args, batch) for batch in args.batches],
        'gpu_after': protocol.gpu_state(),
    }
    protocol.dump(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
