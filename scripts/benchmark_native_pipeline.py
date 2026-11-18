"""Compare pipeline stages with preserved source and native wheels."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

from experiments.rust_trt_benchmark import benchmark as protocol

ROOT = Path(__file__).resolve().parents[1]


class StageBenchmark:
    def __init__(self, args):
        import numpy as np
        if args.stage == 'inference':
            from scripts.benchmark_frame_buffers import BufferBenchmark
            self.stage = BufferBenchmark(args)
            self.operation = self.infer
        elif args.stage == 'geometry':
            from src.danger_detector import DangerDetector
            self.stage = DangerDetector({
                'detect_no_safety_vest_or_helmet': False,
                'detect_near_machinery_or_vehicle': False,
                'detect_in_restricted_area': True,
                'detect_in_utility_pole_restricted_area': True,
                'detect_machinery_close_to_pole': False,
            })
            self.scene = [
                [x, y, x + 10, y + 10, .9, 6] for x, y in
                ((0, 0), (200, 0), (0, 200), (200, 200))
            ] + [[x, 0, x + 20, 200, .9, 9] for x in (300, 500, 700)]
            self.scene += [[x, 100, x + 10, 120, .9, 5]
                           for x in range(20, 700, 20)]
            self.operation = self.geometry
        else:
            from examples.streaming_web.overlay_models import OverlaySnapshot
            from examples.streaming_web.overlay_renderer import OverlayRenderer
            self.stage = OverlayRenderer()
            self.snapshot = OverlaySnapshot(
                sequence=(1, 1),
                frame=np.full((1080, 1920, 3), 100, dtype=np.uint8),
                warnings={'warning_no_hardhat': {'count': 8}},
                track_data=[
                    [x, y, x + 80, y + 100, .9, 2, index, 0]
                    for index, (x, y) in enumerate(
                        (x, y) for y in (100, 400, 700)
                        for x in (100, 500, 900, 1300)
                    )
                ],
                cone_polys=[[[10, 10], [1000, 10], [1000, 600], [10, 600]]],
            )
            self.operation = self.overlay
        for _ in range(8):
            self.operation()

    def infer(self):
        return self.stage.execute({'mode': 'buffer', 'repeats': 1})

    def geometry(self):
        return self.stage.detect_danger(self.scene)

    def overlay(self):
        return self.stage.render(self.snapshot, 'zh-TW')

    def execute(self, command):
        if command['mode'] == 'verify':
            result = self.operation()
            if isinstance(result, dict) and 'times_ms' in result:
                return self.stage.execute({'mode': 'verify'})
            encoded = (
                result.tobytes() if hasattr(result, 'tobytes')
                else json.dumps(result, sort_keys=True).encode()
            )
            return {'sha256': hashlib.sha256(encoded).hexdigest()}
        if command['mode'] == 'info':
            engine = getattr(
                getattr(
                    self.stage,
                    'model',
                    None,
                ),
                'engine',
                None,
            )
            return {
                'workspace_growths': getattr(
                    engine, 'workspace_growths', None,
                ), 'workspace_bytes': getattr(
                    engine, 'workspace_bytes', None,
                ), 'geometry_cache_hits': getattr(
                    self.stage, 'geometry_cache_hits', None,
                ), 'geometry_cache_misses': getattr(
                    self.stage, 'geometry_cache_misses', None,
                ), }
        times = []
        for _ in range(command['repeats']):
            started = time.perf_counter_ns()
            self.operation()
            times.append((time.perf_counter_ns() - started) / 1e6)
        count = 8 if hasattr(self.stage, 'batches') else 1
        return {'times_ms': times, 'images': command['repeats'] * count}

    def close(self):
        if hasattr(self.stage, 'close'):
            self.stage.close()


def child_main(args):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(args.source_root))
    if args.wheel_root:
        sys.path.insert(0, str(args.wheel_root))
    import examples
    examples.__path__ = [
        str(args.source_root / 'examples'), *examples.__path__,
    ]
    stream = sys.stdout
    sys.stdout = sys.stderr
    worker = StageBenchmark(args)
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
        worker.close()


def run_case(args, stage, batch):
    children = {}
    try:
        for name, source, wheel in (
            ('before', args.baseline, args.baseline_wheel),
            ('after', ROOT, args.current_wheel),
        ):
            children[name] = protocol.Child(
                [
                    sys.executable, '-m', 'scripts.benchmark_native_pipeline',
                    '--worker', '--stage', stage, '--batch', str(batch),
                    '--source-root', str(source), '--wheel-root', str(wheel),
                    '--engine', str(args.engine),
                    '--manifest', str(args.manifest),
                ], f'{stage}.b{batch}.{name}',
            )
        parity = {
            name: worker.call('verify')
            for name, worker in children.items()
        }
        assert parity['before'] == parity['after'], parity
        repeats = {
            name: max(
                1, math.ceil(
                    args.target_ms /
                    sum(worker.call('detect')['times_ms']),
                ),
            )
            for name, worker in children.items()
        }
        rounds = []
        for index in range(args.rounds):
            row = {}
            for name in (
                ('before', 'after') if index %
                2 == 0 else (
                    'after', 'before',
                )
            ):
                raw = children[name].call('detect', repeats[name])
                protocol.dump(
                    args.data /
                    f'{stage}.b{batch}.{index}.{name}.json', raw,
                )
                row[name] = protocol.stats(raw)
            rounds.append(row)
        result = {
            'stage': stage, 'batch': batch, 'parity': parity,
            'before_ms': statistics.median(
                row['before']['mean_ms_per_image'] for row in rounds
            ),
            'after_ms': statistics.median(
                row['after']['mean_ms_per_image'] for row in rounds
            ),
            'paired_speedup': statistics.median(
                row['before']['mean_ms_per_image'] /
                row['after']['mean_ms_per_image']
                for row in rounds
            ),
            'info': {
                name: worker.call('info') for name, worker in children.items()
            },
            'rounds': rounds,
        }
        print(
            stage,
            batch,
            result['before_ms'],
            result['after_ms'],
            result['paired_speedup'],
            flush=True,
        )
        return result
    finally:
        for worker in children.values():
            worker.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument(
        '--stage',
        choices=(
            'inference',
            'geometry',
            'overlay',
        ),
    )
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--wheel-root', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument(
        '--baseline-wheel',
        type=Path,
        help='Directory containing the installed baseline native extension',
    )
    parser.add_argument(
        '--current-wheel',
        type=Path,
        help='Directory containing the installed current native extension',
    )
    parser.add_argument(
        '--engine',
        type=Path,
        default=ROOT /
        'models/int8_engine/best_yolo26n.engine',
    )
    parser.add_argument(
        '--manifest',
        type=Path,
        default=ROOT /
        'experiments/rust_trt_benchmark/data/manifest.json',
    )
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--rounds', type=int, default=7)
    parser.add_argument('--target-ms', type=float, default=300)
    parser.add_argument('--core', type=int, default=4)
    parser.add_argument(
        '--data',
        type=Path,
        default=ROOT /
        'outputs/native_pipeline_benchmark',
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Result JSON path, defaults to results.json in --data',
    )
    args = parser.parse_args()
    if args.worker:
        child_main(args)
        return
    if args.baseline is None or not args.baseline.is_dir():
        parser.error('--baseline requires a preserved source snapshot')
    for option, directory in (
        ('--baseline-wheel', args.baseline_wheel),
        ('--current-wheel', args.current_wheel),
    ):
        if directory is None or not directory.is_dir():
            parser.error(
                f'{option} requires an installed native extension directory',
            )
    args.output = args.output or args.data / 'results.json'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.sched_setaffinity(0, {args.core})
    args.data.mkdir(parents=True, exist_ok=True)
    protocol.DATA = args.data
    result = {
        'scope': (
            'Paired stage timings; 1080p shared inference, stable geometry, '
            '1080p zh-TW overlay'
        ),
        'python': sys.version,
        'cpu_affinity': sorted(
            os.sched_getaffinity(0),
        ),
        'gpu_before': protocol.gpu_state(),
        'cases': [],
        'engine_sha256': protocol.sha(args.engine),
        'manifest_sha256': protocol.sha(args.manifest),
        'source_sha256': {
            name: {
                str(path.relative_to(source)): protocol.sha(path)
                for path in (
                    source / 'src/danger_detector.py',
                    source / 'src/geometry.py',
                    source / 'native/inference/src/engine.rs',
                    source / 'native/inference/bridge.cpp',
                    source / 'examples/streaming_web/overlay_text.py',
                    source / 'examples/streaming_web/overlay_renderer.py',
                )
            } for name, source in (
                ('before', args.baseline), ('after', ROOT),
            )
        },
    }
    for stage, batch in (
        ('inference', 1), ('inference', 8),
        ('geometry', 1), ('overlay', 1),
    ):
        result['cases'].append(run_case(args, stage, batch))
        result['gpu_after'] = protocol.gpu_state()
        protocol.dump(args.output, result)


if __name__ == '__main__':
    main()
