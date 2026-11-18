"""Compare full-frame snapshots with leased shared-worker buffer views."""
from __future__ import annotations

import argparse
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
from src.yolo_shared_frames import _SharedFrameRing
from src.yolo_worker import YoloWorker
from src.yolo_worker_protocol import _WorkerRequest

ROOT = Path(__file__).resolve().parents[1]


class BufferBenchmark:
    def __init__(self, args):
        import cv2

        cv2.setNumThreads(1)
        metadata = json.loads(args.manifest.read_text())
        self.worker = YoloWorker(None)
        self.ring = _SharedFrameRing.create(
            (args.height, args.height * 16 // 9, 3), np.dtype(np.uint8), 8,
        )
        self.frames = []
        self.model = None
        try:
            for slot, item in enumerate(metadata['frames'][:8]):
                original = np.fromfile(
                    args.manifest.parent / item['file'], dtype=np.uint8,
                ).reshape(item['height'], item['width'], 3)
                frame = cv2.resize(original, self.ring.shape[1::-1])
                self.ring.write(slot, frame)
                self.frames.append(
                    YoloWorker._read_frame(
                        _WorkerRequest.from_mapping({
                            'id': str(slot), 'camera_id': str(slot),
                            'model_key': 'benchmark',
                            'shm_name': self.ring.name, 'slot': slot,
                            'shape': self.ring.shape, 'dtype': 'uint8',
                        }),
                    ),
                )
            self.batches = [
                self.frames[start:start + args.batch]
                for start in range(0, 8, args.batch)
            ]
            self.model = create_rust_model(
                args.engine, WorkerSettings(
                    backend='rust', batch_size=args.batch,
                ),
                device='cuda:0', rect=False,
            )
            assert self.model._buffer_api, 'Rebuild with BUFFER_API_VERSION=1'
            for mode in ('snapshot', 'buffer'):
                self.execute({'mode': mode, 'repeats': 4})
        except BaseException:
            self.close()
            raise

    def execute(self, command):
        mode = command['mode']
        if mode == 'info':
            return {
                'borrowed_frames': self.model.borrowed_frames,
                'snapshot_bytes': self.model.snapshot_bytes,
                'torch_loaded': 'torch' in sys.modules,
            }
        self.model._buffer_api = mode != 'snapshot'
        if mode == 'verify':
            self.model._buffer_api = False
            expected = [
                self.model.predict_detections(
                    batch,
                ) for batch in self.batches
            ]
            self.model._buffer_api = True
            actual = [
                self.model.predict_detections(
                    batch,
                ) for batch in self.batches
            ]
            assert expected == actual
            return {'frames': len(self.frames), 'exactly_equal': True}
        times, checksum = [], 0.0
        borrowed = self.model.borrowed_frames
        copied = self.model.snapshot_bytes
        for _ in range(command['repeats']):
            for batch in self.batches:
                started = time.perf_counter_ns()
                rows = self.worker._predict_batch(self.model, batch)
                times.append((time.perf_counter_ns() - started) / 1e6)
                checksum += sum(row[4] for image in rows for row in image)
        return {
            'times_ms': times, 'checksum': checksum,
            'images': command['repeats'] * len(self.frames),
            'borrowed_frames': self.model.borrowed_frames - borrowed,
            'snapshot_bytes': self.model.snapshot_bytes - copied,
        }

    def close(self):
        if self.model is not None:
            self.model.close()
        self.worker._close_shared_frames(self.frames)
        self.ring.close()


def child_main(args):
    stream = sys.stdout
    sys.stdout = sys.stderr
    worker = BufferBenchmark(args)
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


def run_case(args, height, batch):
    child = protocol.Child(
        [
            sys.executable, '-m', 'scripts.benchmark_frame_buffers',
            '--worker', '--engine', str(args.engine),
            '--height', str(height), '--batch', str(batch),
            '--manifest', str(args.manifest),
        ], f'{height}.b{batch}',
    )
    try:
        parity = child.call('verify')
        warm = child.call('snapshot')
        repeats = max(1, math.ceil(args.target_ms / sum(warm['times_ms'])))
        rounds = []
        for index in range(args.rounds):
            row = {'gpu_before': protocol.gpu_state()}
            order = ('snapshot', 'buffer') if index % 2 == 0 else (
                'buffer', 'snapshot',
            )
            for mode in order:
                raw = child.call(mode, repeats)
                protocol.dump(
                    args.data / f'{height}.b{batch}.{index}.{mode}.json', raw,
                )
                row[mode] = protocol.stats(raw)
                row[mode]['snapshot_bytes'] = raw['snapshot_bytes']
                row[mode]['borrowed_frames'] = raw['borrowed_frames']
            row['gpu_after'] = protocol.gpu_state()
            rounds.append(row)
        result = {
            'height': height, 'width': height * 16 // 9, 'batch': batch,
            'parity': parity, 'rounds': rounds,
            'snapshot_ms': statistics.median(
                r['snapshot']['mean_ms_per_image'] for r in rounds
            ),
            'buffer_ms': statistics.median(
                r['buffer']['mean_ms_per_image'] for r in rounds
            ),
            'paired_speedup': statistics.median(
                r['snapshot']['mean_ms_per_image'] /
                r['buffer']['mean_ms_per_image']
                for r in rounds
            ),
            'snapshot_bytes_removed_per_frame': (
                height * (height * 16 // 9) * 3
            ),
            'backend_info': child.call('info'),
        }
        print(height, batch, result['paired_speedup'], flush=True)
        return result
    finally:
        child.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument(
        '--engine', type=Path, default=ROOT /
        'models/int8_engine/best_yolo26n.engine',
    )
    parser.add_argument(
        '--manifest', type=Path, default=ROOT /
        'experiments/rust_trt_benchmark/data/manifest.json',
    )
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--rounds', type=int, default=7)
    parser.add_argument('--target-ms', type=float, default=600)
    parser.add_argument('--core', type=int, default=4)
    parser.add_argument(
        '--data', type=Path, default=ROOT /
        'outputs/frame_buffer_benchmark',
    )
    parser.add_argument(
        '--output', type=Path, default=ROOT /
        'native/inference/frame_buffer_results.json',
    )
    args = parser.parse_args()
    if args.worker:
        child_main(args)
        return
    os.sched_setaffinity(0, {args.core})
    args.data.mkdir(parents=True, exist_ok=True)
    protocol.DATA = args.data
    result = {
        'scope': (
            'Rust adapter + YoloWorker._predict_batch; leased shared memory, '
            'one engine, alternating snapshot/buffer'
        ),
        'input': (
            'First 8 manifest frames resized to 16:9 720p/1080p; '
            'resize excluded from timing'
        ),
        'rect': False, 'imgsz': 640, 'python': sys.version,
        'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'engine_sha256': protocol.sha(args.engine),
        'manifest_sha256': protocol.sha(args.manifest),
        'source_sha256': {
            str(path.relative_to(ROOT)): protocol.sha(path)
            for path in (
                Path(__file__), ROOT / 'src/native_inference.py', ROOT /
                'src/yolo_worker.py',
                *sorted((ROOT / 'native/inference/src').glob('*.rs')),
            )
        },
        'cases': [],
    }
    for height in (720, 1080):
        for batch in (1, 8):
            result['cases'].append(run_case(args, height, batch))
            protocol.dump(args.output, result)


if __name__ == '__main__':
    main()
