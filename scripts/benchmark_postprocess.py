"""Compare production Python and Rust bindings on saved detection frames."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import time
from pathlib import Path

from src.danger_detector import DangerDetector
from src.native_postprocess import load_postprocess_backend
from src.native_postprocess import NativeExtension
from src.native_postprocess import RustDangerDetector
from src.native_postprocess import RustYoloDetector
from src.yolo_detector import YoloDetector

FLAGS = {
    'detect_no_safety_vest_or_helmet': True,
    'detect_near_machinery_or_vehicle': True,
    'detect_in_restricted_area': False,
    'detect_in_utility_pole_restricted_area': False,
    'detect_machinery_close_to_pole': False,
}


def make_pipeline(extension: NativeExtension | None):
    if extension is None:
        return YoloDetector(), DangerDetector(FLAGS)
    return (
        RustYoloDetector(native_extension=extension),
        RustDangerDetector(FLAGS, native_extension=extension),
    )


def verify(frames: list, extension: NativeExtension) -> None:
    reference, reference_rules = make_pipeline(None)
    native, native_rules = make_pipeline(extension)
    for index, frame in enumerate(frames):
        expected = reference.track_detections(frame)
        actual = native.track_detections(frame)
        if expected != actual:
            raise AssertionError(f'Tracking differs at frame {index}')
        expected_warnings = reference_rules.detect_danger(expected)
        if expected_warnings != native_rules.detect_danger(actual):
            raise AssertionError(f'Warnings/polygons differ at frame {index}')


def measure(
    frames: list, extension: NativeExtension | None, repeats: int,
) -> float:
    elapsed = 0
    for _ in range(repeats):
        detector, rules = make_pipeline(extension)
        started = time.perf_counter_ns()
        for frame in frames:
            rules.detect_danger(detector.track_detections(frame))
        elapsed += time.perf_counter_ns() - started
    return elapsed / (repeats * len(frames) * 1_000_000)


def benchmark(
    case: dict, extension: NativeExtension, rounds: int, target_ms: float,
):
    frames = case['frames']
    if not frames:
        raise ValueError('Each case needs at least one frame')
    verify(frames, extension)
    calibration = measure(frames, None, 1)
    repeats = max(
        1, min(5000, math.ceil(target_ms / (calibration * len(frames)))),
    )
    measure(frames, extension, 2)
    samples = []
    for index in range(rounds):
        row = {}
        order = ('python', 'rust') if index % 2 == 0 else ('rust', 'python')
        for backend in order:
            gc.collect()
            row[backend] = measure(
                frames, extension if backend == 'rust' else None, repeats,
            )
        samples.append(row)
    ratios = [row['python'] / row['rust'] for row in samples]
    result = {
        'name': case['name'], 'frames': len(frames), 'repeats': repeats,
        'python_ms': statistics.median(row['python'] for row in samples),
        'rust_ms': statistics.median(row['rust'] for row in samples),
        'median_paired_speedup': statistics.median(ratios),
        'paired_speedup_range': [min(ratios), max(ratios)],
        'rounds_ms_per_frame': samples,
    }
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=7)
    parser.add_argument('--target-ms', type=float, default=250.0)
    parser.add_argument('--core', type=int)
    args = parser.parse_args()
    if (
        args.rounds < 1 or not math.isfinite(args.target_ms)
        or args.target_ms <= 0
    ):
        parser.error('rounds and target-ms must be positive')
    if args.core is not None:
        os.sched_setaffinity(0, {args.core})
    extension = load_postprocess_backend('rust')
    assert extension is not None
    corpus = json.loads(args.corpus.read_text())
    result = {
        'python': platform.python_version(), 'platform': platform.platform(),
        'corpus_sha256': hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        'source_sha256': source_hashes(),
        'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'scope': (
            'centroid + normalization + PPE/proximity, including PyO3 calls'
        ),
        'excludes': (
            'inference, video I/O, polygons, encoding, storage, notifications'
        ),
        'cases': [
            benchmark(case, extension, args.rounds, args.target_ms)
            for case in corpus['cases']
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n',
    )


def source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = (
        'src/yolo_detector.py', 'src/danger_detector.py',
        'src/native_postprocess.py', 'native/postprocess-core/src/lib.rs',
        'bindings/python/postprocess/src/lib.rs',
        'bindings/python/postprocess/Cargo.lock',
        'scripts/benchmark_postprocess.py',
    )
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in paths
    }


if __name__ == '__main__':
    main()
