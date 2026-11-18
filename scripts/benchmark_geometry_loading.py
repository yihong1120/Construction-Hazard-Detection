"""Compare stream startup and safety outputs with a source snapshot."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLAGS = (
    'detect_no_safety_vest_or_helmet',
    'detect_near_machinery_or_vehicle',
    'detect_in_restricted_area',
    'detect_in_utility_pole_restricted_area',
    'detect_machinery_close_to_pole',
)


def memory_kib():
    values = {}
    for line in Path('/proc/self/smaps_rollup').read_text().splitlines():
        parts = line.split()
        if parts[0] in {'Rss:', 'Pss:', 'Private_Clean:', 'Private_Dirty:'}:
            values[parts[0][:-1]] = int(parts[1])
    values['Private'] = values['Private_Clean'] + values['Private_Dirty']
    return values


def scene_cases():
    randomizer = random.Random(32)
    scenes = [[], [[0, 0, 10, 10, .9, 2, 0, 0]]]
    for count in (1, 2, 3, 8, 16, 32):
        for _ in range(4):
            rows = []
            for track in range(count):
                x, y = randomizer.randrange(300), randomizer.randrange(300)
                w, h = randomizer.randrange(1, 50), randomizer.randrange(1, 60)
                rows.append([
                    x, y, x + w, y + h, .9,
                    randomizer.choice((2, 4, 5, 6, 8, 9, 10)), track, 0,
                ])
            scenes.append(rows)
    return scenes


def worker(args):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(args.source_root.resolve()))
    started = time.perf_counter()
    import src.stream_processor  # noqa: F401 (measure full stream imports)
    from src.native_postprocess import load_postprocess_backend
    from src.native_postprocess import RustDangerDetector

    extension = load_postprocess_backend('rust')
    imported = time.perf_counter()
    if args.parity:
        results = []
        for combination in itertools.product((False, True), repeat=5):
            detector = RustDangerDetector(
                dict(zip(FLAGS, combination, strict=True)),
                native_extension=extension,
            )
            for scene in scene_cases():
                results.append(detector.detect_danger(scene))
        encoded = json.dumps(results, sort_keys=True).encode()
        return {
            'cases': len(results),
            'sha256': hashlib.sha256(encoded).hexdigest(),
        }
    flags = dict.fromkeys(FLAGS, False)
    flags[FLAGS[0]] = True
    flags[FLAGS[1]] = True
    detectors = [
        RustDangerDetector(flags, native_extension=extension)
        for _ in range(32)
    ]
    for detector in detectors:
        detector.detect_danger([
            [0, 0, 10, 20, .9, 5, 0, 0],
            [0, 0, 10, 20, .9, 2, 1, 0],
        ])
    return {
        'import_seconds': imported - started,
        'ready_seconds': time.perf_counter() - started,
        'memory_kib': memory_kib(),
        'loaded': {
            name: name in sys.modules
            for name in ('shapely', 'scipy', 'sklearn', 'torch')
        },
    }


def child(source_root, parity=False):
    command = [
        sys.executable, str(Path(__file__).resolve()), '--worker',
        '--source-root', str(source_root),
    ]
    if parity:
        command.append('--parity')
    result = subprocess.run(
        command, check=True,
        capture_output=True, text=True,
    )
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--parity', action='store_true')
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--core', type=int, default=4)
    parser.add_argument(
        '--output', type=Path,
        default=ROOT / 'native/inference/geometry_loading_results.json',
    )
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args)))
        return
    if args.baseline is None:
        parser.error(
            '--baseline requires the source snapshot before the changes',
        )
    os.sched_setaffinity(0, {args.core})
    roots = {'before': args.baseline, 'after': ROOT}
    raw = {name: [] for name in roots}
    for index in range(args.rounds):
        order = list(roots) if index % 2 == 0 else list(reversed(roots))
        for name in order:
            raw[name].append(child(roots[name]))
    parity = {name: child(root, parity=True) for name, root in roots.items()}
    assert parity['before'] == parity['after'], parity
    result = {
        'scope': (
            'Fresh src.stream_processor import + 32 Rust PPE/proximity '
            'detectors; no cameras started'
        ),
        'python': sys.version, 'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'parity': parity, 'rounds': raw,
        'medians': {
            name: {
                'ready_seconds': statistics.median(
                    row['ready_seconds'] for row in rows
                ),
                'pss_kib': statistics.median(
                    row['memory_kib']['Pss'] for row in rows
                ),
                'private_kib': statistics.median(
                    row['memory_kib']['Private'] for row in rows
                ),
            } for name, rows in raw.items()
        },
        'source_sha256': {
            name: {
                str(path.relative_to(root)): hashlib.sha256(
                    path.read_bytes(),
                ).hexdigest()
                for path in (
                    root / 'src/danger_detector.py', root / 'src/geometry.py',
                )
            } for name, root in roots.items()
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['medians'], indent=2))


if __name__ == '__main__':
    main()
