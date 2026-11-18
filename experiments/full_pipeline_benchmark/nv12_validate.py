"""Finite CPU/NV12 output-difference checks, separate from capacity timing."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from experiments.full_pipeline_benchmark.benchmark import RULES

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1]),
    )
    area = (a[2] - a[0]) * (a[3] - a[1])
    area += (b[2] - b[0]) * (b[3] - b[1])
    return intersection / max(area - intersection, 1e-12)


def compare(cpu, gpu):
    pairs = sorted(
        (
            (iou(a, b), i, j) for i, a in enumerate(cpu)
            for j, b in enumerate(gpu) if a[5] == b[5]
        ), reverse=True,
    )
    used_cpu, used_gpu, matches = set(), set(), []
    for overlap, i, j in pairs:
        if overlap < .5 or i in used_cpu or j in used_gpu:
            continue
        used_cpu.add(i)
        used_gpu.add(j)
        matches.append((cpu[i], gpu[j], overlap))
    return {
        'cpu_detections': len(cpu), 'gpu_detections': len(gpu),
        'matched_same_class_iou_05': len(matches),
        'unmatched_cpu': len(cpu) - len(matches),
        'unmatched_gpu': len(gpu) - len(matches),
        'max_matched_coordinate_error': max(
            (abs(a[k] - b[k]) for a, b, _ in matches for k in range(4)),
            default=0,
        ),
        'max_matched_confidence_error': max(
            (abs(a[4] - b[4]) for a, b, _ in matches), default=0,
        ),
        'track_id_differences': sum(a[6] != b[6] for a, b, _ in matches),
    }


def trial(args, cameras, backend):
    root = args.output / f'{backend}-{cameras}'
    root.mkdir(parents=True, exist_ok=False)
    engine = ROOT / 'models/int8_engine/best_yolo26n.engine'
    config = {
        'state_dir': str(root / 'state'), 'lossless': True,
        'max_frames': args.frames, 'record_detections': True,
        'batch_size': 8, 'preprocess_workers': 8,
        'gpu': {'asynchronous': True, 'pipeline_depth': 1},
        'models': [{
            'key': 'fixture', 'artifact': str(engine),
            'sha256': hashlib.sha256(engine.read_bytes()).hexdigest(),
            'version': 'isolated-nv12-validation',
        }],
        'streams': [{
            'id': f'camera-{i}', 'site_id': 1, 'site': 'nv12-validation',
            'name': f'camera-{i}', 'source': str(
                args.mixed_source
                if args.mixed_source and i % 2 else args.source,
            ),
            'model': 'fixture',
            'width': (
                args.mixed_width if args.mixed_source and i % 2 else args.width
            ),
            'height': (
                args.mixed_height
                if args.mixed_source and i % 2 else args.height
            ),
            'fps': 15, 'rules': RULES,
            'work_start_hour': 0, 'work_end_hour': 24,
            'output': {
                'clean': str(root / f'clean-{i}.mp4'),
                'annotated': str(root / f'annotated-{i}.mp4'),
            } if i < 4 else {},
        } for i in range(cameras)],
    }
    if backend == 'nv12':
        config['gpu']['direct_pinned'] = args.direct_pinned
        config['capture'] = {'host_yuv': True} if args.host_yuv else {
            'nvdec_device': 0,
        }
    path = root / 'config.json'
    path.write_text(json.dumps(config, indent=2))
    binary = args.binary if backend == 'nv12' else args.before_binary
    with (root / 'stdout.log').open('w') as out, (
        root / 'stderr.log'
    ).open('w') as err:
        subprocess.run(
            [str(binary), 'run', '--config', str(path)],
            cwd=ROOT, stdout=out, stderr=err, timeout=180, check=True,
        )
    report = read(root / 'state/summary.json')
    assert all(
        c['processed'] == args.frames and c['failures'] == 0
        for c in report['cameras']
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--before-binary', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--frames', type=int, default=16)
    parser.add_argument('--host-yuv', action='store_true')
    parser.add_argument('--direct-pinned', action='store_true')
    parser.add_argument('--mixed-source', type=Path)
    parser.add_argument('--mixed-width', type=int, default=1280)
    parser.add_argument('--mixed-height', type=int, default=720)
    parser.add_argument('--cameras', type=int, nargs='+', default=[1, 3, 32])
    args = parser.parse_args()
    if args.mixed_source:
        args.mixed_source = args.mixed_source.resolve()
    for name in ('binary', 'before_binary', 'source', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        'scope': 'Output differences, not labelled accuracy or capacity. '
        'Finite lossless video; 4 viewed cameras, five enabled hazard rules.',
        'binary_sha256': hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        'before_sha256': hashlib.sha256(
            args.before_binary.read_bytes(),
        ).hexdigest(),
        'trials': [],
    }
    for cameras in args.cameras:
        reports = {b: trial(args, cameras, b) for b in ('cpu', 'nv12')}
        differences = []
        for i in range(cameras):
            rows = {}
            for b in reports:
                p = args.output / f'{b}-{cameras}' / 'state'
                rows[b] = [
                    json.loads(line) for line in (
                        p / f'detections-camera-{i}.jsonl'
                    ).read_text().splitlines()
                ]
            for cpu, gpu in zip(rows['cpu'], rows['nv12'], strict=True):
                assert cpu['sequence'] == gpu['sequence']
                differences.append({
                    'camera': i, 'sequence': cpu['sequence'],
                    **compare(cpu['tracks'], gpu['tracks']),
                    'warnings_differ': cpu['warnings'] != gpu['warnings'],
                    'tracks_exact': cpu['tracks'] == gpu['tracks'],
                    'nv12_preprocess': gpu['inference_timing'][
                        'nv12_preprocess'
                    ],
                })
        result['trials'].append({
            'cameras': cameras, 'reports': reports,
            'frame_differences': differences,
        })
        (args.output / 'results.json').write_text(
            json.dumps(result, indent=2) + '\n',
        )
        print(json.dumps({'cameras': cameras, 'complete': True}), flush=True)


if __name__ == '__main__':
    main()
