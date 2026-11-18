"""Compare the standalone Rust pipeline with the existing Python rules."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLAGS = (
    'detect_no_safety_vest_or_helmet',
    'detect_near_machinery_or_vehicle',
    'detect_in_restricted_area',
    'detect_in_utility_pole_restricted_area',
    'detect_machinery_close_to_pole',
)


def scenes():
    rng = random.Random(1720)
    result = [
        [], [[0, 0, 10, 20, .9, 2, 0, 0]], [
            [0, 0, 10, 20, .9, 4, 0, 0],
            [95, 90, 105, 100, .9, 5, 1, 0],
            [95, 0, 105, 100, .9, 9, 2, 0],
        ],
    ]
    for count in [1, 2, 3, 4, 8]:
        result.append([
            [x, 0, x + 20, 100, .9, 9, index, 0]
            for index, x in enumerate(range(0, count * 60, 60))
        ] + [[25, 80, 35, 100, .9, 5, 100, 0]])
    points = [
        (0, 0), (10, 0), (0, 10), (10, 10),
        (200, 200), (210, 200), (200, 210), (210, 210),
    ]
    for class_id in [6, 9]:
        result.append([
            [x - 1, y - 1, x + 1, y + 1, .9, class_id, index, 0]
            for index, (x, y) in enumerate(points)
        ] + [[4, 4, 6, 6, .9, 5, 100, 0]])
    result.append([
        [0, 0, 10, 100, .9, 9, 0, 0],
        [0, 40, 10, 100, .9, 9, 1, 0],
        [100, 20, 110, 100, .9, 9, 2, 0],
        [200, 0, 210, 100, .9, 9, 3, 0],
        [54, 98, 56, 100, .9, 5, 4, 0],
    ])
    for _ in range(30):
        rows = []
        for index in range(24):
            x, y = rng.randrange(500), rng.randrange(300)
            width, height = rng.randrange(5, 100), rng.randrange(5, 180)
            rows.append([
                x, y, x + width, y + height, .9,
                rng.choice([2, 4, 5, 5, 6, 8, 9, 10]), index, index % 2,
            ])
        result.append(rows)
    return result


def rules_parity(binary, output):
    from src.danger_detector import DangerDetector
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    inputs, expected = [], []
    for values in itertools.product([False, True], repeat=5):
        flags = dict(zip(FLAGS, values, strict=True))
        detector = DangerDetector(flags)
        for scene in scenes():
            inputs.append({'tracked': scene, 'rules': flags, 'reset': True})
            warnings, cones, poles = detector.detect_danger(scene)
            expected.append({
                'warnings': warnings, 'cone_polys': cones, 'pole_polys': poles,
            })
    source = output / 'rules-input.jsonl'
    target = output / 'rules-output.jsonl'
    source.write_text(''.join(json.dumps(row) + '\n' for row in inputs))
    subprocess.run(
        [
            str(binary), 'analyze', '--input', str(source),
            '--output', str(target),
        ], check=True,
    )
    actual = [json.loads(line) for line in target.read_text().splitlines()]
    assert len(expected) == len(actual)
    mismatches = []
    geometry_mismatches = []
    maximum_area_delta = 0.
    maximum_boundary_delta = 0.
    for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
        if left['warnings'] != right['warnings']:
            mismatches.append({
                'index': index, 'input': inputs[index],
                'expected': left['warnings'], 'actual': right['warnings'],
            })
        for key in ['cone_polys', 'pole_polys']:
            expected_polygon = unary_union([
                Polygon(points) for points in left[key] if len(points) >= 3
            ])
            actual_polygon = unary_union([
                Polygon(points) for points in right[key] if len(points) >= 3
            ])
            area = expected_polygon.symmetric_difference(actual_polygon).area
            boundary = 0. if expected_polygon.is_empty and (
                actual_polygon.is_empty
            ) else expected_polygon.hausdorff_distance(
                actual_polygon,
            )
            if expected_polygon.is_empty != actual_polygon.is_empty:
                boundary = float('inf')
            maximum_area_delta = max(maximum_area_delta, area)
            maximum_boundary_delta = max(maximum_boundary_delta, boundary)
            if area > .05 or boundary > .01:
                geometry_mismatches.append({
                    'index': index, 'kind': key,
                    'symmetric_difference_pixels_squared': area,
                    'hausdorff_pixels': boundary,
                })
    (output / 'rules-mismatches.json').write_text(
        json.dumps(mismatches, indent=2),
    )
    (output / 'geometry-mismatches.json').write_text(
        json.dumps(geometry_mismatches, indent=2),
    )
    return {
        'cases': len(expected), 'warning_mismatches': len(mismatches),
        'geometry_mismatches': len(geometry_mismatches),
        'maximum_polygon_area_delta_pixels_squared': maximum_area_delta,
        'maximum_polygon_hausdorff_pixels': maximum_boundary_delta,
        'geometry_tolerance': {
            'area_pixels_squared': .05, 'hausdorff_pixels': .01,
        },
    }


def tracking_parity(binary, output):
    from src.yolo_detector import YoloDetector

    inputs, expected, modes = [], [], []
    for mode in ['centroid', 'hungarian']:
        detector = YoloDetector(remote_tracker=mode)
        for frame in range(80):
            detections = [
                [
                    index * 55 + frame % 20, index * 20,
                    index * 55 + frame % 20 + 20, index * 20 + 40,
                    .9, [5, 8, 5, 10][index % 4],
                ]
                for index in range(8) if (index + frame) % 7
            ] if frame % 15 else []
            inputs.append({
                'detections': detections, 'tracker': mode, 'reset': frame == 0,
                'rules': dict.fromkeys(FLAGS, False),
            })
            expected.append(detector.track_detections(detections))
            modes.append(mode)
    source = output / 'tracking-input.jsonl'
    target = output / 'tracking-output.jsonl'
    source.write_text(''.join(json.dumps(row) + '\n' for row in inputs))
    subprocess.run(
        [
            str(binary), 'analyze', '--input', str(source),
            '--output', str(target),
        ], check=True,
    )
    actual = [
        json.loads(line)['tracks']
        for line in target.read_text().splitlines()
    ]
    mismatches = [
        {
            'index': index, 'mode': modes[index],
            'expected': left, 'actual': right,
        }
        for index, (left, right) in enumerate(
            zip(expected, actual, strict=True),
        )
        if left != right
    ]
    (output / 'tracking-mismatches.json').write_text(
        json.dumps(mismatches, indent=2),
    )
    return {'frames': len(expected), 'mismatches': len(mismatches)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--binary', type=Path, default=ROOT /
        'native/pipeline/target/release/construction-pipeline',
    )
    parser.add_argument(
        '--output', type=Path, default=ROOT / 'outputs/rust_pipeline_parity',
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ['PYTHON_DOTENV_DISABLED'] = '1'
    os.environ['POSTPROCESS_BACKEND'] = 'python'
    result = {
        'rules': rules_parity(args.binary, args.output),
        'tracking': tracking_parity(args.binary, args.output),
        'binary_sha256': hashlib.sha256(args.binary.read_bytes()).hexdigest(),
    }
    (args.output / 'results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    if (
        result['rules']['warning_mismatches']
        or result['rules']['geometry_mismatches']
        or result['tracking']['mismatches']
    ):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
