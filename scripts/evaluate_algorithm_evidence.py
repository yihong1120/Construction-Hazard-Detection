from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from shapely.geometry import MultiPoint
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0,
    str(ROOT / 'source') if (ROOT / 'source/src').exists() else str(ROOT),
)
from src.danger_detector import DangerDetector  # noqa: E402
from src.geometry import calculate_people_in_controlled_area  # noqa: E402
from src.geometry import (  # noqa: E402
    detect_polygon_from_cones,
    polygons_to_coords,
)  # noqa: E402


class NoOccupantExclusion(DangerDetector):
    """Counterfactual: preserve every rule except the occupant predicate."""

    @staticmethod
    def _is_driver_detection(person, vehicle):
        return False


def csv_write(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def hull(points):
    return MultiPoint(points).convex_hull if points else None


def inside(geometry, point):
    return geometry is not None and geometry.contains(point)


def flags(data, detector, p):
    machines = [row for row in data if row[5] in {8, 10}]
    idx = detector._build_spatial_index(machines)
    return detector._person_proximity_flags(p, idx)


def confusion(rows, prediction_key):
    values = dict(tp=0, fp=0, tn=0, fn=0)
    for row in rows:
        truth, pred = bool(
            row['declared_expected_alert'],
        ), bool(row[prediction_key])
        values[
            (
                'tp'
                if truth and pred
                else 'fn' if truth else 'fp' if pred else 'tn'
            )
        ] += 1
    return values


def box(cx, cy, cls=6, w=2, h=2):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, 0.9, cls]


def synthetic_experiments():
    cone_cases, proximity_cases = [], []
    layouts = [(1, 0, 0), (1, 250, 80), (2, 0, 0)]
    points = [
        (x, y)
        for ox in [0, 1000]
        for x, y in [(ox, 0), (ox, 100), (ox + 100, 0), (ox + 100, 100)]
    ]
    samples = [
        ('left_inside', 50, 50, 20, 20, True),
        ('right_inside', 1050, 50, 20, 20, True),
        ('between_groups', 550, 50, 20, 20, False),
        ('outside', 550, 200, 20, 20, False),
        ('exact_boundary', 0, 50, 20, 20, False),
        ('bbox_overlaps_but_center_outside', 110, 50, 40, 20, False),
        ('feet_inside_center_outside', 50, -10, 20, 160, False),
        ('center_inside_feet_outside', 50, 50, 20, 200, True),
    ]
    for layout_id, (scale, tx, ty) in enumerate(layouts):
        cones = [box(x * scale + tx, y * scale + ty) for x, y in points]
        detector = DangerDetector()
        regions = detect_polygon_from_cones(cones, detector.clusterer)
        assert len(regions) == 2
        global_region = hull(
            [(x * scale + tx, y * scale + ty) for x, y in points],
        )
        for name, cx, cy, w, h, expected in samples:
            p = box(cx * scale + tx, cy * scale + ty, 5, w * scale, h * scale)
            point = Point((p[0] + p[2]) / 2, (p[1] + p[3]) / 2)
            current = bool(
                calculate_people_in_controlled_area(
                    regions,
                    cones + [p],
                ),
            )
            assert current == expected
            cone_cases.append(
                dict(
                    case=f"L{layout_id}_{name}",
                    kind='declared_synthetic_center_rule',
                    declared_expected_alert=expected,
                    current_alert=current,
                    single_all_cones_hull_alert=bool(
                        global_region.contains(point),
                    ),
                    single_clustered_points_hull_alert=bool(
                        global_region.contains(point),
                    ),
                    cone_count=8,
                    region_count=2,
                    scale=scale,
                    tx=tx,
                    ty=ty,
                    person_bbox=p[:4],
                ),
            )
    cones = [box(x, y) for x, y in points + [(5000, 5000)]]
    d = DangerDetector()
    regions = detect_polygon_from_cones(cones, d.clusterer)
    p = box(2500, 2500, 5, 20, 20)
    clustered = hull([v for region in regions for v in region.exterior.coords])
    cone_cases.append(
        dict(
            case='noise_outlier_bridge',
            kind='declared_synthetic_center_rule',
            declared_expected_alert=False,
            current_alert=bool(
                calculate_people_in_controlled_area(regions, cones + [p]),
            ),
            single_all_cones_hull_alert=bool(
                hull(points + [(5000, 5000)]).contains(Point(2500, 2500)),
            ),
            single_clustered_points_hull_alert=bool(
                clustered.contains(Point(2500, 2500)),
            ),
            cone_count=9,
            region_count=len(regions),
            scale=1,
            tx=0,
            ty=0,
            person_bbox=p[:4],
        ),
    )
    vehicle = [0, 0, 200, 200, 0.9, 10]
    examples = [
        (
            'occupant_inside',
            [80, 10, 100, 30, 0.9, 5],
            False,
            'synthetic_proximity_core',
            'Declared occupant; proximity warning is not desired.',
        ),
        (
            'outside_near_vehicle',
            [210, 150, 230, 190, 0.9, 5],
            True,
            'synthetic_proximity_core',
            'Declared exterior nearby person.',
        ),
        (
            'outside_far_vehicle',
            [600, 150, 620, 190, 0.9, 5],
            False,
            'synthetic_proximity_core',
            'Declared exterior far person.',
        ),
        (
            'exterior_with_occupant_like_projection',
            [
                90,
                12,
                114,
                36,
                0.9,
                5,
            ],
            True,
            'synthetic_projection_ambiguity',
            (
                'Hypothetical exterior person projected inside the vehicle '
                'bbox. Tests a limitation, not a measured real-world '
                'event.'
            ),
        ),
    ]
    for name, person, expected, scope, explanation in examples:
        current = flags([person, vehicle], DangerDetector(), person)
        baseline = flags([person, vehicle], NoOccupantExclusion(), person)
        proximity_cases.append(
            dict(
                case=name,
                scope=scope,
                kind='declared_synthetic_semantic_scenario',
                declared_expected_alert=expected,
                current_alert=bool(
                    current[1] or current[2],
                ),
                no_occupant_exclusion_alert=bool(baseline[1] or baseline[2]),
                occupant_excluded=bool(current[0]),
                person_bbox=person[:4],
                vehicle_bbox=vehicle[:4],
                explanation=explanation,
            ),
        )
    observations = []
    for name, pts in [
        ('single_square', [(0, 0), (0, 100), (100, 0), (100, 100)]),
        (
            'two_collinear_groups',
            [(x, 0) for x in [0, 10, 20, 30, 1000, 1010, 1020, 1030]],
        ),
    ]:
        ds = [box(x, y) for x, y in pts]
        detector = DangerDetector()
        regions = detect_polygon_from_cones(ds, detector.clusterer)
        observations.append(
            dict(
                probe=name,
                cone_count=len(ds),
                labels=detector.clusterer.labels_.tolist(),
                returned_types=[p.geom_type for p in regions],
                exported_coordinate_regions=len(polygons_to_coords(regions)),
                counted_people_with_center_15_0=(
                    calculate_people_in_controlled_area(
                        regions,
                        ds + [box(15, 0, 5)],
                    )
                ),
                status='OBSERVED_LIMITATION',
                note=(
                    'Default single-cluster selection can yield all noise.'
                    if name == 'single_square'
                    else (
                        'Collinear hulls are LineStrings: contains may count '
                        'a person although polygons_to_coords omits the '
                        'shapes.'
                    )
                ),
            ),
        )
    return cone_cases, proximity_cases, observations


def evaluate(dataset_path, output):
    output.mkdir(parents=True, exist_ok=True)
    dataset = json.loads(dataset_path.read_text())
    rows, scenes, pending_labels = [], [], []
    detector, baseline_detector = DangerDetector(), NoOccupantExclusion()
    for item in dataset['records']:
        data = item['detections']
        regions = detect_polygon_from_cones(data, detector.clusterer)
        cone_points = [
            ((d[0] + d[2]) / 2, (d[1] + d[3]) / 2) for d in data if d[5] == 6
        ]
        global_region = hull(cone_points)
        clustered_hull = hull(
            [v for region in regions for v in region.exterior.coords],
        )
        people = [p for p in data if p[5] == 5]
        persons_in_regions = 0
        for i, p in enumerate(people):
            point = Point((p[0] + p[2]) / 2, (p[1] + p[3]) / 2)
            current = bool(any(region.contains(point) for region in regions))
            persons_in_regions += int(current)
            a, b = flags(data, detector, p), flags(data, baseline_detector, p)
            archived = item['people'][i]
            assert p[:4] == archived['bbox']
            assert a == (
                archived['occupant_excluded'],
                archived['near_machinery_alert'],
                archived['near_vehicle_alert'],
            )
            assert current == archived['cone_alert']
            row = dict(
                scene=item['scene_id'],
                person=f"P{i + 1}",
                source_path=item['source']['path'],
                source_sha256=item['source_sha256'],
                formal_image=item['formal_image'],
                cx=point.x,
                cy=point.y,
                bbox=json.dumps(p[:4]),
                confidence=p[4],
                current_cone_alert=current,
                single_all_cones_hull_alert=bool(inside(global_region, point)),
                single_clustered_points_hull_alert=bool(
                    inside(clustered_hull, point),
                ),
                occupant_excluded=bool(a[0]),
                current_machinery_alert=bool(a[1]),
                current_vehicle_alert=bool(a[2]),
                current_any_proximity_alert=bool(a[1] or a[2]),
                no_exclusion_machinery_alert=bool(b[1]),
                no_exclusion_vehicle_alert=bool(b[2]),
                no_exclusion_any_proximity_alert=bool(b[1] or b[2]),
                independent_ground_truth_available=False,
            )
            rows.append(row)
            pending_labels.append(
                dict(
                    scene=item['scene_id'],
                    person=f"P{i + 1}",
                    source_path=item['source']['path'],
                    bbox=json.dumps(p[:4]),
                    ground_truth_cone_intrusion='',
                    ground_truth_occupant='',
                    ground_truth_proximity_alert='',
                    annotator='',
                    review_status='UNREVIEWED',
                    annotation_notes='',
                ),
            )
        assert (
            calculate_people_in_controlled_area(
                regions,
                data,
            )
            == persons_in_regions
        )
        scenes.append(
            dict(
                scene=item['scene_id'],
                source_path=item['source']['path'],
                source_sha256=item['source_sha256'],
                formal_image=item['formal_image'],
                width=item['size'][0],
                height=item['size'][1],
                cone_count=len(cone_points),
                region_count=len(regions),
                people_count=len(people),
                ground_truth_status='NO_INDEPENDENT_REGION_OR_OCCUPANT_LABELS',
            ),
        )
    summary = []
    for scope, chosen in [
        ('formal_11', [r for r in rows if r['formal_image']]),
        ('expanded_64_includes_formal', rows),
    ]:
        scene_count = (
            sum(s['formal_image'] for s in scenes)
            if scope == 'formal_11'
            else len(scenes)
        )
        for method, key in [
            ('current_clustered_regions', 'current_cone_alert'),
            ('single_all_cones_hull', 'single_all_cones_hull_alert'),
            (
                'single_clustered_points_hull',
                'single_clustered_points_hull_alert',
            ),
            (
                'current_proximity_with_exclusion',
                'current_any_proximity_alert',
            ),
            (
                'proximity_without_exclusion',
                'no_exclusion_any_proximity_alert',
            ),
        ]:
            summary.append(
                dict(
                    scope=scope,
                    method=method,
                    image_count=scene_count,
                    detected_person_boxes=len(
                        chosen,
                    ),
                    alert_person_boxes=sum(r[key] for r in chosen),
                    fp='NA',
                    fn='NA',
                    precision='NA',
                    recall='NA',
                    metric_status='UNAVAILABLE_NO_INDEPENDENT_GROUND_TRUTH',
                    unit=(
                        'One detected person bbox per source image; not '
                        'unique tracked '
                        'people.'
                    ),
                ),
            )
    csv_write(output / 'real_person_comparisons.csv', rows)
    csv_write(output / 'real_scene_conditions.csv', scenes)
    csv_write(output / 'real_comparison_summary.csv', summary)
    csv_write(output / 'manual_label_template.csv', pending_labels)
    changed = [
        r
        for r in rows
        if r['current_cone_alert'] != r['single_clustered_points_hull_alert']
        or r['current_any_proximity_alert']
        != r['no_exclusion_any_proximity_alert']
    ]
    csv_write(output / 'changed_decisions.csv', changed)
    c_cases, p_cases, observations = synthetic_experiments()
    csv_write(output / 'synthetic_cone_cases.csv', c_cases)
    csv_write(output / 'synthetic_proximity_cases.csv', p_cases)
    (output / 'edge_case_observations.json').write_text(
        json.dumps(observations, indent=2) + '\n',
    )
    csv_write(output / 'edge_case_observations.csv', observations)
    synthetic_summary = []
    for method in [
        'current_alert',
        'single_all_cones_hull_alert',
        'single_clustered_points_hull_alert',
    ]:
        synthetic_summary.append(
            dict(
                scope='synthetic_cone_25',
                method=method,
                samples=len(c_cases),
                truth_kind=(
                    'DECLARED_SYNTHETIC_RULE_LABELS_NOT_FIELD_GROUND_TRUTH'
                ),
                **confusion(c_cases, method),
            ),
        )
    for scope in [
        'synthetic_proximity_core',
        'synthetic_projection_ambiguity',
    ]:
        subset = [r for r in p_cases if r['scope'] == scope]
        for method in ['current_alert', 'no_occupant_exclusion_alert']:
            synthetic_summary.append(
                dict(
                    scope=scope,
                    method=method,
                    samples=len(subset),
                    truth_kind=(
                        'DECLARED_SYNTHETIC_SCENARIO_NOT_FIELD_GROUND_TRUTH'
                    ),
                    **confusion(subset, method),
                ),
            )
    csv_write(output / 'synthetic_confusion_counts.csv', synthetic_summary)
    report = dict(
        unique_images=len(scenes),
        detected_person_boxes=len(rows),
        formal_images=sum(s['formal_image'] for s in scenes),
        actual_comparisons=summary,
        synthetic_comparisons=synthetic_summary,
        edge_cases=observations,
        validation=(
            'Every current per-person flag matches the archived production '
            'results.'
        ),
        real_world_fp_fn_available=False,
    )
    (output / 'summary.json').write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
        + '\n',
    )
    return report


def main():
    parser = argparse.ArgumentParser()
    base = (
        ROOT
        if (ROOT / 'inputs/replay_dataset.json').exists()
        else ROOT / 'outputs/algorithm_evidence_20261022'
    )
    parser.add_argument(
        '--dataset',
        type=Path,
        default=base / 'inputs/replay_dataset.json',
    )
    parser.add_argument('--output', type=Path, default=base / 'recomputed')
    args = parser.parse_args()
    r = evaluate(args.dataset, args.output)
    print(
        json.dumps(
            dict(
                images=r['unique_images'],
                person_boxes=r['detected_person_boxes'],
                real_world_fp_fn_available=r['real_world_fp_fn_available'],
            ),
            indent=2,
        ),
    )


if __name__ == '__main__':
    main()
