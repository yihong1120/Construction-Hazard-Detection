from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

from examples.shared.overlay_geometry import clip_pixel_bbox
from examples.violation_records import violation_responses as module


def test_invalid_historical_rows_keep_original_ids_and_feedback_targets():
    raw = json.dumps(
        [
            [None, 0, 10, 10, 0.9, 0, 1],
            [10, 10, 20, 20, 0.9, 0, 2],
            [30, 30, 40, 40, 0.8, 0, -1],
        ],
    )
    parsed = module.decode_indexed_detections(raw)
    assert [row.object_id for row in parsed] == ['det_1', 'det_2']
    detections = [module.feedback_detection_from_item(row) for row in parsed]
    feedback = SimpleNamespace(
        target_detection_id='det_1',
        type='false_positive',
        note='wrong detection',
    )
    overlays = module.overlay_objects_from_feedback(
        detections, [feedback], (100, 100),
    )
    assert [(obj.object_id, obj.is_flagged) for obj in overlays] == [
        ('det_1', True),
        ('det_2', False),
    ]
    assert module.feedback_detection_ids_from_json(raw) == {
        'det_1',
        'det_2',
        '2',
    }
    assert 'det_0' not in module.feedback_detection_ids_from_json(raw)


@pytest.mark.parametrize(
    ('box', 'expected'),
    [
        ([90, 10, 110, 30], (0.9, 0.1, 0.1, 0.2)),
        ([-10, 10, 10, 30], (0, 0.1, 0.1, 0.2)),
        ([10, -10, 30, 10], (0.1, 0, 0.2, 0.1)),
        ([10, 90, 30, 110], (0.1, 0.9, 0.2, 0.1)),
    ],
)
def test_historical_boxes_clip_endpoints_before_computing_extent(
    box, expected,
):
    result = module.bbox_to_normalized(box, (100, 100))
    assert (result.x, result.y, result.w, result.h) == pytest.approx(expected)
    assert result.x + result.w <= 1 and result.y + result.h <= 1
    clipped = clip_pixel_bbox(box, (100, 100))
    assert result.w == pytest.approx((clipped[2] - clipped[0]) / 100)


@pytest.mark.parametrize(
    'box',
    [
        [110, 10, 120, 30],
        [-20, 10, -10, 30],
        [10, 10, 10, 20],
        [1, 1, 0, 0],
        [math.nan, 0, 10, 10],
        [0, 0, math.inf, 10],
    ],
)
def test_degenerate_offscreen_or_nonfinite_boxes_are_not_rendered(box):
    assert module.bbox_to_normalized(box, (100, 100)) is None
