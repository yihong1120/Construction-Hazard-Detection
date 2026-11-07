from __future__ import annotations

import pytest
from pydantic import ValidationError

from examples.violation_records.violation_responses import (
    decode_detection_items,
)
from src.detection_rows import TrackedDetectionRows


@pytest.mark.parametrize(
    'row',
    [
        [1, 2, 3, 4, 0.9, 0, float('nan')],
        [1, 2, float('inf'), 4, 0.9, 0, 1],
        [3, 2, 1, 4, 0.9, 0, 1],
        [1, 4, 3, 2, 0.9, 0, 1],
        [1, 2, 3, 4, 1.1, 0, 1],
        [1, 2, 3, 4, 0.9, 0.5, 1],
        [1, 2, 3, 4, 0.9, 0, 1.5],
        [1, 2, 3, 4, 0.9, -1, 1],
        [1, 2, 3, 4, 0.9, 0, -2],
    ],
)
def test_invalid_tracking_rows_are_rejected_before_serialization(row):
    with pytest.raises(ValidationError):
        TrackedDetectionRows.model_validate([row])


def test_unassigned_track_ids_and_clipped_coordinates_remain_supported():
    rows = TrackedDetectionRows.model_validate([[-1, 2, 3, 4, 0.9, 0, -1]])
    assert (
        TrackedDetectionRows.model_validate_json(rows.model_dump_json())
        == rows
    )


def test_corrupt_historical_rows_do_not_hide_the_evidence_record():
    assert decode_detection_items('[[1,2,3,4,0.9,0,null]]') is None
    assert decode_detection_items('not json') is None
    assert (
        len(decode_detection_items('[[1,2,3,4,0.9,0,null],[1,2,3,4,0.9,0,2]]'))
        == 1
    )
