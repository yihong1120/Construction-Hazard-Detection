from __future__ import annotations

from typing import Annotated

from pydantic import Field
from pydantic import FiniteFloat
from pydantic import model_validator
from pydantic import RootModel

DetectionNumber = Annotated[FiniteFloat, Field(strict=True)]
TrackedDetectionRow = Annotated[list[DetectionNumber], Field(min_length=7)]


class TrackedDetectionRows(RootModel[list[TrackedDetectionRow]]):
    """Tracked pixel boxes: x1, y1, x2, y2, confidence, class ID, track ID."""

    @model_validator(mode='after')
    def validate_rows(self):
        for row in self.root:
            x1, y1, x2, y2, confidence, class_id, track_id = row[:7]
            if x2 <= x1 or y2 <= y1:
                raise ValueError('Detection boxes must have positive area')
            if not 0 <= confidence <= 1:
                raise ValueError(
                    'Detection confidence must be between zero and one',
                )
            if class_id < 0 or not class_id.is_integer():
                raise ValueError(
                    'Detection class ID must be a nonnegative integer',
                )
            if track_id < -1 or not track_id.is_integer():
                raise ValueError('Detection track ID must be an integer >= -1')
        return self
