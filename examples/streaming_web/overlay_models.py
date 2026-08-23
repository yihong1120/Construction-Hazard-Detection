from __future__ import annotations

from dataclasses import dataclass
from typing import NotRequired
from typing import TypeAlias
from typing import TypedDict

import numpy as np


TrackingDetection: TypeAlias = list[float]
TrackingDetections: TypeAlias = list[TrackingDetection]
WarningBoundingBox: TypeAlias = list[float | int]


class WarningDetails(TypedDict):
    """One detector warning and its optional proximity evidence."""

    count: int
    person_bboxes: NotRequired[list[WarningBoundingBox]]
    person_track_ids: NotRequired[list[str]]


WarningPayload: TypeAlias = dict[str, WarningDetails]
PolygonCoordinates: TypeAlias = list[list[float]]
PolygonCollection: TypeAlias = list[PolygonCoordinates]


@dataclass(frozen=True)
class DetectionOverlay:
    """One normalised detection ready for an overlay drawing backend."""

    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]
    track_id: str | None = None
    is_warning: bool = False


@dataclass(frozen=True)
class RenderedTextBitmap:
    """Cached multilingual text pixels ready for ROI alpha blending."""

    bgra: np.ndarray
    width: int
    height: int
