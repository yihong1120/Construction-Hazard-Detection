from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
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
    display_name: str | None = None
    color: str | None = None


@dataclass(frozen=True)
class RenderedTextBitmap:
    """Cached multilingual text pixels ready for ROI alpha blending."""

    bgra: np.ndarray
    width: int
    height: int


@dataclass(frozen=True)
class OverlaySnapshot:
    """Source frame and metadata used to render one overlay generation."""

    sequence: tuple[int, int]
    frame: np.ndarray
    warnings: WarningPayload = field(default_factory=dict)
    cone_polys: PolygonCollection = field(default_factory=list)
    pole_polys: PolygonCollection = field(default_factory=list)
    track_data: TrackingDetections | None = None
    class_metadata: list[dict] | None = None
