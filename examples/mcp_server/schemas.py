from __future__ import annotations

from typing import TypeAlias
from typing import TypedDict

from src.warning_types import MutableWarnings


# Note: Avoid providing a catch-all JSON alias here to encourage
# using precise TypedDicts for request/response shapes where feasible.


class TransportConfig(TypedDict):
    """MCP transport configuration."""

    transport: str
    host: str
    port: int
    path: str
    sse_path: str
    debug: bool


# Detection shapes
FloatBBox: TypeAlias = list[float]
PolygonsCoords: TypeAlias = list[list[list[float]]]


class DetectionLikeDict(TypedDict, total=False):
    """Dictionary-shaped detection accepted by MCP tools."""

    bbox: list[float]
    box: list[float]
    confidence: float
    conf: float
    class_: int  # use alias below when reading 'class'
    cls: int


class InferenceMeta(TypedDict):
    """Metadata returned with an inference response."""

    model_key: str
    engine: str
    tracker: str
    confidence_threshold: float
    track_objects: bool
    frame_size: list[int]  # [width, height]


class InferenceResponse(TypedDict):
    """YOLO inference response payload."""

    detections: list[FloatBBox]
    tracked: list[list[float]]
    meta: InferenceMeta


class HazardMeta(TypedDict):
    """Metadata returned with a hazard response."""

    image_width: int | None
    image_height: int | None
    working_hour_only: bool | None
    site_config_provided: bool


class HazardResponse(TypedDict):
    """Safety hazard analysis response payload."""

    warnings: MutableWarnings
    cone_polygons: PolygonsCoords
    pole_polygons: PolygonsCoords
    meta: HazardMeta
