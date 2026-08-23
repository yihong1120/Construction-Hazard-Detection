from __future__ import annotations

import asyncio
from dataclasses import dataclass
from dataclasses import field
from typing import TypedDict

import numpy as np

from examples.streaming_web.overlay_renderer import PolygonCollection
from examples.streaming_web.overlay_renderer import TrackingDetections
from examples.streaming_web.overlay_renderer import WarningPayload
from src.media_stream_publisher import MediaStreamPublisher


class StreamConfig(TypedDict, total=False):
    """Configuration for one video stream from the database."""

    stream_id: int
    video_url: str
    updated_at: str
    model_key: str
    site: str
    stream_name: str
    recognition_enabled: bool
    expire_date: str | None
    detection_items: dict[str, bool]
    work_start_hour: int
    work_end_hour: int


@dataclass
class LatestFrameState:
    """Latest camera frame shared by capture, detection, and publishing."""

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    event: asyncio.Event = field(default_factory=asyncio.Event)
    frame: np.ndarray | None = None
    timestamp: float = 0.0
    sequence: int = 0
    generation: int = 0


@dataclass
class LatestDetectionState:
    """Latest detection metadata used to render shared overlay variants."""

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    event: asyncio.Event = field(default_factory=asyncio.Event)
    frame: np.ndarray | None = None
    timestamp: float = 0.0
    sequence: int = 0
    warnings: WarningPayload = field(default_factory=dict)
    cone_polys: PolygonCollection = field(default_factory=list)
    pole_polys: PolygonCollection = field(default_factory=list)
    track_data: TrackingDetections | None = None


@dataclass(frozen=True)
class OverlaySnapshot:
    """Source frame and metadata used to render one overlay generation."""

    sequence: tuple[int, int]
    frame: np.ndarray
    warnings: WarningPayload = field(default_factory=dict)
    cone_polys: PolygonCollection = field(default_factory=list)
    pole_polys: PolygonCollection = field(default_factory=list)
    track_data: TrackingDetections | None = None


@dataclass
class OverlayPublisherVariant:
    """Publishers and demand state for one overlay media rendition."""

    media_path: str
    rendition: str
    publishers: dict[str, MediaStreamPublisher] = field(default_factory=dict)
    ready_started_at: dict[str, float] = field(default_factory=dict)


class PreviewPublisherKwargs(TypedDict):
    """Media publisher settings for a lower-bandwidth preview rendition."""

    fps: float
    width: int
    height: int
    bitrate: str
    maxrate: str
    bufsize: str
