from __future__ import annotations

import asyncio
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import TypedDict

import numpy as np

from examples.streaming_web.overlay_models import (
    OverlaySnapshot as OverlaySnapshot,
)
from examples.streaming_web.overlay_models import PolygonCollection
from examples.streaming_web.overlay_models import TrackingDetections
from examples.streaming_web.overlay_models import WarningPayload
from src.media_stream_publisher import MediaStreamPublisher
from src.settings.media import MediaPublishSettings

if TYPE_CHECKING:
    from src.stream_capture import StreamCapture
    from src.yolo_detector import YoloDetector
    from src.danger_detector import DangerDetector
    from src.redis_client import RedisManager
    from src.violation_delivery import ViolationQueue


class StreamConfig(TypedDict, total=False):
    """Configuration for one video stream from the database."""

    stream_id: int
    site_id: int
    video_url: str
    updated_at: str
    model_key: str
    model_revision: str
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
    class_metadata: list[dict] | None = None


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


@dataclass(frozen=True)
class DetectionResult:
    frame: np.ndarray
    timestamp: float
    sequence: int
    track_data: object
    model_metadata: dict | None = None
    warnings: WarningPayload = field(default_factory=dict)
    cone_polys: PolygonCollection = field(default_factory=list)
    pole_polys: PolygonCollection = field(default_factory=list)


@dataclass
class DetectionClock:
    last_notification_time: int = 0
    last_warning_event_time: int | None = None
    warning_event_throttle_seconds: int = 30
    last_model_status_at: float = 0.0


@dataclass(frozen=True)
class StreamMediaSettings:
    path: str
    settings: MediaPublishSettings = field(
        default_factory=MediaPublishSettings,
    )

    @property
    def publish_base(self) -> str:
        return self.settings.publish_base

    @property
    def clean(self) -> bool:
        return self.settings.clean

    @property
    def overlay(self) -> bool:
        return self.settings.overlay

    @property
    def restream_clean(self) -> bool:
        return self.settings.restream_clean


@dataclass
class StreamRuntime:
    cfg: StreamConfig
    capture: StreamCapture
    detector: YoloDetector
    danger: DangerDetector
    redis: RedisManager
    delivery: ViolationQueue
    metadata_key: str
    media: StreamMediaSettings
    latest_frame: LatestFrameState = field(default_factory=LatestFrameState)
    latest_detection: LatestDetectionState = field(
        default_factory=LatestDetectionState,
    )
    stop: asyncio.Event = field(default_factory=asyncio.Event)
    clock: DetectionClock = field(default_factory=DetectionClock)

    @property
    def site(self) -> str:
        return self.cfg['site']

    @property
    def stream_name(self) -> str:
        return self.cfg['stream_name']


__all__ = [
    'DetectionClock',
    'DetectionResult',
    'LatestFrameState',
    'LatestDetectionState',
    'OverlayPublisherVariant',
    'OverlaySnapshot',
    'PreviewPublisherKwargs',
    'StreamConfig',
    'StreamMediaSettings',
    'StreamRuntime',
]
