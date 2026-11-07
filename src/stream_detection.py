from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime
from typing import cast

from examples.streaming_web.overlay_models import PolygonCollection
from examples.streaming_web.overlay_models import TrackingDetections
from examples.streaming_web.overlay_models import WarningPayload
from src.evidence import EvidenceMetadata
from src.image_utils import encode_frame
from src.model_publication_status import report_loaded_model
from src.redis_client import RedisManager
from src.runtime_utils import should_notify
from src.settings.media import MediaPublishSettings
from src.settings.media import WarningEventSettings
from src.stream_capture import StreamCapture
from src.stream_overlay_frames import _mark_frame_readonly
from src.stream_runtime_state import DetectionResult
from src.stream_runtime_state import (
    LatestDetectionState as _LatestDetectionState,
)
from src.stream_runtime_state import LatestFrameState as _LatestFrameState
from src.stream_runtime_state import StreamRuntime

logger = logging.getLogger(__name__)


async def _capture_latest_frames(
    streaming_capture: StreamCapture,
    latest_frame: _LatestFrameState,
    stop_event: asyncio.Event,
) -> None:
    """Continuously capture frames without waiting for YOLO."""
    source_fps = MediaPublishSettings().source_fps
    capture_interval = 1.0 / source_fps if source_fps else 0.0
    streaming_capture.update_capture_interval(capture_interval)
    async for frame, ts in streaming_capture.execute_capture():
        if stop_event.is_set():
            return
        _mark_frame_readonly(frame)
        async with latest_frame.lock:
            latest_frame.frame = frame
            latest_frame.timestamp = ts
            latest_frame.sequence += 1
            latest_frame.event.set()


def _capture_reconnect_event(
    streaming_capture: StreamCapture,
) -> asyncio.Event | None:
    """Return the CPU RTSP reconnect signal when the capture exposes one."""
    event = getattr(streaming_capture, 'reconnect_event', None)
    return event if isinstance(event, asyncio.Event) else None


async def _synchronise_capture_reconnects(
    reconnect_event: asyncio.Event,
    clean_reconnect_event: asyncio.Event,
    latest_frame: _LatestFrameState,
    latest_detection: _LatestDetectionState,
    stop_event: asyncio.Event,
) -> None:
    """Invalidate stale frames and notify the direct source restreamer."""
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(reconnect_event.wait(), timeout=1.0)
        except TimeoutError:
            continue
        reconnect_event.clear()
        async with latest_frame.lock:
            latest_frame.frame = None
            latest_frame.timestamp = 0.0
            latest_frame.sequence += 1
            latest_frame.generation += 1
            latest_frame.event.set()
        async with latest_detection.lock:
            latest_detection.frame = None
            latest_detection.timestamp = 0.0
            latest_detection.sequence = 0
            latest_detection.warnings.clear()
            latest_detection.cone_polys.clear()
            latest_detection.pole_polys.clear()
            latest_detection.track_data = None
            latest_detection.event.clear()
        clean_reconnect_event.set()


async def _detect_latest_frames(runtime: StreamRuntime) -> None:
    latest_frame = runtime.latest_frame
    yolo_detector = runtime.detector
    stop_event = runtime.stop
    site, stream_name = runtime.site, runtime.stream_name
    last_sequence = 0
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(latest_frame.event.wait(), timeout=1.0)
        except TimeoutError:
            continue

        async with latest_frame.lock:
            if (
                latest_frame.sequence == last_sequence
                or latest_frame.frame is None
            ):
                latest_frame.event.clear()
                continue
            frame = latest_frame.frame
            ts = latest_frame.timestamp
            last_sequence = latest_frame.sequence
            source_generation = latest_frame.generation
            latest_frame.event.clear()

        try:
            _datas, track_data = await yolo_detector.generate_detections(
                frame,
            )
        except Exception as exc:
            logger.info(
                f'[{site}:{stream_name}] Detection error, keeping stream '
                f'alive: {exc}',
            )
            await asyncio.sleep(1.0)
            continue
        async with latest_frame.lock:
            if (
                latest_frame.generation != source_generation
                or latest_frame.frame is None
            ):
                continue
        try:
            await _record_detection_result(
                runtime=runtime,
                result=DetectionResult(
                    frame,
                    ts,
                    last_sequence,
                    track_data,
                    getattr(yolo_detector, 'model_metadata', None),
                ),
            )
        except Exception as exc:
            logger.info(
                f'[{site}:{stream_name}] Metadata/notification error, keeping '
                f'stream alive: {exc}',
            )
            await asyncio.sleep(0.2)


async def _record_detection_result(
    runtime: StreamRuntime,
    result: DetectionResult,
) -> tuple[int, int | None]:
    frame, timestamp, sequence, track_data = (
        result.frame,
        result.timestamp,
        result.sequence,
        result.track_data,
    )
    model_metadata = result.model_metadata
    if model_metadata:
        await report_loaded_model(runtime, model_metadata)
    danger_detector, redis_manager = runtime.danger, runtime.redis
    latest_detection = runtime.latest_detection
    work_start_hour, work_end_hour = (
        runtime.cfg['work_start_hour'],
        runtime.cfg['work_end_hour'],
    )
    metadata_key = runtime.metadata_key
    last_notification_time = runtime.clock.last_notification_time
    last_warning_event_time = runtime.clock.last_warning_event_time
    warning_event_throttle_seconds = (
        runtime.clock.warning_event_throttle_seconds
    )
    detection_time = datetime.fromtimestamp(int(timestamp))
    is_working = work_start_hour <= detection_time.hour < work_end_hour
    current_timestamp = int(timestamp)
    warning_event_throttle_seconds = (
        warning_event_throttle_seconds
        if warning_event_throttle_seconds is not None
        else _warning_event_throttle_seconds()
    )
    warnings, cone_polys, pole_polys = danger_detector.detect_danger(
        cast(list[list[float]], track_data),
    )

    async with latest_detection.lock:
        latest_detection.frame = frame
        latest_detection.timestamp = timestamp
        latest_detection.sequence = sequence
        latest_detection.warnings = cast(WarningPayload, warnings)
        latest_detection.cone_polys = cast(PolygonCollection, cone_polys)
        latest_detection.pole_polys = cast(PolygonCollection, pole_polys)
        latest_detection.track_data = cast(TrackingDetections, track_data)
        latest_detection.class_metadata = (model_metadata or {}).get(
            'class_metadata',
        )
        latest_detection.event.set()

    should_send_violation = (
        is_working
        and bool(warnings)
        and should_notify(
            current_timestamp,
            last_notification_time,
        )
    )
    if is_working and _warning_event_due(
        warnings,
        current_timestamp,
        last_warning_event_time,
        warning_event_throttle_seconds,
    ):
        await _store_media_server_viewer_data(
            redis_manager=redis_manager,
            metadata_key=metadata_key,
            warnings=warnings,
        )
        last_warning_event_time = current_timestamp
        runtime.clock.last_warning_event_time = current_timestamp
    if should_send_violation:
        await queue_violation_event(
            runtime=runtime,
            result=replace(
                result,
                warnings=cast(WarningPayload, warnings),
                cone_polys=cast(PolygonCollection, cone_polys),
                pole_polys=cast(PolygonCollection, pole_polys),
            ),
        )
        last_notification_time = current_timestamp
    runtime.clock.last_notification_time = last_notification_time
    runtime.clock.last_warning_event_time = last_warning_event_time
    return last_notification_time, last_warning_event_time


async def queue_violation_event(
    runtime: StreamRuntime,
    result: DetectionResult,
) -> int:
    frame_bytes = await asyncio.to_thread(
        encode_frame,
        result.frame,
        'jpeg',
        85,
    )
    payload = {
        'site': runtime.site,
        'stream_name': runtime.stream_name,
        'site_id': runtime.cfg.get('site_id'),
        'stream_config_id': runtime.cfg.get('stream_id'),
        'detection_time': datetime.fromtimestamp(int(result.timestamp))
        .astimezone()
        .isoformat(),
        'warnings': result.warnings,
        'detections': result.track_data,
        'cone_polygon': result.cone_polys,
        'pole_polygon': result.pole_polys,
        **{
            key: value
            for key, value in (result.model_metadata or {}).items()
            if key in {'model_id', 'model_version'}
        },
    }
    await runtime.delivery.submit(
        EvidenceMetadata.from_legacy_payload(payload),
        frame_bytes,
    )
    return int(result.timestamp)


def _warning_event_throttle_seconds() -> int:
    """Return the minimum spacing between live warning metadata events."""
    return WarningEventSettings().throttle_seconds


def _warning_event_due(
    warnings: object,
    current_timestamp: int,
    last_warning_event_time: int | None,
    throttle_seconds: int,
) -> bool:
    """Return whether a warning event should be emitted to live viewers."""
    if not warnings:
        return False
    return (
        last_warning_event_time is None
        or current_timestamp - last_warning_event_time >= throttle_seconds
    )


async def _store_media_server_viewer_data(
    redis_manager: RedisManager,
    metadata_key: str,
    warnings: object,
) -> None:
    """Store one compact warning event for MediaMTX viewers."""
    if not warnings:
        return
    metadata: dict[str, str] = {
        'has_warning': '1',
    }
    event_id = await redis_manager.redis.xadd(
        metadata_key,
        metadata,
        maxlen=10,
    )
    warning_keys = (
        ','.join(sorted(str(key) for key in warnings))
        if isinstance(warnings, Mapping)
        else type(warnings).__name__
    )
    logger.info(
        (
            f'[Warning-Metadata] XADD {metadata_key} id={event_id} '
            f'has_warning=1 warnings={warning_keys}'
        ),
    )
