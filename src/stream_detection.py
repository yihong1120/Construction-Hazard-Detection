"""Capture/inference loops and persisted warning events for one camera."""
from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Mapping
from datetime import datetime
from typing import cast

import numpy as np

from examples.streaming_web.overlay_models import PolygonCollection
from examples.streaming_web.overlay_models import TrackingDetections
from examples.streaming_web.overlay_models import WarningPayload
from src.danger_detector import DangerDetector
from src.image_utils import encode_frame
from src.notifiers.fcm_notifier import FCMSender
from src.redis_client import RedisManager
from src.runtime_utils import should_notify
from src.stream_capture import StreamCapture
from src.stream_overlay_frames import _mark_frame_readonly
from src.stream_runtime_state import (
    LatestDetectionState as _LatestDetectionState,
)
from src.stream_runtime_state import LatestFrameState as _LatestFrameState
from src.violation_sender import ViolationSender
from src.yolo_detector import YoloDetector

logger = logging.getLogger(__name__)
_default_warning_event_throttle_seconds = 30


async def _capture_latest_frames(
    streaming_capture: StreamCapture,
    latest_frame: _LatestFrameState,
    stop_event: asyncio.Event,
) -> None:
    """Continuously capture frames without waiting for YOLO."""
    capture_interval = 1.0 / max(
        1.0,
        float(
            os.getenv(
                'MEDIA_PUBLISH_SOURCE_FPS',
                os.getenv('MEDIA_PUBLISH_FPS', '15.0'),
            ),
        ),
    )
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
        except asyncio.TimeoutError:
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


async def _detect_latest_frames(
    latest_frame: _LatestFrameState,
    yolo_detector: YoloDetector,
    danger_detector: DangerDetector,
    fcm_sender: FCMSender,
    violation_sender: ViolationSender,
    redis_manager: RedisManager,
    latest_detection: _LatestDetectionState,
    site: str,
    stream_name: str,
    work_start_hour: int,
    work_end_hour: int,
    metadata_key: str,
    stop_event: asyncio.Event,
) -> None:
    """Run YOLO and publish overlays on the same frame that was detected."""
    last_sequence = 0
    last_notification_time = 0
    last_warning_event_time: int | None = None
    warning_event_throttle_seconds = _warning_event_throttle_seconds()
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(latest_frame.event.wait(), timeout=1.0)
        except asyncio.TimeoutError:
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
            (
                last_notification_time,
                last_warning_event_time,
            ) = await _record_detection_result(
                frame=frame,
                timestamp=ts,
                sequence=last_sequence,
                track_data=track_data,
                model_metadata=getattr(yolo_detector, 'model_metadata', None),
                danger_detector=danger_detector,
                fcm_sender=fcm_sender,
                violation_sender=violation_sender,
                redis_manager=redis_manager,
                latest_detection=latest_detection,
                site=site,
                stream_name=stream_name,
                work_start_hour=work_start_hour,
                work_end_hour=work_end_hour,
                metadata_key=metadata_key,
                last_notification_time=last_notification_time,
                last_warning_event_time=last_warning_event_time,
                warning_event_throttle_seconds=warning_event_throttle_seconds,
            )
        except Exception as exc:
            logger.info(
                f'[{site}:{stream_name}] Metadata/notification error, keeping '
                f'stream alive: {exc}',
            )
            await asyncio.sleep(0.2)


async def _record_detection_result(
    frame: np.ndarray,
    timestamp: float,
    sequence: int,
    track_data: object,
    danger_detector: DangerDetector,
    fcm_sender: FCMSender,
    violation_sender: ViolationSender,
    redis_manager: RedisManager,
    latest_detection: _LatestDetectionState,
    site: str,
    stream_name: str,
    work_start_hour: int,
    work_end_hour: int,
    metadata_key: str,
    last_notification_time: int,
    last_warning_event_time: int | None,
    warning_event_throttle_seconds: int | None = None,
    model_metadata: dict | None = None,
) -> tuple[int, int | None]:
    """Store tracking, warnings, and notifications for one detected frame."""
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
    if should_send_violation:
        last_notification_time = await _send_violation_and_notification(
            fcm_sender=fcm_sender,
            violation_sender=violation_sender,
            model_metadata=model_metadata,
            site=site,
            stream_name=stream_name,
            warnings=warnings,
            detection_time=detection_time,
            frame=frame,
            track_data=track_data,
            cone_polys=cone_polys,
            pole_polys=pole_polys,
            current_timestamp=current_timestamp,
        )
    return last_notification_time, last_warning_event_time


async def _send_violation_and_notification(
    fcm_sender: FCMSender,
    violation_sender: ViolationSender,
    site: str,
    stream_name: str,
    warnings: object,
    detection_time: datetime,
    frame: np.ndarray,
    track_data: object,
    cone_polys: object,
    pole_polys: object,
    current_timestamp: int,
    model_metadata: dict | None = None,
) -> int:
    """Persist one violation and notify subscribed site users."""
    frame_bytes = encode_frame(frame, 'jpeg', 85)
    violation_id_str = await violation_sender.send_violation(
        site=site,
        stream_name=stream_name,
        warnings=warnings,
        detection_time=detection_time,
        image_bytes=frame_bytes,
        detections=track_data,
        cone_polygon=cone_polys,
        pole_polygon=pole_polys,
        **{
            k: v
            for k, v in (model_metadata or {}).items()
            if k in {'model_id', 'model_version'}
        },
    )
    try:
        violation_id: int | None = (
            int(violation_id_str) if violation_id_str is not None else None
        )
    except Exception:
        violation_id = None

    await fcm_sender.send_fcm_message_to_site(
        site=site,
        stream_name=stream_name,
        message=cast(Mapping[str, Mapping[str, object]], warnings),
        image_path=None,
        violation_id=violation_id,
    )
    return current_timestamp


def _warning_event_throttle_seconds() -> int:
    """Return the minimum spacing between live warning metadata events."""
    raw_value = os.getenv(
        'WARNING_EVENT_THROTTLE_SECONDS',
        str(_default_warning_event_throttle_seconds),
    )
    try:
        return max(1, int(raw_value))
    except ValueError:
        return _default_warning_event_throttle_seconds


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
