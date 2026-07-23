from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from typing import cast
from typing import Final
from typing import TYPE_CHECKING
from typing import TypedDict

import cv2
import numpy as np
from dotenv import load_dotenv

from examples.streaming_web.media_paths import (
    build_annotated_media_path,
)
from examples.streaming_web.media_paths import build_clean_demand_key
from examples.streaming_web.media_paths import build_media_path
from examples.streaming_web.media_paths import build_overlay_ready_key
from examples.streaming_web.media_paths import build_preview_media_path
from examples.streaming_web.media_paths import CLEAN_DEMAND_PREFIX
from examples.streaming_web.media_paths import decode_media_segment
from examples.streaming_web.media_paths import OVERLAY_DEMAND_PREFIX
from examples.streaming_web.overlay_renderer import (
    normalise_label_language,
)
from examples.streaming_web.overlay_renderer import (
    render_overlay_array,
)
from examples.streaming_web.overlay_renderer import (
    SUPPORTED_LABEL_LANGUAGES,
)
from src.danger_detector import DangerDetector
from src.media_restreamer import MediaSourceRestreamer
from src.media_stream_publisher import MediaStreamPublisher
from src.notifiers.fcm_notifier import FCMSender
from src.stream_capture import StreamCapture
from src.utils import RedisManager
from src.utils import Utils
from src.violation_sender import ViolationSender
from src.yolo_detector import YoloDetector
from src.yolo_worker import ResultStore
from src.yolo_worker import WorkerQueue
from src.yolo_worker import YoloWorkerClient

if TYPE_CHECKING:
    RedisPrimitive = bytes | bytearray | memoryview[int] | str | int | float
else:
    RedisPrimitive = bytes | bytearray | memoryview | str | int | float

_default_warning_event_throttle_seconds: Final[int] = 30


class StreamConfig(TypedDict, total=False):
    """Configuration for one video stream from the database."""

    video_url: str
    updated_at: str
    model_key: str
    site: str
    stream_name: str
    detect_with_server: bool
    expire_date: str | None
    detection_items: dict[str, bool]
    work_start_hour: int
    work_end_hour: int
    store_in_redis: bool


@dataclass
class _LatestFrameState:
    """Latest camera frame shared by capture, detection, and publishing."""

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    event: asyncio.Event = field(default_factory=asyncio.Event)
    frame: np.ndarray | None = None
    timestamp: float = 0.0
    sequence: int = 0


@dataclass
class _LatestDetectionState:
    """Latest detection metadata used to render shared overlay variants."""

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    event: asyncio.Event = field(default_factory=asyncio.Event)
    frame: np.ndarray | None = None
    timestamp: float = 0.0
    sequence: int = 0
    warnings: object = None
    cone_polys: object = None
    pole_polys: object = None
    track_data: object = None


@dataclass(frozen=True)
class _OverlaySnapshot:
    """Source frame and metadata used to render one overlay generation."""

    sequence: int
    frame: np.ndarray
    warnings: object = None
    cone_polys: object = None
    pole_polys: object = None
    track_data: object = None


def _preview_publisher_kwargs() -> dict[str, object]:
    """Return the bounded encoder budget used by multi-camera walls."""
    return {
        'fps': max(
            1.0,
            float(
                os.getenv(
                    'MEDIA_PREVIEW_FPS',
                    os.getenv('MEDIA_PUBLISH_FPS', '15'),
                ),
            ),
        ),
        'width': max(2, int(os.getenv('MEDIA_PREVIEW_WIDTH', '640'))),
        'height': max(2, int(os.getenv('MEDIA_PREVIEW_HEIGHT', '360'))),
        'bitrate': os.getenv('MEDIA_PREVIEW_BITRATE', '500k'),
        'maxrate': os.getenv('MEDIA_PREVIEW_MAXRATE', '700k'),
        'bufsize': os.getenv('MEDIA_PREVIEW_BUFSIZE', '1400k'),
    }


def _media_publisher(
    publish_url: str,
    *,
    rendition: str,
) -> MediaStreamPublisher:
    """Create a publisher for exactly one detail or preview rendition."""
    if rendition == 'preview':
        return MediaStreamPublisher(
            publish_url=publish_url,
            **_preview_publisher_kwargs(),
        )
    if rendition == 'detail':
        return MediaStreamPublisher(publish_url=publish_url)
    raise ValueError(f'unsupported media rendition: {rendition}')


async def delete_stream_live_metadata(cfg: StreamConfig) -> None:
    """Delete compact live metadata for one configured camera."""
    redis_manager = RedisManager()
    await redis_manager.delete(
        _stream_metadata_key(cfg['site'], cfg['stream_name']),
    )


def process_single_stream(
    cfg: StreamConfig,
    yolo_request_queue: object | None = None,
    yolo_result_store: object | None = None,
) -> None:
    """Run one configured stream inside a child process."""
    load_dotenv(override=True)
    asyncio.run(
        _run_single_stream(
            cfg,
            yolo_request_queue=yolo_request_queue,
            yolo_result_store=yolo_result_store,
        ),
    )


async def _run_single_stream(
    cfg: StreamConfig,
    yolo_request_queue: object | None = None,
    yolo_result_store: object | None = None,
) -> None:
    """Run one stream processing coroutine for the given configuration."""
    video_url = cfg['video_url']
    model_key = cfg['model_key']
    site = cfg['site']
    stream_name = cfg['stream_name']
    detect_with_server = _resolve_detect_with_server(cfg['detect_with_server'])
    detection_items = cfg['detection_items']
    work_start_hour = cfg['work_start_hour']
    work_end_hour = cfg['work_end_hour']
    live_view_enabled = _live_view_enabled(cfg)
    print(
        f'[{site}:{stream_name}] Streaming output mode: media_server',
        flush=True,
    )
    print(
        f'[{site}:{stream_name}] Detection mode: server',
        flush=True,
    )

    streaming_capture = StreamCapture(stream_url=video_url)
    worker_client = (
        YoloWorkerClient(
            cast(WorkerQueue, yolo_request_queue),
            cast(ResultStore, yolo_result_store),
            camera_id=f'{site}|{stream_name}',
            timeout_seconds=float(
                os.getenv('YOLO_WORKER_TIMEOUT_SECONDS', '30.0'),
            ),
        )
        if (
            detect_with_server
            and os.getenv(
                'YOLO_WORKER_ENABLED',
                'true',
            ).strip().lower() in {'1', 'true', 'yes', 'on'}
            and yolo_request_queue is not None
            and yolo_result_store is not None
        )
        else None
    )

    yolo_detector = YoloDetector(
        model_key=model_key,
        output_folder=site,
        detect_with_server=detect_with_server,
        worker_client=worker_client,
    )
    if detect_with_server:
        _validate_server_model_key(model_key)
        if worker_client is None:
            raise RuntimeError(
                'YOLO_WORKER_ENABLED=true and shared worker queues are '
                'required for server detection mode',
            )
        print(
            f'[{site}:{stream_name}] YOLO detection worker enabled',
            flush=True,
        )
    danger_detector = DangerDetector(detection_items)
    fcm_sender = FCMSender(api_url=os.getenv('FCM_API_URL') or '')
    violation_sender = ViolationSender(
        api_url=os.getenv('VIOLATION_RECORD_API_URL') or '',
    )
    metadata_key = _stream_metadata_key(site, stream_name)
    media_path = build_media_path(site, stream_name)
    media_publish_base = os.getenv(
        'MEDIA_PUBLISH_RTSP_BASE_URL',
    ) or 'rtsp://media-server:8554'
    media_publish_base = media_publish_base.rstrip('/')
    publish_clean_stream = os.getenv(
        'MEDIA_PUBLISH_CLEAN_STREAM',
        'true',
    ).strip().lower() in {'1', 'true', 'yes', 'on'}
    publish_annotated_stream = os.getenv(
        'MEDIA_PUBLISH_ANNOTATED_STREAM',
        'true',
    ).strip().lower() in {'1', 'true', 'yes', 'on'}
    restream_clean_source = os.getenv(
        'MEDIA_PUBLISH_CLEAN_SOURCE_RESTREAM',
        'true',
    ).strip().lower() in {'1', 'true', 'yes', 'on'}
    clean_source_restreamer: MediaSourceRestreamer | None = None
    clean_media_publisher: MediaStreamPublisher | None = None
    overlay_media_publishers: dict[str, MediaStreamPublisher] = {}
    preview_clean_media_publisher: MediaStreamPublisher | None = None
    preview_overlay_media_publishers: dict[str, MediaStreamPublisher] = {}
    if live_view_enabled and publish_clean_stream:
        print(
            f'[{site}:{stream_name}] Clean media stream is on-demand; '
            f'path {media_path}',
            flush=True,
        )
    if live_view_enabled and publish_annotated_stream:
        print(
            f'[{site}:{stream_name}] Overlay media streams are on-demand; '
            f'base path {media_path}',
            flush=True,
        )
    redis_manager = RedisManager() if live_view_enabled else None

    try:
        if (
            live_view_enabled
            and redis_manager is not None
            and publish_annotated_stream
            and os.getenv(
                'MEDIA_PUBLISH_DECOUPLED_ANNOTATED',
                'true',
            ).strip().lower() in {'1', 'true', 'yes', 'on'}
        ):
            await _run_decoupled_media_server_loop(
                streaming_capture=streaming_capture,
                yolo_detector=yolo_detector,
                danger_detector=danger_detector,
                fcm_sender=fcm_sender,
                violation_sender=violation_sender,
                redis_manager=redis_manager,
                clean_media_publisher=clean_media_publisher,
                media_publish_base=media_publish_base,
                media_path=media_path,
                publish_overlay_streams=publish_annotated_stream,
                site=site,
                stream_name=stream_name,
                work_start_hour=work_start_hour,
                work_end_hour=work_end_hour,
                metadata_key=metadata_key,
                publish_clean_stream=publish_clean_stream,
                restream_clean_source=restream_clean_source,
                video_url=video_url,
            )
            return

        await _run_inline_stream_loop(
            streaming_capture=streaming_capture,
            yolo_detector=yolo_detector,
            danger_detector=danger_detector,
            fcm_sender=fcm_sender,
            violation_sender=violation_sender,
            redis_manager=redis_manager,
            clean_source_restreamer=clean_source_restreamer,
            clean_media_publisher=clean_media_publisher,
            overlay_media_publishers=overlay_media_publishers,
            preview_clean_media_publisher=preview_clean_media_publisher,
            preview_overlay_media_publishers=preview_overlay_media_publishers,
            media_publish_base=media_publish_base,
            media_path=media_path,
            publish_annotated_stream=publish_annotated_stream,
            live_view_enabled=live_view_enabled,
            site=site,
            stream_name=stream_name,
            work_start_hour=work_start_hour,
            work_end_hour=work_end_hour,
            metadata_key=metadata_key,
            publish_clean_stream=publish_clean_stream,
            restream_clean_source=restream_clean_source,
            video_url=video_url,
        )
    finally:
        await yolo_detector.close()
        await streaming_capture.release_resources()
        if clean_source_restreamer is not None:
            await clean_source_restreamer.close()
        if clean_media_publisher is not None:
            await clean_media_publisher.close()
        await _close_overlay_publishers(overlay_media_publishers)
        if preview_clean_media_publisher is not None:
            await preview_clean_media_publisher.close()
        await _close_overlay_publishers(preview_overlay_media_publishers)
        if redis_manager is not None:
            try:
                await redis_manager.delete(metadata_key)
            except Exception as e:
                print(f'[WARN] Failed to delete redis key {metadata_key}: {e}')


async def _run_inline_stream_loop(
    streaming_capture: StreamCapture,
    yolo_detector: YoloDetector,
    danger_detector: DangerDetector,
    fcm_sender: FCMSender,
    violation_sender: ViolationSender,
    redis_manager: RedisManager | None,
    clean_source_restreamer: MediaSourceRestreamer | None,
    clean_media_publisher: MediaStreamPublisher | None,
    overlay_media_publishers: dict[str, MediaStreamPublisher],
    media_publish_base: str,
    media_path: str,
    publish_annotated_stream: bool,
    live_view_enabled: bool,
    site: str,
    stream_name: str,
    work_start_hour: int,
    work_end_hour: int,
    metadata_key: str,
    publish_clean_stream: bool,
    restream_clean_source: bool,
    video_url: str,
    preview_clean_media_publisher: MediaStreamPublisher | None = None,
    preview_overlay_media_publishers: dict[
        str,
        MediaStreamPublisher,
    ] | None = None,
) -> None:
    """Process capture/detection in a single loop for one camera."""
    if preview_overlay_media_publishers is None:
        preview_overlay_media_publishers = {}
    last_notification_time = 0
    last_warning_event_time: int | None = None
    warning_event_throttle_seconds = _warning_event_throttle_seconds()
    overlay_ready_started_at: dict[str, float] = {}
    preview_overlay_ready_started_at: dict[str, float] = {}
    preview_media_path = build_preview_media_path(media_path)
    async for frame, ts in streaming_capture.execute_capture():
        detection_time = datetime.fromtimestamp(int(ts))
        is_working = work_start_hour <= detection_time.hour < work_end_hour
        current_timestamp = int(ts)

        if (
            live_view_enabled
            and redis_manager is not None
            and publish_annotated_stream
            and os.getenv(
                'MEDIA_PUBLISH_PRIME_ANNOTATED_STREAM',
                'true',
            ).strip().lower() in {'1', 'true', 'yes', 'on'}
        ):
            try:
                await _publish_requested_overlay_snapshot(
                    redis_manager=redis_manager,
                    overlay_media_publishers=overlay_media_publishers,
                    media_publish_base=media_publish_base,
                    media_path=media_path,
                    site=site,
                    stream_name=stream_name,
                    source_frame=frame,
                    warnings=None,
                    cone_polys=None,
                    pole_polys=None,
                    track_data=None,
                    overlay_ready_started_at=overlay_ready_started_at,
                    rendition='detail',
                )
                await _publish_requested_overlay_snapshot(
                    redis_manager=redis_manager,
                    overlay_media_publishers=preview_overlay_media_publishers,
                    media_publish_base=media_publish_base,
                    media_path=preview_media_path,
                    site=site,
                    stream_name=stream_name,
                    source_frame=frame,
                    warnings=None,
                    cone_polys=None,
                    pole_polys=None,
                    track_data=None,
                    overlay_ready_started_at=preview_overlay_ready_started_at,
                    rendition='preview',
                )
            except Exception as e:
                print(
                    f'[{site}:{stream_name}] Overlay stream prime error: {e}',
                    flush=True,
                )

        _datas, track_data = await yolo_detector.generate_detections(
            frame,
        )
        warnings, cone_polys, pole_polys = danger_detector.detect_danger(
            track_data,
        )
        warnings = Utils.filter_warnings_by_working_hour(warnings, is_working)

        should_send_violation = bool(warnings) and Utils.should_notify(
            current_timestamp,
            last_notification_time,
        )
        if live_view_enabled and redis_manager is not None and (
            clean_source_restreamer is not None
            or clean_media_publisher is not None
            or publish_annotated_stream
            or publish_clean_stream
        ):
            try:
                clean_requested = (
                    publish_clean_stream
                    and await _clean_stream_requested(
                        redis_manager,
                        media_path,
                    )
                )
                if clean_requested:
                    if clean_source_restreamer is None and (
                        restream_clean_source
                    ):
                        clean_source_restreamer = MediaSourceRestreamer(
                            source_url=video_url,
                            publish_url=f'{media_publish_base}/{media_path}',
                        )
                        print(
                            f'[{site}:{stream_name}] Starting clean source '
                            f'restream: {media_publish_base}/{media_path}',
                            flush=True,
                        )
                        await clean_source_restreamer.start()
                    elif (
                        clean_media_publisher is None
                        and clean_source_restreamer is None
                    ):
                        clean_media_publisher = _media_publisher(
                            f'{media_publish_base}/{media_path}',
                            rendition='detail',
                        )
                        print(
                            f'[{site}:{stream_name}] Starting clean stream: '
                            f'{media_publish_base}/{media_path}',
                            flush=True,
                        )
                    if clean_media_publisher is not None:
                        await clean_media_publisher.publish(frame)
                else:
                    if clean_source_restreamer is not None:
                        await clean_source_restreamer.close()
                        clean_source_restreamer = None
                    if clean_media_publisher is not None:
                        await clean_media_publisher.close()
                        clean_media_publisher = None

                preview_clean_requested = (
                    publish_clean_stream
                    and await _clean_stream_requested(
                        redis_manager,
                        preview_media_path,
                    )
                )
                if preview_clean_requested:
                    if preview_clean_media_publisher is None:
                        preview_clean_media_publisher = _media_publisher(
                            f'{media_publish_base}/{preview_media_path}',
                            rendition='preview',
                        )
                        print(
                            f'[{site}:{stream_name}] Starting preview clean '
                            f'stream: {media_publish_base}/'
                            f'{preview_media_path}',
                            flush=True,
                        )
                    await preview_clean_media_publisher.publish(frame)
                elif preview_clean_media_publisher is not None:
                    await preview_clean_media_publisher.close()
                    preview_clean_media_publisher = None
                if publish_annotated_stream:
                    await _publish_requested_overlay_snapshot(
                        redis_manager=redis_manager,
                        overlay_media_publishers=overlay_media_publishers,
                        media_publish_base=media_publish_base,
                        media_path=media_path,
                        site=site,
                        stream_name=stream_name,
                        source_frame=frame,
                        warnings=warnings,
                        cone_polys=cone_polys,
                        pole_polys=pole_polys,
                        track_data=track_data,
                        overlay_ready_started_at=overlay_ready_started_at,
                        rendition='detail',
                    )
                    await _publish_requested_overlay_snapshot(
                        redis_manager=redis_manager,
                        overlay_media_publishers=(
                            preview_overlay_media_publishers
                        ),
                        media_publish_base=media_publish_base,
                        media_path=preview_media_path,
                        site=site,
                        stream_name=stream_name,
                        source_frame=frame,
                        warnings=warnings,
                        cone_polys=cone_polys,
                        pole_polys=pole_polys,
                        track_data=track_data,
                        overlay_ready_started_at=(
                            preview_overlay_ready_started_at
                        ),
                        rendition='preview',
                    )
                if _warning_event_due(
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
            except Exception as e:
                print(f'[{site}:{stream_name}] Media publish error: {e}')

        if should_send_violation:
            last_notification_time = await _send_violation_and_notification(
                fcm_sender=fcm_sender,
                violation_sender=violation_sender,
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

        streaming_capture.update_capture_interval(0.2)

    if clean_source_restreamer is not None:
        await clean_source_restreamer.close()
    if clean_media_publisher is not None:
        await clean_media_publisher.close()
    if preview_clean_media_publisher is not None:
        await preview_clean_media_publisher.close()
    await streaming_capture.release_resources()


async def _run_decoupled_media_server_loop(
    streaming_capture: StreamCapture,
    yolo_detector: YoloDetector,
    danger_detector: DangerDetector,
    fcm_sender: FCMSender,
    violation_sender: ViolationSender,
    redis_manager: RedisManager,
    clean_media_publisher: MediaStreamPublisher | None,
    media_publish_base: str,
    media_path: str,
    publish_overlay_streams: bool,
    site: str,
    stream_name: str,
    work_start_hour: int,
    work_end_hour: int,
    metadata_key: str,
    publish_clean_stream: bool,
    restream_clean_source: bool,
    video_url: str,
) -> None:
    """Run capture, detection, and overlay publishing independently."""
    latest_frame = _LatestFrameState()
    latest_detection = _LatestDetectionState()
    stop_event = asyncio.Event()
    tasks = {
        asyncio.create_task(
            _capture_latest_frames(
                streaming_capture=streaming_capture,
                latest_frame=latest_frame,
                stop_event=stop_event,
            ),
        ),
        asyncio.create_task(
            _detect_latest_frames(
                latest_frame=latest_frame,
                yolo_detector=yolo_detector,
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
                stop_event=stop_event,
            ),
        ),
    }
    if publish_overlay_streams:
        tasks.add(
            asyncio.create_task(
                _publish_requested_overlay_frames(
                    latest_frame=latest_frame,
                    latest_detection=latest_detection,
                    redis_manager=redis_manager,
                    media_publish_base=media_publish_base,
                    media_path=media_path,
                    site=site,
                    stream_name=stream_name,
                    stop_event=stop_event,
                    rendition='detail',
                ),
            ),
        )
        tasks.add(
            asyncio.create_task(
                _publish_requested_overlay_frames(
                    latest_frame=latest_frame,
                    latest_detection=latest_detection,
                    redis_manager=redis_manager,
                    media_publish_base=media_publish_base,
                    media_path=build_preview_media_path(media_path),
                    site=site,
                    stream_name=stream_name,
                    stop_event=stop_event,
                    rendition='preview',
                ),
            ),
        )
    if publish_clean_stream:
        tasks.add(
            asyncio.create_task(
                _publish_requested_clean_frames(
                    latest_frame=latest_frame,
                    redis_manager=redis_manager,
                    media_publish_base=media_publish_base,
                    media_path=media_path,
                    site=site,
                    stream_name=stream_name,
                    source_url=video_url,
                    use_source_restreamer=restream_clean_source,
                    stop_event=stop_event,
                    rendition='detail',
                ),
            ),
        )
        tasks.add(
            asyncio.create_task(
                _publish_requested_clean_frames(
                    latest_frame=latest_frame,
                    redis_manager=redis_manager,
                    media_publish_base=media_publish_base,
                    media_path=build_preview_media_path(media_path),
                    site=site,
                    stream_name=stream_name,
                    source_url=video_url,
                    use_source_restreamer=False,
                    stop_event=stop_event,
                    rendition='preview',
                ),
            ),
        )
    done, pending = await asyncio.wait(
        tasks,
        return_when=asyncio.FIRST_EXCEPTION,
    )
    stop_event.set()
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    for task in done:
        exception = task.exception()
        if exception is not None:
            raise exception


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
            latest_frame.event.clear()

        detection_time = datetime.fromtimestamp(int(ts))
        is_working = work_start_hour <= detection_time.hour < work_end_hour
        current_timestamp = int(ts)
        try:
            _datas, track_data = await yolo_detector.generate_detections(
                frame,
            )
        except Exception as exc:
            print(
                f'[{site}:{stream_name}] Detection error, keeping stream '
                f'alive: {exc}',
                flush=True,
            )
            await asyncio.sleep(1.0)
            continue
        try:
            warnings, cone_polys, pole_polys = danger_detector.detect_danger(
                track_data,
            )
            warnings = Utils.filter_warnings_by_working_hour(
                warnings,
                is_working,
            )

            async with latest_detection.lock:
                latest_detection.frame = frame
                latest_detection.timestamp = ts
                latest_detection.sequence = last_sequence
                latest_detection.warnings = warnings
                latest_detection.cone_polys = cone_polys
                latest_detection.pole_polys = pole_polys
                latest_detection.track_data = track_data
                latest_detection.event.set()

            should_send_violation = bool(warnings) and Utils.should_notify(
                current_timestamp,
                last_notification_time,
            )
            if _warning_event_due(
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
                last_notification_time = await (
                    _send_violation_and_notification(
                        fcm_sender=fcm_sender,
                        violation_sender=violation_sender,
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
                )
        except Exception as exc:
            print(
                f'[{site}:{stream_name}] Metadata/notification error, keeping '
                f'stream alive: {exc}',
                flush=True,
            )
            await asyncio.sleep(0.2)


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
) -> int:
    """Persist one violation and notify subscribed site users."""
    frame_bytes = Utils.encode_frame(frame, 'jpeg', 85)
    violation_id_str = await violation_sender.send_violation(
        site=site,
        stream_name=stream_name,
        warnings=warnings,
        detection_time=detection_time,
        image_bytes=frame_bytes,
        detections=track_data,
        cone_polygon=cone_polys,
        pole_polygon=pole_polys,
    )
    try:
        violation_id: int | None = (
            int(violation_id_str)
            if violation_id_str is not None
            else None
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


async def _publish_requested_overlay_frames(
    latest_frame: _LatestFrameState,
    latest_detection: _LatestDetectionState,
    redis_manager: RedisManager,
    media_publish_base: str,
    media_path: str,
    site: str,
    stream_name: str,
    stop_event: asyncio.Event,
    rendition: str = 'detail',
) -> None:
    """Publish only the shared overlay languages for one rendition."""
    fps = (
        float(_preview_publisher_kwargs()['fps'])
        if rendition == 'preview'
        else max(1.0, float(os.getenv('MEDIA_PUBLISH_FPS', '15.0')))
    )
    frame_interval = 1.0 / fps
    overlay_publishers: dict[str, MediaStreamPublisher] = {}
    rendered_overlay_cache: dict[str, tuple[int, np.ndarray]] = {}
    overlay_ready_started_at: dict[str, float] = {}
    try:
        while not stop_event.is_set():
            try:
                requested_languages = await _requested_overlay_languages(
                    redis_manager,
                    media_path,
                )
                await _close_unrequested_overlay_publishers(
                    overlay_publishers,
                    requested_languages,
                )
                _drop_unrequested_overlay_cache(
                    rendered_overlay_cache,
                    requested_languages,
                )
                _drop_unrequested_overlay_start_times(
                    overlay_ready_started_at,
                    requested_languages,
                )
                if requested_languages:
                    snapshot = await _latest_overlay_snapshot(
                        latest_frame=latest_frame,
                        latest_detection=latest_detection,
                    )
                    if snapshot is None:
                        snapshot = _OverlaySnapshot(
                            sequence=-1,
                            frame=_build_media_startup_frame(
                                site,
                                stream_name,
                            ),
                        )
                    for language in sorted(requested_languages):
                        await _publish_overlay_language_snapshot(
                            redis_manager=redis_manager,
                            overlay_media_publishers=overlay_publishers,
                            rendered_overlay_cache=rendered_overlay_cache,
                            media_publish_base=media_publish_base,
                            media_path=media_path,
                            site=site,
                            stream_name=stream_name,
                            label_language=language,
                            snapshot=snapshot,
                            overlay_ready_started_at=overlay_ready_started_at,
                            rendition=rendition,
                        )
            except Exception as exc:
                print(
                    f'[{site}:{stream_name}] Overlay publish loop error: '
                    f'{exc}',
                    flush=True,
                )
                await asyncio.sleep(0.5)
            await asyncio.sleep(frame_interval)
    finally:
        await _close_overlay_publishers(overlay_publishers)


async def _latest_overlay_snapshot(
    latest_frame: _LatestFrameState,
    latest_detection: _LatestDetectionState,
) -> _OverlaySnapshot | None:
    """Return latest source frame plus detection metadata if available."""
    async with latest_detection.lock:
        if latest_detection.frame is not None:
            return _OverlaySnapshot(
                sequence=latest_detection.sequence,
                frame=latest_detection.frame,
                warnings=latest_detection.warnings,
                cone_polys=latest_detection.cone_polys,
                pole_polys=latest_detection.pole_polys,
                track_data=latest_detection.track_data,
            )
    async with latest_frame.lock:
        if latest_frame.frame is not None:
            return _OverlaySnapshot(
                sequence=latest_frame.sequence,
                frame=latest_frame.frame,
            )
    return None


async def _requested_overlay_languages(
    redis_manager: RedisManager,
    media_path: str,
) -> set[str]:
    """Read active overlay language demand keys for one clean media path."""
    pattern = f'{OVERLAY_DEMAND_PREFIX}:{media_path}:*'
    allowed_languages = set(_allowed_overlay_languages())
    languages: set[str] = set()
    async for raw_key in redis_manager.redis.scan_iter(match=pattern):
        key = raw_key.decode('utf-8') if isinstance(
            raw_key,
            bytes,
        ) else str(raw_key)
        encoded_language = key.rsplit(':', 1)[-1]
        try:
            language = normalise_label_language(
                decode_media_segment(encoded_language),
            )
        except Exception:
            continue
        if language in allowed_languages:
            languages.add(language)
    return languages


async def _publish_requested_overlay_snapshot(
    redis_manager: RedisManager,
    overlay_media_publishers: dict[str, MediaStreamPublisher],
    media_publish_base: str,
    media_path: str,
    site: str,
    stream_name: str,
    source_frame: np.ndarray,
    warnings: object,
    cone_polys: object,
    pole_polys: object,
    track_data: object,
    overlay_ready_started_at: dict[str, float] | None = None,
    rendition: str = 'detail',
) -> None:
    """Publish one snapshot to all currently requested overlay languages."""
    requested_languages = await _requested_overlay_languages(
        redis_manager,
        media_path,
    )
    rendered_overlay_cache: dict[str, tuple[int, np.ndarray]] = {}
    await _close_unrequested_overlay_publishers(
        overlay_media_publishers,
        requested_languages,
    )
    if overlay_ready_started_at is not None:
        _drop_unrequested_overlay_start_times(
            overlay_ready_started_at,
            requested_languages,
        )
    snapshot = _OverlaySnapshot(
        sequence=0,
        frame=source_frame,
        warnings=warnings,
        cone_polys=cone_polys,
        pole_polys=pole_polys,
        track_data=track_data,
    )
    for language in sorted(requested_languages):
        await _publish_overlay_language_snapshot(
            redis_manager=redis_manager,
            overlay_media_publishers=overlay_media_publishers,
            rendered_overlay_cache=rendered_overlay_cache,
            media_publish_base=media_publish_base,
            media_path=media_path,
            site=site,
            stream_name=stream_name,
            label_language=language,
            snapshot=snapshot,
            overlay_ready_started_at=overlay_ready_started_at,
            rendition=rendition,
        )


async def _publish_overlay_language_snapshot(
    redis_manager: RedisManager,
    overlay_media_publishers: dict[str, MediaStreamPublisher],
    rendered_overlay_cache: dict[str, tuple[int, np.ndarray]],
    media_publish_base: str,
    media_path: str,
    site: str,
    stream_name: str,
    label_language: str,
    snapshot: _OverlaySnapshot,
    overlay_ready_started_at: dict[str, float] | None = None,
    rendition: str = 'detail',
) -> None:
    """Render and publish one language variant backed by a shared publisher."""
    overlay_path = build_annotated_media_path(media_path, label_language)
    publisher = overlay_media_publishers.get(label_language)
    if publisher is None:
        publisher = _media_publisher(
            f'{media_publish_base}/{overlay_path}',
            rendition=rendition,
        )
        overlay_media_publishers[label_language] = publisher
        print(
            f'[{site}:{stream_name}] Starting shared overlay stream '
            f'{label_language}: {media_publish_base}/{overlay_path}',
            flush=True,
        )

    if snapshot.track_data is None:
        publish_frame = snapshot.frame
    else:
        cached = rendered_overlay_cache.get(label_language)
        if cached is not None and cached[0] == snapshot.sequence:
            publish_frame = cached[1]
        else:
            publish_frame = _build_media_publish_frame(
                frame=snapshot.frame,
                warnings=snapshot.warnings,
                cone_polys=snapshot.cone_polys,
                pole_polys=snapshot.pole_polys,
                track_data=snapshot.track_data,
                label_language=label_language,
            )
            rendered_overlay_cache[label_language] = (
                snapshot.sequence,
                publish_frame,
            )
    await publisher.publish(publish_frame)
    if _overlay_ready_grace_elapsed(overlay_ready_started_at, label_language):
        await _mark_overlay_ready(redis_manager, overlay_path)


def _overlay_ready_grace_elapsed(
    overlay_ready_started_at: dict[str, float] | None,
    label_language: str,
) -> bool:
    """Return True once a new overlay publisher had time to open HLS."""
    if overlay_ready_started_at is None:
        return True
    loop = asyncio.get_running_loop()
    now = loop.time()
    first_publish_at = overlay_ready_started_at.setdefault(
        label_language,
        now,
    )
    return now - first_publish_at >= _overlay_ready_grace_seconds()


def _overlay_ready_grace_seconds() -> float:
    """Return how long to wait before advertising a new overlay as ready."""
    try:
        return max(
            0.0,
            float(os.getenv('MEDIA_OVERLAY_READY_GRACE_SECONDS', '2.0')),
        )
    except ValueError:
        return 2.0


def _drop_unrequested_overlay_cache(
    rendered_overlay_cache: dict[str, tuple[int, np.ndarray]],
    requested_languages: set[str],
) -> None:
    """Free rendered overlay frames for languages no longer being viewed."""
    for language in list(rendered_overlay_cache):
        if language not in requested_languages:
            rendered_overlay_cache.pop(language, None)


def _drop_unrequested_overlay_start_times(
    overlay_ready_started_at: dict[str, float],
    requested_languages: set[str],
) -> None:
    """Forget readiness timing for overlay languages no longer requested."""
    for language in list(overlay_ready_started_at):
        if language not in requested_languages:
            overlay_ready_started_at.pop(language, None)


async def _mark_overlay_ready(
    redis_manager: RedisManager,
    overlay_media_path: str,
) -> None:
    """Mark an overlay path ready while frames are actively published."""
    ttl_seconds = max(
        5,
        int(os.getenv('MEDIA_OVERLAY_READY_TTL_SECONDS', '15')),
    )
    await redis_manager.redis.set(
        build_overlay_ready_key(overlay_media_path),
        b'1',
        ex=ttl_seconds,
    )


async def _close_unrequested_overlay_publishers(
    overlay_media_publishers: dict[str, MediaStreamPublisher],
    requested_languages: set[str],
) -> None:
    """Close overlay publishers no longer requested by viewers.

    Args:
        overlay_media_publishers: Active publishers keyed by language.
        requested_languages: Languages still needed by connected clients.
    """
    for language in list(overlay_media_publishers):
        if language in requested_languages:
            continue
        publisher = overlay_media_publishers.pop(language)
        await publisher.close()


async def _close_overlay_publishers(
    overlay_media_publishers: dict[str, MediaStreamPublisher],
) -> None:
    """Close every overlay publisher in a stream session."""
    for language in list(overlay_media_publishers):
        publisher = overlay_media_publishers.pop(language)
        await publisher.close()


async def _publish_requested_clean_frames(
    latest_frame: _LatestFrameState,
    redis_manager: RedisManager,
    media_publish_base: str,
    media_path: str,
    site: str,
    stream_name: str,
    source_url: str,
    use_source_restreamer: bool,
    stop_event: asyncio.Event,
    rendition: str = 'detail',
) -> None:
    """Publish one clean detail or preview rendition while requested."""
    fps = (
        float(_preview_publisher_kwargs()['fps'])
        if rendition == 'preview'
        else max(1.0, float(os.getenv('MEDIA_PUBLISH_FPS', '15.0')))
    )
    frame_interval = 1.0 / fps
    clean_publisher: MediaStreamPublisher | None = None
    clean_restreamer: MediaSourceRestreamer | None = None
    publish_url = f'{media_publish_base}/{media_path}'
    try:
        while not stop_event.is_set():
            try:
                requested = await _clean_stream_requested(
                    redis_manager,
                    media_path,
                )
                if not requested:
                    if clean_restreamer is not None:
                        await clean_restreamer.close()
                        clean_restreamer = None
                    if clean_publisher is not None:
                        await clean_publisher.close()
                        clean_publisher = None
                    await asyncio.sleep(frame_interval)
                    continue

                if use_source_restreamer and rendition == 'detail':
                    if clean_restreamer is None:
                        clean_restreamer = MediaSourceRestreamer(
                            source_url=source_url,
                            publish_url=publish_url,
                        )
                        print(
                            f'[{site}:{stream_name}] Starting clean source '
                            f'restream: {publish_url}',
                            flush=True,
                        )
                        await clean_restreamer.start()
                    await asyncio.sleep(frame_interval)
                    continue

                if clean_publisher is None:
                    clean_publisher = _media_publisher(
                        publish_url,
                        rendition=rendition,
                    )
                    print(
                        f'[{site}:{stream_name}] Starting {rendition} clean '
                        f'stream: '
                        f'{publish_url}',
                        flush=True,
                    )

                async with latest_frame.lock:
                    frame = latest_frame.frame
                if frame is not None:
                    await clean_publisher.publish(frame)
            except Exception as exc:
                print(
                    f'[{site}:{stream_name}] Clean media publish error, '
                    f'keeping stream alive: {exc}',
                    flush=True,
                )
                await asyncio.sleep(0.5)
            await asyncio.sleep(frame_interval)
    finally:
        if clean_restreamer is not None:
            await clean_restreamer.close()
        if clean_publisher is not None:
            await clean_publisher.close()


async def _clean_stream_requested(
    redis_manager: RedisManager,
    media_path: str,
) -> bool:
    """Return True while at least one viewer wants the clean stream."""
    return bool(
        await redis_manager.redis.exists(build_clean_demand_key(media_path)),
    )


def _live_view_enabled(cfg: StreamConfig) -> bool:
    """Return whether this stream should publish live MediaMTX outputs."""
    return bool(cfg.get('store_in_redis'))


def _stream_metadata_key(site: str, stream_name: str) -> str:
    """Build the compact live metadata stream key."""
    return f'stream_metadata:{Utils.encode(site)}|{Utils.encode(stream_name)}'


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
    metadata: dict[RedisPrimitive, RedisPrimitive] = {
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
    print(
        (
            f'[Warning-Metadata] XADD {metadata_key} id={event_id} '
            f'has_warning=1 warnings={warning_keys}'
        ),
        flush=True,
    )


def _build_media_publish_frame(
    frame: np.ndarray,
    warnings: object,
    cone_polys: object,
    pole_polys: object,
    track_data: object,
    label_language: str = 'en',
) -> np.ndarray:
    """Return the annotated frame published to MediaMTX."""
    # Overlay rendering draws onto the frame, so copy exactly once here while
    # the rest of the live pipeline can pass frame references around.
    return render_overlay_array(
        frame.copy(),
        detection_items=track_data,
        warnings=warnings,
        cone_polygons=cone_polys,
        pole_polygons=pole_polys,
        overlay_mode='backend',
        label_language=label_language,
        min_confidence=float(
            os.getenv(
                'MEDIA_PUBLISH_OVERLAY_MIN_CONFIDENCE',
                '0.25',
            ),
        ),
        box_thickness=max(
            1,
            int(float(os.getenv('MEDIA_PUBLISH_BOX_THICKNESS', '2'))),
        ),
    )


def _build_media_startup_frame(site: str, stream_name: str) -> np.ndarray:
    """Return a startup frame for the annotated media path."""
    width = int(os.getenv('MEDIA_PUBLISH_STARTUP_WIDTH', '1280'))
    height = int(os.getenv('MEDIA_PUBLISH_STARTUP_HEIGHT', '720'))
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    title = f'{site} / {stream_name}'
    subtitle = 'Starting live analysis...'
    cv2.putText(
        frame,
        title[:80],
        (48, max(80, height // 2 - 30)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        subtitle,
        (48, max(130, height // 2 + 30)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (160, 200, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def _csv_env(name: str, default: str) -> list[str]:
    """Read a comma-separated environment setting.

    Args:
        name: Environment variable name.
        default: Default comma-separated value.

    Returns:
        Normalised non-empty entries.
    """
    value = os.getenv(name, default)
    return [item.strip() for item in value.split(',') if item.strip()]


def _allowed_overlay_languages() -> tuple[str, ...]:
    """Return enabled overlay languages supported by the renderer."""
    configured = _csv_env(
        'MEDIA_OVERLAY_ALLOWED_LANGUAGES',
        ','.join(SUPPORTED_LABEL_LANGUAGES),
    )
    allowed = []
    for language in configured:
        normalised = normalise_label_language(language)
        if (
            normalised in SUPPORTED_LABEL_LANGUAGES
            and normalised not in allowed
        ):
            allowed.append(normalised)
    return tuple(allowed or ('en',))


def _mark_frame_readonly(frame: np.ndarray) -> None:
    """Mark a captured frame immutable so it can be shared without copies."""
    try:
        frame.setflags(write=False)
    except ValueError:
        pass


def _resolve_detect_with_server(_configured: bool) -> bool:
    """Return the runtime detection mode for streams.

    Stream processing is server-only. The database ``detect_with_server`` value
    is still accepted for compatibility with existing records, but local
    inference is disabled for the main runtime path.
    """
    return True


def _validate_server_model_key(model_key: str) -> None:
    """Ensure a server model key is configured.

    Args:
        model_key: YOLO server model key.

    Raises:
        RuntimeError: If the model key is not configured.
    """
    supported = _server_model_keys()
    if model_key in supported:
        return
    raise RuntimeError(
        f"YOLO server model_key '{model_key}' is not supported. "
        f"Use one of: {', '.join(supported)}",
    )


def _server_model_keys() -> tuple[str, ...]:
    """Return configured YOLO server model keys in stable order."""
    configured = _csv_env(
        'DETECT_SERVER_MODEL_KEYS',
        'yolo26n,yolo26s,yolo26m,yolo26l,yolo26x',
    )
    return tuple(dict.fromkeys(configured))
