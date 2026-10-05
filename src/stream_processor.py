from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from dataclasses import field
from typing import cast

import numpy as np
from dotenv import load_dotenv

from examples.streaming_web.media_paths import (
    build_annotated_media_path,
)
from examples.streaming_web.media_paths import build_clean_demand_key
from examples.streaming_web.media_paths import build_media_path
from examples.streaming_web.media_paths import build_overlay_ready_key
from examples.streaming_web.media_paths import build_preview_media_path
from examples.streaming_web.metadata_keys import build_metadata_key
from examples.streaming_web.metadata_keys import get_metadata_site_generation
from examples.streaming_web.playback_demand import active_overlay_languages
from src import stream_detection
from src import stream_overlay_frames
from src.async_tasks import cancel_on_first_failure
from src.danger_detector import DangerDetector
from src.media_publish_config import create_media_publisher
from src.media_publish_config import env_enabled
from src.media_publish_config import preview_publisher_options
from src.media_restreamer import MediaSourceRestreamer
from src.media_stream_publisher import MediaStreamPublisher
from src.notifiers.fcm_notifier import FCMSender
from src.redis_client import RedisManager
from src.stream_capture import StreamCapture
from src.stream_runtime_state import (
    LatestDetectionState as _LatestDetectionState,
)
from src.stream_runtime_state import LatestFrameState as _LatestFrameState
from src.stream_runtime_state import (
    OverlayPublisherVariant as _OverlayPublisherVariant,
)
from src.stream_runtime_state import OverlaySnapshot as _OverlaySnapshot
from src.stream_runtime_state import StreamConfig
from src.violation_sender import ViolationSender
from src.yolo_detector import YoloDetector
from src.yolo_worker import WorkerQueue
from src.yolo_worker import WorkerResultReceiver
from src.yolo_worker import YoloWorkerClient

logger = logging.getLogger(__name__)


def _media_demand_cache_seconds() -> float:
    """Return the time that viewer-demand reads may be reused per camera."""
    try:
        return max(
            0.0,
            float(os.getenv('MEDIA_DEMAND_CACHE_SECONDS', '0.5')),
        )
    except ValueError:
        return 0.5


@dataclass
class _MediaDemandCache:
    """Share short-lived clean and overlay demand reads for one camera."""

    refresh_seconds: float = field(default_factory=_media_demand_cache_seconds)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _clean_requested: dict[str, bool] = field(default_factory=dict)
    _clean_refreshed_at: dict[str, float] = field(default_factory=dict)
    _overlay_languages: dict[str, set[str]] = field(default_factory=dict)
    _overlay_refreshed_at: dict[str, float] = field(default_factory=dict)

    async def clean_requested(
        self,
        redis_manager: RedisManager,
        media_path: str,
    ) -> bool:
        """Return cached clean-stream demand, refreshing only when needed."""

        async with self._lock:
            now = time.monotonic()
            last_refreshed_at = self._clean_refreshed_at.get(media_path)
            if (
                last_refreshed_at is None
                or now - last_refreshed_at >= self.refresh_seconds
            ):
                self._clean_requested[
                    media_path
                ] = await _clean_stream_requested(redis_manager, media_path)
                self._clean_refreshed_at[media_path] = time.monotonic()
            return self._clean_requested.get(media_path, False)

    async def overlay_languages(
        self,
        redis_manager: RedisManager,
        media_path: str,
    ) -> set[str]:
        """Return cached overlay demand, refreshing only when needed."""

        async with self._lock:
            now = time.monotonic()
            last_refreshed_at = self._overlay_refreshed_at.get(media_path)
            if (
                last_refreshed_at is None
                or now - last_refreshed_at >= self.refresh_seconds
            ):
                self._overlay_languages[
                    media_path
                ] = await _requested_overlay_languages(
                    redis_manager,
                    media_path,
                )
                self._overlay_refreshed_at[media_path] = time.monotonic()
            return set(self._overlay_languages.get(media_path, set()))


async def delete_stream_live_metadata(cfg: StreamConfig) -> None:
    """Delete compact live metadata for one configured camera."""
    redis_manager = RedisManager()
    await redis_manager.delete(
        build_metadata_key(cfg['site'], cfg['stream_name']),
    )


def process_single_stream(
    cfg: StreamConfig,
    yolo_request_queue: object | None = None,
    yolo_result_queue: object | None = None,
) -> None:
    """Run one configured stream inside a child process."""
    load_dotenv(override=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s %(message)s',
    )
    asyncio.run(
        _run_single_stream(
            cfg,
            yolo_request_queue=yolo_request_queue,
            yolo_result_queue=yolo_result_queue,
        ),
    )


async def _run_single_stream(
    cfg: StreamConfig,
    yolo_request_queue: object | None = None,
    yolo_result_queue: object | None = None,
) -> None:
    """Run one camera through the shared-worker streaming pipeline."""
    if not cfg.get('recognition_enabled', True):
        logger.info(
            f'[{cfg["site"]}:{cfg["stream_name"]}] Recognition disabled; '
            'skipping stream processor startup',
        )
        return
    if yolo_request_queue is None or yolo_result_queue is None:
        raise RuntimeError('Shared YOLO worker queues are required')

    video_url = cfg['video_url']
    model_key = cfg['model_key']
    site = cfg['site']
    stream_name = cfg['stream_name']
    media_path = build_media_path(site, stream_name)
    streaming_capture = StreamCapture(stream_url=video_url)
    yolo_detector = YoloDetector(
        model_key=model_key,
        output_folder=site,
        worker_client=YoloWorkerClient(
            cast(WorkerQueue, yolo_request_queue),
            cast(WorkerResultReceiver, yolo_result_queue),
            camera_id=f'{site}|{stream_name}',
            timeout_seconds=float(
                os.getenv('YOLO_WORKER_TIMEOUT_SECONDS', '30.0'),
            ),
        ),
    )
    _validate_server_model_key(model_key)
    logger.info(
        f'[{site}:{stream_name}] Streaming output mode: media_server',
    )
    logger.info(
        f'[{site}:{stream_name}] YOLO detection worker enabled',
    )

    media_publish_base = (
        os.getenv('MEDIA_PUBLISH_RTSP_BASE_URL') or 'rtsp://media-server:8554'
    ).rstrip('/')
    publish_clean_stream = env_enabled('MEDIA_PUBLISH_CLEAN_STREAM', True)
    publish_annotated_stream = env_enabled(
        'MEDIA_PUBLISH_ANNOTATED_STREAM',
        True,
    )
    restream_clean_source = env_enabled(
        'MEDIA_PUBLISH_CLEAN_SOURCE_RESTREAM',
        True,
    )
    if publish_clean_stream:
        logger.info(
            f'[{site}:{stream_name}] Clean media stream is on-demand; '
            f'path {media_path}',
        )
    if publish_annotated_stream:
        logger.info(
            f'[{site}:{stream_name}] Overlay media streams are on-demand; '
            f'base path {media_path}',
        )

    redis_manager = RedisManager()
    metadata_key = build_metadata_key(
        site,
        stream_name,
        await get_metadata_site_generation(redis_manager.redis, site),
    )
    try:
        await _run_decoupled_media_server_loop(
            streaming_capture=streaming_capture,
            yolo_detector=yolo_detector,
            danger_detector=DangerDetector(cfg['detection_items']),
            fcm_sender=FCMSender(api_url=os.getenv('FCM_API_URL') or ''),
            violation_sender=ViolationSender(
                api_url=os.getenv('VIOLATION_RECORD_API_URL') or '',
            ),
            redis_manager=redis_manager,
            media_publish_base=media_publish_base,
            media_path=media_path,
            publish_overlay_streams=publish_annotated_stream,
            site=site,
            stream_name=stream_name,
            work_start_hour=cfg['work_start_hour'],
            work_end_hour=cfg['work_end_hour'],
            metadata_key=metadata_key,
            publish_clean_stream=publish_clean_stream,
            restream_clean_source=restream_clean_source,
            video_url=video_url,
        )
    finally:
        await yolo_detector.close()
        await streaming_capture.release_resources()
        try:
            await redis_manager.delete(metadata_key)
        except Exception as exc:
            logger.info(
                f'[WARN] Failed to delete redis key {metadata_key}: {exc}',
            )


async def _run_decoupled_media_server_loop(
    streaming_capture: StreamCapture,
    yolo_detector: YoloDetector,
    danger_detector: DangerDetector,
    fcm_sender: FCMSender,
    violation_sender: ViolationSender,
    redis_manager: RedisManager,
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
    source_reconnect_event = stream_detection._capture_reconnect_event(
        streaming_capture,
    )
    clean_reconnect_event = asyncio.Event()
    demand_cache = _MediaDemandCache()
    tasks = {
        asyncio.create_task(
            stream_detection._capture_latest_frames(
                streaming_capture=streaming_capture,
                latest_frame=latest_frame,
                stop_event=stop_event,
            ),
        ),
        asyncio.create_task(
            stream_detection._detect_latest_frames(
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
    if source_reconnect_event is not None:
        tasks.add(
            asyncio.create_task(
                stream_detection._synchronise_capture_reconnects(
                    reconnect_event=source_reconnect_event,
                    clean_reconnect_event=clean_reconnect_event,
                    latest_frame=latest_frame,
                    latest_detection=latest_detection,
                    stop_event=stop_event,
                ),
            ),
        )
    if publish_overlay_streams:
        tasks.add(
            asyncio.create_task(
                _publish_requested_overlay_variants(
                    latest_frame=latest_frame,
                    latest_detection=latest_detection,
                    redis_manager=redis_manager,
                    media_publish_base=media_publish_base,
                    variants=_overlay_publisher_variants(media_path),
                    site=site,
                    stream_name=stream_name,
                    stop_event=stop_event,
                    demand_cache=demand_cache,
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
                    source_reconnect_event=clean_reconnect_event,
                    demand_cache=demand_cache,
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
                    source_reconnect_event=None,
                    demand_cache=demand_cache,
                    rendition='preview',
                ),
            ),
        )
    try:
        await cancel_on_first_failure(cast(set[asyncio.Task[object]], tasks))
    finally:
        stop_event.set()


def _overlay_publisher_variants(
    media_path: str,
) -> list[_OverlayPublisherVariant]:
    """Build the detail and preview variants that share overlay rendering."""
    return [
        _OverlayPublisherVariant(media_path, 'detail'),
        _OverlayPublisherVariant(
            build_preview_media_path(media_path),
            'preview',
        ),
    ]


def _overlay_variant_fps(variants: list[_OverlayPublisherVariant]) -> float:
    """Return one source update rate sufficient for every rendition."""
    rates = [
        (
            preview_publisher_options()['fps']
            if variant.rendition == 'preview'
            else max(1.0, float(os.getenv('MEDIA_PUBLISH_FPS', '15.0')))
        )
        for variant in variants
    ]
    return max(rates, default=1.0)


async def _publish_requested_overlay_variants(
    latest_frame: _LatestFrameState,
    latest_detection: _LatestDetectionState,
    redis_manager: RedisManager,
    media_publish_base: str,
    variants: list[_OverlayPublisherVariant],
    site: str,
    stream_name: str,
    stop_event: asyncio.Event,
    demand_cache: _MediaDemandCache | None = None,
) -> None:
    """Render each language once and publish it to requested variants."""
    demand_cache = demand_cache or _MediaDemandCache()
    frame_interval = 1.0 / _overlay_variant_fps(variants)
    rendered_overlay_cache: dict[
        str,
        tuple[tuple[int, int], np.ndarray],
    ] = {}
    try:
        while not stop_event.is_set():
            try:
                requested_by_variant = {
                    variant.media_path: await demand_cache.overlay_languages(
                        redis_manager,
                        variant.media_path,
                    )
                    for variant in variants
                }
                for variant in variants:
                    requested_languages = requested_by_variant[
                        variant.media_path
                    ]
                    await _close_unrequested_overlay_publishers(
                        variant.publishers,
                        requested_languages,
                    )
                    _drop_unrequested_overlay_start_times(
                        variant.ready_started_at,
                        requested_languages,
                    )
                requested_languages = set().union(
                    *requested_by_variant.values(),
                )
                _drop_unrequested_overlay_cache(
                    rendered_overlay_cache,
                    requested_languages,
                )
                if requested_languages:
                    snapshot = await _latest_overlay_snapshot(
                        latest_frame=latest_frame,
                        latest_detection=latest_detection,
                    )
                    if snapshot is None:
                        snapshot = _OverlaySnapshot(
                            sequence=(-1, -1),
                            frame=(
                                stream_overlay_frames
                                ._build_media_startup_frame(site, stream_name)
                            ),
                        )
                    publish_frames = (
                        stream_overlay_frames._overlay_publish_frames(
                            snapshot,
                            requested_languages,
                            rendered_overlay_cache,
                        )
                    )
                    for variant in variants:
                        for language in sorted(
                            requested_by_variant[variant.media_path],
                        ):
                            await _publish_overlay_language_snapshot(
                                redis_manager=redis_manager,
                                overlay_media_publishers=variant.publishers,
                                media_publish_base=media_publish_base,
                                media_path=variant.media_path,
                                site=site,
                                stream_name=stream_name,
                                label_language=language,
                                publish_frame=publish_frames[language],
                                overlay_ready_started_at=(
                                    variant.ready_started_at
                                ),
                                rendition=variant.rendition,
                            )
            except Exception as exc:
                logger.info(
                    f'[{site}:{stream_name}] Overlay publish loop error: '
                    f'{exc}',
                )
                await asyncio.sleep(0.5)
            await asyncio.sleep(frame_interval)
    finally:
        for variant in variants:
            await _close_overlay_publishers(variant.publishers)


async def _latest_overlay_snapshot(
    latest_frame: _LatestFrameState,
    latest_detection: _LatestDetectionState,
) -> _OverlaySnapshot | None:
    """Return the freshest frame with the most recent detection metadata."""

    async with latest_frame.lock:
        source_frame = latest_frame.frame
        source_sequence = latest_frame.sequence
    async with latest_detection.lock:
        detection_frame = latest_detection.frame
        detection_sequence = latest_detection.sequence
        warnings = latest_detection.warnings
        cone_polys = latest_detection.cone_polys
        pole_polys = latest_detection.pole_polys
        track_data = latest_detection.track_data
        class_metadata = latest_detection.class_metadata

    if source_frame is not None:
        return _OverlaySnapshot(
            sequence=(source_sequence, detection_sequence),
            frame=source_frame,
            warnings=warnings,
            cone_polys=cone_polys,
            pole_polys=pole_polys,
            track_data=track_data,
            class_metadata=class_metadata,
        )
    if detection_frame is not None:
        return _OverlaySnapshot(
            sequence=(source_sequence, detection_sequence),
            frame=detection_frame,
            warnings=warnings,
            cone_polys=cone_polys,
            pole_polys=pole_polys,
            track_data=track_data,
            class_metadata=class_metadata,
        )
    return None


async def _requested_overlay_languages(
    redis_manager: RedisManager,
    media_path: str,
) -> set[str]:
    """Read demand with bounded ``MGET`` instead of scanning Redis keys."""
    return await active_overlay_languages(
        redis_manager.redis,
        media_path,
        stream_overlay_frames._allowed_overlay_languages(),
    )


async def _publish_overlay_language_snapshot(
    redis_manager: RedisManager,
    overlay_media_publishers: dict[str, MediaStreamPublisher],
    media_publish_base: str,
    media_path: str,
    site: str,
    stream_name: str,
    label_language: str,
    publish_frame: np.ndarray,
    overlay_ready_started_at: dict[str, float] | None = None,
    rendition: str = 'detail',
) -> None:
    """Publish one pre-rendered language variant through a shared publisher."""
    overlay_path = build_annotated_media_path(media_path, label_language)
    publisher = overlay_media_publishers.get(label_language)
    if publisher is None:
        publisher = create_media_publisher(
            f'{media_publish_base}/{overlay_path}',
            rendition=rendition,
        )
        overlay_media_publishers[label_language] = publisher
        logger.info(
            f'[{site}:{stream_name}] Starting shared overlay stream '
            f'{label_language}: {media_publish_base}/{overlay_path}',
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
    rendered_overlay_cache: dict[
        str,
        tuple[tuple[int, int], np.ndarray],
    ],
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
    source_reconnect_event: asyncio.Event | None = None,
    demand_cache: _MediaDemandCache | None = None,
    rendition: str = 'detail',
) -> None:
    """Publish one clean detail or preview rendition while requested.

    A capture reconnect intentionally invalidates ``latest_frame`` so stale
    images are never analysed or annotated. An already-running clean encoder
    is retained during that interval: its latest-frame writer repeats the last
    valid image at a constant rate, preserving the RTSP and HLS session until
    fresh source video arrives.
    """
    demand_cache = demand_cache or _MediaDemandCache()
    fps = (
        preview_publisher_options()['fps']
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
                requested = await demand_cache.clean_requested(
                    redis_manager,
                    media_path,
                )
                if not requested:
                    (
                        clean_restreamer,
                        clean_publisher,
                    ) = await _close_clean_media_publishers(
                        clean_restreamer,
                        clean_publisher,
                    )
                    await asyncio.sleep(frame_interval)
                    continue

                if use_source_restreamer and rendition == 'detail':
                    clean_restreamer = await _ensure_clean_restreamer(
                        clean_restreamer,
                        source_reconnect_event,
                        source_url,
                        publish_url,
                        site,
                        stream_name,
                    )
                    await asyncio.sleep(frame_interval)
                    continue

                async with latest_frame.lock:
                    frame = latest_frame.frame
                if frame is None:
                    await asyncio.sleep(frame_interval)
                    continue
                clean_publisher = _ensure_clean_publisher(
                    clean_publisher,
                    publish_url,
                    rendition,
                    site,
                    stream_name,
                )
                await clean_publisher.publish(frame)
            except Exception as exc:
                logger.info(
                    f'[{site}:{stream_name}] Clean media publish error, '
                    f'keeping stream alive: {exc}',
                )
                await asyncio.sleep(0.5)
            await asyncio.sleep(frame_interval)
    finally:
        await _close_clean_media_publishers(clean_restreamer, clean_publisher)


async def _close_clean_media_publishers(
    clean_restreamer: MediaSourceRestreamer | None,
    clean_publisher: MediaStreamPublisher | None,
) -> tuple[None, None]:
    """Close clean-stream publishers and reset their local state."""
    if clean_restreamer is not None:
        await clean_restreamer.close()
    if clean_publisher is not None:
        await clean_publisher.close()
    return None, None


async def _ensure_clean_restreamer(
    clean_restreamer: MediaSourceRestreamer | None,
    source_reconnect_event: asyncio.Event | None,
    source_url: str,
    publish_url: str,
    site: str,
    stream_name: str,
) -> MediaSourceRestreamer:
    """Restart or start the clean source restream requested by a viewer."""
    if source_reconnect_event is not None and source_reconnect_event.is_set():
        source_reconnect_event.clear()
        if clean_restreamer is not None:
            logger.info(
                f'[{site}:{stream_name}] Restarting clean source restream '
                'after capture-watchdog reconnect',
            )
            await clean_restreamer.restart()
    if clean_restreamer is not None:
        return clean_restreamer
    restreamer = MediaSourceRestreamer(
        source_url=source_url,
        publish_url=publish_url,
    )
    logger.info(
        f'[{site}:{stream_name}] Starting clean source restream: '
        f'{publish_url}',
    )
    await restreamer.start()
    return restreamer


def _ensure_clean_publisher(
    clean_publisher: MediaStreamPublisher | None,
    publish_url: str,
    rendition: str,
    site: str,
    stream_name: str,
) -> MediaStreamPublisher:
    """Create a requested clean-frame publisher once per media path."""
    if clean_publisher is not None:
        return clean_publisher
    publisher = create_media_publisher(publish_url, rendition=rendition)
    logger.info(
        f'[{site}:{stream_name}] Starting {rendition} clean stream: '
        f'{publish_url}',
    )
    return publisher


async def _clean_stream_requested(
    redis_manager: RedisManager,
    media_path: str,
) -> bool:
    """Return True while at least one viewer wants the clean stream."""
    return bool(
        await redis_manager.redis.exists(build_clean_demand_key(media_path)),
    )


def _validate_server_model_key(model_key: str) -> None:
    """Ensure a server model key is configured.

    Args:
        model_key: YOLO server model key.

    Raises:
        RuntimeError: If the model key is not configured.
    """
    import re

    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,119}', model_key):
        raise RuntimeError('Invalid model key')
    # The shared worker resolves and validates publication in the DB catalog.


def _server_model_keys() -> tuple[str, ...]:
    """Return configured YOLO server model keys in stable order."""
    configured = stream_overlay_frames._csv_env(
        'DETECT_SERVER_MODEL_KEYS',
        'yolo26n,yolo26s,yolo26m,yolo26l,yolo26x',
    )
    return tuple(dict.fromkeys(configured))
