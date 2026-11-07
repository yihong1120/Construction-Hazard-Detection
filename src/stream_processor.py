from __future__ import annotations

import asyncio
import logging
import re
from typing import cast

from dotenv import load_dotenv

from examples.streaming_web.media_paths import build_media_path
from examples.streaming_web.metadata_keys import build_metadata_key
from examples.streaming_web.metadata_keys import get_metadata_site_generation
from src import stream_detection
from src.async_tasks import run_coupled_tasks
from src.danger_detector import DangerDetector
from src.redis_client import RedisManager
from src.settings.inference import ModelCatalogSettings
from src.settings.inference import WorkerSettings
from src.settings.media import MediaPublishSettings
from src.stream_capture import StreamCapture
from src.stream_identity import stream_identity
from src.stream_media import StreamPublisherGroup
from src.stream_runtime_state import StreamConfig
from src.stream_runtime_state import StreamMediaSettings
from src.stream_runtime_state import StreamRuntime
from src.violation_delivery import ViolationQueue
from src.yolo_detector import YoloDetector
from src.yolo_worker import WorkerQueue
from src.yolo_worker import WorkerResultReceiver
from src.yolo_worker import YoloWorkerClient

logger = logging.getLogger(__name__)


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
            f"[{cfg['site']}:{cfg['stream_name']}] Recognition disabled; "
            'skipping stream processor startup',
        )
        return
    if yolo_request_queue is None or yolo_result_queue is None:
        raise RuntimeError('Shared YOLO worker queues are required')
    media_settings = MediaPublishSettings()
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
            cast(
                WorkerResultReceiver,
                yolo_result_queue,
            ),
            camera_id=stream_identity(cfg),
            timeout_seconds=WorkerSettings().timeout_seconds,
        ),
    )
    _validate_server_model_key(model_key)
    logger.info(f'[{site}:{stream_name}] Streaming output mode: media_server')
    logger.info(f'[{site}:{stream_name}] YOLO detection worker enabled')
    publish_clean_stream = media_settings.clean
    publish_annotated_stream = media_settings.overlay
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
    delivery = ViolationQueue(
        str(cfg.get('stream_id') or f'{site}|{stream_name}'),
    )
    try:
        runtime = StreamRuntime(
            cfg=cfg,
            capture=streaming_capture,
            detector=yolo_detector,
            danger=DangerDetector(cfg['detection_items']),
            redis=redis_manager,
            delivery=delivery,
            metadata_key=metadata_key,
            media=StreamMediaSettings(media_path, media_settings),
        )
        runtime.clock.warning_event_throttle_seconds = (
            stream_detection._warning_event_throttle_seconds()
        )
        await _run_decoupled_media_server_loop(runtime)
    finally:
        delivery.close()
        await yolo_detector.close()
        await streaming_capture.release_resources()
        try:
            await redis_manager.delete(metadata_key)
        except Exception as exc:
            logger.info(
                f'[WARN] Failed to delete redis key {metadata_key}: {exc}',
            )


async def _run_decoupled_media_server_loop(runtime: StreamRuntime) -> None:
    publishers = StreamPublisherGroup(runtime)
    streaming_capture = runtime.capture
    latest_frame, latest_detection, stop_event = (
        runtime.latest_frame,
        runtime.latest_detection,
        runtime.stop,
    )
    publish_clean_stream, publish_overlay_streams = (
        runtime.media.clean,
        runtime.media.overlay,
    )
    source_reconnect_event = stream_detection._capture_reconnect_event(
        streaming_capture,
    )
    clean_reconnect_event = publishers.clean_reconnect_event
    tasks = {
        asyncio.create_task(
            stream_detection._capture_latest_frames(
                streaming_capture=streaming_capture,
                latest_frame=latest_frame,
                stop_event=stop_event,
            ),
        ),
        asyncio.create_task(
            stream_detection._detect_latest_frames(runtime=runtime),
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
        tasks.add(asyncio.create_task(publishers.publish_overlays()))
    if publish_clean_stream:
        tasks.add(asyncio.create_task(publishers.publish_clean('detail')))
        tasks.add(asyncio.create_task(publishers.publish_clean('preview')))
    try:
        await run_coupled_tasks(cast(set[asyncio.Task[object]], tasks))
    finally:
        stop_event.set()


def _validate_server_model_key(model_key: str) -> None:
    """Ensure a server model key is configured.

    Args:
        model_key: YOLO server model key.

    Raises:
        RuntimeError: If the model key is not configured.
    """
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,119}', model_key):
        raise RuntimeError('Invalid model key')


def _server_model_keys() -> tuple[str, ...]:
    """Return configured YOLO server model keys in stable order."""
    return ModelCatalogSettings().model_keys
