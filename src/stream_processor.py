from __future__ import annotations

import asyncio
import gc
import logging
import os
import tempfile
import time
from collections.abc import Iterator
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from typing import cast
from typing import Final
from typing import Protocol
from typing import TYPE_CHECKING
from typing import TypedDict
from urllib.parse import urlsplit

import cv2
import numpy as np
import torch
from dotenv import load_dotenv

from examples.streaming_web.media_paths import (
    build_annotated_media_path,
)
from examples.streaming_web.media_paths import build_clean_demand_key
from examples.streaming_web.media_paths import build_media_path
from examples.streaming_web.media_paths import build_overlay_demand_key
from examples.streaming_web.media_paths import build_overlay_ready_key
from examples.streaming_web.media_paths import build_preview_media_path
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
from src.gpu_rtsp_relay import GpuRtspRelay
from src.gpu_stream_capture import GpuFrame
from src.gpu_stream_capture import GpuStreamCapture
from src.gpu_yolo_worker import GpuYoloBatcher
from src.gpu_yolo_worker import GpuYoloWorkerClient
from src.media_restreamer import MediaSourceRestreamer
from src.media_stream_publisher import MediaStreamPublisher
from src.notifiers.fcm_notifier import FCMSender
from src.stream_capture import StreamCapture
from src.ultralytics_args import precision_kwargs
from src.ultralytics_args import PrecisionValue
from src.utils import RedisManager
from src.utils import Utils
from src.violation_sender import ViolationSender
from src.yolo_detector import YoloDetector
from src.yolo_worker import WorkerQueue
from src.yolo_worker import WorkerResultReceiver
from src.yolo_worker import YoloWorkerClient

if TYPE_CHECKING:
    RedisPrimitive = bytes | bytearray | memoryview[int] | str | int | float
else:
    RedisPrimitive = bytes | bytearray | memoryview | str | int | float

_default_warning_event_throttle_seconds: Final[int] = 30


def _media_demand_cache_seconds() -> float:
    """Return the time that viewer-demand reads may be reused per camera."""
    try:
        return max(
            0.0,
            float(os.getenv('MEDIA_DEMAND_CACHE_SECONDS', '0.5')),
        )
    except ValueError:
        return 0.5


class StreamConfig(TypedDict, total=False):
    """Configuration for one video stream from the database."""

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
class _LatestFrameState:
    """Latest camera frame shared by capture, detection, and publishing."""

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    event: asyncio.Event = field(default_factory=asyncio.Event)
    frame: np.ndarray | GpuFrame | None = None
    timestamp: float = 0.0
    sequence: int = 0


@dataclass
class _LatestDetectionState:
    """Latest detection metadata used to render shared overlay variants."""

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    event: asyncio.Event = field(default_factory=asyncio.Event)
    frame: np.ndarray | GpuFrame | None = None
    timestamp: float = 0.0
    sequence: int = 0
    warnings: object = None
    cone_polys: object = None
    pole_polys: object = None
    track_data: object = None


@dataclass(frozen=True)
class _OverlaySnapshot:
    """Source frame and metadata used to render one overlay generation."""

    sequence: tuple[int, int]
    frame: np.ndarray
    warnings: object = None
    cone_polys: object = None
    pole_polys: object = None
    track_data: object = None


@dataclass
class _OverlayPublisherVariant:
    """Publishers and demand state for one overlay media rendition."""

    media_path: str
    rendition: str
    publishers: dict[str, MediaStreamPublisher] = field(default_factory=dict)
    ready_started_at: dict[str, float] = field(default_factory=dict)


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
                self._clean_requested[media_path] = (
                    await _clean_stream_requested(redis_manager, media_path)
                )
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
                self._overlay_languages[media_path] = (
                    await _requested_overlay_languages(
                        redis_manager,
                        media_path,
                    )
                )
                self._overlay_refreshed_at[media_path] = time.monotonic()
            return set(self._overlay_languages.get(media_path, set()))


class _PreviewPublisherKwargs(TypedDict):
    """Media publisher settings for a lower-bandwidth preview rendition."""

    fps: float
    width: int
    height: int
    bitrate: str
    maxrate: str
    bufsize: str


@dataclass
class _UltralyticsStreamRuntime:
    """Per-camera state fed by one grouped Ultralytics stream predictor."""

    cfg: StreamConfig
    yolo_detector: YoloDetector
    danger_detector: DangerDetector
    fcm_sender: FCMSender
    violation_sender: ViolationSender
    redis_manager: RedisManager
    latest_frame: _LatestFrameState = field(default_factory=_LatestFrameState)
    latest_detection: _LatestDetectionState = field(
        default_factory=_LatestDetectionState,
    )
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    publisher_tasks: list[asyncio.Task[None]] = field(default_factory=list)
    last_notification_time: int = 0
    last_warning_event_time: int | None = None


class _UltralyticsStreamReadTimeout(TimeoutError):
    """Raised when an Ultralytics stream loader stops yielding frames."""


class _UltralyticsStreamSourceRetry(RuntimeError):
    """Raised when a healthy shard should retry a quarantined RTSP source."""


class _UltralyticsStreamModel(Protocol):
    """Minimal direct-stream interface provided by an Ultralytics model."""

    predictor: object | None

    def predict(self, **kwargs: object) -> Iterator[object]:
        """Run stream prediction and yield one result per input frame."""


def _preview_publisher_kwargs() -> _PreviewPublisherKwargs:
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
    yolo_result_queue: object | None = None,
) -> None:
    """Run one configured stream inside a child process."""
    load_dotenv(override=True)
    asyncio.run(
        _run_single_stream(
            cfg,
            yolo_request_queue=yolo_request_queue,
            yolo_result_queue=yolo_result_queue,
        ),
    )


def process_gpu_stream_group(configs: list[StreamConfig]) -> None:
    """Run NVDEC camera tasks and one shared CUDA YOLO batcher in a process."""
    load_dotenv(override=True)
    asyncio.run(_run_gpu_stream_group(configs))


def process_ultralytics_stream_group(configs: list[StreamConfig]) -> None:
    """Run grouped RTSP inference through Ultralytics' stream loader."""
    load_dotenv(override=True)
    asyncio.run(_run_ultralytics_stream_group(configs))


async def _run_ultralytics_stream_group(configs: list[StreamConfig]) -> None:
    """Keep one direct Ultralytics predictor alive for each model key."""
    grouped_configs: dict[str, list[StreamConfig]] = {}
    for cfg in configs:
        model_key = cfg['model_key']
        _validate_server_model_key(model_key)
        grouped_configs.setdefault(model_key, []).append(cfg)

    max_sources = _ultralytics_stream_max_sources_per_model()
    startup_lock = asyncio.Lock()
    tasks: list[asyncio.Task[None]] = []
    for model_key, model_configs in sorted(grouped_configs.items()):
        sorted_configs = sorted(
            model_configs,
            key=lambda cfg: cfg['video_url'],
        )
        shards = [
            sorted_configs[index:index + max_sources]
            for index in range(0, len(sorted_configs), max_sources)
        ]
        for shard_index, shard_configs in enumerate(shards, start=1):
            tasks.append(
                asyncio.create_task(
                    _run_ultralytics_model_stream_group(
                        model_key,
                        shard_configs,
                        shard_index=shard_index,
                        shard_count=len(shards),
                        startup_lock=startup_lock,
                    ),
                ),
            )
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def _run_ultralytics_model_stream_group(
    model_key: str,
    configs: list[StreamConfig],
    *,
    shard_index: int = 1,
    shard_count: int = 1,
    startup_lock: asyncio.Lock | None = None,
) -> None:
    """Read same-model RTSP sources directly through ``model.predict``."""
    restart_seconds = _ultralytics_stream_restart_seconds()
    source_retry_seconds = _ultralytics_stream_failed_source_retry_seconds()
    frame_timeout_seconds = _ultralytics_stream_frame_timeout_seconds()
    shard_label = f'{shard_index}/{shard_count}'
    excluded_source_urls: set[str] = set()
    retry_excluded_sources_at: float | None = None
    while True:
        restart_delay = restart_seconds
        active_configs = (
            [
                cfg
                for cfg in configs
                if cfg['video_url'] not in excluded_source_urls
            ]
            or configs
        )
        runtimes: list[_UltralyticsStreamRuntime] = []
        source_file: str | None = None
        model: _UltralyticsStreamModel | None = None
        result_stream: Iterator[object] | None = None
        try:
            for cfg in active_configs:
                runtimes.append(await _start_ultralytics_stream_runtime(cfg))

            source_file = _write_ultralytics_stream_sources(
                [runtime.cfg['video_url'] for runtime in runtimes],
            )
            _configure_ultralytics_stream_capture()
            model = _load_ultralytics_stream_model(model_key)
            result_stream = model.predict(
                source=source_file,
                stream=True,
                stream_buffer=False,
                vid_stride=_ultralytics_stream_vid_stride(),
                imgsz=int(os.getenv('YOLO_WORKER_IMGSZ', '640')),
                device=os.getenv('YOLO_WORKER_DEVICE', 'cuda:0'),
                verbose=False,
                **_ultralytics_stream_precision_kwargs(),
            )
            result_index = 0
            first_result = True
            metrics_started_at = time.monotonic()
            metrics_result_count = 0
            while True:
                if first_result and startup_lock is not None:
                    async with startup_lock:
                        result = (
                            await _next_ultralytics_stream_result_with_timeout(
                                result_stream,
                                model,
                                frame_timeout_seconds,
                            )
                        )
                else:
                    result = (
                        await _next_ultralytics_stream_result_with_timeout(
                            result_stream,
                            model,
                            frame_timeout_seconds,
                        )
                    )
                first_result = False
                if result is None:
                    raise RuntimeError('Ultralytics stream ended')
                if (
                    retry_excluded_sources_at is not None
                    and time.monotonic() >= retry_excluded_sources_at
                ):
                    excluded_source_urls.clear()
                    retry_excluded_sources_at = None
                    raise _UltralyticsStreamSourceRetry(
                        'retrying previously unavailable RTSP source',
                    )
                runtime = runtimes[result_index % len(runtimes)]
                result_index += 1
                await _apply_ultralytics_stream_result(runtime, result)
                metrics_result_count += 1
                elapsed = time.monotonic() - metrics_started_at
                if elapsed >= _ultralytics_stream_metrics_interval_seconds():
                    print(
                        '[Ultralytics-Stream] throughput '
                        f'model={model_key} shard={shard_label} '
                        'results_per_second='
                        f'{metrics_result_count / elapsed:.1f} '
                        f'cameras={len(runtimes)}',
                        flush=True,
                    )
                    metrics_started_at = time.monotonic()
                    metrics_result_count = 0
        except asyncio.CancelledError:
            raise
        except _UltralyticsStreamSourceRetry as exc:
            print(
                '[Ultralytics-Stream] model='
                f'{model_key} shard={shard_label} {exc}',
                flush=True,
            )
        except ConnectionError as exc:
            failed_source_urls = _ultralytics_stream_failed_source_urls(
                exc,
                active_configs,
            )
            if failed_source_urls:
                excluded_source_urls.update(failed_source_urls)
                retry_excluded_sources_at = (
                    time.monotonic() + source_retry_seconds
                )
                failed_streams = ', '.join(
                    f"{cfg['site']}:{cfg['stream_name']}"
                    for cfg in active_configs
                    if cfg['video_url'] in failed_source_urls
                )
                print(
                    '[Ultralytics-Stream] model='
                    f'{model_key} shard={shard_label} temporarily excludes '
                    f'unavailable source(s): {failed_streams}; retrying in '
                    f'{source_retry_seconds:.0f}s',
                    flush=True,
                )
                if len(failed_source_urls) == len(active_configs):
                    restart_delay = source_retry_seconds
            print(
                '[Ultralytics-Stream] model='
                f'{model_key} shard={shard_label} restarting after '
                f'{_ultralytics_stream_error_message(exc)}',
                flush=True,
            )
        except _UltralyticsStreamReadTimeout as exc:
            restart_delay = source_retry_seconds
            print(
                '[Ultralytics-Stream] model='
                f'{model_key} shard={shard_label} restarting after '
                f'{_ultralytics_stream_error_message(exc)}',
                flush=True,
            )
        except Exception as exc:
            print(
                '[Ultralytics-Stream] model='
                f'{model_key} shard={shard_label} restarting after '
                f'{_ultralytics_stream_error_message(exc)}',
                flush=True,
            )
        finally:
            result_stream = None
            if model is not None:
                await _release_ultralytics_stream_model(model)
            model = None
            if source_file is not None:
                try:
                    os.unlink(source_file)
                except FileNotFoundError:
                    pass
            await _close_ultralytics_stream_runtimes(runtimes)
        await asyncio.sleep(restart_delay)


def _write_ultralytics_stream_sources(sources: list[str]) -> str:
    """Write a private ``.streams`` source list understood by Ultralytics."""
    with tempfile.NamedTemporaryFile(
        mode='w',
        encoding='utf-8',
        prefix='hazard-ultralytics-',
        suffix='.streams',
        delete=False,
    ) as source_file:
        source_file.write('\n'.join(sources))
        source_file.write('\n')
        return source_file.name


def _configure_ultralytics_stream_capture() -> None:
    """Apply RTSP/TCP settings before LoadStreams opens the video source."""
    os.environ.setdefault(
        'OPENCV_FFMPEG_CAPTURE_OPTIONS',
        'rtsp_transport;tcp|stimeout;5000000|max_delay;5000000',
    )


def _load_ultralytics_stream_model(
    model_key: str,
) -> _UltralyticsStreamModel:
    """Load one local YOLO model for direct RTSP stream prediction."""
    precision = os.getenv('YOLO_WORKER_PRECISION', 'f16').strip().lower()
    if precision in {'int8', '8'}:
        model_dir = os.getenv(
            'ULTRALYTICS_STREAM_MODEL_DIR',
            'models/int8_engine',
        )
        suffix = '.engine'
    else:
        model_dir = os.getenv('ULTRALYTICS_STREAM_MODEL_DIR', 'models/pt')
        suffix = '.pt'
    model_path = os.path.join(model_dir, f'best_{model_key}{suffix}')
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f'Ultralytics stream model does not exist: {model_path}',
        )

    # LoadStreams logs full RTSP URLs at INFO and noisy wait loops at WARNING.
    # Keep direct-stream workers quiet and let our watchdog report restarts.
    logging.getLogger('ultralytics').setLevel(logging.ERROR)
    from ultralytics import YOLO

    return cast(_UltralyticsStreamModel, YOLO(model_path, task='detect'))


def _ultralytics_stream_precision_kwargs() -> dict[str, PrecisionValue]:
    """Return inference precision options for direct RTSP predictor mode."""
    precision = os.getenv('YOLO_WORKER_PRECISION', 'f16').strip().lower()
    if precision in {'int8', '8', 'engine'}:
        return {'rect': False}
    if precision in {'f32', 'fp32', '32'}:
        return precision_kwargs(False)
    return precision_kwargs(True)


def _ultralytics_stream_error_message(exc: Exception) -> str:
    """Turn TensorRT context failures into an actionable restart message."""
    message = str(exc)
    if (
        isinstance(exc, AttributeError)
        and 'NoneType' in message
        and 'set_input_shape' in message
    ):
        return (
            'TensorRT execution context was not created. '
            'Check available GPU memory and stop concurrent model-export '
            'or inference jobs before retrying.'
        )
    if isinstance(exc, ConnectionError):
        return 'ConnectionError: an RTSP source could not be opened'
    return f'{type(exc).__name__}: {message}'


def _next_ultralytics_stream_result(
    result_stream: Iterator[object],
) -> object | None:
    """Get one result without leaking ``StopIteration`` through a Future."""
    return next(result_stream, None)


async def _next_ultralytics_stream_result_with_timeout(
    result_stream: Iterator[object],
    model: object,
    timeout_seconds: float,
) -> object | None:
    """Read one Ultralytics result, restarting instead of waiting forever."""
    next_task = asyncio.create_task(
        asyncio.to_thread(_next_ultralytics_stream_result, result_stream),
    )
    done, _pending = await asyncio.wait({next_task}, timeout=timeout_seconds)
    if next_task in done:
        return next_task.result()

    await _close_ultralytics_stream_loader(model)
    close_grace_seconds = _ultralytics_stream_close_grace_seconds()
    done, _pending = await asyncio.wait(
        {next_task},
        timeout=close_grace_seconds,
    )
    if next_task in done:
        return next_task.result()

    next_task.add_done_callback(_discard_ultralytics_next_task)
    raise _UltralyticsStreamReadTimeout(
        'no frame yielded for '
        f'{timeout_seconds:.1f}s; closing stuck RTSP loader',
    )


def _discard_ultralytics_next_task(
    task: asyncio.Future[object | None],
) -> None:
    """Consume late task failures from a timed-out ``next()`` call."""
    try:
        task.result()
    except asyncio.CancelledError:
        pass
    except Exception:
        pass


async def _close_ultralytics_stream_loader(
    model: object,
) -> None:
    """Release the OpenCV readers owned by an Ultralytics predictor."""
    predictor = getattr(model, 'predictor', None)
    dataset = getattr(predictor, 'dataset', None)
    close = getattr(dataset, 'close', None)
    if callable(close):
        try:
            await asyncio.to_thread(close)
        except Exception:
            pass


async def _release_ultralytics_stream_model(
    model: object,
) -> None:
    """Release a predictor's TensorRT context before recreating its shard."""
    await _close_ultralytics_stream_loader(model)
    predictor = getattr(model, 'predictor', None)
    if predictor is not None:
        try:
            predictor.dataset = None
            predictor.model = None
        except Exception:
            pass
    try:
        setattr(model, 'predictor', None)
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _ultralytics_stream_failed_source_urls(
    exc: ConnectionError,
    configs: list[StreamConfig],
) -> set[str]:
    """Match a LoadStreams connection error to configured RTSP sources."""
    message = str(exc)
    failed_urls: set[str] = set()
    for cfg in configs:
        parts = urlsplit(cfg['video_url'])
        source_id = f'{parts.hostname or ""}{parts.path}'
        if source_id and source_id in message:
            failed_urls.add(cfg['video_url'])
    return failed_urls


async def _start_ultralytics_stream_runtime(
    cfg: StreamConfig,
) -> _UltralyticsStreamRuntime:
    """Start one camera's publishing tasks for direct Ultralytics results."""
    site = cfg['site']
    stream_name = cfg['stream_name']
    model_key = cfg['model_key']
    media_path = build_media_path(site, stream_name)
    media_publish_base = (
        os.getenv('MEDIA_PUBLISH_RTSP_BASE_URL') or 'rtsp://media-server:8554'
    ).rstrip('/')
    publish_clean_stream = _env_enabled('MEDIA_PUBLISH_CLEAN_STREAM', True)
    publish_annotated_stream = _env_enabled(
        'MEDIA_PUBLISH_ANNOTATED_STREAM',
        True,
    )
    restream_clean_source = _env_enabled(
        'MEDIA_PUBLISH_CLEAN_SOURCE_RESTREAM',
        True,
    )
    runtime = _UltralyticsStreamRuntime(
        cfg=cfg,
        yolo_detector=YoloDetector(
            model_key=model_key,
            output_folder=site,
            detect_with_server=True,
        ),
        danger_detector=DangerDetector(cfg['detection_items']),
        fcm_sender=FCMSender(api_url=os.getenv('FCM_API_URL') or ''),
        violation_sender=ViolationSender(
            api_url=os.getenv('VIOLATION_RECORD_API_URL') or '',
        ),
        redis_manager=RedisManager(),
    )
    demand_cache = _MediaDemandCache()
    print(
        f'[{site}:{stream_name}] Detection mode: ultralytics_stream '
        f'model={model_key}',
        flush=True,
    )
    if publish_clean_stream:
        print(
            f'[{site}:{stream_name}] Clean media stream is on-demand; '
            f'path {media_path}',
            flush=True,
        )
    if publish_annotated_stream:
        print(
            f'[{site}:{stream_name}] Overlay media streams are on-demand; '
            f'base path {media_path}',
            flush=True,
        )

    if publish_annotated_stream:
        runtime.publisher_tasks.append(
            asyncio.create_task(
                _publish_requested_overlay_variants(
                    latest_frame=runtime.latest_frame,
                    latest_detection=runtime.latest_detection,
                    redis_manager=runtime.redis_manager,
                    media_publish_base=media_publish_base,
                    variants=_overlay_publisher_variants(media_path),
                    site=site,
                    stream_name=stream_name,
                    stop_event=runtime.stop_event,
                    demand_cache=demand_cache,
                ),
            ),
        )
    if publish_clean_stream:
        runtime.publisher_tasks.extend([
            asyncio.create_task(
                _publish_requested_clean_frames(
                    latest_frame=runtime.latest_frame,
                    redis_manager=runtime.redis_manager,
                    media_publish_base=media_publish_base,
                    media_path=media_path,
                    site=site,
                    stream_name=stream_name,
                    source_url=cfg['video_url'],
                    use_source_restreamer=restream_clean_source,
                    stop_event=runtime.stop_event,
                    demand_cache=demand_cache,
                    rendition='detail',
                ),
            ),
            asyncio.create_task(
                _publish_requested_clean_frames(
                    latest_frame=runtime.latest_frame,
                    redis_manager=runtime.redis_manager,
                    media_publish_base=media_publish_base,
                    media_path=build_preview_media_path(media_path),
                    site=site,
                    stream_name=stream_name,
                    source_url=cfg['video_url'],
                    use_source_restreamer=False,
                    stop_event=runtime.stop_event,
                    demand_cache=demand_cache,
                    rendition='preview',
                ),
            ),
        ])
    return runtime


async def _close_ultralytics_stream_runtimes(
    runtimes: list[_UltralyticsStreamRuntime],
) -> None:
    """Stop publishers and remove live metadata for a direct stream group."""
    for runtime in runtimes:
        runtime.stop_event.set()
        for task in runtime.publisher_tasks:
            task.cancel()
    await asyncio.gather(
        *[
            task
            for runtime in runtimes
            for task in runtime.publisher_tasks
        ],
        return_exceptions=True,
    )
    for runtime in runtimes:
        await runtime.yolo_detector.close()
        try:
            await runtime.redis_manager.delete(
                _stream_metadata_key(
                    runtime.cfg['site'],
                    runtime.cfg['stream_name'],
                ),
            )
        except Exception as exc:
            print(
                '[WARN] Failed to delete Ultralytics stream metadata: '
                f'{type(exc).__name__}',
                flush=True,
            )


async def _apply_ultralytics_stream_result(
    runtime: _UltralyticsStreamRuntime,
    result: object,
) -> None:
    """Store one yielded result and publish its warning metadata."""
    source_frame = getattr(result, 'orig_img', None)
    if not isinstance(source_frame, np.ndarray):
        raise RuntimeError('Ultralytics stream result has no image frame')
    frame = np.ascontiguousarray(source_frame.copy())
    _mark_frame_readonly(frame)
    timestamp = time.time()
    async with runtime.latest_frame.lock:
        runtime.latest_frame.frame = frame
        runtime.latest_frame.timestamp = timestamp
        runtime.latest_frame.sequence += 1
        sequence = runtime.latest_frame.sequence
        runtime.latest_frame.event.set()

    detection_rows = _ultralytics_result_rows(result)
    track_data = runtime.yolo_detector.track_detections(detection_rows)
    try:
        (
            runtime.last_notification_time,
            runtime.last_warning_event_time,
        ) = await _record_detection_result(
            frame=frame,
            timestamp=timestamp,
            sequence=sequence,
            track_data=track_data,
            danger_detector=runtime.danger_detector,
            fcm_sender=runtime.fcm_sender,
            violation_sender=runtime.violation_sender,
            redis_manager=runtime.redis_manager,
            latest_detection=runtime.latest_detection,
            site=runtime.cfg['site'],
            stream_name=runtime.cfg['stream_name'],
            work_start_hour=runtime.cfg['work_start_hour'],
            work_end_hour=runtime.cfg['work_end_hour'],
            metadata_key=_stream_metadata_key(
                runtime.cfg['site'],
                runtime.cfg['stream_name'],
            ),
            last_notification_time=runtime.last_notification_time,
            last_warning_event_time=runtime.last_warning_event_time,
        )
    except Exception as exc:
        print(
            f"[{runtime.cfg['site']}:{runtime.cfg['stream_name']}] "
            'Metadata/notification error, keeping stream alive: '
            f'{type(exc).__name__}',
            flush=True,
        )


def _ultralytics_result_rows(result: object) -> list[list[float]]:
    """Convert an Ultralytics ``Results`` object's boxes to tracker rows."""
    boxes = getattr(result, 'boxes', None)
    box_data = getattr(boxes, 'data', None)
    if box_data is None:
        return []
    if hasattr(box_data, 'cpu'):
        box_data = box_data.cpu()
    rows = box_data.tolist()
    return [
        [
            float(row[0]),
            float(row[1]),
            float(row[2]),
            float(row[3]),
            float(row[-2]),
            int(row[-1]),
        ]
        for row in rows
        if len(row) >= 6
    ]


def _ultralytics_stream_restart_seconds() -> float:
    """Return the delay before a failed direct predictor reconnects."""
    try:
        return max(
            1.0,
            float(os.getenv('ULTRALYTICS_STREAM_RESTART_SECONDS', '5.0')),
        )
    except ValueError:
        return 5.0


def _ultralytics_stream_frame_timeout_seconds() -> float:
    """Return how long direct RTSP mode may wait for the next result."""
    try:
        return max(
            1.0,
            float(
                os.getenv(
                    'ULTRALYTICS_STREAM_FRAME_TIMEOUT_SECONDS',
                    '15.0',
                ),
            ),
        )
    except ValueError:
        return 15.0


def _ultralytics_stream_close_grace_seconds() -> float:
    """Return the grace period for a closed Ultralytics loader to unblock."""
    try:
        return max(
            0.1,
            float(
                os.getenv(
                    'ULTRALYTICS_STREAM_CLOSE_GRACE_SECONDS',
                    '10.0',
                ),
            ),
        )
    except ValueError:
        return 10.0


def _ultralytics_stream_failed_source_retry_seconds() -> float:
    """Return the delay before retrying an unavailable RTSP source."""
    try:
        return max(
            5.0,
            float(
                os.getenv(
                    'ULTRALYTICS_STREAM_FAILED_SOURCE_RETRY_SECONDS',
                    '300.0',
                ),
            ),
        )
    except ValueError:
        return 300.0


def _ultralytics_stream_vid_stride() -> int:
    """Return the frame stride passed to Ultralytics' RTSP loader."""
    try:
        return max(1, int(os.getenv('ULTRALYTICS_STREAM_VID_STRIDE', '1')))
    except ValueError:
        return 1


def _ultralytics_stream_max_sources_per_model() -> int:
    """Return the maximum number of RTSP sources in one predictor shard."""
    try:
        return max(
            1,
            int(os.getenv('ULTRALYTICS_STREAM_MAX_SOURCES_PER_MODEL', '3')),
        )
    except ValueError:
        return 3


def _ultralytics_stream_metrics_interval_seconds() -> float:
    """Return the periodic direct-predictor throughput report interval."""
    try:
        return max(
            1.0,
            float(
                os.getenv(
                    'ULTRALYTICS_STREAM_METRICS_INTERVAL_SECONDS',
                    '10.0',
                ),
            ),
        )
    except ValueError:
        return 10.0


def _env_enabled(name: str, default: bool) -> bool:
    """Read a boolean environment variable with an explicit default."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {'1', 'true', 'yes', 'on'}


async def _run_gpu_stream_group(configs: list[StreamConfig]) -> None:
    """Keep camera tasks alive around one shared CUDA model cache."""
    batcher = GpuYoloBatcher()
    await batcher.start()
    tasks = [
        asyncio.create_task(
            _run_gpu_stream_with_restart(
                cfg,
                batcher.client(f"{cfg['site']}|{cfg['stream_name']}"),
            ),
        )
        for cfg in configs
    ]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await batcher.close()


async def _run_gpu_stream_with_restart(
    cfg: StreamConfig,
    worker_client: GpuYoloWorkerClient,
) -> None:
    """Keep one camera on its relay-to-NVDEC processing path."""
    retry_seconds = max(
        1.0,
        float(os.getenv('GPU_DECODE_STREAM_RETRY_SECONDS', '5.0')),
    )
    relay = GpuRtspRelay(cfg['video_url'])
    try:
        while True:
            try:
                await relay.start()
                if not relay.is_running:
                    raise RuntimeError('GPU TCP relay is not running')
                await asyncio.sleep(_gpu_relay_startup_seconds())
                await _run_single_stream(
                    cfg,
                    gpu_yolo_client=worker_client,
                    gpu_decode_stream_url=relay.publish_url,
                )
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(
                    f"[{cfg['site']}:{cfg['stream_name']}] GPU relay/NVDEC "
                    f'task restarting after {type(exc).__name__}',
                    flush=True,
                )
                await asyncio.sleep(retry_seconds)
    finally:
        await relay.close()


async def _run_single_stream(
    cfg: StreamConfig,
    yolo_request_queue: object | None = None,
    yolo_result_queue: object | None = None,
    gpu_yolo_client: GpuYoloWorkerClient | None = None,
    gpu_decode_stream_url: str | None = None,
) -> None:
    """Run one stream processing coroutine for the given configuration."""
    if not cfg.get('recognition_enabled', True):
        print(
            f"[{cfg['site']}:{cfg['stream_name']}] Recognition disabled; "
            'skipping stream processor startup',
            flush=True,
        )
        return

    video_url = cfg['video_url']
    model_key = cfg['model_key']
    site = cfg['site']
    stream_name = cfg['stream_name']
    detection_items = cfg['detection_items']
    work_start_hour = cfg['work_start_hour']
    work_end_hour = cfg['work_end_hour']
    # Every active recognition stream also publishes its MediaMTX live view.
    live_view_enabled = True
    print(
        f'[{site}:{stream_name}] Streaming output mode: media_server',
        flush=True,
    )
    print(
        f'[{site}:{stream_name}] Detection mode: server',
        flush=True,
    )

    gpu_decode_enabled = _gpu_decode_enabled()
    if gpu_yolo_client is not None:
        if not gpu_decode_enabled or not GpuStreamCapture.is_available():
            raise RuntimeError(
                'GPU relay streams require TorchCodec NVDEC support',
            )
        if gpu_decode_stream_url is None:
            raise RuntimeError(
                'GPU relay streams require a local relay URL',
            )
        streaming_capture: StreamCapture | GpuStreamCapture = GpuStreamCapture(
            stream_url=gpu_decode_stream_url,
        )
        yolo_detector = YoloDetector(
            model_key=model_key,
            output_folder=site,
            detect_with_server=True,
            worker_client=gpu_yolo_client,
        )
        print(
            f'[{site}:{stream_name}] GPU TCP relay, NVDEC, and shared '
            'batch YOLO enabled',
            flush=True,
        )
    else:
        if gpu_decode_enabled:
            print(
                f'[{site}:{stream_name}] GPU decode unavailable; '
                'using shared YOLO worker',
                flush=True,
            )
        streaming_capture = StreamCapture(stream_url=video_url)
        worker_client = (
            YoloWorkerClient(
                cast(WorkerQueue, yolo_request_queue),
                cast(WorkerResultReceiver, yolo_result_queue),
                camera_id=f'{site}|{stream_name}',
                timeout_seconds=float(
                    os.getenv('YOLO_WORKER_TIMEOUT_SECONDS', '30.0'),
                ),
            )
            if (
                os.getenv(
                    'YOLO_WORKER_ENABLED',
                    'true',
                ).strip().lower() in {'1', 'true', 'yes', 'on'}
                and yolo_request_queue is not None
                and yolo_result_queue is not None
            )
            else None
        )
        yolo_detector = YoloDetector(
            model_key=model_key,
            output_folder=site,
            detect_with_server=True,
            worker_client=worker_client,
        )
        _validate_server_model_key(model_key)
        if worker_client is None:
            raise RuntimeError(
                'YOLO_WORKER_ENABLED=true and shared worker result queues are '
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
        await _close_overlay_publishers(overlay_media_publishers)
        await _close_overlay_publishers(preview_overlay_media_publishers)
        if redis_manager is not None:
            try:
                await redis_manager.delete(metadata_key)
            except Exception as e:
                print(f'[WARN] Failed to delete redis key {metadata_key}: {e}')


async def _run_inline_stream_loop(
    streaming_capture: StreamCapture | GpuStreamCapture,
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
    demand_cache = _MediaDemandCache()
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
                await _publish_requested_overlay_snapshot_variants(
                    redis_manager=redis_manager,
                    variants=_inline_overlay_publisher_variants(
                        media_path=media_path,
                        overlay_media_publishers=overlay_media_publishers,
                        overlay_ready_started_at=overlay_ready_started_at,
                        preview_overlay_media_publishers=(
                            preview_overlay_media_publishers
                        ),
                        preview_overlay_ready_started_at=(
                            preview_overlay_ready_started_at
                        ),
                    ),
                    media_publish_base=media_publish_base,
                    site=site,
                    stream_name=stream_name,
                    source_frame=frame,
                    warnings=None,
                    cone_polys=None,
                    pole_polys=None,
                    track_data=None,
                    demand_cache=demand_cache,
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
        should_send_violation = (
            is_working
            and bool(warnings)
            and Utils.should_notify(
                current_timestamp,
                last_notification_time,
            )
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
                    and await demand_cache.clean_requested(
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
                        await clean_media_publisher.publish(
                            _frame_for_cpu_consumers(frame),
                        )
                else:
                    if clean_source_restreamer is not None:
                        await clean_source_restreamer.close()
                        clean_source_restreamer = None
                    if clean_media_publisher is not None:
                        await clean_media_publisher.close()
                        clean_media_publisher = None

                preview_clean_requested = (
                    publish_clean_stream
                    and await demand_cache.clean_requested(
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
                    await preview_clean_media_publisher.publish(
                        _frame_for_cpu_consumers(frame),
                    )
                elif preview_clean_media_publisher is not None:
                    await preview_clean_media_publisher.close()
                    preview_clean_media_publisher = None
                if publish_annotated_stream:
                    await _publish_requested_overlay_snapshot_variants(
                        redis_manager=redis_manager,
                        variants=_inline_overlay_publisher_variants(
                            media_path=media_path,
                            overlay_media_publishers=overlay_media_publishers,
                            overlay_ready_started_at=overlay_ready_started_at,
                            preview_overlay_media_publishers=(
                                preview_overlay_media_publishers
                            ),
                            preview_overlay_ready_started_at=(
                                preview_overlay_ready_started_at
                            ),
                        ),
                        media_publish_base=media_publish_base,
                        site=site,
                        stream_name=stream_name,
                        source_frame=frame,
                        warnings=warnings,
                        cone_polys=cone_polys,
                        pole_polys=pole_polys,
                        track_data=track_data,
                        demand_cache=demand_cache,
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
                frame=_frame_for_cpu_consumers(frame),
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
    streaming_capture: StreamCapture | GpuStreamCapture,
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
    source_reconnect_event = _capture_reconnect_event(streaming_capture)
    demand_cache = _MediaDemandCache()
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
                    source_reconnect_event=source_reconnect_event,
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
                    source_reconnect_event=source_reconnect_event,
                    demand_cache=demand_cache,
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
    streaming_capture: StreamCapture | GpuStreamCapture,
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
        if isinstance(frame, np.ndarray):
            _mark_frame_readonly(frame)
        async with latest_frame.lock:
            latest_frame.frame = frame
            latest_frame.timestamp = ts
            latest_frame.sequence += 1
            latest_frame.event.set()


def _capture_reconnect_event(
    streaming_capture: StreamCapture | GpuStreamCapture,
) -> asyncio.Event | None:
    """Return the CPU RTSP reconnect signal when the capture exposes one."""
    event = getattr(streaming_capture, 'reconnect_event', None)
    return event if isinstance(event, asyncio.Event) else None


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
            (
                last_notification_time,
                last_warning_event_time,
            ) = await _record_detection_result(
                frame=frame,
                timestamp=ts,
                sequence=last_sequence,
                track_data=track_data,
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
            print(
                f'[{site}:{stream_name}] Metadata/notification error, keeping '
                f'stream alive: {exc}',
                flush=True,
            )
            await asyncio.sleep(0.2)


async def _record_detection_result(
    frame: np.ndarray | GpuFrame,
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
        latest_detection.warnings = warnings
        latest_detection.cone_polys = cone_polys
        latest_detection.pole_polys = pole_polys
        latest_detection.track_data = track_data
        latest_detection.event.set()

    should_send_violation = (
        is_working
        and bool(warnings)
        and Utils.should_notify(
            current_timestamp,
            last_notification_time,
        )
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
        last_notification_time = await _send_violation_and_notification(
            fcm_sender=fcm_sender,
            violation_sender=violation_sender,
            site=site,
            stream_name=stream_name,
            warnings=warnings,
            detection_time=detection_time,
            frame=_frame_for_cpu_consumers(frame),
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
    demand_cache: _MediaDemandCache | None = None,
    rendition: str = 'detail',
) -> None:
    """Compatibility wrapper for publishing one overlay rendition."""
    await _publish_requested_overlay_variants(
        latest_frame=latest_frame,
        latest_detection=latest_detection,
        redis_manager=redis_manager,
        media_publish_base=media_publish_base,
        variants=[_OverlayPublisherVariant(media_path, rendition)],
        site=site,
        stream_name=stream_name,
        stop_event=stop_event,
        demand_cache=demand_cache,
    )


def _overlay_publisher_variants(
    media_path: str,
) -> list[_OverlayPublisherVariant]:
    """Build the detail and preview variants that share overlay rendering."""
    return [
        _OverlayPublisherVariant(media_path, 'detail'),
        _OverlayPublisherVariant(
            build_preview_media_path(media_path), 'preview',
        ),
    ]


def _inline_overlay_publisher_variants(
    media_path: str,
    overlay_media_publishers: dict[str, MediaStreamPublisher],
    overlay_ready_started_at: dict[str, float],
    preview_overlay_media_publishers: dict[str, MediaStreamPublisher],
    preview_overlay_ready_started_at: dict[str, float],
) -> list[_OverlayPublisherVariant]:
    """Bind inline-loop publisher maps to their detail and preview paths."""
    return [
        _OverlayPublisherVariant(
            media_path=media_path,
            rendition='detail',
            publishers=overlay_media_publishers,
            ready_started_at=overlay_ready_started_at,
        ),
        _OverlayPublisherVariant(
            media_path=build_preview_media_path(media_path),
            rendition='preview',
            publishers=preview_overlay_media_publishers,
            ready_started_at=preview_overlay_ready_started_at,
        ),
    ]


def _overlay_variant_fps(variants: list[_OverlayPublisherVariant]) -> float:
    """Return one source update rate sufficient for every rendition."""
    rates = [
        _preview_publisher_kwargs()['fps']
        if variant.rendition == 'preview'
        else max(1.0, float(os.getenv('MEDIA_PUBLISH_FPS', '15.0')))
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
                            frame=_build_media_startup_frame(
                                site,
                                stream_name,
                            ),
                        )
                    publish_frames = _overlay_publish_frames(
                        snapshot,
                        requested_languages,
                        rendered_overlay_cache,
                    )
                    for variant in variants:
                        for language in sorted(
                            requested_by_variant[variant.media_path],
                        ):
                            await _publish_overlay_language_snapshot(
                                redis_manager=redis_manager,
                                overlay_media_publishers=variant.publishers,
                                rendered_overlay_cache=rendered_overlay_cache,
                                media_publish_base=media_publish_base,
                                media_path=variant.media_path,
                                site=site,
                                stream_name=stream_name,
                                label_language=language,
                                snapshot=snapshot,
                                overlay_ready_started_at=(
                                    variant.ready_started_at
                                ),
                                rendition=variant.rendition,
                                publish_frame=publish_frames[language],
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

    if source_frame is not None:
        return _OverlaySnapshot(
            sequence=(source_sequence, detection_sequence),
            frame=_frame_for_cpu_consumers(source_frame),
            warnings=warnings,
            cone_polys=cone_polys,
            pole_polys=pole_polys,
            track_data=track_data,
        )
    if detection_frame is not None:
        return _OverlaySnapshot(
            sequence=(source_sequence, detection_sequence),
            frame=_frame_for_cpu_consumers(detection_frame),
            warnings=warnings,
            cone_polys=cone_polys,
            pole_polys=pole_polys,
            track_data=track_data,
        )
    return None


async def _requested_overlay_languages(
    redis_manager: RedisManager,
    media_path: str,
) -> set[str]:
    """Read demand with bounded ``MGET`` instead of scanning Redis keys."""
    allowed_languages = _allowed_overlay_languages()
    demand_keys = [
        build_overlay_demand_key(media_path, language)
        for language in allowed_languages
    ]
    values = await redis_manager.redis.mget(demand_keys)
    return {
        language
        for language, value in zip(allowed_languages, values, strict=True)
        if value is not None
    }


async def _publish_requested_overlay_snapshot(
    redis_manager: RedisManager,
    overlay_media_publishers: dict[str, MediaStreamPublisher],
    media_publish_base: str,
    media_path: str,
    site: str,
    stream_name: str,
    source_frame: np.ndarray | GpuFrame,
    warnings: object,
    cone_polys: object,
    pole_polys: object,
    track_data: object,
    overlay_ready_started_at: dict[str, float] | None = None,
    demand_cache: _MediaDemandCache | None = None,
    rendition: str = 'detail',
) -> None:
    """Compatibility wrapper for publishing one overlay rendition snapshot."""
    await _publish_requested_overlay_snapshot_variants(
        redis_manager=redis_manager,
        variants=[
            _OverlayPublisherVariant(
                media_path=media_path,
                rendition=rendition,
                publishers=overlay_media_publishers,
                ready_started_at=(
                    overlay_ready_started_at
                    if overlay_ready_started_at is not None
                    else {}
                ),
            ),
        ],
        media_publish_base=media_publish_base,
        site=site,
        stream_name=stream_name,
        source_frame=source_frame,
        warnings=warnings,
        cone_polys=cone_polys,
        pole_polys=pole_polys,
        track_data=track_data,
        demand_cache=demand_cache,
    )


async def _publish_requested_overlay_snapshot_variants(
    redis_manager: RedisManager,
    variants: list[_OverlayPublisherVariant],
    media_publish_base: str,
    site: str,
    stream_name: str,
    source_frame: np.ndarray | GpuFrame,
    warnings: object,
    cone_polys: object,
    pole_polys: object,
    track_data: object,
    demand_cache: _MediaDemandCache | None = None,
) -> None:
    """Publish one rendered language frame to all requested media variants."""
    demand_cache = demand_cache or _MediaDemandCache()
    requested_by_variant = {
        variant.media_path: await demand_cache.overlay_languages(
            redis_manager,
            variant.media_path,
        )
        for variant in variants
    }
    for variant in variants:
        requested_languages = requested_by_variant[variant.media_path]
        await _close_unrequested_overlay_publishers(
            variant.publishers,
            requested_languages,
        )
        _drop_unrequested_overlay_start_times(
            variant.ready_started_at,
            requested_languages,
        )
    requested_languages = set().union(*requested_by_variant.values())
    if not requested_languages:
        return
    snapshot = _OverlaySnapshot(
        sequence=(0, 0),
        frame=_frame_for_cpu_consumers(source_frame),
        warnings=warnings,
        cone_polys=cone_polys,
        pole_polys=pole_polys,
        track_data=track_data,
    )
    rendered_overlay_cache: dict[
        str,
        tuple[tuple[int, int], np.ndarray],
    ] = {}
    publish_frames = _overlay_publish_frames(
        snapshot,
        requested_languages,
        rendered_overlay_cache,
    )
    for variant in variants:
        for language in sorted(requested_by_variant[variant.media_path]):
            await _publish_overlay_language_snapshot(
                redis_manager=redis_manager,
                overlay_media_publishers=variant.publishers,
                rendered_overlay_cache=rendered_overlay_cache,
                media_publish_base=media_publish_base,
                media_path=variant.media_path,
                site=site,
                stream_name=stream_name,
                label_language=language,
                snapshot=snapshot,
                overlay_ready_started_at=variant.ready_started_at,
                rendition=variant.rendition,
                publish_frame=publish_frames[language],
            )


async def _publish_overlay_language_snapshot(
    redis_manager: RedisManager,
    overlay_media_publishers: dict[str, MediaStreamPublisher],
    rendered_overlay_cache: dict[
        str,
        tuple[tuple[int, int], np.ndarray],
    ],
    media_publish_base: str,
    media_path: str,
    site: str,
    stream_name: str,
    label_language: str,
    snapshot: _OverlaySnapshot,
    overlay_ready_started_at: dict[str, float] | None = None,
    rendition: str = 'detail',
    publish_frame: np.ndarray | None = None,
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

    if publish_frame is None and snapshot.track_data is None:
        publish_frame = snapshot.frame
    elif publish_frame is None:
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
    assert publish_frame is not None
    await publisher.publish(publish_frame)
    if _overlay_ready_grace_elapsed(overlay_ready_started_at, label_language):
        await _mark_overlay_ready(redis_manager, overlay_path)


def _overlay_publish_frames(
    snapshot: _OverlaySnapshot,
    requested_languages: set[str],
    rendered_overlay_cache: dict[
        str,
        tuple[tuple[int, int], np.ndarray],
    ],
) -> dict[str, np.ndarray]:
    """Build at most one immutable overlay frame per requested language."""
    if snapshot.track_data is None:
        return {language: snapshot.frame for language in requested_languages}
    publish_frames: dict[str, np.ndarray] = {}
    for language in requested_languages:
        cached = rendered_overlay_cache.get(language)
        if cached is not None and cached[0] == snapshot.sequence:
            publish_frames[language] = cached[1]
            continue
        publish_frame = _build_media_publish_frame(
            frame=snapshot.frame,
            warnings=snapshot.warnings,
            cone_polys=snapshot.cone_polys,
            pole_polys=snapshot.pole_polys,
            track_data=snapshot.track_data,
            label_language=language,
        )
        rendered_overlay_cache[language] = (snapshot.sequence, publish_frame)
        publish_frames[language] = publish_frame
    return publish_frames


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
    """Publish one clean detail or preview rendition while requested."""
    demand_cache = demand_cache or _MediaDemandCache()
    fps = (
        _preview_publisher_kwargs()['fps']
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
                    if clean_restreamer is not None:
                        await clean_restreamer.close()
                        clean_restreamer = None
                    if clean_publisher is not None:
                        await clean_publisher.close()
                        clean_publisher = None
                    await asyncio.sleep(frame_interval)
                    continue

                if use_source_restreamer and rendition == 'detail':
                    if (
                        source_reconnect_event is not None
                        and source_reconnect_event.is_set()
                    ):
                        source_reconnect_event.clear()
                        if clean_restreamer is not None:
                            print(
                                f'[{site}:{stream_name}] Restarting clean '
                                'source restream after frozen-frame '
                                'reconnect',
                                flush=True,
                            )
                            await clean_restreamer.restart()
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
                    await clean_publisher.publish(
                        _frame_for_cpu_consumers(frame),
                    )
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


def _gpu_decode_enabled() -> bool:
    """Return whether a stream should use the NVDEC local-inference path."""
    return os.getenv('GPU_DECODE_ENABLED', 'false').strip().lower() in {
        '1',
        'true',
        'yes',
        'on',
    }


def _gpu_relay_startup_seconds() -> float:
    """Return the time allowed for MediaMTX to expose a new relay path."""
    try:
        return max(
            0.1,
            float(os.getenv('GPU_DECODE_RELAY_STARTUP_SECONDS', '1.0')),
        )
    except ValueError:
        return 1.0


def _frame_for_cpu_consumers(frame: np.ndarray | GpuFrame) -> np.ndarray:
    """Return BGR data only when publishing or storing a CPU-side image."""
    if isinstance(frame, GpuFrame):
        return frame.to_bgr()
    return frame


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
