from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from dataclasses import field
from functools import partial

import numpy as np

from examples.streaming_web.media_paths import build_annotated_media_path
from examples.streaming_web.media_paths import build_clean_demand_key
from examples.streaming_web.media_paths import build_overlay_ready_key
from examples.streaming_web.media_paths import build_preview_media_path
from examples.streaming_web.overlay_renderer import OverlayRenderer
from examples.streaming_web.playback_demand import active_overlay_languages
from src import stream_overlay_frames as frames
from src.media_publish_config import MediaPublisherFactory
from src.media_restreamer import MediaSourceRestreamer
from src.media_stream_publisher import MediaStreamPublisher
from src.overlay_pipeline import LatestOverlayPublication
from src.overlay_pipeline import OverlayRenderPool
from src.redis_client import RedisManager
from src.settings.media import MediaPublishSettings
from src.settings.media import OverlaySettings
from src.settings.overlay_language import get_overlay_language_policy
from src.stream_runtime_state import (
    LatestDetectionState as _LatestDetectionState,
)
from src.stream_runtime_state import LatestFrameState as _LatestFrameState
from src.stream_runtime_state import (
    OverlayPublisherVariant as _OverlayPublisherVariant,
)
from src.stream_runtime_state import OverlaySnapshot as _OverlaySnapshot
from src.stream_runtime_state import StreamRuntime

logger = logging.getLogger(__name__)


def _demand_cache_seconds() -> float:
    return MediaPublishSettings().demand_cache_seconds


def _default_overlay_languages() -> tuple[str, ...]:
    return get_overlay_language_policy().allowed


@dataclass
class _MediaDemandCache:
    """Share short-lived clean and overlay demand reads for one camera."""

    refresh_seconds: float = field(
        default_factory=_demand_cache_seconds,
    )
    allowed_languages: tuple[str, ...] = field(
        default_factory=_default_overlay_languages,
    )
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
                    await active_overlay_languages(
                        redis_manager.redis,
                        media_path,
                        self.allowed_languages,
                    )
                )
                self._overlay_refreshed_at[media_path] = time.monotonic()
            return set(self._overlay_languages.get(media_path, set()))


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


def _drop_unrequested_overlay_cache(
    rendered_overlay_cache: dict[str, tuple[tuple[int, int], np.ndarray]],
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


async def _close_clean_media_publishers(
    clean_restreamer: MediaSourceRestreamer | None,
    clean_publisher: MediaStreamPublisher | None,
) -> tuple[None, None]:
    """Close clean-stream publishers and reset their local state."""
    if clean_restreamer is not None:
        await clean_restreamer.close()
    if clean_publisher is not None:
        await clean_publisher.close()
    return (None, None)


async def _clean_stream_requested(
    redis_manager: RedisManager,
    media_path: str,
) -> bool:
    """Return True while at least one viewer wants the clean stream."""
    return bool(
        await redis_manager.redis.exists(build_clean_demand_key(media_path)),
    )


class StreamPublisherGroup:

    def __init__(self, runtime: StreamRuntime) -> None:
        self.runtime = runtime
        self.settings = runtime.media.settings
        self.factory = MediaPublisherFactory(self.settings, runtime.capture)
        self.overlays = frames.OverlayFrameCache(
            OverlayRenderer(
                OverlaySettings(
                    min_confidence=self.settings.overlay_min_confidence,
                    box_thickness=self.settings.box_thickness,
                ),
            ),
        )
        languages = get_overlay_language_policy()
        languages.default_language()
        self.demand_cache = _MediaDemandCache(
            self.settings.demand_cache_seconds,
            languages.allowed,
        )
        self.variants = _overlay_publisher_variants(runtime.media.path)
        self.clean_reconnect_event = asyncio.Event()

    async def publish_overlays(self) -> None:
        redis_manager = self.runtime.redis
        """Render each language once and publish it to requested variants."""
        latest_frame = self.runtime.latest_frame
        latest_detection = self.runtime.latest_detection
        variants = self.variants
        site, stream_name = (self.runtime.site, self.runtime.stream_name)
        stop_event = self.runtime.stop
        cache = self.demand_cache
        render_pool = OverlayRenderPool(self.overlays)
        publications: dict[tuple[str, str], LatestOverlayPublication] = {}
        try:
            while not stop_event.is_set():
                started_at = asyncio.get_running_loop().time()
                frame_interval = 1.0 / max(
                    self.factory.fps(v.rendition) for v in variants
                )
                try:
                    requested_by_variant = {
                        variant.media_path: await cache.overlay_languages(
                            redis_manager,
                            variant.media_path,
                        )
                        for variant in variants
                    }
                    requested_pairs = {
                        (variant.media_path, language)
                        for variant in variants
                        for language in requested_by_variant[
                            variant.media_path
                        ]
                    }
                    retired = [
                        publications.pop(key) for key in list(publications)
                        if key not in requested_pairs
                    ]
                    await asyncio.gather(
                        *(worker.close() for worker in retired),
                    )
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
                    if requested_languages:
                        snapshot = await _latest_overlay_snapshot(
                            latest_frame=latest_frame,
                            latest_detection=latest_detection,
                        )
                        if snapshot is None:
                            snapshot = _OverlaySnapshot(
                                sequence=(-1, -1),
                                frame=frames._build_media_startup_frame(
                                    site,
                                    stream_name,
                                    (
                                        self.settings.startup_width,
                                        self.settings.startup_height,
                                    ),
                                ),
                            )
                        publish_frames = await render_pool.render(
                            snapshot, requested_languages,
                        )
                        for variant in variants:
                            for language in sorted(
                                requested_by_variant[variant.media_path],
                            ):
                                key = (variant.media_path, language)
                                if key not in publications:
                                    worker = LatestOverlayPublication(
                                        partial(
                                            self.publish_overlay,
                                            variant, language,
                                        ),
                                    )
                                    publications[key] = worker
                                publications[key].offer(
                                    publish_frames[language],
                                )
                    else:
                        await render_pool.render(None, set())
                except Exception as exc:
                    logger.info(
                        f'[{site}:{stream_name}] Overlay publish loop error: '
                        f'{exc}',
                    )
                    await asyncio.sleep(0.5)
                await asyncio.sleep(
                    max(
                        0.0,
                        frame_interval
                        - (asyncio.get_running_loop().time() - started_at),
                    ),
                )
        finally:
            await asyncio.gather(
                *(worker.close() for worker in publications.values()),
            )
            render_pool.close()
            for variant in variants:
                await _close_overlay_publishers(variant.publishers)

    async def publish_overlay(
        self,
        variant: _OverlayPublisherVariant,
        language: str,
        publish_frame: np.ndarray,
    ) -> None:
        """Publish a rendered language frame through its shared publisher."""
        media_publish_base = self.runtime.media.publish_base
        media_path = variant.media_path
        site, stream_name = (self.runtime.site, self.runtime.stream_name)
        label_language = language
        overlay_media_publishers = variant.publishers
        overlay_ready_started_at = variant.ready_started_at
        rendition = variant.rendition
        overlay_path = build_annotated_media_path(media_path, label_language)
        publisher = overlay_media_publishers.get(label_language)
        previous = publisher
        publisher = await self.factory.ensure(
            f'{media_publish_base}/{overlay_path}',
            rendition=rendition, current=publisher,
        )
        overlay_media_publishers[label_language] = publisher
        if publisher is not previous:
            if previous is not None:
                overlay_ready_started_at.pop(label_language, None)
            logger.info(
                f'[{site}:{stream_name}] Starting shared overlay stream '
                f'{label_language}: {media_publish_base}/{overlay_path}',
            )
        await publisher.publish(publish_frame)
        if self._overlay_ready_grace_elapsed(
            overlay_ready_started_at,
            label_language,
        ):
            await self._mark_overlay_ready(overlay_path)

    async def publish_clean(self, rendition: str = 'detail') -> None:
        redis_manager = self.runtime.redis
        """Publish one clean detail or preview rendition while requested.

        A capture reconnect intentionally invalidates ``latest_frame`` so stale
        An existing encoder repeats its last valid frame during reconnects,
        preserving the RTSP and HLS session until
        fresh source video arrives.
        """
        latest_frame = self.runtime.latest_frame
        media_path = (
            self.runtime.media.path
            if rendition == 'detail'
            else build_preview_media_path(
                self.runtime.media.path,
            )
        )
        site, stream_name = (self.runtime.site, self.runtime.stream_name)
        use_source_restreamer = (
            self.runtime.media.restream_clean and rendition == 'detail'
        )
        stop_event = self.runtime.stop
        cache = self.demand_cache
        clean_publisher: MediaStreamPublisher | None = None
        clean_restreamer: MediaSourceRestreamer | None = None
        try:
            while not stop_event.is_set():
                started_at = asyncio.get_running_loop().time()
                frame_interval = 1.0 / self.factory.fps(rendition)
                try:
                    requested = await cache.clean_requested(
                        redis_manager,
                        media_path,
                    )
                    if not requested:
                        clean_restreamer, clean_publisher = (
                            await _close_clean_media_publishers(
                                clean_restreamer,
                                clean_publisher,
                            )
                        )
                        await asyncio.sleep(frame_interval)
                        continue
                    if use_source_restreamer and rendition == 'detail':
                        clean_restreamer = await self._ensure_clean_restreamer(
                            clean_restreamer,
                        )
                        await asyncio.sleep(frame_interval)
                        continue
                    async with latest_frame.lock:
                        frame = latest_frame.frame
                    if frame is None:
                        await asyncio.sleep(frame_interval)
                        continue
                    clean_publisher = await self._ensure_clean_publisher(
                        clean_publisher, rendition,
                    )
                    await clean_publisher.publish(frame)
                except Exception as exc:
                    logger.info(
                        f'[{site}:{stream_name}] Clean media publish error, '
                        f'keeping stream alive: {exc}',
                    )
                    await asyncio.sleep(0.5)
                await asyncio.sleep(
                    max(
                        0.0,
                        frame_interval
                        - (asyncio.get_running_loop().time() - started_at),
                    ),
                )
        finally:
            await _close_clean_media_publishers(
                clean_restreamer,
                clean_publisher,
            )

    def _overlay_ready_grace_elapsed(
        self, started_at: dict[str, float] | None, language: str,
    ) -> bool:
        if started_at is None:
            return True
        now = asyncio.get_running_loop().time()
        first = started_at.setdefault(language, now)
        return now - first >= self.settings.ready_grace_seconds

    async def _mark_overlay_ready(self, path: str) -> None:
        await self.runtime.redis.redis.set(
            build_overlay_ready_key(path),
            b'1',
            ex=self.settings.ready_ttl_seconds,
        )

    async def _ensure_clean_publisher(
        self, current: MediaStreamPublisher | None, rendition: str,
    ) -> MediaStreamPublisher:
        path = self.runtime.media.path
        if rendition == 'preview':
            path = build_preview_media_path(path)
        return await self.factory.ensure(
            f'{self.runtime.media.publish_base}/{path}', rendition=rendition,
            current=current,
        )

    async def _ensure_clean_restreamer(
        self, current: MediaSourceRestreamer | None,
    ) -> MediaSourceRestreamer:
        if self.clean_reconnect_event.is_set():
            self.clean_reconnect_event.clear()
            if current is not None:
                await current.restart()
        if current is None:
            current = MediaSourceRestreamer(
                source_url=self.runtime.cfg['video_url'],
                publish_url=(
                    f'{self.runtime.media.publish_base}/'
                    f'{self.runtime.media.path}'
                ),
                encoding=self.factory.encoding,
            )
            await current.start()
        return current
