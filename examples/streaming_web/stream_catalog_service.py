from __future__ import annotations

import redis.asyncio as redis
from fastapi import HTTPException
from sqlalchemy import Select
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.jwt_config import JwtAuthorizationCredentials
from examples.auth.models import Site
from examples.auth.models import StreamConfig as StreamConfigModel
from examples.auth.user_service import load_user_access_context
from examples.streaming_web.media_paths import build_annotated_media_path
from examples.streaming_web.media_paths import build_media_hls_url
from examples.streaming_web.media_paths import build_media_path
from examples.streaming_web.media_paths import build_media_webrtc_url
from examples.streaming_web.media_paths import decode_media_segment
from examples.streaming_web.metadata_keys import encode_stream_id
from examples.streaming_web.overlay_renderer import normalise_label_language
from examples.streaming_web.playback_service import _overlay_is_ready
from examples.streaming_web.playback_service import _touch_overlay_demand
from examples.streaming_web.playback_service import OVERLAY_DEMAND_TTL_SECONDS
from examples.streaming_web.schemas import LabelListResponse


async def get_visible_labels(
    credentials: JwtAuthorizationCredentials,
    db: AsyncSession,
) -> LabelListResponse:
    """Return site labels visible to the authenticated user.

    Args:
        credentials: Verified JWT credentials for the caller.
        db: Database session used to resolve access and sites.

    Returns:
        Validated labels available to the caller.

    Raises:
        HTTPException: If the token does not identify a username.
    """
    username = credentials.subject.get('username')
    if not username:
        raise HTTPException(status_code=401, detail='Invalid token: no subject')

    _, user_site_names, user_role = await load_user_access_context(
        db,
        username,
    )
    result = await db.execute(select(Site.name).order_by(Site.name))
    all_labels = list(result.scalars().all())
    return LabelListResponse(
        labels=(
            all_labels
            if user_role == 'super_admin'
            else [name for name in all_labels if name in user_site_names]
        ),
    )


def _visible_stream_names_query(label: str) -> Select[tuple[str]]:
    """Build the query for streams enabled for live playback.

    Args:
        label: Site label whose configured streams are selected.

    Returns:
        SQLAlchemy select statement yielding configured stream names.
    """
    return (
        select(StreamConfigModel.stream_name)
        .join(Site)
        .where(
            Site.name == label,
            StreamConfigModel.recognition_enabled.is_(True),
        )
    )


async def _resolve_configured_stream_name(
    db: AsyncSession,
    label: str,
    stream_id: str | None,
    key: str | None,
) -> str:
    """Resolve and validate an enabled configured stream name.

    Args:
        db: Database session used to query configured streams.
        label: Site label containing the requested stream.
        stream_id: Optional encoded configured stream identifier.
        key: Optional decoded configured stream name.

    Returns:
        Validated decoded configured stream name.

    Raises:
        HTTPException: If the stream identifier is absent or not enabled.
    """
    requested_name = key or (
        decode_media_segment(stream_id) if stream_id else ''
    )
    if not requested_name:
        raise HTTPException(
            status_code=422, detail='stream_id_or_key_required',
        )

    result = await db.execute(_visible_stream_names_query(label))
    stream_names = set(result.scalars().all())
    if requested_name not in stream_names:
        raise HTTPException(status_code=404, detail='stream_not_found')
    return requested_name


async def _get_configured_media_streams(
    db: AsyncSession,
    label: str,
    rds: redis.Redis | None = None,
    overlay_mode: str = 'none',
    overlay_language: str | None = None,
) -> list[dict[str, object]]:
    """Return configured streams enriched with media-server URLs.

    Args:
        db: Database session used to query configured streams.
        label: Site label containing the streams.
        rds: Optional Redis connection for overlay readiness state.
        overlay_mode: Requested overlay mode.
        overlay_language: Optional requested overlay label language.

    Returns:
        Public playback descriptors for enabled configured streams.
    """
    result = await db.execute(_visible_stream_names_query(label))
    stream_names = list(result.scalars().all())
    streams: list[dict[str, object]] = []
    selected_language = normalise_label_language(overlay_language)
    for stream_name in stream_names:
        stream = _build_stream_listing(
            label,
            stream_name,
            encode_stream_id(stream_name),
        )
        if overlay_mode == 'backend' and rds is not None:
            await _apply_overlay_listing_state(
                rds=rds,
                stream=stream,
                language=selected_language,
            )
        streams.append(stream)
    return streams


async def _apply_overlay_listing_state(
    rds: redis.Redis,
    stream: dict[str, object],
    language: str,
) -> None:
    """Request and describe the shared overlay stream for one listing.

    Args:
        rds: Redis connection used to maintain demand and readiness.
        stream: Mutable public descriptor to enrich in place.
        language: Canonical overlay label language.
    """
    media_path = str(stream['media_path'])
    overlay_path = build_annotated_media_path(media_path, language)
    overlay_hls_url = build_media_hls_url(overlay_path)

    # Listing a backend overlay keeps its producer warm for the first viewer.
    await _touch_overlay_demand(rds, media_path, language)
    ready = await _overlay_is_ready(rds, overlay_path)

    stream.update(
        {
            'language': language,
            'profile': 'overlay',
            'status': 'ready' if ready else 'starting',
            'state': 'ready' if ready else 'starting',
            'overlay_ready': ready,
            'playback_ready': True,
            'media_path': overlay_path,
            'media_hls_url': overlay_hls_url,
            'demand_ttl_seconds': OVERLAY_DEMAND_TTL_SECONDS,
            'playback_url': overlay_hls_url,
        },
    )


def _build_stream_listing(
    label: str,
    key: str,
    stream_id: str,
) -> dict[str, object]:
    """Build the public clean-stream descriptor for one camera.

    Args:
        label: Site label containing the camera.
        key: Decoded configured stream name.
        stream_id: Encoded configured stream identifier.

    Returns:
        Public descriptor with HLS and WebRTC endpoints.
    """
    media_path = build_media_path(label, key)
    stream: dict[str, object] = {
        'key': key,
        'stream_id': stream_id,
        'media_path': media_path,
    }
    hls_url = build_media_hls_url(media_path)
    webrtc_url = build_media_webrtc_url(media_path)
    playback_url = hls_url
    stream.update(
        {
            'transport': 'hls',
            'playback_url': playback_url,
            'media_hls_url': hls_url,
            'webrtc_url': webrtc_url,
            'profile': 'clean',
            'status': 'ready',
            'state': 'ready',
            'playback_ready': True,
        },
    )
    return stream
