from __future__ import annotations

from typing import cast

import redis.asyncio as redis
from fastapi import HTTPException

from examples.streaming_web import playback_config as config
from examples.streaming_web.media_paths import parse_annotated_media_path
from examples.streaming_web.overlay_labels import normalise_label_language
from examples.streaming_web.playback_demand import touch_clean_demand
from examples.streaming_web.playback_demand import touch_overlay_demand
from examples.streaming_web.playback_languages import (
    _allowed_overlay_languages,
)
from examples.streaming_web.playback_languages import _default_overlay_language
from examples.streaming_web.playback_languages import _language_alias_map
from examples.streaming_web.playback_languages import _overlay_language_options
from examples.streaming_web.schemas import OverlayLanguageListResponse
from examples.streaming_web.schemas import PlaybackProfile
from examples.streaming_web.schemas import PlaybackRendition
from examples.streaming_web.schemas import PlaybackSession
from src.settings.media import MediaGatewaySettings


async def touch_overlay_demand_from_media_path(
    rds: redis.Redis,
    media_path: str,
    ttl_seconds: int = config.OVERLAY_DEMAND_TTL_SECONDS,
) -> None:
    """Refresh overlay demand from an annotated media path.

    Args:
        rds: Redis connection used to write the demand lease.
        media_path: Annotated MediaMTX stream path.
        ttl_seconds: Lease duration to set in seconds.
    """
    parsed = parse_annotated_media_path(media_path)
    if parsed is None:
        return
    base_path, raw_language = parsed
    language = normalise_label_language(raw_language)
    if language not in _allowed_overlay_languages():
        return
    await touch_overlay_demand(
        rds,
        base_path,
        language,
        ttl_seconds=ttl_seconds,
    )


async def touch_media_demand_from_media_path(
    rds: redis.Redis,
    media_path: str,
    ttl_seconds: int | None = None,
) -> None:
    """Refresh the demand key matching a clean or overlay media path.

    Args:
        rds: Redis connection used to write the demand lease.
        media_path: Requested clean or annotated MediaMTX path.
        ttl_seconds: Optional lease duration overriding the profile default.
    """
    parsed = parse_annotated_media_path(media_path)
    if parsed is not None:
        await touch_overlay_demand_from_media_path(
            rds,
            media_path,
            ttl_seconds=(ttl_seconds or config.OVERLAY_DEMAND_TTL_SECONDS),
        )
        return
    if media_path.startswith('hazard_'):
        await touch_clean_demand(
            rds,
            media_path,
            ttl_seconds=(ttl_seconds or config.CLEAN_DEMAND_TTL_SECONDS),
        )


def normalise_playback_profile(profile: str | None) -> PlaybackProfile:
    """Normalise and validate a requested playback profile.

    Args:
        profile: Optional raw profile supplied by the client.

    Returns:
        Canonical clean or overlay profile.

    Raises:
        HTTPException: If the profile is unsupported.
    """
    value = (profile or 'clean').strip().lower()
    if value in {'clean', 'overlay'}:
        return cast(PlaybackProfile, value)
    raise HTTPException(status_code=422, detail='unsupported_profile')


def normalise_playback_rendition(
    rendition: str | None,
) -> PlaybackRendition:
    """Normalise and validate a requested HLS rendition.

    Args:
        rendition: Optional raw rendition supplied by the client.

    Returns:
        Canonical detail or preview rendition.

    Raises:
        HTTPException: If the rendition is unsupported.
    """
    value = (rendition or 'detail').strip().lower()
    if value in {'detail', 'preview'}:
        return cast(PlaybackRendition, value)
    raise HTTPException(status_code=422, detail='unsupported_rendition')


def build_session_playback_url(session_id: str) -> str:
    """Build the stable public playlist URL for a playback session.

    Args:
        session_id: Opaque playback session identifier.

    Returns:
        Relative public HLS playlist URL.
    """
    return (
        f"{config.STREAM_PLAYBACK_PUBLIC_BASE_PATH}/sessions/"
        f"{session_id}/index.m3u8"
    )


def session_selected_media_path(session: PlaybackSession) -> str:
    """Return the media path currently selected by a playback session.

    Args:
        session: Clean or overlay playback session.

    Returns:
        Clean base path or language-specific annotated path.
    """
    if session['profile'] == 'overlay':
        return session['overlay_media_path']
    return session['base_media_path']


def session_possible_media_paths(session: PlaybackSession) -> set[str]:
    """Return media paths that may have reverse indexes for a session.

    Args:
        session: Clean or overlay playback session.

    Returns:
        Set containing the base path and the current selected path.
    """
    return {
        session['base_media_path'],
        session_selected_media_path(session),
    }


def build_overlay_language_response() -> OverlayLanguageListResponse:
    """Build the public overlay-language capability response.

    Returns:
        Canonical language options, aliases, and lease limits for clients.
    """
    allowed_languages = _allowed_overlay_languages()
    return OverlayLanguageListResponse(
        default_language=_default_overlay_language(),
        allowed_language_codes=list(allowed_languages),
        aliases=_language_alias_map(),
        languages=_overlay_language_options(allowed_languages),
        stream_playback_endpoint='/hazard/api/stream-playback',
        max_active_languages_per_stream=config.OVERLAY_MAX_ACTIVE_LANGUAGES,
        demand_ttl_seconds=config.OVERLAY_DEMAND_TTL_SECONDS,
        ready_ttl_seconds=MediaGatewaySettings().ready_ttl_seconds,
    )
