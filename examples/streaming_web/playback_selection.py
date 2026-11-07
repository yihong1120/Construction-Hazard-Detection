from __future__ import annotations

import asyncio
from datetime import datetime
from datetime import timezone

import redis.asyncio as redis
from fastapi import HTTPException

from examples.streaming_web import playback_config as config
from examples.streaming_web.media_paths import build_media_hls_url
from examples.streaming_web.playback_demand import overlay_is_ready
from examples.streaming_web.playback_session_store import (
    renew_playback_selection,
)
from examples.streaming_web.schemas import PlaybackSession
from examples.streaming_web.schemas import PlaybackSessionState


async def select_session_playback(
    rds: redis.Redis,
    session: PlaybackSession,
) -> PlaybackSessionState:
    """Resolve the current media path for a stable playback session.

    Args:
        rds: Redis connection used to renew demand and readiness state.
        session: Clean or overlay playback session to resolve.

    Returns:
        Current media path, HLS URL, and client-visible readiness state.
    """
    if not await renew_playback_selection(rds, session):
        raise HTTPException(409, 'playback_session_changed')
    if session['profile'] == 'clean':
        media_path = session['base_media_path']
        return {
            'status': 'ready',
            'overlay_ready': False,
            'media_path': media_path,
            'hls_url': build_media_hls_url(media_path),
        }
    overlay_media_path = session['overlay_media_path']
    overlay_hls_url = build_media_hls_url(overlay_media_path)
    ready = await overlay_is_ready(rds, overlay_media_path)
    return {
        'status': 'ready' if ready else 'starting',
        'overlay_ready': ready,
        'media_path': overlay_media_path,
        'hls_url': overlay_hls_url,
    }


async def wait_for_session_startup(session: PlaybackSession) -> None:
    """Wait briefly for an on-demand publisher to create its HLS playlist.

    Args:
        session: Session whose creation time bounds the startup wait.
    """
    if config.STREAM_PLAYBACK_STARTUP_WAIT_SECONDS <= 0:
        return
    created = datetime.fromisoformat(session['created_at'])
    elapsed = (datetime.now(timezone.utc) - created).total_seconds()
    remaining = config.STREAM_PLAYBACK_STARTUP_WAIT_SECONDS - elapsed
    if remaining > 0:
        await asyncio.sleep(remaining)
