from __future__ import annotations

from typing import cast

import redis.asyncio as redis

from examples.streaming_web import playback_config as config
from examples.streaming_web.media_paths import build_media_hls_url
from examples.streaming_web.media_paths import build_overlay_ready_key
from examples.streaming_web.playback_policy import build_session_playback_url
from examples.streaming_web.playback_policy import session_selected_media_path
from examples.streaming_web.playback_selection import select_session_playback
from examples.streaming_web.schemas import PlaybackSession
from examples.streaming_web.schemas import PlaybackSessionResponse
from examples.streaming_web.schemas import PlaybackSessionState


async def build_playback_session_response_body(
    rds: redis.Redis,
    session: PlaybackSession,
) -> PlaybackSessionResponse:
    """Build the API payload returned to frontend playback controllers.

    Args:
        rds: Redis connection used to resolve live demand and readiness.
        session: Clean or overlay playback session to expose.

    Returns:
        Public stable playback descriptor for the frontend.
    """
    state = await select_session_playback(rds, session)
    return playback_session_response_body(session, state)


def playback_session_response_body(
    session: PlaybackSession,
    state: PlaybackSessionState,
) -> PlaybackSessionResponse:
    """Build the public response after demand/readiness state is known."""
    session_id = session['session_id']
    stable_url = build_session_playback_url(session_id)
    return {
        'session_id': session_id,
        'stream_id': session['stream_id'],
        'key': session['stream_name'],
        'label': session['label'],
        'transport': 'hls',
        'status': state['status'],
        'profile': session['profile'],
        'rendition': session['rendition'],
        'playback_ready': True,
        'playback_url': stable_url,
        'media_hls_url': state['hls_url'],
        'language': session['language'],
        'overlay_ready': state['overlay_ready'],
        'media_path': state['media_path'],
        'expires_at': session['expires_at'],
        'expires_in': config.STREAM_PLAYBACK_SESSION_TTL_SECONDS,
        'demand_ttl_seconds': config.OVERLAY_DEMAND_TTL_SECONDS,
    }


async def build_playback_session_response_bodies(
    rds: redis.Redis,
    sessions: list[PlaybackSession],
) -> list[PlaybackSessionResponse]:
    """Build descriptors for an already committed camera-wall batch."""
    return await read_playback_session_responses(rds, sessions)


async def read_playback_session_responses(
    rds: redis.Redis, sessions: list[PlaybackSession],
) -> list[PlaybackSessionResponse]:
    """Read volatile readiness after the session batch has committed."""
    overlay_ready_keys = [
        build_overlay_ready_key(session_selected_media_path(session))
        for session in sessions if session['profile'] == 'overlay'
    ]
    readiness_values = (
        cast(list[bytes | None], await rds.mget(overlay_ready_keys))
        if overlay_ready_keys
        else []
    )
    readiness_by_path = dict(
        zip(overlay_ready_keys, readiness_values, strict=True),
    )
    bodies: list[PlaybackSessionResponse] = []
    for session in sessions:
        media_path = session_selected_media_path(session)
        if session['profile'] == 'clean':
            state: PlaybackSessionState = {
                'status': 'ready',
                'overlay_ready': False,
                'media_path': media_path,
                'hls_url': build_media_hls_url(media_path),
            }
        else:
            ready = (
                readiness_by_path.get(
                    build_overlay_ready_key(media_path),
                )
                is not None
            )
            state = {
                'status': 'ready' if ready else 'starting',
                'overlay_ready': ready,
                'media_path': media_path,
                'hls_url': build_media_hls_url(media_path),
            }
        bodies.append(playback_session_response_body(session, state))
    return bodies
