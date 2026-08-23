from __future__ import annotations

import asyncio
import json
import os
import secrets
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import cast

import redis.asyncio as redis
from fastapi import HTTPException

from examples.streaming_web.media_paths import build_annotated_media_path
from examples.streaming_web.media_paths import build_clean_demand_key
from examples.streaming_web.media_paths import build_media_hls_url
from examples.streaming_web.media_paths import build_media_path
from examples.streaming_web.media_paths import build_overlay_demand_key
from examples.streaming_web.media_paths import build_overlay_ready_key
from examples.streaming_web.media_paths import build_preview_media_path
from examples.streaming_web.media_paths import encode_media_segment
from examples.streaming_web.media_paths import parse_annotated_media_path
from examples.streaming_web.metadata_keys import encode_stream_id
from examples.streaming_web.overlay_renderer import normalise_label_language
from examples.streaming_web.playback_demand import overlay_is_ready
from examples.streaming_web.playback_demand import touch_clean_demand
from examples.streaming_web.playback_demand import touch_overlay_demand
from examples.streaming_web.playback_languages import _allowed_overlay_languages
from examples.streaming_web.playback_languages import _default_overlay_language
from examples.streaming_web.playback_languages import _language_alias_map
from examples.streaming_web.playback_languages import _overlay_language_options
from examples.streaming_web.schemas import CleanPlaybackSession
from examples.streaming_web.schemas import OverlayLanguageListResponse
from examples.streaming_web.schemas import OverlayPlaybackSession
from examples.streaming_web.schemas import PlaybackProfile
from examples.streaming_web.schemas import PlaybackRendition
from examples.streaming_web.schemas import PlaybackSession
from examples.streaming_web.schemas import PlaybackSessionResponse
from examples.streaming_web.schemas import PlaybackSessionState


OVERLAY_DEMAND_TTL_SECONDS = int(
    os.getenv('MEDIA_OVERLAY_DEMAND_TTL_SECONDS', '90'),
)
CLEAN_DEMAND_TTL_SECONDS = int(
    os.getenv(
        'MEDIA_CLEAN_DEMAND_TTL_SECONDS',
        str(OVERLAY_DEMAND_TTL_SECONDS),
    ),
)
OVERLAY_MAX_ACTIVE_LANGUAGES = max(
    1,
    int(os.getenv('MEDIA_OVERLAY_MAX_ACTIVE_LANGUAGES_PER_STREAM', '5')),
)
STREAM_PLAYBACK_SESSION_TTL_SECONDS = int(
    os.getenv(
        'STREAM_PLAYBACK_SESSION_TTL_SECONDS',
        str(75 * 60),
    ),
)
STREAM_PLAYBACK_SESSION_PREFIX = 'stream_playback_session'
STREAM_PLAYBACK_MEDIA_SESSION_PREFIX = 'stream_playback_media_session'
STREAM_PLAYBACK_DEMAND_SESSION_PREFIX = 'stream_playback_demand_session'
STREAM_PLAYBACK_SESSION_REFRESH_PREFIX = 'stream_playback_session_refresh'
STREAM_PLAYBACK_PUBLIC_BASE_PATH = os.getenv(
    'STREAM_PLAYBACK_PUBLIC_BASE_PATH',
    '/hazard/api/stream-playback',
).rstrip('/')
STREAM_PLAYBACK_STARTUP_WAIT_SECONDS = max(
    0.0,
    float(os.getenv('STREAM_PLAYBACK_STARTUP_WAIT_SECONDS', '3.0')),
)
STREAM_PLAYBACK_SESSION_REFRESH_INTERVAL_SECONDS = max(
    1,
    int(os.getenv('STREAM_PLAYBACK_SESSION_REFRESH_INTERVAL_SECONDS', '20')),
)


async def _touch_overlay_demand_from_media_path(
    rds: redis.Redis,
    media_path: str,
    ttl_seconds: int = OVERLAY_DEMAND_TTL_SECONDS,
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


async def _touch_media_demand_from_media_path(
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
        await _touch_overlay_demand_from_media_path(
            rds,
            media_path,
            ttl_seconds=(ttl_seconds or OVERLAY_DEMAND_TTL_SECONDS),
        )
        return
    if media_path.startswith('hazard_'):
        await touch_clean_demand(
            rds,
            media_path,
            ttl_seconds=(ttl_seconds or CLEAN_DEMAND_TTL_SECONDS),
        )


def _normalise_playback_profile(profile: str | None) -> PlaybackProfile:
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


def _normalise_playback_rendition(
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


def _playback_session_key(session_id: str) -> str:
    """Build the Redis key for a browser-facing playback session.

    Args:
        session_id: Opaque playback session identifier.

    Returns:
        Canonical Redis key for the session payload.
    """
    return f'{STREAM_PLAYBACK_SESSION_PREFIX}:{session_id}'


def _playback_media_sessions_key(media_path: str) -> str:
    """Build the expiring reverse-session index for one media path.

    Args:
        media_path: Selected clean or annotated MediaMTX path.

    Returns:
        Redis ZSET key whose members are playback-session identifiers.
    """
    return f'{STREAM_PLAYBACK_MEDIA_SESSION_PREFIX}:{media_path}'


def _playback_session_refresh_key(media_path: str) -> str:
    """Build the short lease that coalesces fragment-driven renewals.

    Args:
        media_path: Selected clean or annotated MediaMTX path.

    Returns:
        Redis key used to ensure one renewal per bounded interval.
    """
    return f'{STREAM_PLAYBACK_SESSION_REFRESH_PREFIX}:{media_path}'


def _playback_demand_session_key(
    base_media_path: str,
    profile: PlaybackProfile,
    language: str | None = None,
) -> str:
    """Build the active-session index for one shared producer demand."""
    if profile == 'overlay':
        if language is None:
            raise ValueError('overlay playback demand requires a language')
        return (
            f'{STREAM_PLAYBACK_DEMAND_SESSION_PREFIX}:overlay:'
            f'{base_media_path}:{encode_media_segment(language)}'
        )
    return f'{STREAM_PLAYBACK_DEMAND_SESSION_PREFIX}:clean:{base_media_path}'


def _build_session_playback_url(session_id: str) -> str:
    """Build the stable public playlist URL for a playback session.

    Args:
        session_id: Opaque playback session identifier.

    Returns:
        Relative public HLS playlist URL.
    """
    return (
        f'{STREAM_PLAYBACK_PUBLIC_BASE_PATH}/sessions/'
        f'{session_id}/index.m3u8'
    )


def _decode_playback_session_payload(
    raw: bytes | None,
) -> PlaybackSession | None:
    """Decode an optional Redis playback-session payload.

    Args:
        raw: Optional UTF-8 JSON payload retrieved from Redis.

    Returns:
        Typed playback session, or ``None`` when no Redis value exists.
    """
    if raw is None:
        return None
    return json.loads(raw)


async def _load_playback_session(
    rds: redis.Redis,
    session_id: str,
) -> PlaybackSession | None:
    """Load one short-lived playback session from Redis.

    Args:
        rds: Redis connection used to retrieve the session.
        session_id: Opaque playback session identifier.

    Returns:
        Decoded session, or ``None`` when it has expired or never existed.
    """
    return _decode_playback_session_payload(
        await rds.get(_playback_session_key(session_id)),
    )


def _session_selected_media_path(session: PlaybackSession) -> str:
    """Return the media path currently selected by a playback session.

    Args:
        session: Clean or overlay playback session.

    Returns:
        Clean base path or language-specific annotated path.
    """
    if session['profile'] == 'overlay':
        return session['overlay_media_path']
    return session['base_media_path']


def _session_possible_media_paths(session: PlaybackSession) -> set[str]:
    """Return media paths that may have reverse indexes for a session.

    Args:
        session: Clean or overlay playback session.

    Returns:
        Set containing the base path and the current selected path.
    """
    return {
        session['base_media_path'],
        _session_selected_media_path(session),
    }


async def _delete_playback_session_media_indexes(
    rds: redis.Redis,
    session: PlaybackSession,
) -> None:
    """Remove all reverse indexes for one playback session.

    Args:
        rds: Redis connection used to delete reverse indexes.
        session: Session whose indexed media paths are removed.
    """
    session_id = session['session_id']
    async with rds.pipeline(transaction=False) as pipeline:
        for media_path in _session_possible_media_paths(session):
            pipeline.zrem(
                _playback_media_sessions_key(media_path),
                session_id,
            )
        pipeline.zrem(
            _playback_demand_session_key(
                session['base_media_path'],
                session['profile'],
                session.get('language'),
            ),
            session_id,
        )
        await pipeline.execute()


async def _register_playback_session_demand(
    rds: redis.Redis,
    session: PlaybackSession,
) -> None:
    """Index an active session by its shared producer demand.

    The score is its expiry time, so release can discard stale memberships in
    Redis rather than scanning every playback-session payload.
    """
    expires_at = (
        datetime.now(timezone.utc).timestamp()
        + STREAM_PLAYBACK_SESSION_TTL_SECONDS
    )
    key = _playback_demand_session_key(
        session['base_media_path'],
        session['profile'],
        session.get('language'),
    )
    await rds.zadd(key, {session['session_id']: expires_at})
    await rds.expire(key, STREAM_PLAYBACK_SESSION_TTL_SECONDS)


async def _register_playback_session_media_path(
    rds: redis.Redis,
    session: PlaybackSession,
    media_path: str,
) -> None:
    """Index a media path so media authorisation can renew a session.

    Args:
        rds: Redis connection used to store the reverse index.
        session: Session to renew when the path is read.
        media_path: Current selected MediaMTX path.
    """
    expires_at = (
        datetime.now(timezone.utc).timestamp()
        + STREAM_PLAYBACK_SESSION_TTL_SECONDS
    )
    key = _playback_media_sessions_key(media_path)
    async with rds.pipeline(transaction=False) as pipeline:
        pipeline.zadd(
            key,
            {session['session_id']: expires_at},
        )
        pipeline.expire(
            key,
            STREAM_PLAYBACK_SESSION_TTL_SECONDS,
        )
        await pipeline.execute()


async def _refresh_playback_sessions_for_media_path(
    rds: redis.Redis,
    media_path: str,
) -> None:
    """Renew sessions whose selected proxied HLS media is being read.

    Args:
        rds: Redis connection used to renew the indexed session group.
        media_path: MediaMTX path just authorised by the media proxy.
    """
    renewed = await rds.set(
        _playback_session_refresh_key(media_path),
        b'1',
        ex=STREAM_PLAYBACK_SESSION_REFRESH_INTERVAL_SECONDS,
        nx=True,
    )
    if not renewed:
        return

    now = datetime.now(timezone.utc).timestamp()
    media_key = _playback_media_sessions_key(media_path)
    raw_session_ids = cast(
        list[bytes | str],
        await rds.zrangebyscore(media_key, now, '+inf'),
    )
    if not raw_session_ids:
        return
    session_ids = [
        raw_id.decode('utf-8') if isinstance(raw_id, bytes) else str(raw_id)
        for raw_id in raw_session_ids
    ]
    raw_sessions = cast(
        list[bytes | None],
        await rds.mget([
            _playback_session_key(session_id) for session_id in session_ids
        ]),
    )
    active_session_ids: list[str] = []
    stale_session_ids: list[str] = []
    for session_id, raw_session in zip(
        session_ids,
        raw_sessions,
        strict=True,
    ):
        session = _decode_playback_session_payload(raw_session)
        if session is None or _session_selected_media_path(session) != media_path:
            stale_session_ids.append(session_id)
        else:
            active_session_ids.append(session_id)

    expires_at = now + STREAM_PLAYBACK_SESSION_TTL_SECONDS
    parsed_overlay_path = parse_annotated_media_path(media_path)
    if parsed_overlay_path is None:
        demand_key = _playback_demand_session_key(media_path, 'clean')
    else:
        base_media_path, language = parsed_overlay_path
        demand_key = _playback_demand_session_key(
            base_media_path,
            'overlay',
            normalise_label_language(language),
        )

    async with rds.pipeline(transaction=False) as pipeline:
        if stale_session_ids:
            pipeline.zrem(
                media_key,
                *stale_session_ids,
            )
        for session_id in active_session_ids:
            pipeline.expire(
                _playback_session_key(session_id),
                STREAM_PLAYBACK_SESSION_TTL_SECONDS,
            )
        if active_session_ids:
            scores: dict[str | bytes, str | bytes | float | int] = {
                session_id: expires_at
                for session_id in active_session_ids
            }
            pipeline.zadd(media_key, scores)
            pipeline.expire(
                media_key,
                STREAM_PLAYBACK_SESSION_TTL_SECONDS,
            )
            pipeline.zadd(demand_key, scores)
            pipeline.expire(
                demand_key,
                STREAM_PLAYBACK_SESSION_TTL_SECONDS,
            )
        await pipeline.execute()


async def _create_or_update_playback_session(
    rds: redis.Redis,
    session_id: str | None,
    username: str,
    label: str,
    stream_name: str,
    profile: PlaybackProfile,
    rendition: PlaybackRendition,
    language: str | None,
) -> PlaybackSession:
    """Create or update a short-lived playback session for one camera.

    Args:
        rds: Redis connection used to store session state and indexes.
        session_id: Optional existing session identifier to update.
        username: Authenticated owner of the session.
        label: Site label containing the stream.
        stream_name: Validated configured stream name.
        profile: Requested clean or overlay profile.
        rendition: Requested detail or preview rendition.
        language: Canonical overlay language, if the profile is overlay.

    Returns:
        Newly created or updated typed playback session.

    Raises:
        HTTPException: If an existing session is absent or owned by another user.
    """
    if session_id:
        existing = await _load_playback_session(rds, session_id)
        if existing is None:
            raise HTTPException(status_code=404, detail='session_not_found')
        if existing.get('username') != username:
            raise HTTPException(status_code=403, detail='session_forbidden')
        await _delete_playback_session_media_indexes(rds, existing)
    else:
        # Generate server-side identifiers; clients may only refresh a session
        # that is already present and owned by the authenticated user.
        session_id = secrets.token_urlsafe(24)

    session = _new_playback_session(
        session_id=session_id,
        username=username,
        label=label,
        stream_name=stream_name,
        profile=profile,
        rendition=rendition,
        language=language,
    )
    await rds.set(
        _playback_session_key(session_id),
        json.dumps(session, ensure_ascii=False).encode('utf-8'),
        ex=STREAM_PLAYBACK_SESSION_TTL_SECONDS,
    )
    return session


def _new_playback_session(
    *,
    session_id: str,
    username: str,
    label: str,
    stream_name: str,
    profile: PlaybackProfile,
    rendition: PlaybackRendition,
    language: str | None,
) -> PlaybackSession:
    """Build one new session payload without performing Redis I/O."""
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(seconds=STREAM_PLAYBACK_SESSION_TTL_SECONDS)
    detail_media_path = build_media_path(label, stream_name)
    base_media_path = (
        build_preview_media_path(detail_media_path)
        if rendition == 'preview'
        else detail_media_path
    )
    if profile == 'clean':
        return CleanPlaybackSession(
            session_id=session_id,
            username=username,
            label=label,
            stream_name=stream_name,
            stream_id=encode_stream_id(stream_name),
            profile='clean',
            rendition=rendition,
            language=None,
            base_media_path=base_media_path,
            overlay_media_path=None,
            created_at=now.isoformat(),
            expires_at=expires_at.isoformat(),
        )
    if language is None:
        raise HTTPException(status_code=422, detail='overlay_language_required')
    return OverlayPlaybackSession(
        session_id=session_id,
        username=username,
        label=label,
        stream_name=stream_name,
        stream_id=encode_stream_id(stream_name),
        profile='overlay',
        rendition=rendition,
        language=language,
        base_media_path=base_media_path,
        overlay_media_path=build_annotated_media_path(
            base_media_path,
            language,
        ),
        created_at=now.isoformat(),
        expires_at=expires_at.isoformat(),
    )


async def create_playback_sessions_for_streams(
    rds: redis.Redis,
    *,
    username: str,
    label: str,
    stream_names: list[str],
    profile: PlaybackProfile,
    rendition: PlaybackRendition,
    language: str | None,
) -> list[PlaybackSession]:
    """Create many new playback sessions in one Redis pipeline.

    Camera-wall requests never need to read or update an existing session.
    Building their payloads locally and storing them in one pipeline removes a
    Redis round trip per camera while keeping the single-session API intact.
    """
    return await create_playback_sessions(
        rds,
        username=username,
        requests=[
            (label, stream_name, profile, rendition, language)
            for stream_name in stream_names
        ],
    )


async def create_playback_sessions(
    rds: redis.Redis,
    *,
    username: str,
    requests: list[
        tuple[str, str, PlaybackProfile, PlaybackRendition, str | None]
    ],
) -> list[PlaybackSession]:
    """Create heterogeneous new playback sessions in one Redis pipeline.

    Args:
        rds: Redis connection used to persist the short-lived sessions.
        username: Authenticated owner for every created session.
        requests: ``(label, stream_name, profile, rendition, language)`` rows.

    Returns:
        New sessions in the same order as ``requests``.
    """
    sessions = [
        _new_playback_session(
            session_id=secrets.token_urlsafe(24),
            username=username,
            label=label,
            stream_name=stream_name,
            profile=profile,
            rendition=rendition,
            language=language,
        )
        for label, stream_name, profile, rendition, language in requests
    ]
    if not sessions:
        return sessions
    async with rds.pipeline(transaction=False) as pipeline:
        for session in sessions:
            pipeline.set(
                _playback_session_key(session['session_id']),
                json.dumps(session, ensure_ascii=False).encode('utf-8'),
                ex=STREAM_PLAYBACK_SESSION_TTL_SECONDS,
            )
        await pipeline.execute()
    return sessions


async def _refresh_playback_session_ttl(
    rds: redis.Redis,
    session_id: str,
) -> None:
    """Renew a playback session while its HLS playlist is read.

    Args:
        rds: Redis connection used to extend the session expiry.
        session_id: Opaque session identifier to renew.
    """
    await rds.expire(
        _playback_session_key(session_id),
        STREAM_PLAYBACK_SESSION_TTL_SECONDS,
    )


async def _select_session_playback(
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
    if session['profile'] == 'clean':
        base_media_path = session['base_media_path']
        await touch_clean_demand(
            rds,
            base_media_path,
            CLEAN_DEMAND_TTL_SECONDS,
        )
        await _register_playback_session_media_path(
            rds, session, base_media_path,
        )
        await _register_playback_session_demand(rds, session)
        return {
            'status': 'ready',
            'overlay_ready': False,
            'media_path': base_media_path,
            'hls_url': build_media_hls_url(base_media_path),
        }

    base_media_path = session['base_media_path']
    language = session['language']
    overlay_media_path = session['overlay_media_path']
    overlay_hls_url = build_media_hls_url(overlay_media_path)
    await touch_overlay_demand(
        rds,
        base_media_path,
        language,
        OVERLAY_DEMAND_TTL_SECONDS,
    )
    await _register_playback_session_media_path(
        rds, session, overlay_media_path,
    )
    await _register_playback_session_demand(rds, session)
    ready = await overlay_is_ready(rds, overlay_media_path)
    return {
        'status': 'ready' if ready else 'starting',
        'overlay_ready': ready,
        'media_path': overlay_media_path,
        'hls_url': overlay_hls_url,
    }


async def _wait_for_session_startup(session: PlaybackSession) -> None:
    """Wait briefly for an on-demand publisher to create its HLS playlist.

    Args:
        session: Session whose creation time bounds the startup wait.
    """
    if STREAM_PLAYBACK_STARTUP_WAIT_SECONDS <= 0:
        return
    created = datetime.fromisoformat(session['created_at'])
    elapsed = (datetime.now(timezone.utc) - created).total_seconds()
    remaining = STREAM_PLAYBACK_STARTUP_WAIT_SECONDS - elapsed
    if remaining > 0:
        await asyncio.sleep(remaining)


async def _build_playback_session_response_body(
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
    state = await _select_session_playback(rds, session)
    return _playback_session_response_body(session, state)


def _playback_session_response_body(
    session: PlaybackSession,
    state: PlaybackSessionState,
) -> PlaybackSessionResponse:
    """Build the public response after demand/readiness state is known."""
    session_id = session['session_id']
    stable_url = _build_session_playback_url(session_id)
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
        'expires_in': STREAM_PLAYBACK_SESSION_TTL_SECONDS,
        'demand_ttl_seconds': OVERLAY_DEMAND_TTL_SECONDS,
    }


async def build_playback_session_response_bodies(
    rds: redis.Redis,
    sessions: list[PlaybackSession],
) -> list[PlaybackSessionResponse]:
    """Activate a camera-wall session list with bounded Redis round trips.

    This is the batch counterpart of ``_select_session_playback``. It writes
    every producer lease and reverse index in one pipeline, then obtains all
    overlay readiness flags in one ``MGET``.
    """
    if not sessions:
        return []

    expires_at = (
        datetime.now(timezone.utc).timestamp()
        + STREAM_PLAYBACK_SESSION_TTL_SECONDS
    )
    demand_timestamp = str(int(datetime.now(timezone.utc).timestamp())).encode(
        'ascii',
    )
    overlay_ready_keys: list[str] = []
    async with rds.pipeline(transaction=False) as pipeline:
        for session in sessions:
            media_path = _session_selected_media_path(session)
            media_index_key = _playback_media_sessions_key(media_path)
            demand_key = _playback_demand_session_key(
                session['base_media_path'],
                session['profile'],
                session.get('language'),
            )
            if session['profile'] == 'clean':
                pipeline.set(
                    build_clean_demand_key(session['base_media_path']),
                    demand_timestamp,
                    ex=CLEAN_DEMAND_TTL_SECONDS,
                )
            else:
                pipeline.set(
                    build_overlay_demand_key(
                        session['base_media_path'],
                        session['language'],
                    ),
                    demand_timestamp,
                    ex=OVERLAY_DEMAND_TTL_SECONDS,
                )
                overlay_ready_keys.append(build_overlay_ready_key(media_path))
            scores: dict[str | bytes, str | bytes | float | int] = {
                session['session_id']: expires_at,
            }
            pipeline.zadd(media_index_key, scores)
            pipeline.expire(
                media_index_key,
                STREAM_PLAYBACK_SESSION_TTL_SECONDS,
            )
            pipeline.zadd(demand_key, scores)
            pipeline.expire(
                demand_key,
                STREAM_PLAYBACK_SESSION_TTL_SECONDS,
            )
        await pipeline.execute()

    readiness_values = (
        cast(list[bytes | None], await rds.mget(overlay_ready_keys))
        if overlay_ready_keys
        else []
    )
    readiness_by_path = dict(zip(overlay_ready_keys, readiness_values, strict=True))
    bodies: list[PlaybackSessionResponse] = []
    for session in sessions:
        media_path = _session_selected_media_path(session)
        if session['profile'] == 'clean':
            state: PlaybackSessionState = {
                'status': 'ready',
                'overlay_ready': False,
                'media_path': media_path,
                'hls_url': build_media_hls_url(media_path),
            }
        else:
            ready = readiness_by_path.get(
                build_overlay_ready_key(media_path),
            ) is not None
            state = {
                'status': 'ready' if ready else 'starting',
                'overlay_ready': ready,
                'media_path': media_path,
                'hls_url': build_media_hls_url(media_path),
            }
        bodies.append(_playback_session_response_body(session, state))
    return bodies


async def _has_other_playback_session(
    rds: redis.Redis,
    base_media_path: str,
    profile: PlaybackProfile,
    language: str | None = None,
) -> bool:
    """Determine whether another active session uses the same producer demand.

    Args:
        rds: Redis connection used to read the producer-demand session index.
        base_media_path: Shared clean base path to compare.
        profile: Clean or overlay profile to compare.
        language: Overlay language to compare for overlay sessions.

    Returns:
        ``True`` when another session still requires the same producer lease.
    """
    key = _playback_demand_session_key(base_media_path, profile, language)
    await rds.zremrangebyscore(
        key,
        '-inf',
        datetime.now(timezone.utc).timestamp(),
    )
    return bool(await rds.zcard(key))


def _build_overlay_language_response() -> OverlayLanguageListResponse:
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
        max_active_languages_per_stream=OVERLAY_MAX_ACTIVE_LANGUAGES,
        demand_ttl_seconds=OVERLAY_DEMAND_TTL_SECONDS,
        ready_ttl_seconds=int(
            os.getenv('MEDIA_OVERLAY_READY_TTL_SECONDS', '15'),
        ),
    )
