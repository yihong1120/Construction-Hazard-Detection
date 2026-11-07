from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from datetime import timezone

import redis.asyncio as redis
from fastapi import HTTPException
from redis.asyncio.client import Pipeline

from examples.auth.redis_transactions import optimistic
from examples.streaming_web import playback_config as config
from examples.streaming_web.media_paths import build_annotated_media_path
from examples.streaming_web.media_paths import build_clean_demand_key
from examples.streaming_web.media_paths import build_media_path
from examples.streaming_web.media_paths import build_overlay_demand_key
from examples.streaming_web.media_paths import build_preview_media_path
from examples.streaming_web.media_paths import encode_media_segment
from examples.streaming_web.media_paths import parse_annotated_media_path
from examples.streaming_web.metadata_keys import encode_stream_id
from examples.streaming_web.overlay_admission import OverlayAdmission
from examples.streaming_web.overlay_labels import normalise_label_language
from examples.streaming_web.playback_policy import session_possible_media_paths
from examples.streaming_web.playback_policy import session_selected_media_path
from examples.streaming_web.schemas import CleanPlaybackSession
from examples.streaming_web.schemas import OverlayPlaybackSession
from examples.streaming_web.schemas import PlaybackProfile
from examples.streaming_web.schemas import PlaybackRendition
from examples.streaming_web.schemas import PlaybackSession


@dataclass(frozen=True)
class PlaybackSessionPlan:
    label: str
    stream_name: str
    profile: PlaybackProfile
    rendition: PlaybackRendition
    language: str | None
    session_id: str | None = None

    def session(self, username: str) -> PlaybackSession:
        return new_playback_session(
            session_id=self.session_id or secrets.token_urlsafe(24),
            username=username, label=self.label, stream_name=self.stream_name,
            profile=self.profile, rendition=self.rendition,
            language=self.language,
        )


async def apply_playback_session_plans(
    rds: redis.Redis, username: str, plans: list[PlaybackSessionPlan],
) -> list[PlaybackSession]:
    """Commit a validated batch with its demand and media indexes."""
    if not plans:
        return []
    return await optimistic(rds, _PlaybackBatch(username, plans).apply)


def playback_session_key(session_id: str) -> str:
    """Build the Redis key for a browser-facing playback session.

    Args:
        session_id: Opaque playback session identifier.

    Returns:
        Canonical Redis key for the session payload.
    """
    return f"{config.STREAM_PLAYBACK_SESSION_PREFIX}:{session_id}"


def playback_media_sessions_key(media_path: str) -> str:
    """Build the expiring reverse-session index for one media path.

    Args:
        media_path: Selected clean or annotated MediaMTX path.

    Returns:
        Redis ZSET key whose members are playback-session identifiers.
    """
    return f"{config.STREAM_PLAYBACK_MEDIA_SESSION_PREFIX}:{media_path}"


def playback_session_refresh_key(media_path: str) -> str:
    """Build the short lease that coalesces fragment-driven renewals.

    Args:
        media_path: Selected clean or annotated MediaMTX path.

    Returns:
        Redis key used to ensure one renewal per bounded interval.
    """
    return f"{config.STREAM_PLAYBACK_SESSION_REFRESH_PREFIX}:{media_path}"


def playback_demand_session_key(
    base_media_path: str,
    profile: PlaybackProfile,
    language: str | None = None,
) -> str:
    """Build the active-session index for one shared producer demand."""
    if profile == 'overlay':
        if language is None:
            raise ValueError('overlay playback demand requires a language')
        return (
            f"{config.STREAM_PLAYBACK_DEMAND_SESSION_PREFIX}:overlay:"
            f"{base_media_path}:{encode_media_segment(language)}"
        )
    return (
        f"{config.STREAM_PLAYBACK_DEMAND_SESSION_PREFIX}:clean:"
        f"{base_media_path}"
    )


def decode_playback_session_payload(
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


async def load_playback_session(
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
    return decode_playback_session_payload(
        await rds.get(playback_session_key(session_id)),
    )


def _remove_indexes(pipe: Pipeline, session: PlaybackSession) -> None:
    for path in session_possible_media_paths(session):
        pipe.zrem(playback_media_sessions_key(path), session['session_id'])
    pipe.zrem(
        playback_demand_session_key(
            session['base_media_path'],
            session['profile'],
            session['language'],
        ),
        session['session_id'],
    )


def _producer_key(session: PlaybackSession) -> str:
    if session['profile'] == 'overlay':
        return build_overlay_demand_key(
            session['base_media_path'],
            session['language'],
        )
    return build_clean_demand_key(session['base_media_path'])


async def delete_playback_session(
    rds: redis.Redis,
    session: PlaybackSession,
) -> bool:
    return await optimistic(rds, _PlaybackSessionMutation(session).delete)


def _same_revision(
    current: PlaybackSession | None,
    selected: PlaybackSession,
) -> bool:
    return current is not None and all(
        current.get(key) == selected.get(key)
        for key in (
            'revision',
            'username',
            'base_media_path',
            'profile',
            'language',
            'rendition',
        )
    )


async def renew_playback_selection(
    rds: redis.Redis,
    session: PlaybackSession,
) -> bool:
    return await optimistic(rds, _PlaybackSessionMutation(session).renew)


async def register_playback_session_demand(
    rds: redis.Redis,
    session: PlaybackSession,
) -> None:
    key = playback_demand_session_key(
        session['base_media_path'],
        session['profile'],
        session['language'],
    )
    await _register_index(rds, session, key, _producer_key(session))


async def register_playback_session_media_path(
    rds: redis.Redis,
    session: PlaybackSession,
    media_path: str,
) -> None:
    await _register_index(
        rds,
        session,
        playback_media_sessions_key(media_path),
    )


async def _register_index(
    rds: redis.Redis,
    session: PlaybackSession,
    index: str,
    producer_key: str | None = None,
) -> None:
    await optimistic(
        rds,
        _PlaybackIndexRegistration(session, index, producer_key).apply,
    )


async def refresh_playback_sessions_for_media_path(
    rds: redis.Redis,
    media_path: str,
) -> None:
    """Renew the bounded group selected by an authorised HLS read."""
    await optimistic(rds, _PlaybackMediaRenewal(media_path).apply)


async def create_or_update_playback_session(
    rds: redis.Redis, username: str, plan: PlaybackSessionPlan,
) -> PlaybackSession:
    return (await apply_playback_session_plans(rds, username, [plan]))[0]


def new_playback_session(
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
    expires_at = now + timedelta(
        seconds=config.STREAM_PLAYBACK_SESSION_TTL_SECONDS,
    )
    detail_media_path = build_media_path(label, stream_name)
    base_media_path = (
        build_preview_media_path(detail_media_path)
        if rendition == 'preview'
        else detail_media_path
    )
    if profile == 'clean':
        return CleanPlaybackSession(
            session_id=session_id,
            revision=secrets.token_hex(12),
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
        raise HTTPException(
            status_code=422,
            detail='overlay_language_required',
        )
    return OverlayPlaybackSession(
        session_id=session_id,
        revision=secrets.token_hex(12),
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
    """Commit new sessions and their producer/index state together."""
    return await apply_playback_session_plans(
        rds, username, [
            PlaybackSessionPlan(label, name, profile, rendition, language)
            for label, name, profile, rendition, language in requests
        ],
    )


async def refresh_playback_session_ttl(
    rds: redis.Redis,
    session_id: str,
) -> None:
    """Renew a playback session while its HLS playlist is read.

    Args:
        rds: Redis connection used to extend the session expiry.
        session_id: Opaque session identifier to renew.
    """
    await rds.expire(
        playback_session_key(session_id),
        config.STREAM_PLAYBACK_SESSION_TTL_SECONDS,
    )


async def has_other_playback_session(
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
    key = playback_demand_session_key(base_media_path, profile, language)
    await rds.zremrangebyscore(
        key,
        '-inf',
        datetime.now(timezone.utc).timestamp(),
    )
    return bool(await rds.zcard(key))


class _PlaybackBatch:
    def __init__(self, username: str, plans: list[PlaybackSessionPlan]):
        self.username = username
        self.plans = plans
        self.sessions = [plan.session(self.username) for plan in self.plans]
        self.keys = [
            playback_session_key(s['session_id']) for s in self.sessions
        ]
        if len(self.keys) != len(set(self.keys)):
            raise HTTPException(422, 'duplicate_playback_session')
        self.admission = OverlayAdmission()
        for session in self.sessions:
            if session['profile'] == 'overlay':
                self.admission.add(
                    session['base_media_path'],
                    session['language'],
                    config.OVERLAY_DEMAND_TTL_SECONDS,
                )

    async def apply(self, pipe: Pipeline) -> list[PlaybackSession]:
        await pipe.watch(*self.keys)
        existing = [
            decode_playback_session_payload(raw)
            for raw in await pipe.mget(self.keys)
        ]
        for plan, previous in zip(self.plans, existing, strict=True):
            if plan.session_id and previous is None:
                raise HTTPException(404, 'session_not_found')
            if previous is not None and previous['username'] != self.username:
                raise HTTPException(403, 'session_forbidden')
            if not plan.session_id and previous is not None:
                raise HTTPException(503, 'session_id_collision')

        now = datetime.now(timezone.utc).timestamp()
        expiry = now + config.STREAM_PLAYBACK_SESSION_TTL_SECONDS
        await self.admission.watch(pipe)
        pipe.multi()
        for previous in existing:
            if previous is not None:
                _remove_indexes(pipe, previous)
        for key, session in zip(self.keys, self.sessions, strict=True):
            pipe.set(
                key,
                json.dumps(session, ensure_ascii=False).encode(),
                ex=config.STREAM_PLAYBACK_SESSION_TTL_SECONDS,
            )
            demand_ttl = (
                config.OVERLAY_DEMAND_TTL_SECONDS
                if session['profile'] == 'overlay'
                else config.CLEAN_DEMAND_TTL_SECONDS
            )
            if session['profile'] == 'clean':
                pipe.set(_producer_key(session), b'1', ex=demand_ttl)
            for index in (
                playback_media_sessions_key(
                    session_selected_media_path(session),
                ),
                playback_demand_session_key(
                    session['base_media_path'],
                    session['profile'],
                    session['language'],
                ),
            ):
                pipe.zadd(index, {session['session_id']: expiry})
                pipe.expire(index, config.STREAM_PLAYBACK_SESSION_TTL_SECONDS)
        self.admission.queue(pipe)
        await pipe.execute()
        return self.sessions


class _PlaybackSessionMutation:
    def __init__(self, session: PlaybackSession):
        self.session = session
        self.key = playback_session_key(session['session_id'])
        self.demand_index = playback_demand_session_key(
            session['base_media_path'],
            session['profile'],
            session['language'],
        )
        self.producer_key = _producer_key(session)
        self._admission: OverlayAdmission | None = None

    async def delete(self, pipe: Pipeline) -> bool:
        await pipe.watch(self.key, self.demand_index, self.producer_key)
        if not _same_revision(
            decode_playback_session_payload(await pipe.get(self.key)),
            self.session,
        ):
            return False
        now = datetime.now(timezone.utc).timestamp()
        others = await pipe.zrangebyscore(self.demand_index, now, '+inf')
        own_id = self.session['session_id'].encode()
        others = [
            value
            for value in others
            if value
            not in (
                own_id,
                self.session['session_id'],
            )
        ]
        pipe.multi()
        _remove_indexes(pipe, self.session)
        pipe.delete(self.key)
        pipe.zremrangebyscore(self.demand_index, '-inf', now)
        if not others:
            pipe.delete(self.producer_key)
        await pipe.execute()
        return True

    async def renew(self, pipe: Pipeline) -> bool:
        if self._admission is None:
            self._admission = OverlayAdmission()
            if self.session['profile'] == 'overlay':
                self._admission.add_key(
                    self.producer_key,
                    config.OVERLAY_DEMAND_TTL_SECONDS,
                )
        admission = self._admission
        await pipe.watch(self.key)
        if not _same_revision(
            decode_playback_session_payload(await pipe.get(self.key)),
            self.session,
        ):
            return False
        await admission.watch(pipe)
        expiry = (
            datetime.now(timezone.utc).timestamp()
            + config.STREAM_PLAYBACK_SESSION_TTL_SECONDS
        )
        pipe.multi()
        if self.session['profile'] == 'clean':
            pipe.set(
                _producer_key(self.session),
                b'1',
                ex=config.CLEAN_DEMAND_TTL_SECONDS,
            )
        admission.queue(pipe)
        for index in (
            playback_media_sessions_key(
                session_selected_media_path(self.session),
            ),
            playback_demand_session_key(
                self.session['base_media_path'],
                self.session['profile'],
                self.session['language'],
            ),
        ):
            pipe.zadd(index, {self.session['session_id']: expiry})
            pipe.expire(index, config.STREAM_PLAYBACK_SESSION_TTL_SECONDS)
        await pipe.execute()
        return True


class _PlaybackIndexRegistration:
    def __init__(
        self,
        session: PlaybackSession,
        index: str,
        producer_key: str | None,
    ):
        self.session = session
        self.index = index
        self.producer_key = producer_key
        self.session_key = playback_session_key(self.session['session_id'])
        self.admission = OverlayAdmission()
        if self.producer_key is not None:
            self.admission.add_key(
                self.producer_key,
                config.OVERLAY_DEMAND_TTL_SECONDS,
            )

    async def apply(self, pipe: Pipeline) -> None:
        await pipe.watch(self.session_key)
        if not _same_revision(
            decode_playback_session_payload(await pipe.get(self.session_key)),
            self.session,
        ):
            return
        expires_at = (
            datetime.now(timezone.utc).timestamp()
            + config.STREAM_PLAYBACK_SESSION_TTL_SECONDS
        )
        await self.admission.watch(pipe)
        pipe.multi()
        if self.producer_key is not None:
            if self.producer_key not in self.admission.requested_keys:
                pipe.set(
                    self.producer_key,
                    b'1',
                    ex=config.CLEAN_DEMAND_TTL_SECONDS,
                )
            self.admission.queue(pipe)
        pipe.zadd(self.index, {self.session['session_id']: expires_at})
        pipe.expire(self.index, config.STREAM_PLAYBACK_SESSION_TTL_SECONDS)
        await pipe.execute()


class _PlaybackMediaRenewal:
    def __init__(self, media_path: str):
        self.media_path = media_path
        self.media_key = playback_media_sessions_key(self.media_path)
        self.refresh_key = playback_session_refresh_key(self.media_path)
        parsed = parse_annotated_media_path(self.media_path)
        if parsed is None:
            self.demand_key = playback_demand_session_key(
                self.media_path,
                'clean',
            )
        else:
            base, language = parsed
            self.demand_key = playback_demand_session_key(
                base,
                'overlay',
                normalise_label_language(language),
            )
        self.batch = config.STREAM_PLAYBACK_REFRESH_BATCH_SIZE

    async def apply(self, pipe: Pipeline):
        await pipe.watch(self.media_key, self.demand_key, self.refresh_key)
        if await pipe.get(self.refresh_key):
            return
        now = datetime.now(timezone.utc).timestamp()
        expired = await pipe.zrangebyscore(
            self.media_key,
            '-inf',
            now,
            start=0,
            num=self.batch,
        )
        expired_demand = await pipe.zrangebyscore(
            self.demand_key,
            '-inf',
            now,
            start=0,
            num=self.batch,
        )
        ids = await pipe.zrangebyscore(
            self.media_key,
            f'({now}',
            '+inf',
            start=0,
            num=self.batch,
        )
        ids = [
            raw.decode() if isinstance(raw, bytes) else str(raw) for raw in ids
        ]
        keys = [
            playback_session_key(
                raw.decode() if isinstance(raw, bytes) else str(raw),
            )
            for raw in ids
        ]
        if keys:
            await pipe.watch(*keys)
            sessions = await pipe.mget(keys)
        else:
            sessions = []
        active: list[str] = []
        stale: list[str] = []
        for raw, payload in zip(ids, sessions, strict=True):
            session = decode_playback_session_payload(payload)
            (
                active
                if session is not None
                and session_selected_media_path(session) == self.media_path
                else stale
            ).append(raw)
        if not ids and not expired and not expired_demand:
            return
        pipe.multi()
        if expired or stale:
            pipe.zrem(self.media_key, *expired, *stale)
        if expired_demand or stale:
            pipe.zrem(self.demand_key, *expired_demand, *stale)
        ttl = config.STREAM_PLAYBACK_SESSION_TTL_SECONDS
        for raw in active:
            pipe.expire(
                playback_session_key(
                    raw.decode() if isinstance(raw, bytes) else str(raw),
                ),
                ttl,
            )
        if active:
            scores = {raw: now + ttl for raw in active}
            pipe.zadd(self.media_key, scores)
            pipe.zadd(self.demand_key, scores)
        pipe.expire(self.media_key, ttl)
        pipe.expire(self.demand_key, ttl)
        pipe.set(
            self.refresh_key,
            b'1',
            ex=config.STREAM_PLAYBACK_SESSION_REFRESH_INTERVAL_SECONDS,
        )
        await pipe.execute()
