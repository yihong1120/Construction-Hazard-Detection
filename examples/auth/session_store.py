from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import time
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any
from typing import cast

import jwt
from cryptography.fernet import Fernet
from redis.asyncio import Redis

from examples.auth.config import Settings

settings = Settings()

AUTH_PREFIX = 'bff:session'
MEDIA_PREFIX = 'media:session'
MEDIA_PUBLIC_PREFIX = 'media:public'
MEDIA_PARENT_PREFIX = 'media:parent'
AUTH_SESSION_TTL_SECONDS = int(
    os.getenv('BFF_SESSION_TTL_SECONDS', str(30 * 24 * 3600)),
)
MEDIA_SESSION_TTL_SECONDS = max(
    300,
    min(900, int(os.getenv('MEDIA_SESSION_TTL_SECONDS', '600'))),
)


def _text(value: bytes | None) -> str | None:
    """Perform text.

    Args:
        value: Value used by this callable.

    Returns:
        The callable result.
    """
    if value is None:
        return None
    return value.decode('utf-8')


def _digest(value: str) -> str:
    """Perform digest.

    Args:
        value: Value used by this callable.

    Returns:
        The callable result.
    """
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _fernet() -> Fernet:
    """Perform fernet.

    Returns:
        The callable result.
    """
    secret = (
        os.getenv('BFF_TOKEN_ENCRYPTION_KEY', '').strip()
        or settings.authjwt_secret_key
    )
    key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest())
    return Fernet(key)


def _encrypt(value: str) -> str:
    """Perform encrypt.

    Args:
        value: Value used by this callable.

    Returns:
        The callable result.
    """
    return _fernet().encrypt(value.encode()).decode()


def _decrypt(value: str) -> str:
    """Perform decrypt.

    Args:
        value: Value used by this callable.

    Returns:
        The callable result.
    """
    return _fernet().decrypt(value.encode()).decode()


def _jwt_exp(token: str) -> int:
    """Perform jwt exp.

    Args:
        token: Value used by this callable.

    Returns:
        The callable result.
    """
    try:
        payload = jwt.decode(
            token,
            options={'verify_signature': False, 'verify_exp': False},
        )
        return int(payload.get('exp') or 0)
    except (ValueError, TypeError, jwt.PyJWTError):
        return 0


def auth_session_key(session_id: str) -> str:
    """Build a Redis key without persisting the opaque cookie value."""
    return f"{AUTH_PREFIX}:{_digest(session_id)}"


def media_session_key(token: str) -> str:
    """Build a Redis key without persisting the opaque bearer/cookie value."""
    return f"{MEDIA_PREFIX}:{_digest(token)}"


async def create_auth_session(
    redis: Redis,
    token_pair: Mapping[str, object],
    user: Mapping[str, object],
) -> tuple[str, dict[str, Any]]:
    """Store server-side tokens and return a 256-bit opaque session id."""
    access_token = str(token_pair['access_token'])
    refresh_token = str(token_pair['refresh_token'])
    feature_names = list(cast(Sequence[str], token_pair['feature_names']))
    session_id = secrets.token_urlsafe(32)
    key = auth_session_key(session_id)
    now = int(time.time())
    data: dict[str, Any] = {
        'session_id_hash': key.rsplit(':', 1)[-1],
        'user': dict(user),
        'feature_names': feature_names,
        'access_token_encrypted': _encrypt(access_token),
        'access_expires_at': _jwt_exp(access_token),
        'refresh_token_encrypted': _encrypt(refresh_token),
        'refresh_expires_at': _jwt_exp(refresh_token),
        'csrf_secret': secrets.token_urlsafe(32),
        'created_at': now,
        'updated_at': now,
        'revoked': False,
    }
    # BFF sessions retain the same non-secret deployment contract as their
    # tokens.  It is checked before reuse and refresh, so deployment changes
    # force a fresh browser sign-in as well.
    if token_pair.get('deployment') is not None:
        data['deployment'] = dict(
            cast(Mapping[str, object], token_pair['deployment']),
        )
    await redis.set(
        key,
        json.dumps(data, separators=(',', ':')).encode('utf-8'),
        ex=AUTH_SESSION_TTL_SECONDS,
    )
    return session_id, data


async def get_auth_session(
    redis: Redis,
    session_id: str | None,
) -> dict[str, Any] | None:
    """Perform get auth session.

    Args:
        redis: Value used by this callable.
        session_id: Value used by this callable.

    Returns:
        The callable result.
    """
    if not session_id:
        return None
    raw = await redis.get(auth_session_key(session_id))
    if raw is None:
        return None
    data = json.loads(raw)
    if data['revoked']:
        return None
    return data


async def touch_auth_session(
    redis: Redis,
    session_id: str,
) -> bool:
    """Extend the idle lifetime of an authenticated BFF session."""
    return bool(
        await redis.expire(
            auth_session_key(session_id),
            AUTH_SESSION_TTL_SECONDS,
        ),
    )


def auth_tokens(session: Mapping[str, object]) -> tuple[str, str]:
    """Decrypt the access and refresh token only at the proxy boundary."""
    return (
        _decrypt(str(session['access_token_encrypted'])),
        _decrypt(str(session['refresh_token_encrypted'])),
    )


async def save_auth_tokens(
    redis: Redis,
    session_id: str,
    session: dict[str, Any],
    access_token: str,
    refresh_token: str,
    feature_names: list[str] | None = None,
) -> None:
    """Perform save auth tokens.

    Args:
        redis: Value used by this callable.
        session_id: Value used by this callable.
        session: Value used by this callable.
        access_token: Value used by this callable.
        refresh_token: Value used by this callable.
        feature_names: Value used by this callable.
    """
    key = auth_session_key(session_id)
    session['access_token_encrypted'] = _encrypt(access_token)
    session['access_expires_at'] = _jwt_exp(access_token)
    session['refresh_token_encrypted'] = _encrypt(refresh_token)
    session['refresh_expires_at'] = _jwt_exp(refresh_token)
    session['updated_at'] = int(time.time())
    if feature_names is not None:
        session['feature_names'] = feature_names
    await redis.set(
        key,
        json.dumps(session, separators=(',', ':')).encode('utf-8'),
        ex=AUTH_SESSION_TTL_SECONDS,
    )


async def delete_auth_session(redis: Redis, session_id: str | None) -> None:
    """Perform delete auth session.

    Args:
        redis: Value used by this callable.
        session_id: Value used by this callable.
    """
    if not session_id:
        return
    parent = auth_session_key(session_id)
    await revoke_media_for_parent(redis, parent)
    await redis.delete(parent, f"{parent}:refresh-lock")


async def acquire_refresh_lock(
    redis: Redis,
    session_id: str,
    ttl_seconds: int = 15,
) -> str | None:
    """Perform acquire refresh lock.

    Args:
        redis: Value used by this callable.
        session_id: Value used by this callable.
        ttl_seconds: Value used by this callable.

    Returns:
        The callable result.
    """
    owner = secrets.token_urlsafe(18)
    acquired = await redis.set(
        f"{auth_session_key(session_id)}:refresh-lock",
        owner.encode('utf-8'),
        ex=ttl_seconds,
        nx=True,
    )
    return owner if acquired else None


async def release_refresh_lock(
    redis: Redis,
    session_id: str,
    owner: str,
) -> None:
    """Perform release refresh lock.

    Args:
        redis: Value used by this callable.
        session_id: Value used by this callable.
        owner: Value used by this callable.
    """
    key = f"{auth_session_key(session_id)}:refresh-lock"
    if _text(await redis.get(key)) == owner:
        await redis.delete(key)


async def create_media_session(
    redis: Redis,
    *,
    user_id: int,
    username: str,
    site: str,
    camera: str | None = None,
    cameras: Sequence[str] | None = None,
    profile: str,
    parent: str,
    platform: str,
    language: str | None = None,
    quality: str | None = None,
    purpose: str | None = None,
    demand_keys: Sequence[str] | None = None,
    playback_sessions: Mapping[str, Mapping[str, object]] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Create a separately revocable, narrowly scoped media capability."""
    is_batch = cameras is not None
    camera_scope = (
        cameras if cameras is not None else ((camera,) if camera else ())
    )
    scoped_cameras = list(dict.fromkeys(camera_scope))
    if not scoped_cameras:
        raise ValueError('media session requires at least one camera')
    token = secrets.token_urlsafe(32)
    public_id = secrets.token_urlsafe(12)
    now = int(time.time())
    data: dict[str, Any] = {
        'id': public_id,
        'user_id': user_id,
        'username': username,
        'site': site,
        'camera': None if is_batch else scoped_cameras[0],
        'cameras': scoped_cameras,
        'scope': 'batch' if is_batch else 'camera',
        'profile': profile,
        'parent': parent,
        'platform': platform,
        'user_active': True,
        'created_at': now,
        'expires_at': now + MEDIA_SESSION_TTL_SECONDS,
    }
    if language:
        data['language'] = language
    if quality:
        data['quality'] = quality
    if purpose:
        data['purpose'] = purpose
    data['demand_keys'] = list(dict.fromkeys(demand_keys or ()))
    data['playback_sessions'] = {
        session_id: dict(descriptor)
        for session_id, descriptor in (playback_sessions or {}).items()
    }
    token_key = media_session_key(token)
    parent_key = f"{MEDIA_PARENT_PREFIX}:{_digest(parent)}"
    payload = json.dumps(data, separators=(',', ':')).encode('utf-8')
    await redis.set(token_key, payload, ex=MEDIA_SESSION_TTL_SECONDS)
    await redis.set(
        f"{MEDIA_PUBLIC_PREFIX}:{public_id}",
        token_key.encode('utf-8'),
        ex=MEDIA_SESSION_TTL_SECONDS,
    )
    await redis.sadd(parent_key, token_key.encode('utf-8'))
    await redis.expire(parent_key, AUTH_SESSION_TTL_SECONDS)
    await _refresh_media_session_demands(redis, data)
    return token, data


def media_session_cameras(session: Mapping[str, object]) -> tuple[str, ...]:
    """Return the exact, de-duplicated camera scope of a media session."""
    return tuple(dict.fromkeys(cast(list[str], session['cameras'])))


def media_session_demand_keys(
    session: Mapping[str, object],
) -> tuple[str, ...]:
    """Return the producer-demand leases owned by a media capability."""
    return tuple(dict.fromkeys(cast(list[str], session['demand_keys'])))


async def _refresh_media_session_demands(
    redis: Redis,
    session: Mapping[str, object],
) -> None:
    """Keep publishers alive for the full authenticated playback lease."""
    for key in media_session_demand_keys(session):
        await redis.set(key, b'1', ex=MEDIA_SESSION_TTL_SECONDS)


async def get_media_session(
    redis: Redis,
    token: str | None,
) -> dict[str, Any] | None:
    """Perform get media session.

    Args:
        redis: Value used by this callable.
        token: Value used by this callable.

    Returns:
        The callable result.
    """
    if not token:
        return None
    raw = await redis.get(media_session_key(token))
    if raw is None:
        return None
    data = json.loads(raw)
    if int(data['expires_at']) <= int(time.time()):
        return None
    return data


async def get_media_session_by_id(
    redis: Redis,
    public_id: str | None,
) -> dict[str, Any] | None:
    """Load a media capability by its non-secret public identifier."""
    if not public_id:
        return None
    token_key = _text(
        await redis.get(
            f"{MEDIA_PUBLIC_PREFIX}:{public_id}",
        ),
    )
    if token_key is None:
        return None
    raw = await redis.get(token_key)
    if raw is None:
        return None
    data = json.loads(raw)
    if int(data['expires_at']) <= int(time.time()):
        return None
    return data


async def renew_media_session(
    redis: Redis,
    public_id: str,
    *,
    owner: str,
) -> dict[str, Any] | None:
    """Extend one owned media capability without changing its opaque token.

    Playback URLs embed the opaque capability as ``mt``. Keeping that token
    stable lets HLS continue fetching segments while an authenticated client
    renews its lease, instead of forcing every player to load a new manifest.
    """
    public_key = f"{MEDIA_PUBLIC_PREFIX}:{public_id}"
    token_key = _text(await redis.get(public_key))
    if token_key is None:
        return None
    raw = await redis.get(token_key)
    if raw is None:
        await redis.delete(public_key)
        return None
    data = json.loads(raw)
    if data['parent'] != owner:
        return None

    now = int(time.time())
    if int(data['expires_at']) <= now:
        return None
    data['expires_at'] = now + MEDIA_SESSION_TTL_SECONDS
    payload = json.dumps(data, separators=(',', ':')).encode('utf-8')
    await redis.set(token_key, payload, ex=MEDIA_SESSION_TTL_SECONDS)
    await redis.set(
        public_key,
        token_key.encode('utf-8'),
        ex=MEDIA_SESSION_TTL_SECONDS,
    )
    await _refresh_media_session_demands(redis, data)
    return data


async def delete_media_session(
    redis: Redis,
    public_id: str,
    *,
    owner: str | None = None,
) -> bool:
    """Perform delete media session.

    Args:
        redis: Value used by this callable.
        public_id: Value used by this callable.
        owner: Value used by this callable.

    Returns:
        The callable result.
    """
    public_key = f"{MEDIA_PUBLIC_PREFIX}:{public_id}"
    token_key = _text(await redis.get(public_key))
    if token_key is None:
        return False
    raw = await redis.get(token_key)
    if raw is None:
        await redis.delete(public_key)
        return False
    data = json.loads(raw)
    if owner is not None and data['parent'] != owner:
        return False
    parent = data['parent']
    await redis.delete(token_key, public_key)
    await redis.srem(f"{MEDIA_PARENT_PREFIX}:{_digest(parent)}", token_key)
    return True


async def revoke_media_for_parent(redis: Redis, parent: str) -> None:
    """Perform revoke media for parent.

    Args:
        redis: Value used by this callable.
        parent: Value used by this callable.
    """
    parent_key = f"{MEDIA_PARENT_PREFIX}:{_digest(parent)}"
    members = await redis.smembers(parent_key)
    for member in members:
        token_key = _text(member)
        raw = await redis.get(token_key)
        if raw is not None:
            public_id = json.loads(raw)['id']
            await redis.delete(f"{MEDIA_PUBLIC_PREFIX}:{public_id}")
        await redis.delete(token_key)
    await redis.delete(parent_key)
