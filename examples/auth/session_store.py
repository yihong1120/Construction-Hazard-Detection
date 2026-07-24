from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import time
from collections.abc import Mapping
from typing import Any

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


def _text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode('utf-8')
    return str(value)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _fernet() -> Fernet:
    secret = (
        os.getenv('BFF_TOKEN_ENCRYPTION_KEY', '').strip()
        or settings.authjwt_secret_key
    )
    key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest())
    return Fernet(key)


def _encrypt(value: str) -> str:
    return _fernet().encrypt(value.encode()).decode()


def _decrypt(value: str) -> str:
    return _fernet().decrypt(value.encode()).decode()


def _jwt_exp(token: str) -> int:
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
    return f'{AUTH_PREFIX}:{_digest(session_id)}'


def media_session_key(token: str) -> str:
    """Build a Redis key without persisting the opaque bearer/cookie value."""
    return f'{MEDIA_PREFIX}:{_digest(token)}'


async def create_auth_session(
    redis: Redis,
    token_pair: Mapping[str, object],
    user: Mapping[str, object],
) -> tuple[str, dict[str, Any]]:
    """Store server-side tokens and return a 256-bit opaque session id."""
    access_token = str(token_pair['access_token'])
    refresh_token = str(token_pair['refresh_token'])
    raw_features = token_pair.get('feature_names')
    feature_names = (
        [str(value) for value in raw_features]
        if isinstance(raw_features, (list, tuple))
        else []
    )
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
    await redis.set(
        key,
        json.dumps(data, separators=(',', ':')),
        ex=AUTH_SESSION_TTL_SECONDS,
    )
    return session_id, data


async def get_auth_session(
    redis: Redis,
    session_id: str | None,
) -> dict[str, Any] | None:
    if not session_id:
        return None
    raw = _text(await redis.get(auth_session_key(session_id)))
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or data.get('revoked') is True:
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
        json.dumps(session, separators=(',', ':')),
        ex=AUTH_SESSION_TTL_SECONDS,
    )


async def delete_auth_session(redis: Redis, session_id: str | None) -> None:
    if not session_id:
        return
    parent = auth_session_key(session_id)
    await revoke_media_for_parent(redis, parent)
    await redis.delete(parent, f'{parent}:refresh-lock')


async def acquire_refresh_lock(
    redis: Redis,
    session_id: str,
    ttl_seconds: int = 15,
) -> str | None:
    owner = secrets.token_urlsafe(18)
    acquired = await redis.set(
        f'{auth_session_key(session_id)}:refresh-lock',
        owner,
        ex=ttl_seconds,
        nx=True,
    )
    return owner if acquired else None


async def release_refresh_lock(
    redis: Redis,
    session_id: str,
    owner: str,
) -> None:
    key = f'{auth_session_key(session_id)}:refresh-lock'
    if _text(await redis.get(key)) == owner:
        await redis.delete(key)


async def create_media_session(
    redis: Redis,
    *,
    user_id: int,
    username: str,
    site: str,
    camera: str | None = None,
    cameras: list[str] | tuple[str, ...] | None = None,
    profile: str,
    parent: str,
    platform: str,
    language: str | None = None,
    quality: str | None = None,
    purpose: str | None = None,
    demand_keys: list[str] | tuple[str, ...] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Create a separately revocable, narrowly scoped media capability."""
    is_batch = cameras is not None
    scoped_cameras = list(
        dict.fromkeys(
            cameras if cameras is not None else ([camera] if camera else []),
        ),
    )
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
    scoped_demand_keys = list(
        dict.fromkeys(
            key for key in (demand_keys or ())
            if isinstance(key, str) and key
        ),
    )
    if scoped_demand_keys:
        data['demand_keys'] = scoped_demand_keys
    token_key = media_session_key(token)
    parent_key = f'{MEDIA_PARENT_PREFIX}:{_digest(parent)}'
    payload = json.dumps(data, separators=(',', ':'))
    await redis.set(token_key, payload, ex=MEDIA_SESSION_TTL_SECONDS)
    await redis.set(
        f'{MEDIA_PUBLIC_PREFIX}:{public_id}',
        token_key,
        ex=MEDIA_SESSION_TTL_SECONDS,
    )
    await redis.sadd(parent_key, token_key)
    await redis.expire(parent_key, AUTH_SESSION_TTL_SECONDS)
    await _refresh_media_session_demands(redis, data)
    return token, data


def media_session_cameras(session: Mapping[str, object]) -> tuple[str, ...]:
    """Return the exact, de-duplicated camera scope of a media session."""
    raw_cameras = session.get('cameras')
    cameras: list[str] = []
    if isinstance(raw_cameras, (list, tuple)):
        cameras.extend(
            camera
            for camera in raw_cameras
            if isinstance(camera, str) and camera
        )
    if not cameras:
        camera = session.get('camera')
        if isinstance(camera, str) and camera:
            cameras.append(camera)
    return tuple(dict.fromkeys(cameras))


def media_session_demand_keys(
    session: Mapping[str, object],
) -> tuple[str, ...]:
    """Return the producer-demand leases owned by a media capability."""
    raw_keys = session.get('demand_keys')
    if not isinstance(raw_keys, (list, tuple)):
        return ()
    return tuple(
        dict.fromkeys(
            key for key in raw_keys
            if isinstance(key, str) and key
        ),
    )


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
    if not token:
        return None
    raw = _text(await redis.get(media_session_key(token)))
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    if int(data.get('expires_at') or 0) <= int(time.time()):
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
            f'{MEDIA_PUBLIC_PREFIX}:{public_id}',
        ),
    )
    if not token_key:
        return None
    raw = _text(await redis.get(token_key))
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    if int(data.get('expires_at') or 0) <= int(time.time()):
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
    public_key = f'{MEDIA_PUBLIC_PREFIX}:{public_id}'
    token_key = _text(await redis.get(public_key))
    if not token_key:
        return None
    raw = _text(await redis.get(token_key))
    if not raw:
        await redis.delete(public_key)
        return None
    try:
        data = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    if data.get('parent') != owner:
        return None

    now = int(time.time())
    if int(data.get('expires_at') or 0) <= now:
        return None
    data['expires_at'] = now + MEDIA_SESSION_TTL_SECONDS
    payload = json.dumps(data, separators=(',', ':'))
    await redis.set(token_key, payload, ex=MEDIA_SESSION_TTL_SECONDS)
    await redis.set(public_key, token_key, ex=MEDIA_SESSION_TTL_SECONDS)
    await _refresh_media_session_demands(redis, data)
    return data


async def delete_media_session(
    redis: Redis,
    public_id: str,
    *,
    owner: str | None = None,
) -> bool:
    public_key = f'{MEDIA_PUBLIC_PREFIX}:{public_id}'
    token_key = _text(await redis.get(public_key))
    if not token_key:
        return False
    raw = _text(await redis.get(token_key))
    if not raw:
        await redis.delete(public_key)
        return False
    try:
        data = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return False
    if owner is not None and data.get('parent') != owner:
        return False
    parent = str(data.get('parent') or '')
    await redis.delete(token_key, public_key)
    if parent:
        await redis.srem(f'{MEDIA_PARENT_PREFIX}:{_digest(parent)}', token_key)
    return True


async def revoke_media_for_parent(redis: Redis, parent: str) -> None:
    parent_key = f'{MEDIA_PARENT_PREFIX}:{_digest(parent)}'
    members = await redis.smembers(parent_key)
    for member in members or ():
        token_key = _text(member)
        if not token_key:
            continue
        raw = _text(await redis.get(token_key))
        if raw:
            try:
                public_id = str(json.loads(raw).get('id') or '')
            except (TypeError, json.JSONDecodeError):
                public_id = ''
            if public_id:
                await redis.delete(f'{MEDIA_PUBLIC_PREFIX}:{public_id}')
        await redis.delete(token_key)
    await redis.delete(parent_key)
