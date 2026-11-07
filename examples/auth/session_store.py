from __future__ import annotations

import base64
import hashlib
import json
import secrets
import time
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from functools import lru_cache
from functools import partial
from typing import Any
from typing import cast

import jwt
from cryptography.fernet import Fernet
from fastapi import HTTPException
from redis.asyncio import Redis
from redis.asyncio.client import Pipeline

from examples.auth.media_session_request import MediaSessionRequest
from examples.auth.redis_transactions import optimistic
from examples.streaming_web.overlay_admission import OverlayAdmission
from src.settings.sessions import SessionCryptoSettings
from src.settings.sessions import SessionStoreSettings

AUTH_PREFIX = 'bff:session'
OIDC_LOGOUT_STATE_PREFIX = 'bff:oidc:logout'
MEDIA_PREFIX = 'media:session'
MEDIA_PUBLIC_PREFIX = 'media:public'
MEDIA_PARENT_PREFIX = 'media:parent'
# Browser BFF sessions use ordinary OIDC refresh tokens.  They therefore share
# one online Keycloak SSO session with the Account Console instead of keeping
# an independent offline login alive after browser SSO has expired.
OIDC_ONLINE_SSO_SESSION_MODE = 'online_sso_v1'
session_settings = SessionStoreSettings()
AUTH_SESSION_TTL_SECONDS = session_settings.session_ttl_seconds
OIDC_LOGOUT_STATE_TTL_SECONDS = 120
MEDIA_SESSION_TTL_SECONDS = session_settings.media_session_ttl_seconds


def _text(value: bytes | None) -> str | None:
    if value is None:
        return None
    return value.decode('utf-8')


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


@lru_cache(maxsize=1)
def _fernet() -> Fernet:
    secret = SessionCryptoSettings().token_encryption_key.get_secret_value()
    if not secret:
        raise RuntimeError('BFF_TOKEN_ENCRYPTION_KEY is required')
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
    return f"{AUTH_PREFIX}:{_digest(session_id)}"


def oidc_logout_state_key(state: str) -> str:
    """Build a Redis key without persisting a raw OIDC logout state."""
    return f"{OIDC_LOGOUT_STATE_PREFIX}:{_digest(state)}"


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
        'auth_provider': 'oidc',
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
        'revision': 1,
    }
    data['oidc_session_mode'] = OIDC_ONLINE_SSO_SESSION_MODE
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
    if not session_id:
        return None
    raw = await redis.get(auth_session_key(session_id))
    if raw is None:
        return None
    data = json.loads(raw)
    if data['revoked'] or data.get('auth_provider') != 'oidc':
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


async def create_oidc_logout_state(
    redis: Redis,
) -> str:
    """Persist a one-use, token-free bridge for browser SSO logout."""
    state = secrets.token_urlsafe(32)
    await redis.set(
        oidc_logout_state_key(state),
        b'1',
        ex=OIDC_LOGOUT_STATE_TTL_SECONDS,
    )
    return state


async def consume_oidc_logout_state(
    redis: Redis,
    state: str | None,
) -> bool:
    """Consume a global-logout state exactly once."""
    if not state:
        return False
    raw = await redis.getdel(oidc_logout_state_key(state))
    return raw == b'1'


@dataclass(frozen=True)
class AuthTokenUpdate:
    session_id: str
    owner: str = field(repr=False)
    session: Mapping[str, Any] = field(repr=False)
    access_token: str | None = field(default=None, repr=False)
    refresh_token: str | None = field(default=None, repr=False)
    revoke: bool = False


class AuthRefreshConflict(HTTPException):
    def __init__(self):
        super().__init__(503, 'session_refresh_busy')


_COMMIT_REFRESH = """
local raw = redis.call('GET',KEYS[1])
if not raw then return -1 end
local current = cjson.decode(raw)
if current.revoked then return -1 end
if redis.call('GET',KEYS[2]) ~= ARGV[1] then return 0 end
if tonumber(current.revision or 0) ~= tonumber(ARGV[2]) then return 0 end
redis.call('SET',KEYS[1],ARGV[3],'EX',ARGV[4])
return 1
"""


async def save_auth_tokens(
    redis: Redis,
    update: AuthTokenUpdate,
) -> dict[str, Any]:
    revision = int(update.session.get('revision') or 0)
    data = dict(update.session)
    data['revision'] = revision + 1
    data['updated_at'] = int(time.time())
    if update.revoke:
        data['revoked'] = True
    else:
        if not update.access_token or not update.refresh_token:
            raise ValueError('A refreshed token pair is required')
        data.update(
            access_token_encrypted=_encrypt(update.access_token),
            access_expires_at=_jwt_exp(update.access_token),
            refresh_token_encrypted=_encrypt(update.refresh_token),
            refresh_expires_at=_jwt_exp(update.refresh_token),
        )
    key = auth_session_key(update.session_id)
    result = await redis.eval(
        _COMMIT_REFRESH,
        2,
        key,
        key + ':refresh-lock',
        update.owner,
        revision,
        json.dumps(data, separators=(',', ':')),
        AUTH_SESSION_TTL_SECONDS,
    )
    if result == -1:
        raise HTTPException(401, 'app_session_expired')
    if result != 1:
        raise AuthRefreshConflict()
    return data


async def delete_auth_session(redis: Redis, session_id: str | None) -> None:
    if not session_id:
        return
    parent = auth_session_key(session_id)
    await revoke_media_for_parent(redis, parent)


async def acquire_refresh_lock(
    redis: Redis,
    session_id: str,
    ttl_seconds: int = 15,
) -> str | None:
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
    key = f"{auth_session_key(session_id)}:refresh-lock"

    await optimistic(
        redis,
        partial(_release_refresh_owner, key=key, owner=owner),
    )


async def create_media_session(
    redis: Redis,
    request: MediaSessionRequest,
) -> tuple[str, dict[str, Any]]:
    operation = _MediaSessionCreation(request)
    await optimistic(redis, operation.apply)
    return operation.token, operation.data


def media_session_cameras(session: Mapping[str, object]) -> tuple[str, ...]:
    """Return the exact, de-duplicated camera scope of a media session."""
    return tuple(dict.fromkeys(cast(list[str], session['cameras'])))


def media_session_demand_keys(
    session: Mapping[str, object],
) -> tuple[str, ...]:
    """Return the producer-demand leases owned by a media capability."""
    return tuple(dict.fromkeys(cast(list[str], session['demand_keys'])))


async def get_media_session(
    redis: Redis,
    token: str | None,
) -> dict[str, Any] | None:
    if not token:
        return None
    raw = await redis.get(media_session_key(token))
    if raw is None:
        return None
    data = json.loads(raw)
    if int(data['expires_at']) <= int(
        time.time(),
    ) or not await _media_parent_current(redis, data):
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
    if int(data['expires_at']) <= int(
        time.time(),
    ) or not await _media_parent_current(redis, data):
        return None
    return data


def media_parent_generation_key(parent: str) -> str:
    return f"{MEDIA_PARENT_PREFIX}:generation:{_digest(parent)}"


async def media_parent_generation(redis: Redis, parent: str) -> int:
    return int(await redis.get(media_parent_generation_key(parent)) or 0)


async def _media_parent_current(redis: Redis, data: Mapping[str, Any]) -> bool:
    parent = str(data['parent'])
    keys = [media_parent_generation_key(parent)]
    if parent.startswith(AUTH_PREFIX + ':'):
        keys.append(parent)
    values = await redis.mget(keys)
    return int(values[0] or 0) == int(data.get('parent_generation', 0)) and (
        len(values) == 1 or values[1] is not None
    )


async def renew_media_session(
    redis: Redis,
    public_id: str,
    *,
    owner: str,
) -> dict[str, Any] | None:
    return await optimistic(
        redis,
        _MediaSessionMutation(public_id, owner).renew,
    )


async def delete_media_session(
    redis: Redis,
    public_id: str,
    *,
    owner: str | None = None,
) -> bool:
    return await optimistic(
        redis,
        _MediaSessionMutation(public_id, owner).delete,
    )


async def revoke_media_for_parent(redis: Redis, parent: str) -> None:
    await optimistic(redis, _MediaParentRevocation(parent).apply)


class _MediaSessionCreation:
    def __init__(self, request: MediaSessionRequest):
        self.owner = request.owner
        is_batch = request.batch
        scoped_cameras = list(dict.fromkeys(request.cameras))
        if not scoped_cameras:
            raise ValueError('media session requires at least one camera')
        self.token = secrets.token_urlsafe(32)
        self.public_id = secrets.token_urlsafe(12)
        now = int(time.time())
        self.data: dict[str, Any] = {
            'id': self.public_id,
            'user_id': self.owner.user_id,
            'username': self.owner.username,
            'site': request.site,
            'camera': None if is_batch else scoped_cameras[0],
            'cameras': scoped_cameras,
            'scope': 'batch' if is_batch else 'camera',
            'profile': request.profile,
            'parent': self.owner.parent,
            'platform': self.owner.platform,
            'user_active': True,
            'created_at': now,
            'expires_at': now + MEDIA_SESSION_TTL_SECONDS,
        }
        if request.language:
            self.data['language'] = request.language
        if request.quality:
            self.data['quality'] = request.quality
        if request.purpose:
            self.data['purpose'] = request.purpose
        self.data['demand_keys'] = list(
            dict.fromkeys(request.demand_keys or ()),
        )
        self.data['playback_sessions'] = {
            session_id: dict(descriptor)
            for session_id, descriptor in (
                request.playback_sessions or {}
            ).items()
        }
        self.token_key = media_session_key(self.token)
        self.parent_key = f"{MEDIA_PARENT_PREFIX}:{_digest(self.owner.parent)}"
        self.admission = OverlayAdmission()
        for key in media_session_demand_keys(self.data):
            self.admission.add_key(key, MEDIA_SESSION_TTL_SECONDS)

    async def apply(self, pipe: Pipeline) -> None:
        generation_key = media_parent_generation_key(self.owner.parent)
        await pipe.watch(generation_key, self.owner.parent)
        generation = int(await pipe.get(generation_key) or 0)
        if (
            self.owner.generation is not None
            and generation != self.owner.generation
        ):
            raise HTTPException(401, 'app_session_expired')
        if self.owner.parent.startswith(
            AUTH_PREFIX + ':',
        ) and not await pipe.exists(self.owner.parent):
            raise HTTPException(401, 'app_session_expired')
        self.data['parent_generation'] = generation
        await self.admission.watch(pipe)
        pipe.multi()
        pipe.set(
            self.token_key,
            json.dumps(
                self.data,
                separators=(
                    ',',
                    ':',
                ),
            ).encode(),
            ex=MEDIA_SESSION_TTL_SECONDS,
        )
        pipe.set(
            f"{MEDIA_PUBLIC_PREFIX}:{self.public_id}",
            self.token_key.encode(),
            ex=MEDIA_SESSION_TTL_SECONDS,
        )
        pipe.sadd(self.parent_key, self.token_key.encode())
        pipe.expire(self.parent_key, AUTH_SESSION_TTL_SECONDS)
        for key in media_session_demand_keys(self.data):
            if key not in self.admission.requested_keys:
                pipe.set(key, b'1', ex=MEDIA_SESSION_TTL_SECONDS)
        self.admission.queue(pipe)
        await pipe.execute()


class _MediaSessionMutation:
    def __init__(self, public_id: str, owner: str | None):
        self.public_key = f"{MEDIA_PUBLIC_PREFIX}:{public_id}"
        self.owner = owner

    async def renew(self, pipe: Pipeline) -> dict[str, Any] | None:
        assert self.owner is not None
        await pipe.watch(self.public_key)
        token_key = _text(await pipe.get(self.public_key))
        if token_key is None:
            return None
        generation_key = media_parent_generation_key(self.owner)
        await pipe.watch(token_key, generation_key, self.owner)
        raw = await pipe.get(token_key)
        if raw is None:
            pipe.multi()
            pipe.delete(self.public_key)
            await pipe.execute()
            return None
        data = json.loads(raw)
        now = int(time.time())
        if data['parent'] != self.owner or int(data['expires_at']) <= now:
            return None
        if int(data.get('parent_generation', 0)) != int(
            await pipe.get(generation_key) or 0,
        ):
            return None
        if self.owner.startswith(AUTH_PREFIX + ':') and not await pipe.exists(
            self.owner,
        ):
            return None
        data['expires_at'] = now + MEDIA_SESSION_TTL_SECONDS
        admission = OverlayAdmission()
        for key in media_session_demand_keys(data):
            admission.add_key(key, MEDIA_SESSION_TTL_SECONDS)
        await admission.watch(pipe)
        pipe.multi()
        pipe.set(
            token_key,
            json.dumps(
                data,
                separators=(
                    ',',
                    ':',
                ),
            ).encode(),
            ex=MEDIA_SESSION_TTL_SECONDS,
        )
        pipe.set(
            self.public_key,
            token_key.encode(),
            ex=MEDIA_SESSION_TTL_SECONDS,
        )
        pipe.sadd(
            f"{MEDIA_PARENT_PREFIX}:{_digest(self.owner)}",
            token_key.encode(),
        )
        pipe.expire(
            f"{MEDIA_PARENT_PREFIX}:{_digest(self.owner)}",
            AUTH_SESSION_TTL_SECONDS,
        )
        for key in media_session_demand_keys(data):
            if key not in admission.requested_keys:
                pipe.set(key, b'1', ex=MEDIA_SESSION_TTL_SECONDS)
        admission.queue(pipe)
        await pipe.execute()
        return data

    async def delete(self, pipe: Pipeline) -> bool:
        await pipe.watch(self.public_key)
        token_key = _text(await pipe.get(self.public_key))
        if token_key is None:
            return False
        await pipe.watch(token_key)
        raw = await pipe.get(token_key)
        if raw is None:
            pipe.multi()
            pipe.delete(self.public_key)
            await pipe.execute()
            return False
        data = json.loads(raw)
        if self.owner is not None and data['parent'] != self.owner:
            return False
        pipe.multi()
        pipe.delete(token_key, self.public_key)
        pipe.srem(
            f"{MEDIA_PARENT_PREFIX}:{_digest(data['parent'])}",
            token_key,
        )
        await pipe.execute()
        return True


class _MediaParentRevocation:
    def __init__(self, parent: str):
        self.parent = parent
        self.parent_key = f"{MEDIA_PARENT_PREFIX}:{_digest(self.parent)}"
        self.generation_key = media_parent_generation_key(self.parent)

    async def apply(self, pipe: Pipeline) -> None:
        await pipe.watch(self.parent_key, self.generation_key, self.parent)
        generation = int(await pipe.get(self.generation_key) or 0) + 1
        members = await pipe.smembers(self.parent_key)
        tokens = [_text(member) for member in members]
        tokens = [key for key in tokens if key is not None]
        if tokens:
            await pipe.watch(*tokens)
            records = await pipe.mget(tokens)
        else:
            records = []
        pipe.multi()
        pipe.set(
            self.generation_key,
            str(generation).encode(),
            ex=AUTH_SESSION_TTL_SECONDS,
        )
        for token_key, raw in zip(tokens, records):
            if raw is not None:
                data = json.loads(raw)
                if data.get('parent', self.parent) == self.parent:
                    pipe.delete(
                        f"{MEDIA_PUBLIC_PREFIX}:{data['id']}",
                        token_key,
                    )
        pipe.delete(self.parent_key)
        if self.parent.startswith(AUTH_PREFIX + ':'):
            pipe.delete(self.parent, f'{self.parent}:refresh-lock')
        await pipe.execute()


async def _release_refresh_owner(
    pipe: Pipeline,
    *,
    key: str,
    owner: str,
) -> None:
    await pipe.watch(key)
    if _text(await pipe.get(key)) == owner:
        pipe.multi()
        pipe.delete(key)
        await pipe.execute()
