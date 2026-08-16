from __future__ import annotations

import datetime
import hashlib
import json
from typing import cast
from uuid import uuid4

import httpx
import jwt
from fastapi import HTTPException
from redis.asyncio import Redis
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from examples.auth.cache import get_user_data
from examples.auth.cache import set_user_data
from examples.auth.config import Settings
from examples.auth.jwt_config import access_token_subject_from_payload
from examples.auth.jwt_config import jwt_access
from examples.auth.jwt_config import jwt_refresh
from examples.auth.jwt_config import refresh_token_subject_from_payload
from examples.auth.models import Feature
from examples.auth.models import group_features_table
from examples.auth.models import User
from examples.auth.models import USER_STATUS_ACTIVE
from examples.auth.models import USER_STATUS_EMAIL_UNVERIFIED
from examples.auth.models import USER_STATUS_PENDING_ADMIN_APPROVAL
from examples.auth.models import USER_STATUS_REJECTED
from examples.auth.models import USER_STATUS_SUSPENDED
from examples.auth.models import UserProfile
from examples.auth.token_cleanup import prune_user_cache
from examples.auth.token_revocation import AccessTokenRevocationPayload
from examples.auth.token_revocation import revoke_access_token
from examples.auth.token_revocation import revoke_access_token_jtis
from examples.db_management.schemas.auth import DbUserInfo
from examples.db_management.schemas.auth import RefreshRequest
from examples.db_management.schemas.auth import RefreshTokenPayload
from examples.db_management.schemas.auth import TokenPairData
from examples.db_management.schemas.auth import UserCache
from examples.db_management.schemas.auth import UserLogin

# Configuration settings for JWT authentication
settings = Settings()
SECRET_KEY = settings.authjwt_secret_key
HCAPTCHA_SECRET_KEY = settings.hcaptcha_secret_key
HCAPTCHA_SITE_KEY = settings.hcaptcha_site_key
HCAPTCHA_BYPASS_KEY = settings.hcaptcha_bypass_key
HCAPTCHA_VERIFY_URL = 'https://api.hcaptcha.com/siteverify'
ALGORITHM = 'HS256'
SUPER_ADMIN = 'ChangDar'
ACCESS_TTL = datetime.timedelta(minutes=15)  # Short-lived API capability
REFRESH_TTL = datetime.timedelta(days=30)    # Refresh token expiry time


def _hash_account_identifier(identifier: str) -> str:
    """Hash an account identifier before storing it in Redis keys.

    Args:
        identifier: Username or email address from an authentication request.

    Returns:
        Stable non-reversible identifier hash.
    """
    normalized = identifier.strip().lower()
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def _hash_login_pair(identifier: str, client_ip: str | None) -> str:
    """Hash a login identifier and client IP for Redis keying.

    Args:
        identifier: Login identifier.
        client_ip: Requesting client address.

    Returns:
        Stable non-reversible pair hash.
    """
    normalized = identifier.strip().lower()
    source = client_ip or 'unknown'
    return hashlib.sha256(f'{normalized}|{source}'.encode()).hexdigest()


def _login_fail_key(identifier: str, client_ip: str | None) -> str:
    """Build the Redis key for failures from one login pair.

    Args:
        identifier: Login identifier.
        client_ip: Requesting client address.

    Returns:
        Login-pair failure-counter key.
    """
    return f'login_fail:pair:{_hash_login_pair(identifier, client_ip)}'


def _login_cooldown_key(identifier: str, client_ip: str | None) -> str:
    """Build the Redis key for a login-pair cooldown.

    Args:
        identifier: Login identifier.
        client_ip: Requesting client address.

    Returns:
        Login-pair cooldown key.
    """
    return f'login_cooldown:pair:{_hash_login_pair(identifier, client_ip)}'


def _account_fail_key(identifier: str) -> str:
    """Build the Redis key for account-wide login failures.

    Args:
        identifier: Login identifier.

    Returns:
        Account failure-counter key.
    """
    return f'login_fail:account:{_hash_account_identifier(identifier)}'


def _login_lock_key(identifier: str) -> str:
    """Build the Redis key for an account login lock.

    Args:
        identifier: Login identifier.

    Returns:
        Account-lock key.
    """
    return f'login_lock:account:{_hash_account_identifier(identifier)}'


def _login_pair_index_key(identifier: str) -> str:
    """Build the Redis key indexing client-pair login records.

    Args:
        identifier: Login identifier.

    Returns:
        Client-pair index key.
    """
    return f'login_pairs:account:{_hash_account_identifier(identifier)}'


def _login_fail_pair_key(pair_hash: str) -> str:
    """Build the failure-counter key for a client-pair hash.

    Args:
        pair_hash: Hashed login identifier and client IP.

    Returns:
        Login-pair failure-counter key.
    """
    return f'login_fail:pair:{pair_hash}'


def _login_cooldown_pair_key(pair_hash: str) -> str:
    """Build the cooldown key for a client-pair hash.

    Args:
        pair_hash: Hashed login identifier and client IP.

    Returns:
        Login-pair cooldown key.
    """
    return f'login_cooldown:pair:{pair_hash}'


def _utc_iso_after(seconds: int) -> str:
    """Return a UTC ISO timestamp offset by the requested seconds.

    Args:
        seconds: Time offset from the current UTC time.

    Returns:
        Timezone-aware ISO 8601 timestamp.
    """
    expires_at = (
        datetime.datetime.now(datetime.timezone.utc)
        + datetime.timedelta(seconds=seconds)
    )
    return expires_at.replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _decode_redis_value(value: bytes | None) -> str | None:
    """Decode an optional UTF-8 Redis value.

    Args:
        value: Redis value that may be bytes, text, or absent.

    Returns:
        Decoded text, or ``None`` when no value is supplied.
    """
    if value is None:
        return None
    return value.decode('utf-8')


def _decode_redis_members(values: set[bytes]) -> list[str]:
    """Decode and sort UTF-8 Redis set members.

    Args:
        values: Raw Redis set members.

    Returns:
        Sorted decoded set members.
    """
    return sorted(value.decode('utf-8') for value in values)


async def _get_positive_ttl(redis_pool: Redis, key: str, fallback: int) -> int:
    """Return a positive Redis TTL, using a fallback when necessary.

    Args:
        redis_pool: Redis connection used to read expiry state.
        key: Key whose expiry is inspected.
        fallback: TTL used when the key has no positive expiry.

    Returns:
        Positive TTL in seconds.
    """
    ttl = int(await redis_pool.ttl(key))
    return ttl if ttl > 0 else fallback


async def _check_login_guard(
    redis_pool: Redis,
    identifier: str,
    client_ip: str | None,
) -> None:
    """Reject a login while its identifier or IP is guarded.

    Args:
        redis_pool: Redis connection holding login guards.
        identifier: Login identifier being checked.
        client_ip: Requesting client address.

    Raises:
        HTTPException: If a lock or cooldown is active.
    """
    lock_key = _login_lock_key(identifier)
    locked_until = _decode_redis_value(await redis_pool.get(lock_key))
    if locked_until:
        raise HTTPException(
            status_code=423,
            detail={
                'code': 'account_locked',
                'locked_until': locked_until,
            },
        )

    cooldown_key = _login_cooldown_key(identifier, client_ip)
    cooldown_marker = await redis_pool.get(cooldown_key)
    if cooldown_marker is not None:
        retry_after = await _get_positive_ttl(
            redis_pool,
            cooldown_key,
            settings.login_cooldown_seconds,
        )
        raise HTTPException(
            status_code=429,
            detail={
                'code': 'login_cooldown',
                'retry_after_seconds': retry_after,
            },
            headers={'Retry-After': str(retry_after)},
        )


async def _record_failed_login(
    redis_pool: Redis,
    identifier: str,
    client_ip: str | None,
) -> None:
    """Record a failed login and apply the relevant guard.

    Args:
        redis_pool: Redis connection holding guard counters.
        identifier: Login identifier that failed.
        client_ip: Requesting client address.
    """
    pair_hash = _hash_login_pair(identifier, client_ip)
    pair_index_key = _login_pair_index_key(identifier)
    index_ttl = max(
        settings.login_failure_window_seconds,
        settings.login_cooldown_seconds,
        settings.login_lock_seconds,
    )
    await redis_pool.sadd(pair_index_key, pair_hash)
    await redis_pool.expire(pair_index_key, index_ttl)

    fail_key = _login_fail_key(identifier, client_ip)
    pair_fail_count = int(await redis_pool.incr(fail_key))
    if pair_fail_count == 1:
        await redis_pool.expire(
            fail_key,
            settings.login_failure_window_seconds,
        )

    account_fail_key = _account_fail_key(identifier)
    account_fail_count = int(await redis_pool.incr(account_fail_key))
    if account_fail_count == 1:
        await redis_pool.expire(
            account_fail_key,
            settings.login_failure_window_seconds,
        )

    if account_fail_count >= settings.login_lock_threshold:
        locked_until = _utc_iso_after(settings.login_lock_seconds)
        await redis_pool.set(
            _login_lock_key(identifier),
            locked_until,
            ex=settings.login_lock_seconds,
        )
        await redis_pool.delete(
            account_fail_key,
        )
        raise HTTPException(
            status_code=423,
            detail={
                'code': 'account_locked',
                'locked_until': locked_until,
            },
        )

    if pair_fail_count >= settings.login_cooldown_threshold:
        await redis_pool.set(
            _login_cooldown_key(identifier, client_ip),
            '1',
            ex=settings.login_cooldown_seconds,
        )
        raise HTTPException(
            status_code=429,
            detail={
                'code': 'login_cooldown',
                'retry_after_seconds': settings.login_cooldown_seconds,
            },
            headers={'Retry-After': str(settings.login_cooldown_seconds)},
        )

    remaining_attempts = max(
        settings.login_cooldown_threshold - pair_fail_count,
        0,
    )
    raise HTTPException(
        status_code=401,
        detail={
            'code': 'invalid_credentials',
            'remaining_attempts': remaining_attempts,
        },
    )


async def _clear_login_guard(
    redis_pool: Redis,
    identifier: str,
    client_ip: str | None,
) -> None:
    """Clear login failure and cooldown state after authentication.

    Args:
        redis_pool: Redis connection holding guard state.
        identifier: Successfully authenticated login identifier.
        client_ip: Requesting client address.
    """
    await redis_pool.delete(
        _login_fail_key(identifier, client_ip),
        _login_cooldown_key(identifier, client_ip),
        _account_fail_key(identifier),
        _login_lock_key(identifier),
    )


async def clear_login_guard_for_identifier(
    redis_pool: Redis,
    identifier: str,
) -> None:
    """Clear all tracked login guards for an account identifier.

    Args:
        redis_pool: Redis connection holding guard state.
        identifier: Login identifier whose guards are removed.
    """
    pair_index_key = _login_pair_index_key(identifier)
    pair_hashes = _decode_redis_members(
        await redis_pool.smembers(
            pair_index_key,
        ),
    )
    keys = [
        _account_fail_key(identifier),
        _login_lock_key(identifier),
        pair_index_key,
    ]
    for pair_hash in pair_hashes:
        keys.extend([
            _login_fail_pair_key(pair_hash),
            _login_cooldown_pair_key(pair_hash),
        ])
    await redis_pool.delete(*keys)


async def clear_login_guard_for_identifiers(
    redis_pool: Redis,
    identifiers: list[str],
) -> None:
    """Clear login guards for username and email aliases after a reset.

    Args:
        redis_pool: Redis connection holding guard state.
        identifiers: Username and email aliases to clear.
    """
    seen: set[str] = set()
    for identifier in identifiers:
        normalized = identifier.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        await clear_login_guard_for_identifier(redis_pool, normalized)


async def _load_feature_names(
    db: AsyncSession,
    group_id: int | None,
) -> list[str]:
    """Retrieve feature names associated with a specified group.

    Args:
        db (AsyncSession): Database session for queries.
        group_id (Optional[int]): ID of the group.

    Returns:
        list[str]: List of feature names linked to the group.
    """
    if group_id is None:
        return []

    rows = await db.execute(
        select(Feature.feature_name)
        .join(
            group_features_table,
            Feature.id == group_features_table.c.feature_id,
        )
        .where(group_features_table.c.group_id == group_id),
    )

    return [r.feature_name for r in rows]


async def _authenticate(
    db: AsyncSession,
    identifier: str,
    password: str,
) -> User:
    """Authenticate user credentials and verify active status.

    Args:
        db (AsyncSession): Database session for queries.
        identifier (str): User's username or profile e-mail address.
        password (str): User's password.

    Returns:
        User: Authenticated user object.

    Raises:
        HTTPException: When credentials are incorrect or user is inactive.
    """
    login_identifier = identifier.strip()
    user = await db.scalar(
        select(User).where(User.username == login_identifier),
    )

    if user is None:
        user = await db.scalar(
            select(User)
            .join(UserProfile, UserProfile.user_id == User.id)
            .where(func.lower(UserProfile.email) == login_identifier.lower()),
        )

    if not user or not await user.check_password(password):
        raise HTTPException(
            status_code=401, detail='Wrong username/e-mail or password',
        )

    if user.status == USER_STATUS_EMAIL_UNVERIFIED:
        raise HTTPException(
            status_code=403,
            detail={
                'code': 'email_unverified',
                'status': user.status,
                'message': 'Please verify your email before logging in.',
            },
        )

    if user.status == USER_STATUS_PENDING_ADMIN_APPROVAL:
        raise HTTPException(
            status_code=403,
            detail={
                'code': 'pending_admin_approval',
                'status': user.status,
                'message': 'Account is waiting for administrator approval.',
            },
        )

    if user.status == USER_STATUS_REJECTED:
        raise HTTPException(
            status_code=403,
            detail={
                'code': 'account_rejected',
                'status': user.status,
                'message': 'Account application was rejected.',
            },
        )

    if user.status == USER_STATUS_SUSPENDED:
        raise HTTPException(
            status_code=403,
            detail={
                'code': 'account_suspended',
                'status': user.status,
                'message': 'Account is suspended.',
            },
        )

    if user.status != USER_STATUS_ACTIVE:
        raise HTTPException(
            status_code=403,
            detail={'code': 'user_not_active', 'status': user.status},
        )

    return user


async def _verify_hcaptcha(
    hcaptcha_token: str | None,
    hcaptcha_bypass_key: str | None = None,
) -> None:
    """Verify an hCaptcha token before credential authentication.

    Args:
        hcaptcha_token: Client hCaptcha response token.
        hcaptcha_bypass_key: Trusted server-side bypass credential.

    Raises:
        HTTPException: If hCaptcha validation is required and fails.
    """
    if not settings.hcaptcha_enabled:
        return

    if (
        HCAPTCHA_BYPASS_KEY
        and hcaptcha_bypass_key
        and hcaptcha_bypass_key == HCAPTCHA_BYPASS_KEY
    ):
        return

    token = (hcaptcha_token or '').strip()
    if not token:
        raise HTTPException(status_code=400, detail='hCaptcha token required')

    if not HCAPTCHA_SECRET_KEY or not HCAPTCHA_SITE_KEY:
        raise HTTPException(status_code=500, detail='hCaptcha not configured')

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                HCAPTCHA_VERIFY_URL,
                data={
                    'secret': HCAPTCHA_SECRET_KEY,
                    'response': token,
                    'sitekey': HCAPTCHA_SITE_KEY,
                },
            )
            response.raise_for_status()
            result = response.json()
    except (httpx.HTTPError, ValueError):
        raise HTTPException(
            status_code=403, detail='hCaptcha verification failed',
        )

    if result.get('success') is not True:
        raise HTTPException(
            status_code=403, detail='hCaptcha verification failed',
        )


async def verify_refresh_token(
    refresh_token: str,
    redis_pool: Redis,
) -> RefreshTokenPayload:
    """Verify and decode a JWT refresh token.

    Args:
        refresh_token (str): Refresh token to validate.
        redis_pool (Redis): Redis connection pool for caching.

    Returns:
        RefreshTokenPayload: Decoded token payload.

    Raises:
        HTTPException: If token is invalid, expired, or not recognised.
    """
    try:
        # Decode and verify JWT refresh token and its purpose.
        payload = jwt_refresh.decode_token(refresh_token)
        subject = refresh_token_subject_from_payload(payload)
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401, detail='Refresh token has expired',
        )
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail='Invalid refresh token')

    username = subject['username']
    family_id = subject['family_id']
    if await redis_pool.get(
        _refresh_family_revoked_key(family_id),
    ):
        raise HTTPException(status_code=401, detail='Refresh token reused')

    # Retrieve user's data from Redis cache (and prune expired entries)
    await prune_user_cache(redis_pool, username)
    user_data = cast(
        UserCache | None,
        await get_user_data(redis_pool, username),
    )
    if (
        not user_data
        or not _cache_contains_refresh_token(user_data, refresh_token)
    ):
        if family_id:
            await _revoke_refresh_family(redis_pool, family_id)
            await _revoke_user_access_tokens(redis_pool, username)
            raise HTTPException(status_code=401, detail='Refresh token reused')
        raise HTTPException(
            status_code=401, detail='Refresh token not recognised',
        )

    return cast(RefreshTokenPayload, payload)


def _hash_refresh_token(refresh_token: str) -> str:
    """Hash a refresh token before storing web-cookie session state.

    Args:
        refresh_token: Raw refresh token.

    Returns:
        Stable non-reversible token hash.
    """
    return hashlib.sha256(refresh_token.encode('utf-8')).hexdigest()


def _refresh_state_key(refresh_token: str) -> str:
    """Build the Redis key tracking one refresh token's state.

    Args:
        refresh_token: Raw refresh token.

    Returns:
        Refresh-token state key.
    """
    return f'oauth:refresh-state:{_hash_refresh_token(refresh_token)}'


def _refresh_family_revoked_key(family_id: str) -> str:
    """Build the Redis key marking a refresh-token family revoked.

    Args:
        family_id: Refresh-token rotation-family identifier.

    Returns:
        Family-revocation marker key.
    """
    return f'oauth:refresh-family-revoked:{family_id}'


async def _register_refresh_token_state(
    redis_pool: Redis,
    refresh_token: str,
    username: str,
    family_id: str,
    *,
    enforce_family_active: bool = False,
) -> None:
    """Register an active refresh token without storing its bearer value.

    Args:
        redis_pool: Redis connection holding token state.
        refresh_token: Raw refresh token to register.
        username: Username that owns the token.
        family_id: Rotation-family identifier.
        enforce_family_active: Whether a revoked family must be rejected.

    Raises:
        HTTPException: If the refresh-token family is revoked.
    """
    if enforce_family_active and await redis_pool.get(
        _refresh_family_revoked_key(family_id),
    ):
        raise HTTPException(status_code=401, detail='Refresh token reused')
    await redis_pool.set(
        _refresh_state_key(refresh_token),
        json.dumps(
            {
                'status': 'active',
                'username': username,
                'family_id': family_id,
            },
            separators=(',', ':'),
        ).encode('utf-8'),
        ex=int(REFRESH_TTL.total_seconds()),
    )


async def _revoke_refresh_family(
    redis_pool: Redis,
    family_id: str,
) -> None:
    """Mark a refresh-token family revoked for its remaining lifetime.

    Args:
        redis_pool: Redis connection holding token state.
        family_id: Rotation-family identifier to revoke.
    """
    await redis_pool.set(
        _refresh_family_revoked_key(family_id),
        b'1',
        ex=int(REFRESH_TTL.total_seconds()),
    )


async def _revoke_user_access_tokens(
    redis_pool: Redis,
    username: str,
) -> int:
    """Immediately revoke every unexpired access token for a user.

    Args:
        redis_pool: Redis connection holding user token state.
        username: Username whose tokens are revoked.

    Returns:
        Number of access-token identifiers revoked.
    """
    cache = cast(UserCache | None, await get_user_data(redis_pool, username))
    if not cache:
        return 0
    return await revoke_access_token_jtis(redis_pool, cache['jti_meta'])


async def _consume_refresh_token_state(
    redis_pool: Redis,
    refresh_token: str,
    family_id: str,
    username: str,
) -> None:
    """Make a rotating refresh token single-use across all workers.

    Args:
        redis_pool: Redis connection holding token state.
        refresh_token: Raw refresh token to consume.
        family_id: Rotation-family identifier from token claims.
        username: Username from token claims.

    Raises:
        HTTPException: If the token is unknown, reused, or revoked.
    """
    if await redis_pool.get(_refresh_family_revoked_key(family_id)):
        await _revoke_user_access_tokens(redis_pool, username)
        raise HTTPException(status_code=401, detail='Refresh token reused')
    lock_key = f'{_refresh_state_key(refresh_token)}:consume'
    acquired = await redis_pool.set(lock_key, b'1', ex=30, nx=True)
    if not acquired:
        await _revoke_refresh_family(redis_pool, family_id)
        await _revoke_user_access_tokens(redis_pool, username)
        raise HTTPException(status_code=401, detail='Refresh token reused')
    raw = await redis_pool.get(_refresh_state_key(refresh_token))
    if raw is None:
        await _revoke_refresh_family(redis_pool, family_id)
        await _revoke_user_access_tokens(redis_pool, username)
        raise HTTPException(status_code=401, detail='Refresh token reused')
    state = json.loads(raw)
    if (
        state.get('status') != 'active'
        or state.get('family_id') != family_id
    ):
        await _revoke_refresh_family(redis_pool, family_id)
        await _revoke_user_access_tokens(redis_pool, username)
        raise HTTPException(status_code=401, detail='Refresh token reused')
    state['status'] = 'used'
    await redis_pool.set(
        _refresh_state_key(refresh_token),
        json.dumps(state, separators=(',', ':')).encode('utf-8'),
        ex=int(REFRESH_TTL.total_seconds()),
    )


def _cache_contains_refresh_token(
    cache: UserCache,
    refresh_token: str,
) -> bool:
    """Return whether a raw or hashed refresh token is recognised.

    Args:
        cache: Cached user token state.
        refresh_token: Raw refresh token to look up.

    Returns:
        ``True`` when the cache recognises the token.
    """
    if refresh_token in cache['refresh_tokens']:
        return True
    token_hash = _hash_refresh_token(refresh_token)
    return token_hash in cache['refresh_token_hashes']


def _remove_refresh_token_from_cache(
    cache: UserCache,
    refresh_token: str,
) -> None:
    """Remove a refresh token and its hash from a cache payload.

    Args:
        cache: Mutable cached user token state.
        refresh_token: Raw refresh token to remove.
    """
    token_hash = _hash_refresh_token(refresh_token)
    cache['refresh_tokens'] = [
        token
        for token in cache['refresh_tokens']
        if token != refresh_token
    ]
    cache['refresh_token_hashes'] = [
        value
        for value in cache['refresh_token_hashes']
        if value != token_hash
    ]


def _store_refresh_token_in_cache(
    cache: UserCache,
    refresh_token: str,
    *,
    hash_refresh_token: bool = False,
) -> None:
    """Store a refresh token raw for mobile or hashed for web.

    Args:
        cache: Mutable cached user token state.
        refresh_token: Raw refresh token to store.
        hash_refresh_token: Whether to retain only a token hash.
    """
    if hash_refresh_token:
        cache['refresh_token_hashes'].append(
            _hash_refresh_token(refresh_token),
        )
    else:
        cache['refresh_tokens'].append(refresh_token)


async def login_user(
    payload: UserLogin,
    db: AsyncSession,
    redis_pool: Redis,
    hcaptcha_bypass_key: str | None = None,
    client_ip: str | None = None,
    hash_refresh_token: bool = False,
) -> TokenPairData:
    """Authenticate user, issue JWT tokens, and store session in Redis cache.

    Args:
        payload (UserLogin): Login credentials (username/e-mail and password).
        db (AsyncSession): Database session.
        redis_pool (Redis): Redis connection pool for caching sessions.

    Returns:
        TokenPairData: Generated tokens and user-related details.
    """
    await _verify_hcaptcha(payload.hcaptcha_token, hcaptcha_bypass_key)

    await _check_login_guard(redis_pool, payload.identifier, client_ip)

    try:
        user = await _authenticate(
            db, payload.identifier, payload.password,
        )
    except HTTPException as exc:
        if exc.status_code == 401:
            await _record_failed_login(
                redis_pool,
                payload.identifier,
                client_ip,
            )
        raise

    await _clear_login_guard(redis_pool, payload.identifier, client_ip)

    return await issue_token_pair_for_user(
        user,
        db,
        redis_pool,
        hash_refresh_token=hash_refresh_token,
    )


async def issue_token_pair_for_user(
    user: User,
    db: AsyncSession,
    redis_pool: Redis,
    hash_refresh_token: bool = False,
) -> TokenPairData:
    """Issue local JWT tokens for an already verified user.

    Args:
        user: Authenticated user receiving the token pair.
        db: Database session used to resolve permissions.
        redis_pool: Redis connection used to register token state.
        hash_refresh_token: Whether to cache the refresh token as a hash.

    Returns:
        Token-pair data for the authentication response.
    """
    await prune_user_cache(redis_pool, user.username)
    cache = cast(
        UserCache | None,
        await get_user_data(redis_pool, user.username),
    )
    if cache is None:
        cache = UserCache(
            db_user=DbUserInfo(
                id=user.id,
                username=user.username,
                role=user.role,
                group_id=user.group_id,
                status=user.status,
            ),
            jti_list=[],
            jti_meta={},
            refresh_tokens=[],
            refresh_token_hashes=[],
            refresh_token_families={},
            feature_names=[],
        )

    # Load feature names for user's group
    feature_names = await _load_feature_names(db, user.group_id)
    cache['feature_names'] = feature_names

    # Generate JWT tokens
    new_jti = str(uuid4())
    access_token = jwt_access.create_access_token(
        subject={
            'username': user.username,
            'user_id': user.id,
            'role': user.role,
            'jti': new_jti,
            'features': feature_names,
        },
        expires_delta=ACCESS_TTL,
    )
    refresh_family_id = str(uuid4())
    refresh_token = jwt_refresh.create_access_token(
        subject={
            'username': user.username,
            'family_id': refresh_family_id,
            'token_id': str(uuid4()),
        },
        expires_delta=REFRESH_TTL,
    )

    # Update cache and store in Redis
    cache['jti_list'].append(new_jti)

    # store access token expiry timestamp for pruning (epoch seconds)
    at_payload = jwt_access.decode_token(access_token, verify_exp=False)
    cache['jti_meta'][new_jti] = int(at_payload['exp'])
    _store_refresh_token_in_cache(
        cache,
        refresh_token,
        hash_refresh_token=hash_refresh_token,
    )
    await set_user_data(
        redis_pool,
        user.username,
        cast(dict[str, object], cache),
    )
    await _register_refresh_token_state(
        redis_pool,
        refresh_token,
        user.username,
        refresh_family_id,
    )

    return {
        'access_token': access_token,
        'refresh_token': refresh_token,
        'username': user.username,
        'role': user.role,
        'user_id': user.id,
        'group_id': user.group_id,
        'feature_names': feature_names,
    }


async def logout_user(
    refresh_token: str | None,
    authorization: str | None,
    redis_pool: Redis,
) -> None:
    """Invalidate user's tokens on logout.

    Args:
        refresh_token (str): Refresh token to invalidate.
        authorization (Optional[str]): JWT access token from request headers.
        redis_pool (Redis): Redis connection pool.
    """
    username, jti, access_payload = _access_logout_context(authorization)
    refresh_context = _refresh_logout_context(refresh_token)
    if refresh_token and refresh_context is None and access_payload is None:
        return
    if refresh_context is not None:
        refresh_username, refresh_family_id = refresh_context
        if not username and refresh_username:
            username = refresh_username
    else:
        refresh_family_id = None

    if username is None:
        return

    # Revocation must happen before cache maintenance so a logout takes
    # effect immediately, even if the user cache was already evicted.
    if access_payload is not None:
        await revoke_access_token(redis_pool, access_payload)
    if refresh_family_id:
        await _revoke_refresh_family(redis_pool, refresh_family_id)

    # Remove the tokens from Redis cache
    await prune_user_cache(redis_pool, username)
    cache = cast(UserCache | None, await get_user_data(redis_pool, username))
    if not cache:
        return

    # Refresh-only logout requests do not identify a single access token;
    # revoke the user's current access capabilities rather than leaving them
    # valid until their natural expiry.
    if access_payload is None:
        await _revoke_user_access_tokens(redis_pool, username)

    _remove_logout_tokens_from_cache(cache, jti, refresh_token)
    await set_user_data(
        redis_pool,
        username,
        cast(dict[str, object], cache),
    )


def _access_logout_context(
    authorization: str | None,
) -> tuple[str | None, str | None, AccessTokenRevocationPayload | None]:
    """Decode an access token into logout identity and JTI context.

    Args:
        authorization: Optional HTTP authorisation header.

    Returns:
        Username, access-token identifier, and decoded payload when valid.
    """
    if not authorization:
        return None, None, None
    parts = authorization.split()
    if len(parts) != 2:
        return None, None, None
    try:
        payload = jwt_access.decode_token(parts[1], verify_exp=False)
        subject = access_token_subject_from_payload(payload)
    except jwt.PyJWTError:
        return None, None, None
    return subject['username'], subject['jti'], {
        'jti': subject['jti'],
        'exp': cast(int, payload['exp']),
    }


def _refresh_logout_context(
    refresh_token: str | None,
) -> tuple[str | None, str | None] | None:
    """Decode refresh-token identity and family data for logout.

    Args:
        refresh_token: Optional raw refresh token.

    Returns:
        Username and rotation family, or ``None`` when invalid.
    """
    if not refresh_token:
        return None
    try:
        payload = jwt_refresh.decode_token(refresh_token, verify_exp=False)
        subject = refresh_token_subject_from_payload(payload)
    except jwt.PyJWTError:
        return None
    return subject['username'], subject['family_id']


def _remove_logout_tokens_from_cache(
    cache: UserCache,
    jti: str | None,
    refresh_token: str | None,
) -> None:
    """Remove logout token references while preserving cache consistency.

    Args:
        cache: Mutable cached user token state.
        jti: Optional access-token identifier to remove.
        refresh_token: Optional refresh token to remove.
    """
    cache['jti_list'] = [
        token for token in cache['jti_list'] if token != jti
    ]
    if jti:
        cache['jti_meta'].pop(jti, None)
    if refresh_token:
        _remove_refresh_token_from_cache(cache, refresh_token)


async def refresh_tokens(
    payload: RefreshRequest,
    redis_pool: Redis,
    hash_refresh_token: bool = False,
) -> TokenPairData:
    """Issue new JWT tokens using a refresh token.

    Args:
        payload (RefreshRequest): Contains the refresh token.
        redis_pool (Redis): Redis connection pool.

    Returns:
        TokenPairData: New JWT access and refresh tokens.

    Raises:
        HTTPException: If refresh token is invalid or missing.
    """
    old_refresh = payload.refresh_token or ''
    if not old_refresh:
        raise HTTPException(status_code=401, detail='Missing refresh token')

    # Verify provided refresh token
    data = await verify_refresh_token(old_refresh, redis_pool)
    username = data['subject']['username']
    family_id = str(data['subject'].get('family_id') or '')

    await prune_user_cache(redis_pool, username)
    cache = cast(UserCache | None, await get_user_data(redis_pool, username))
    if not cache or not _cache_contains_refresh_token(cache, old_refresh):
        if family_id:
            await _revoke_refresh_family(redis_pool, family_id)
            await _revoke_user_access_tokens(redis_pool, username)
            raise HTTPException(status_code=401, detail='Refresh token reused')
        raise HTTPException(status_code=401, detail='Refresh token invalid')

    if family_id:
        await _consume_refresh_token_state(
            redis_pool,
            old_refresh,
            family_id,
            username,
        )

    _remove_refresh_token_from_cache(cache, old_refresh)

    # Generate new JWT tokens
    new_jti = str(uuid4())
    access_token = jwt_access.create_access_token(
        subject={
            'username': username,
            'user_id': cast(DbUserInfo, cache['db_user'])['id'],
            'role': cast(DbUserInfo, cache['db_user'])['role'],
            'jti': new_jti,
            'features': cache['feature_names'],
        },
        expires_delta=ACCESS_TTL,
    )
    new_family_id = family_id or str(uuid4())
    new_refresh = jwt_refresh.create_access_token(
        subject={
            'username': username,
            'family_id': new_family_id,
            'token_id': str(uuid4()),
        },
        expires_delta=REFRESH_TTL,
    )

    # Update and store new tokens in Redis cache
    cache['jti_list'].append(new_jti)
    # store access token expiry for pruning
    at_payload = jwt_access.decode_token(access_token, verify_exp=False)
    cache['jti_meta'][new_jti] = int(at_payload['exp'])
    _store_refresh_token_in_cache(
        cache,
        new_refresh,
        hash_refresh_token=hash_refresh_token,
    )
    await set_user_data(
        redis_pool,
        username,
        cast(dict[str, object], cache),
    )
    await _register_refresh_token_state(
        redis_pool,
        new_refresh,
        username,
        new_family_id,
        enforce_family_active=bool(family_id),
    )

    return {
        'access_token': access_token,
        'refresh_token': new_refresh,
        'feature_names': cache['feature_names'],
    }
