from __future__ import annotations

import datetime
import hashlib
from typing import Any

from fastapi import HTTPException
from redis.asyncio import Redis


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
    return hashlib.sha256(f"{normalized}|{source}".encode()).hexdigest()


def _login_fail_key(identifier: str, client_ip: str | None) -> str:
    """Build the Redis key for failures from one login pair.

    Args:
        identifier: Login identifier.
        client_ip: Requesting client address.

    Returns:
        Login-pair failure-counter key.
    """
    return f"login_fail:pair:{_hash_login_pair(identifier, client_ip)}"


def _login_cooldown_key(identifier: str, client_ip: str | None) -> str:
    """Build the Redis key for a login-pair cooldown.

    Args:
        identifier: Login identifier.
        client_ip: Requesting client address.

    Returns:
        Login-pair cooldown key.
    """
    return f"login_cooldown:pair:{_hash_login_pair(identifier, client_ip)}"


def _account_fail_key(identifier: str) -> str:
    """Build the Redis key for account-wide login failures.

    Args:
        identifier: Login identifier.

    Returns:
        Account failure-counter key.
    """
    return f"login_fail:account:{_hash_account_identifier(identifier)}"


def _login_lock_key(identifier: str) -> str:
    """Build the Redis key for an account login lock.

    Args:
        identifier: Login identifier.

    Returns:
        Account-lock key.
    """
    return f"login_lock:account:{_hash_account_identifier(identifier)}"


def _login_pair_index_key(identifier: str) -> str:
    """Build the Redis key indexing client-pair login records.

    Args:
        identifier: Login identifier.

    Returns:
        Client-pair index key.
    """
    return f"login_pairs:account:{_hash_account_identifier(identifier)}"


def _login_fail_pair_key(pair_hash: str) -> str:
    """Build the failure-counter key for a client-pair hash.

    Args:
        pair_hash: Hashed login identifier and client IP.

    Returns:
        Login-pair failure-counter key.
    """
    return f"login_fail:pair:{pair_hash}"


def _login_cooldown_pair_key(pair_hash: str) -> str:
    """Build the cooldown key for a client-pair hash.

    Args:
        pair_hash: Hashed login identifier and client IP.

    Returns:
        Login-pair cooldown key.
    """
    return f"login_cooldown:pair:{pair_hash}"


def _utc_iso_after(seconds: int) -> str:
    """Return a UTC ISO timestamp offset by the requested seconds.

    Args:
        seconds: Time offset from the current UTC time.

    Returns:
        Timezone-aware ISO 8601 timestamp.
    """
    expires_at = datetime.datetime.now(
        datetime.timezone.utc,
    ) + datetime.timedelta(seconds=seconds)
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


async def check_login_guard(
    redis_pool: Redis,
    identifier: str,
    client_ip: str | None,
    *,
    policy: Any,
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
            policy.login_cooldown_seconds,
        )
        raise HTTPException(
            status_code=429,
            detail={
                'code': 'login_cooldown',
                'retry_after_seconds': retry_after,
            },
            headers={'Retry-After': str(retry_after)},
        )


async def record_failed_login(
    redis_pool: Redis,
    identifier: str,
    client_ip: str | None,
    *,
    policy: Any,
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
        policy.login_failure_window_seconds,
        policy.login_cooldown_seconds,
        policy.login_lock_seconds,
    )
    await redis_pool.sadd(pair_index_key, pair_hash)
    await redis_pool.expire(pair_index_key, index_ttl)

    fail_key = _login_fail_key(identifier, client_ip)
    pair_fail_count = int(await redis_pool.incr(fail_key))
    if pair_fail_count == 1:
        await redis_pool.expire(
            fail_key,
            policy.login_failure_window_seconds,
        )

    account_fail_key = _account_fail_key(identifier)
    account_fail_count = int(await redis_pool.incr(account_fail_key))
    if account_fail_count == 1:
        await redis_pool.expire(
            account_fail_key,
            policy.login_failure_window_seconds,
        )

    if account_fail_count >= policy.login_lock_threshold:
        locked_until = _utc_iso_after(policy.login_lock_seconds)
        await redis_pool.set(
            _login_lock_key(identifier),
            locked_until,
            ex=policy.login_lock_seconds,
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

    if pair_fail_count >= policy.login_cooldown_threshold:
        await redis_pool.set(
            _login_cooldown_key(identifier, client_ip),
            '1',
            ex=policy.login_cooldown_seconds,
        )
        raise HTTPException(
            status_code=429,
            detail={
                'code': 'login_cooldown',
                'retry_after_seconds': policy.login_cooldown_seconds,
            },
            headers={'Retry-After': str(policy.login_cooldown_seconds)},
        )

    remaining_attempts = max(
        policy.login_cooldown_threshold - pair_fail_count,
        0,
    )
    raise HTTPException(
        status_code=401,
        detail={
            'code': 'invalid_credentials',
            'remaining_attempts': remaining_attempts,
        },
    )


async def clear_login_guard(
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
        keys.extend(
            [
                _login_fail_pair_key(pair_hash),
                _login_cooldown_pair_key(pair_hash),
            ],
        )
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
