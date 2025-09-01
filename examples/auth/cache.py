from __future__ import annotations

import json
from collections.abc import Sequence
from typing import cast

from fastapi import Depends
from fastapi import HTTPException
from fastapi import Request
from fastapi import Response
from fastapi_jwt import JwtAuthorizationCredentials
from redis.asyncio import Redis
from redis.exceptions import NoScriptError

from examples.auth.jwt_config import jwt_access


class RateLimiterService:
    """
    Rate limiting and user cache service.
    """

    # Lua script used to atomically perform INCR and TTL in one round-trip,
    # applying an expiry only when the key is first created or has no TTL.
    _RATE_LIMIT_LUA: str = """
-- KEYS[1] = key, ARGV[1] = window_seconds
local current = redis.call('INCR', KEYS[1])
local ttl = redis.call('TTL', KEYS[1])
-- Apply expiry on first increment or when there is no TTL,
-- to keep a fixed window
if ttl == -1 or ttl == -2 then
    redis.call('EXPIRE', KEYS[1], tonumber(ARGV[1]))
    ttl = tonumber(ARGV[1])
end
return { current, ttl }
"""

    def __init__(
        self,
        project_prefix: str = 'construction-hazard-detection',
        limits: dict[str, tuple[int, int]] | None = None,
    ) -> None:
        """
        Initialise the service.

        Args:
            project_prefix: Project-specific prefix for Redis keys.
            limits: Mapping of role to ``(max_requests, window_seconds)``. If
                omitted, sensible defaults will be used.
        """
        self.project_prefix = project_prefix
        self.limits = limits or {
            'guest': (24, 86400),  # 24 requests / 24 hours
            'user': (3000, 60),    # 3000 requests / minute
        }
        # SHA of the rate-limit Lua script (loaded lazily and cached).
        self._rate_limit_sha: str | None = None

    def _user_key(self, username: str) -> str:
        """
        Compose the Redis key for user cache entries.

        Args:
            username: The username used to compose the key.

        Returns:
            The fully qualified Redis key for the cached user data.
        """
        return f"{self.project_prefix}:user_cache:{username}"

    def _rate_key(
        self,
        role: str,
        username: str,
        method: str,
        path: str,
    ) -> str:
        """
        Compose the Redis key for rate limiting counters.

        Args:
            role: The user's role (e.g., ``'user'`` or ``'guest'``).
            username: The username to scope the counter.
            method: HTTP method. Currently unused in the key for compatibility.
            path: The request path component.

        Returns:
            The Redis key for the rate limiter counter.
        """
        # Backwards-compatible shape: no prefix and no HTTP method included.
        return f"rate_limit:{role}:{username}:{path}"

    async def get_user_data(
        self,
        redis_pool: Redis,
        username: str,
    ) -> dict[str, object] | None:
        """
        Retrieve cached user data by username.

        Args:
            redis_pool: Asynchronous Redis client/connection.
            username: Username used to compose the Redis key.

        Returns:
            If the entry exists and contains valid JSON, a dictionary is
            returned; otherwise ``None`` is returned.
        """
        key: str = self._user_key(username)
        raw_data: bytes | None = await redis_pool.get(key)
        if raw_data is None:
            return None
        try:
            # json.loads accepts bytes (UTF-8 by default)
            return json.loads(raw_data)
        except Exception:
            # Treat corrupted/non-JSON payloads as missing user data
            return None

    async def set_user_data(
        self,
        redis_pool: Redis,
        username: str,
        data: dict[str, object],
    ) -> None:
        """
        Store user data in Redis by username.

        Args:
            redis_pool: Asynchronous Redis client/connection.
            username: Username used to compose the Redis key.
            data: JSON-serialisable user data to persist.
        """
        key: str = self._user_key(username)
        # Compact JSON to avoid unnecessary whitespace.
        await redis_pool.set(key, json.dumps(data))

    async def _ensure_rate_limit_script(self, redis_pool: Redis) -> str:
        """
        Ensure the rate-limit Lua script is loaded and return its SHA.

        Args:
            redis_pool: Asynchronous Redis client/connection.

        Returns:
            The SHA identifier of the loaded Lua script.
        """
        if self._rate_limit_sha:
            return self._rate_limit_sha
        sha = await redis_pool.script_load(self._RATE_LIMIT_LUA)
        self._rate_limit_sha = (
            sha if isinstance(sha, str) else cast(bytes, sha).decode()
        )
        return self._rate_limit_sha

    async def _incr_and_get_ttl(
        self,
        redis_pool: Redis,
        key: str,
        window_seconds: int,
    ) -> tuple[int, int]:
        """
        Increment the counter and obtain the TTL in an efficient manner.

        Args:
            redis_pool: Asynchronous Redis client/connection.
            key: Redis key to increment.
            window_seconds: The fixed window length in seconds.

        Returns:
            A tuple of ``(current_requests, ttl_seconds)``.
        """
        # Fast path: single RTT using EVALSHA; compatible fallbacks below.
        try:
            sha = await self._ensure_rate_limit_script(redis_pool)
            res: Sequence[object] = await redis_pool.evalsha(
                sha,
                1,
                key,
                window_seconds,
            )
            # Validate response shape
            if not isinstance(res, (list, tuple)) or len(res) < 2:
                raise ValueError('Invalid evalsha response shape')
            current = int(res[0])
            ttl = int(res[1])
            return current, ttl
        except NoScriptError:
            # Script was flushed; load and retry once
            sha2 = await redis_pool.script_load(self._RATE_LIMIT_LUA)
            self._rate_limit_sha = (
                sha2 if isinstance(sha2, str) else cast(bytes, sha2).decode()
            )
            res = await redis_pool.evalsha(
                self._rate_limit_sha,
                1,
                key,
                window_seconds,
            )
            if not isinstance(res, (list, tuple)) or len(res) < 2:
                raise ValueError('Invalid evalsha response shape')
            current = int(res[0])
            ttl = int(res[1])
            return current, ttl
        except Exception:
            # Fallback 1: direct INCR + TTL (apply EXPIRE if missing)
            try:
                current_requests = int(await redis_pool.incr(key))
                ttl = int(await redis_pool.ttl(key))
                if ttl == -1:
                    await redis_pool.expire(key, window_seconds)
                    ttl = window_seconds
                return current_requests, ttl
            except Exception:
                # Fallback 2: non-transactional pipeline for single round-trip
                async with redis_pool.pipeline(transaction=False) as pipe:
                    pipe.incr(key)
                    pipe.ttl(key)
                    results = await pipe.execute()
                current_requests = int(results[0])
                ttl = int(results[1])
                if ttl == -1:
                    await redis_pool.expire(key, window_seconds)
                    ttl = window_seconds
                return current_requests, ttl

    async def preload_script(self, redis_pool: Redis) -> None:
        """Optionally pre-load the Lua script at app start.

        This avoids a small latency hit on the first request that requires the
        rate limit script.

        Args:
            redis_pool: Asynchronous Redis client/connection.
        """
        await self._ensure_rate_limit_script(redis_pool)

    async def __call__(
        self,
        request: Request,
        response: Response,
        credentials: JwtAuthorizationCredentials = Depends(jwt_access),
    ) -> int:
        """
        Enforce per-role rate limiting.

        Args:
            request: The incoming FastAPI request.
            response: The outgoing FastAPI response where headers will be set.
            credentials: JWT credential object produced by ``jwt_access``.

        Returns:
            Remaining requests in the current window after this request.

        Raises:
            HTTPException: If credentials are invalid, if the user is not
                found in Redis, if the token JTI has been rotated/revoked, or
                if the rate limit is exceeded.
        """
        payload: dict[str, object] = cast(
            dict[str, object], credentials.subject,
        )
        username_obj: object = payload.get('username')
        token_jti_obj: object = payload.get('jti')

        # Validate presence and type of required token fields
        if not all(
            isinstance(v, str) and v
            for v in (username_obj, token_jti_obj)
        ):
            raise HTTPException(
                status_code=401, detail='Token is missing or invalid fields',
            )
        username: str = cast(str, username_obj)
        token_jti: str = cast(str, token_jti_obj)

        # Obtain Redis connection
        redis_pool: Redis = request.app.state.redis_client.client

        # Load user data and verify JTI list membership
        # Wrapped via a module-level function to ease patching in tests
        user_data: dict[str, object] | None = await get_user_data(
            redis_pool,
            username,
        )
        if not user_data:
            raise HTTPException(
                status_code=401, detail='No such user in Redis',
            )

        jti_src = user_data.get('jti_list', [])
        if isinstance(jti_src, (list, tuple)):
            jti_list: list[str] = [
                s for s in jti_src if isinstance(s, str) and s
            ]
        else:
            jti_list = []

        if token_jti not in jti_list:
            raise HTTPException(
                status_code=401, detail='Token jti is invalid or replaced',
            )

        # Determine role and quotas
        role_any: object = payload.get('role', 'user')
        role: str = (
            role_any if isinstance(role_any, str) and role_any else 'user'
        )
        max_requests, window_seconds = self.limits.get(
            role, self.limits['user'],
        )

        # Compose rate-limit key (keeps legacy shape for compatibility)
        key: str = self._rate_key(
            role, username, request.method, request.url.path,
        )

        # Single RTT via Lua script to get current count and TTL
        current_requests, ttl = await self._incr_and_get_ttl(
            redis_pool,
            key,
            window_seconds,
        )

        if current_requests > max_requests:
            raise HTTPException(status_code=429, detail='Rate limit exceeded')

        remaining = max_requests - current_requests
        remaining = remaining if remaining >= 0 else 0

        # Expose rate-limit metadata in response headers
        response.headers['X-RateLimit-Remaining'] = str(remaining)
        response.headers['X-RateLimit-Limit'] = str(max_requests)
        # Use TTL as reset countdown; if negative/unknown, fall back to window
        reset_seconds = ttl if isinstance(
            ttl, int,
        ) and ttl >= 0 else window_seconds
        response.headers['X-RateLimit-Reset'] = str(int(reset_seconds))
        return remaining


_DEFAULT_SERVICE = RateLimiterService()


PROJECT_PREFIX: str = _DEFAULT_SERVICE.project_prefix

# Centralised role quotas: ``(max_requests, window_seconds)``
LIMITS: dict[str, tuple[int, int]] = _DEFAULT_SERVICE.limits


async def get_user_data(
    redis_pool: Redis,
    username: str,
) -> dict[str, object] | None:
    """
    Backwards-compatible wrapper for ``RateLimiterService.get_user_data``.

    Args:
        redis_pool: Asynchronous Redis client/connection.
        username: Username used to compose the Redis key.

    Returns:
        The cached user dictionary if present and valid, otherwise ``None``.
    """
    return await _DEFAULT_SERVICE.get_user_data(redis_pool, username)


async def set_user_data(
    redis_pool: Redis,
    username: str,
    data: dict[str, object],
) -> None:
    """
    Backwards-compatible wrapper for ``RateLimiterService.set_user_data``.

    Args:
        redis_pool: Asynchronous Redis client/connection.
        username: Username used to compose the Redis key.
        data: JSON-serialisable user data to persist.
    """
    await _DEFAULT_SERVICE.set_user_data(redis_pool, username, data)


async def custom_rate_limiter(
    request: Request,
    response: Response | JwtAuthorizationCredentials,
    credentials: JwtAuthorizationCredentials = Depends(jwt_access),
) -> int:
    """
    Backwards-compatible wrapper for ``RateLimiterService.__call__``.

    Args:
        request: The incoming FastAPI request.
        response: The outgoing FastAPI response (or a credentials object in
            the test-friendly short form).
        credentials: JWT credential object produced by ``jwt_access``.

    Returns:
        Remaining requests in the current window after this request.
    """
    # If the second argument is not a Response, treat it as credentials (test
    # shorthand) and create a temporary Response instance.
    if not isinstance(response, Response):
        credentials = response
        response = Response()

    return await _DEFAULT_SERVICE(
        request, response, credentials=credentials,
    )
