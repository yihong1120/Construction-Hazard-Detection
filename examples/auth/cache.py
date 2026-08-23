from __future__ import annotations

import json
from typing import cast

from fastapi import HTTPException
from fastapi import Request
from fastapi import Response
from fastapi import Security
from redis.asyncio import Redis
from redis.exceptions import NoScriptError

from examples.auth.jwt_config import jwt_access
from examples.auth.jwt_config import JwtAuthorizationCredentials


class RateLimiterService:
    """Rate limiting and user cache service."""

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
        """Initialise the service.

        Args:
            project_prefix: Project-specific prefix for Redis keys.
            limits: Mapping of role to ``(max_requests, window_seconds)``. If
                omitted, sensible defaults will be used.
        """
        self.project_prefix = project_prefix
        self.limits = limits or {
            'guest': (24, 86400),  # 24 requests / 24 hours
            'user': (3000, 60),  # 3000 requests / minute
        }
        # SHA of the rate-limit Lua script (loaded lazily and cached).
        self._rate_limit_sha: str | None = None

    def _user_key(self, username: str) -> str:
        """Compose the Redis key for user cache entries.

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
        """Compose the Redis key for rate limiting counters.

        Args:
            role: The user's role (e.g., ``'user'`` or ``'guest'``).
            username: The username to scope the counter.
            method: HTTP method.
            path: The request path component.

        Returns:
            The Redis key for the rate limiter counter.
        """
        return (
            f"{self.project_prefix}:rate_limit:{role}:{username}:"
            f"{method}:{path}"
        )

    async def get_user_data(
        self,
        redis_pool: Redis,
        username: str,
    ) -> dict[str, object] | None:
        """Retrieve cached user data by username.

        Args:
            redis_pool: Asynchronous Redis client/connection.
            username: Username used to compose the Redis key.

        Returns:
            The cached user dictionary, or ``None`` when no entry exists.
        """
        key: str = self._user_key(username)
        raw_data: bytes | None = await redis_pool.get(key)
        if raw_data is None:
            return None
        return json.loads(raw_data)

    async def set_user_data(
        self,
        redis_pool: Redis,
        username: str,
        data: dict[str, object],
    ) -> None:
        """Store user data in Redis by username.

        Args:
            redis_pool: Asynchronous Redis client/connection.
            username: Username used to compose the Redis key.
            data: JSON-serialisable user data to persist.
        """
        key: str = self._user_key(username)
        # Compact JSON to avoid unnecessary whitespace.
        await redis_pool.set(key, json.dumps(data).encode('utf-8'))

    async def _ensure_rate_limit_script(self, redis_pool: Redis) -> str:
        """Ensure the rate-limit Lua script is loaded and return its SHA.

        Args:
            redis_pool: Asynchronous Redis client/connection.

        Returns:
            The SHA identifier of the loaded Lua script.
        """
        if self._rate_limit_sha:
            return self._rate_limit_sha
        self._rate_limit_sha = cast(
            str,
            await redis_pool.script_load(self._RATE_LIMIT_LUA),
        )
        return self._rate_limit_sha

    async def _incr_and_get_ttl(
        self,
        redis_pool: Redis,
        key: str,
        window_seconds: int,
    ) -> tuple[int, int]:
        """Increment the counter and obtain the TTL in an efficient manner.

        Args:
            redis_pool: Asynchronous Redis client/connection.
            key: Redis key to increment.
            window_seconds: The fixed window length in seconds.

        Returns:
            A tuple of ``(current_requests, ttl_seconds)``.
        """
        # Single RTT using EVALSHA.
        try:
            sha = await self._ensure_rate_limit_script(redis_pool)
            current, ttl = cast(
                list[int],
                await redis_pool.evalsha(
                    sha,
                    1,
                    key,
                    window_seconds,
                ),
            )
            return current, ttl
        except NoScriptError:
            # Script was flushed; load and retry once
            self._rate_limit_sha = cast(
                str,
                await redis_pool.script_load(self._RATE_LIMIT_LUA),
            )
            current, ttl = cast(
                list[int],
                await redis_pool.evalsha(
                    self._rate_limit_sha,
                    1,
                    key,
                    window_seconds,
                ),
            )
            return current, ttl

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
        jwt_creds: JwtAuthorizationCredentials = Security(jwt_access),
    ) -> int:
        """Enforce per-role rate limiting.

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
        subject = jwt_creds.subject
        username = subject['username']
        token_jti = subject['jti']

        # Obtain Redis connection
        redis_pool: Redis = request.app.state.redis_client.client

        # Load user data and verify JTI list membership.
        user_data: dict[str, object] | None = await self.get_user_data(
            redis_pool,
            username,
        )
        if not user_data:
            raise HTTPException(
                status_code=401,
                detail='No such user in Redis',
            )

        jti_list = cast(list[str], user_data['jti_list'])

        if token_jti not in jti_list:
            raise HTTPException(
                status_code=401,
                detail='Token jti is invalid or replaced',
            )

        # Determine role and quotas
        role = subject['role']
        max_requests, window_seconds = self.limits.get(
            role,
            self.limits['user'],
        )

        # Compose rate-limit key
        key: str = self._rate_key(
            role,
            username,
            request.method,
            request.url.path,
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
        reset_seconds = ttl if ttl >= 0 else window_seconds
        response.headers['X-RateLimit-Reset'] = str(int(reset_seconds))
        return remaining


rate_limiter_service = RateLimiterService()

PROJECT_PREFIX: str = rate_limiter_service.project_prefix

# Centralised role quotas: ``(max_requests, window_seconds)``
LIMITS: dict[str, tuple[int, int]] = rate_limiter_service.limits
