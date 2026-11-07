from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from fastapi import HTTPException
from redis.asyncio import Redis
from redis.exceptions import RedisError

from examples.auth.deployment_context import DeploymentBinding
from examples.auth.session_store import acquire_refresh_lock
from examples.auth.session_store import auth_session_key
from examples.auth.session_store import auth_tokens
from examples.auth.session_store import AuthRefreshConflict
from examples.auth.session_store import AuthTokenUpdate
from examples.auth.session_store import delete_auth_session
from examples.auth.session_store import get_auth_session
from examples.auth.session_store import OIDC_ONLINE_SSO_SESSION_MODE
from examples.auth.session_store import release_refresh_lock
from examples.auth.session_store import save_auth_tokens
from examples.bff.oidc_services import refresh_oidc_tokens
from examples.bff.proxy_config import logger
from examples.bff.proxy_config import REFRESH_BEFORE_SECONDS
from src.async_tasks import finish_on_cancel


def _is_terminal_refresh_error(exc: HTTPException) -> bool:
    """Return whether a refresh error permanently invalidates a BFF session.

    Args:
        exc: Refresh-token HTTP exception.

    Returns:
        ``True`` for a permanent authentication failure.
    """
    if exc.status_code == 409:
        return True
    if exc.status_code != 401:
        return False
    detail = str(exc.detail).lower()
    if detail.startswith('oidc_'):
        return True
    return any(
        marker in detail
        for marker in ('expired', 'revoked', 'reused', 'invalid', 'recognised')
    )


async def get_proxy_access_token(
    redis: Redis,
    session_id: str,
    *,
    force_refresh: bool = False,
    deployment: DeploymentBinding | None = None,
) -> tuple[str, dict[str, Any]]:
    """Return a valid BFF access token, refreshing it when necessary.

    Args:
        redis: Redis connection holding BFF session state.
        session_id: Opaque browser session identifier.
        force_refresh: Whether to rotate despite a still-valid access token.

    Returns:
        Access token and current server-side session record.

    Raises:
        HTTPException: If the session is absent, refresh is rejected, or a peer
            refresh does not complete in time.
    """
    session = await get_auth_session(redis, session_id)
    if session is None:
        raise HTTPException(status_code=401, detail='app_session_expired')
    if (
        session.get('auth_provider') == 'oidc'
        and session.get('oidc_session_mode') != OIDC_ONLINE_SSO_SESSION_MODE
    ):
        # Sessions minted before the BFF moved away from offline tokens cannot
        # participate in browser SSO. Clear only that opaque BFF session so
        # the next application navigation performs one normal Keycloak login.
        await delete_auth_session(redis, session_id)
        raise HTTPException(status_code=401, detail='app_session_expired')

    access_token, _ = auth_tokens(session)
    remaining = int(session.get('access_expires_at') or 0) - int(time.time())
    if not force_refresh and remaining > REFRESH_BEFORE_SECONDS:
        return access_token, session

    lock_owner = await acquire_refresh_lock(redis, session_id)
    if lock_owner is None:
        return await _wait_for_proxy_session_refresh(
            redis,
            session_id,
            RefreshObservation(
                int(session.get('revision') or 0),
                access_token,
                force_refresh,
            ),
        )

    return await ProxySessionRefresh(redis, session_id, lock_owner).run(
        access_token,
    )


@dataclass(frozen=True)
class RefreshObservation:
    revision: int
    access_token: str = field(repr=False)
    forced: bool = False


async def _wait_for_proxy_session_refresh(
    redis: Redis,
    session_id: str,
    observed: RefreshObservation,
) -> tuple[str, dict[str, Any]]:
    """Reuse the latest peer token only while its absolute expiry is valid."""
    for _ in range(40):
        await asyncio.sleep(0.05)
        latest = await get_auth_session(redis, session_id)
        if latest is None:
            raise HTTPException(status_code=401, detail='app_session_expired')
        latest_access, _ = auth_tokens(latest)
        if (
            latest_access != observed.access_token
            and (
                not observed.forced
                or int(latest.get('revision') or 0) > observed.revision
            )
            and int(latest.get('access_expires_at') or 0) > time.time()
        ):
            return latest_access, latest
    assert latest is not None
    if (
        not observed.forced
        and int(latest.get('access_expires_at') or 0) > time.time()
    ):
        return latest_access, latest
    raise HTTPException(status_code=503, detail='session_refresh_busy')


class ProxySessionRefresh:
    """Own token rotation, renewal and conditional persistence."""

    lease_seconds = 15

    def __init__(self, redis: Redis, session_id: str, owner: str):
        self.redis = redis
        self.session_id = session_id
        self.owner = owner
        self.snapshot: dict[str, Any] = {}

    async def run(self, observed_access: str):
        try:
            return await self._run(observed_access)
        finally:
            await finish_on_cancel(self._release())

    async def _release(self):
        try:
            async with asyncio.timeout(5):
                await release_refresh_lock(
                    self.redis, self.session_id, self.owner,
                )
        except (RedisError, TimeoutError):
            logger.warning('BFF refresh lease release deferred')

    async def _persist(self, update):
        try:
            async with asyncio.timeout(5):
                return await save_auth_tokens(self.redis, update)
        except TimeoutError as error:
            raise HTTPException(503, 'session_refresh_busy') from error

    async def _run(self, observed_access: str):
        latest = await get_auth_session(self.redis, self.session_id)
        if latest is None:
            raise HTTPException(401, 'app_session_expired')
        self.snapshot = latest
        access, refresh = auth_tokens(latest)
        if (
            access != observed_access
            and int(latest.get('access_expires_at') or 0) > time.time()
        ):
            return access, latest
        heartbeat = asyncio.create_task(self._renew())
        try:
            # A disconnected browser must not discard an accepted rotation.
            return await finish_on_cancel(self._exchange_and_store(refresh))
        finally:
            heartbeat.cancel()
            await finish_on_cancel(
                asyncio.gather(heartbeat, return_exceptions=True),
            )

    async def _renew(self):
        script = """
if redis.call('GET',KEYS[1]) ~= ARGV[1] then return 0 end
return redis.call('EXPIRE',KEYS[1],ARGV[2])
"""
        try:
            while True:
                await asyncio.sleep(self.lease_seconds / 3)
                async with asyncio.timeout(self.lease_seconds / 3):
                    if not await self.redis.eval(
                        script,
                        1,
                        auth_session_key(self.session_id) + ':refresh-lock',
                        self.owner,
                        self.lease_seconds,
                    ):
                        return
        except RedisError:
            return
        except TimeoutError:
            return

    async def _latest_valid(self):
        latest = await get_auth_session(self.redis, self.session_id)
        if latest is None:
            raise HTTPException(401, 'app_session_expired')
        access = auth_tokens(latest)[0]
        if (
            int(latest.get('access_expires_at') or 0) <= time.time()
            or int(latest.get('revision') or 0)
            <= int(self.snapshot.get('revision') or 0)
            or access == auth_tokens(self.snapshot)[0]
        ):
            raise AuthRefreshConflict()
        return access, latest

    async def _exchange_and_store(self, refresh_token):
        try:
            result = await refresh_oidc_tokens(refresh_token)
        except HTTPException as error:
            if not _is_terminal_refresh_error(error):
                raise
            try:
                await self._persist(
                    AuthTokenUpdate(
                        self.session_id,
                        self.owner,
                        self.snapshot,
                        revoke=True,
                    ),
                )
            except AuthRefreshConflict:
                return await self._latest_valid()
            await finish_on_cancel(
                delete_auth_session(self.redis, self.session_id),
            )
            raise HTTPException(401, 'app_session_expired') from error
        access, refresh = str(result['access_token']), str(
            result['refresh_token'],
        )
        try:
            latest = await self._persist(
                AuthTokenUpdate(
                    self.session_id,
                    self.owner,
                    self.snapshot,
                    access,
                    refresh,
                ),
            )
        except AuthRefreshConflict:
            return await self._latest_valid()
        return access, latest


async def _get_proxy_access_token_or_503(
    redis: Redis,
    session_id: str,
    request_path: str,
    *,
    force_refresh: bool = False,
    deployment: DeploymentBinding | None = None,
) -> tuple[str, dict[str, Any]]:
    """Return a BFF access token or translate Redis failure to HTTP 503.

    Args:
        redis: Redis connection holding BFF session state.
        session_id: Opaque browser session identifier.
        request_path: Original BFF request path for observability.
        force_refresh: Whether to rotate despite a valid access token.

    Returns:
        Valid access token and current server-side session record.

    Raises:
        HTTPException: If Redis is unavailable or token refresh fails.
    """
    try:
        return await get_proxy_access_token(
            redis,
            session_id,
            force_refresh=force_refresh,
            deployment=deployment,
        )
    except RedisError as exc:
        logger.warning(
            'BFF Redis connection failed path=%s error_type=%s',
            request_path,
            type(exc).__name__,
            exc_info=True,
        )
        raise HTTPException(
            status_code=503,
            detail='bff_redis_unavailable',
        ) from exc
