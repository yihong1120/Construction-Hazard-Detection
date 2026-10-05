from __future__ import annotations

import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import HTTPException
from fastapi import Request
from fastapi import Response
from redis.asyncio import Redis
from redis.exceptions import NoScriptError

from examples.auth.cache import rate_limiter_service
from examples.auth.cache import RateLimiterService
from examples.auth.jwt_config import JwtAuthorizationCredentials


class CacheTestCase(unittest.IsolatedAsyncioTestCase):
    """Test cases for cache functionalities (get_user_data, set_user_data) and
    the custom_rate_limiter behavior."""

    async def test_oidc_enforces_quota(self) -> None:
        """Verified OIDC users need no legacy login and remain rate limited."""
        request = MagicMock(spec=Request)
        request.method = 'POST'
        request.url.path = '/detect'
        request.app.state.redis_client.client = AsyncMock(spec=Redis)
        credentials = JwtAuthorizationCredentials(
            subject={
                'username': 'alice',
                'user_id': 1,
                'role': 'guest',
                'jti': 'oidc:issuer:external-token',
                'features': [],
            },
            is_oidc=True,
        )
        with (
            patch.object(
                rate_limiter_service, '_incr_and_get_ttl',
                AsyncMock(side_effect=[(1, 60), (25, 60)]),
            ),
        ):
            response = Response()
            remaining = await rate_limiter_service(
                request, response, credentials,
            )
            self.assertEqual(remaining, 23)
            self.assertEqual(response.headers['X-RateLimit-Limit'], '24')
            with self.assertRaises(HTTPException) as raised:
                await rate_limiter_service(request, Response(), credentials)
            self.assertEqual(raised.exception.status_code, 429)

    async def test_preload_and_cached_script(self) -> None:
        """Ensure preload_script loads Lua once and cached path is used
        after."""
        service = RateLimiterService()
        redis_pool = AsyncMock(spec=Redis)
        redis_pool.script_load = AsyncMock(return_value='sha-1')

        # First call should load script
        await service.preload_script(redis_pool)
        # Second ensure call should use cached SHA, not load again
        sha = await service._ensure_rate_limit_script(redis_pool)
        self.assertEqual(sha, 'sha-1')
        redis_pool.script_load.assert_awaited_once()

    async def test_incr_get_ttl_evalsha_success(self) -> None:
        """Cover the fast path where evalsha succeeds."""
        service = RateLimiterService()
        redis_pool = AsyncMock(spec=Redis)
        redis_pool.script_load = AsyncMock(return_value='sha-xyz')
        redis_pool.evalsha = AsyncMock(return_value=[5, 100])

        current, ttl = await service._incr_and_get_ttl(redis_pool, 'k', 60)
        self.assertEqual((current, ttl), (5, 100))
        redis_pool.script_load.assert_awaited_once()
        redis_pool.evalsha.assert_awaited_once()

    async def test_incr_get_ttl_rejects_invalid_initial_response_shape(
        self,
    ) -> None:
        """Malformed Lua replies must not be mistaken for rate-limit data."""
        service = RateLimiterService()
        redis_pool = AsyncMock(spec=Redis)
        redis_pool.script_load = AsyncMock(return_value='sha-invalid')
        redis_pool.evalsha = AsyncMock(return_value=[1])

        with self.assertRaises(ValueError):
            await service._incr_and_get_ttl(redis_pool, 'invalid', 60)

    async def test_incr_get_ttl_noscript_reload(self) -> None:
        """Cover the NoScriptError branch that reloads script and retries."""
        service = RateLimiterService()
        redis_pool = AsyncMock(spec=Redis)
        redis_pool.script_load = AsyncMock(side_effect=['sha-a', 'sha-b'])
        redis_pool.evalsha = AsyncMock(side_effect=[NoScriptError(), [7, 50]])

        current, ttl = await service._incr_and_get_ttl(redis_pool, 'k2', 30)
        self.assertEqual((current, ttl), (7, 50))
        # script_load called twice: initial ensure + reload after FLUSH
        self.assertEqual(redis_pool.script_load.await_count, 2)
        self.assertEqual(redis_pool.evalsha.await_count, 2)

    async def test_incr_get_ttl_noscript_then_invalid_shape_raises(
        self,
    ) -> None:
        """After NoScriptError, the second evalsha returns an invalid shape
        which should raise a ValueError."""
        service = RateLimiterService()
        redis_pool = AsyncMock(spec=Redis)
        redis_pool.script_load = AsyncMock(
            side_effect=['sha-1', 'sha-2'],
        )
        redis_pool.evalsha = AsyncMock(side_effect=[NoScriptError(), [1]])

        with self.assertRaises(ValueError):
            await service._incr_and_get_ttl(redis_pool, 'k5', 10)

    async def test_custom_rate_limiter_with_response_and_negative_ttl(
        self,
    ) -> None:
        """Call wrapper with explicit Response and negative TTL."""
        service_response = Response()
        redis_pool = AsyncMock(spec=Redis)
        # Use fast path; negative TTL should fall back to window_seconds (60)
        redis_pool.script_load = AsyncMock(return_value='sha-fast')
        redis_pool.evalsha = AsyncMock(return_value=[1, -2])

        mock_request = MagicMock(spec=Request)
        mock_request.method = 'GET'
        mock_request.url.path = '/some'
        mock_request.app.state.redis_client.client = redis_pool

        creds = MagicMock()
        creds.subject = {
            'username': 'u1',
            'user_id': 1,
            'role': 'user',
            'jti': 'abc',
            'features': [],
        }

        remaining = await rate_limiter_service(
            mock_request,
            service_response,
            creds,
        )
        self.assertEqual(remaining, 2999)
        self.assertEqual(
            service_response.headers.get(
                'X-RateLimit-Reset',
            ),
            '60',
        )

    async def test_rate_limiter_guest_role_exceeds(
        self,
    ) -> None:
        """Test rate limiter for guest role that exceeds the limit (24/day)."""
        redis_pool = AsyncMock(spec=Redis)
        redis_pool.script_load = AsyncMock(return_value='sha')
        redis_pool.evalsha = AsyncMock(return_value=[25, 86400])

        mock_request = MagicMock(spec=Request)
        mock_request.app.state.redis_client.client = redis_pool
        mock_request.url.path = '/rate_limit_test'

        mock_credentials = MagicMock()
        mock_credentials.subject = {
            'role': 'guest',
            'username': 'test_user',
            'jti': 'test_jti',
        }

        with self.assertRaises(HTTPException) as exc:
            await rate_limiter_service(
                mock_request,
                Response(),
                mock_credentials,
            )
        self.assertEqual(exc.exception.status_code, 429)
        self.assertIn('Rate limit exceeded', exc.exception.detail)

        redis_pool.evalsha.assert_awaited_once()

    async def test_rate_limiter_guest_role_within_limit(
        self,
    ) -> None:
        """Test rate limiter for a guest role within limit (24/day)."""
        redis_pool = AsyncMock(spec=Redis)
        redis_pool.script_load = AsyncMock(return_value='sha')
        redis_pool.evalsha = AsyncMock(return_value=[5, 100])

        mock_request = MagicMock(spec=Request)
        mock_request.app.state.redis_client.client = redis_pool
        mock_request.url.path = '/rate_limit_test'

        mock_credentials = MagicMock()
        mock_credentials.subject = {
            'role': 'guest',
            'username': 'test_user',
            'jti': 'test_jti',
        }

        remaining = await rate_limiter_service(
            mock_request,
            Response(),
            mock_credentials,
        )
        self.assertEqual(remaining, 24 - 5)

        redis_pool.evalsha.assert_awaited_once()

    async def test_rate_limiter_user_role_within_limit(
        self,
    ) -> None:
        """Test rate limiter for user role within limit (3000/min)."""
        redis_pool = AsyncMock(spec=Redis)
        redis_pool.script_load = AsyncMock(return_value='sha')
        redis_pool.evalsha = AsyncMock(return_value=[500, 45])

        mock_request = MagicMock(spec=Request)
        mock_request.app.state.redis_client.client = redis_pool
        mock_request.url.path = '/user_endpoint'

        mock_credentials = MagicMock()
        mock_credentials.subject = {
            'role': 'user',
            'username': 'test_user',
            'jti': 'test_jti',
        }

        remaining = await rate_limiter_service(
            mock_request,
            Response(),
            mock_credentials,
        )
        self.assertEqual(remaining, 3000 - 500)

        redis_pool.evalsha.assert_awaited_once()

    async def test_rate_limiter_user_role_exceeds_limit(
        self,
    ) -> None:
        """Test rate limiter for user role exceeding limit (3000/min)."""
        redis_pool = AsyncMock(spec=Redis)
        redis_pool.script_load = AsyncMock(return_value='sha')
        redis_pool.evalsha = AsyncMock(return_value=[3001, 60])

        mock_request = MagicMock(spec=Request)
        mock_request.app.state.redis_client.client = redis_pool
        mock_request.url.path = '/user_endpoint'

        mock_credentials = MagicMock()
        mock_credentials.subject = {
            'role': 'user',
            'username': 'test_user',
            'jti': 'test_jti',
        }

        with self.assertRaises(HTTPException) as exc:
            await rate_limiter_service(
                mock_request,
                Response(),
                mock_credentials,
            )
        self.assertEqual(exc.exception.status_code, 429)
        self.assertEqual(exc.exception.detail, 'Rate limit exceeded')

    async def test_rate_limiter_with_ttl_expiry(
        self,
    ) -> None:
        """Test rate limiter TTL handling when ttl == -1 (no expiry set
        yet)."""
        redis_pool = AsyncMock(spec=Redis)
        redis_pool.script_load = AsyncMock(return_value='sha')
        redis_pool.evalsha = AsyncMock(return_value=[10, 86400])

        mock_request = MagicMock(spec=Request)
        mock_request.app.state.redis_client.client = redis_pool
        mock_request.url.path = '/rate_limit_test'

        mock_credentials = MagicMock()
        mock_credentials.subject = {
            'role': 'guest',
            'username': 'test_user',
            'jti': 'test_jti',
        }

        remaining = await rate_limiter_service(
            mock_request,
            Response(),
            mock_credentials,
        )
        self.assertEqual(remaining, 24 - 10)

        redis_pool.evalsha.assert_awaited_once()


if __name__ == '__main__':
    unittest.main()
