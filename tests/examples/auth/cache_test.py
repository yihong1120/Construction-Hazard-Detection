from __future__ import annotations

import json
import unittest
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import HTTPException
from fastapi import Request
from fastapi import Response
from redis.asyncio import Redis
from redis.exceptions import NoScriptError

from examples.auth.cache import PROJECT_PREFIX
from examples.auth.cache import rate_limiter_service
from examples.auth.cache import RateLimiterService


class CacheTestCase(unittest.IsolatedAsyncioTestCase):
    """Test cases for cache functionalities (get_user_data, set_user_data) and
    the custom_rate_limiter behavior."""

    async def test_get_user_data(self) -> None:
        """Test retrieving user data from the Redis cache."""
        # Make redis_pool a valid AsyncMock with .get as an async method
        redis_pool = AsyncMock(spec=Redis)
        redis_pool.get = AsyncMock(
            return_value=b'{"username": "test_user", "role": "user"}',
        )

        user_data = await rate_limiter_service.get_user_data(
            redis_pool,
            'test_user',
        )
        self.assertIsInstance(user_data, dict)
        assert user_data is not None
        self.assertEqual(user_data['username'], 'test_user')
        self.assertEqual(user_data['role'], 'user')

        redis_pool.get.assert_awaited_once_with(
            f"{PROJECT_PREFIX}:user_cache:test_user",
        )

    async def test_get_user_data_not_found(self) -> None:
        """Test retrieving user data that does not exist in the Redis cache."""
        redis_pool = AsyncMock(spec=Redis)
        # Return None so it looks like no data found
        redis_pool.get = AsyncMock(return_value=None)

        user_data = await rate_limiter_service.get_user_data(
            redis_pool,
            'nonexistent_user',
        )
        self.assertIsNone(user_data)

        redis_pool.get.assert_awaited_once_with(
            f"{PROJECT_PREFIX}:user_cache:nonexistent_user",
        )

    async def test_set_user_data(self) -> None:
        """Test storing user data in the Redis cache."""
        redis_pool = AsyncMock(spec=Redis)
        # Make sure set is also an async method
        redis_pool.set = AsyncMock(return_value=True)

        user_data_dict: dict[str, object] = {
            'username': 'test_user',
            'role': 'user',
        }
        await rate_limiter_service.set_user_data(
            redis_pool,
            'test_user',
            user_data_dict,
        )

        redis_pool.set.assert_awaited_once_with(
            f"{PROJECT_PREFIX}:user_cache:test_user",
            json.dumps(user_data_dict).encode('utf-8'),
        )

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

    @patch(
        'examples.auth.cache.rate_limiter_service.get_user_data',
        return_value={'jti_list': 'not_a_list'},
    )
    async def test_rate_limiter_jti_list_not_iterable(
        self,
        mock_get_user_data: Any,
    ) -> None:
        """When jti_list is not a list/tuple, it should be treated as empty."""
        redis_pool = AsyncMock(spec=Redis)
        redis_pool.script_load = AsyncMock(return_value='sha-ok')
        redis_pool.evalsha = AsyncMock(return_value=[1, 10])

        mock_request = MagicMock(spec=Request)
        mock_request.app.state.redis_client.client = redis_pool
        mock_request.url.path = '/any'

        creds = MagicMock()
        creds.subject = {
            'role': 'user',
            'username': 'u2',
            'jti': 'abc',
        }

        with self.assertRaises(HTTPException) as exc:
            await rate_limiter_service(mock_request, Response(), creds)
        self.assertEqual(exc.exception.status_code, 401)
        self.assertIn('invalid or replaced', exc.exception.detail)

    @patch(
        'examples.auth.cache.rate_limiter_service.get_user_data',
        return_value={'jti_list': ['abc']},
    )
    async def test_custom_rate_limiter_with_response_and_negative_ttl(
        self,
        mock_get_user_data: Any,
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

    @patch(
        'examples.auth.cache.rate_limiter_service.get_user_data',
        return_value={'jti_list': ['test_jti']},
    )
    async def test_rate_limiter_guest_role_exceeds(
        self,
        mock_get_user_data: Any,
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

    @patch(
        'examples.auth.cache.rate_limiter_service.get_user_data',
        return_value={'jti_list': ['test_jti']},
    )
    async def test_rate_limiter_guest_role_within_limit(
        self,
        mock_get_user_data: Any,
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

    @patch(
        'examples.auth.cache.rate_limiter_service.get_user_data',
        return_value={'jti_list': ['test_jti']},
    )
    async def test_rate_limiter_user_role_within_limit(
        self,
        mock_get_user_data: Any,
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

    @patch(
        'examples.auth.cache.rate_limiter_service.get_user_data',
        return_value={'jti_list': ['test_jti']},
    )
    async def test_rate_limiter_user_role_exceeds_limit(
        self,
        mock_get_user_data: Any,
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

    @patch(
        'examples.auth.cache.rate_limiter_service.get_user_data',
        return_value={'jti_list': ['test_jti']},
    )
    async def test_rate_limiter_with_ttl_expiry(
        self,
        mock_get_user_data: Any,
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

    async def test_rate_limiter_invalid_jti(self) -> None:
        """Test rate limiter with an invalid token jti."""
        redis_pool = AsyncMock(spec=Redis)
        mock_request = MagicMock(spec=Request)
        mock_request.app.state.redis_client.client = redis_pool
        mock_request.url.path = '/rate_limit_test'

        mock_credentials = MagicMock()
        mock_credentials.subject = {
            'role': 'guest',
            'username': 'test_user',
            'jti': 'wrong_jti',
        }

        with patch(
            'examples.auth.cache.rate_limiter_service.get_user_data',
            return_value={'jti_list': ['test_jti']},
        ):
            with self.assertRaises(HTTPException) as exc:
                await rate_limiter_service(
                    mock_request,
                    Response(),
                    mock_credentials,
                )
            self.assertEqual(exc.exception.status_code, 401)
            self.assertEqual(
                exc.exception.detail,
                'Token jti is invalid or replaced',
            )

    async def test_rate_limiter_no_user_in_redis(self) -> None:
        """Test rate limiter when no user data is found in Redis."""
        redis_pool = AsyncMock(spec=Redis)
        redis_pool.incr = AsyncMock()
        mock_request = MagicMock(spec=Request)
        mock_request.app.state.redis_client.client = redis_pool
        mock_request.url.path = '/rate_limit_test'

        mock_credentials = MagicMock()
        mock_credentials.subject = {
            'role': 'guest',
            'username': 'test_user',
            'jti': 'test_jti',
        }

        with patch(
            'examples.auth.cache.rate_limiter_service.get_user_data',
            return_value=None,
        ):
            with self.assertRaises(HTTPException) as exc:
                await rate_limiter_service(
                    mock_request,
                    Response(),
                    mock_credentials,
                )
            self.assertEqual(exc.exception.status_code, 401)
            self.assertEqual(exc.exception.detail, 'No such user in Redis')


if __name__ == '__main__':
    unittest.main()
