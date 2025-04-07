from __future__ import annotations

import json
import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import HTTPException
from fastapi import Request
from redis.asyncio import Redis

from examples.auth.cache import custom_rate_limiter
from examples.auth.cache import get_user_data
from examples.auth.cache import PROJECT_PREFIX
from examples.auth.cache import set_user_data


class CacheTestCase(unittest.IsolatedAsyncioTestCase):
    """
    Test cases for cache functionalities (get_user_data, set_user_data)
    and the custom_rate_limiter behavior.
    """

    async def test_get_user_data(self):
        """
        Test retrieving user data from the Redis cache.
        """
        # Make redis_pool a valid AsyncMock with .get as an async method
        redis_pool = AsyncMock(spec=Redis)
        redis_pool.get = AsyncMock(
            return_value='{"username": "test_user", "role": "user"}',
        )

        user_data = await get_user_data(redis_pool, 'test_user')
        self.assertIsInstance(user_data, dict)
        self.assertEqual(user_data['username'], 'test_user')
        self.assertEqual(user_data['role'], 'user')

        redis_pool.get.assert_awaited_once_with(
            f"{PROJECT_PREFIX}:user_cache:test_user",
        )

    async def test_get_user_data_not_found(self):
        """
        Test retrieving user data that does not exist in the Redis cache.
        """
        redis_pool = AsyncMock(spec=Redis)
        # Return None so it looks like no data found
        redis_pool.get = AsyncMock(return_value=None)

        user_data = await get_user_data(redis_pool, 'nonexistent_user')
        self.assertIsNone(user_data)

        redis_pool.get.assert_awaited_once_with(
            f"{PROJECT_PREFIX}:user_cache:nonexistent_user",
        )

    async def test_set_user_data(self):
        """
        Test storing user data in the Redis cache.
        """
        redis_pool = AsyncMock(spec=Redis)
        # Make sure set is also an async method
        redis_pool.set = AsyncMock(return_value=True)

        user_data_dict = {'username': 'test_user', 'role': 'user'}
        await set_user_data(redis_pool, 'test_user', user_data_dict)

        redis_pool.set.assert_awaited_once_with(
            f"{PROJECT_PREFIX}:user_cache:test_user",
            json.dumps(user_data_dict),
        )

    @patch(
        'examples.auth.cache.get_user_data',
        return_value={'jti_list': ['test_jti']},
    )
    async def test_rate_limiter_guest_role_exceeds(self, mock_get_user_data):
        """
        Test rate limiter for guest role that exceeds the limit (24/day).
        """
        redis_pool = AsyncMock(spec=Redis)
        # Make these methods async
        redis_pool.incr = AsyncMock(return_value=25)  # Exceeds limit
        redis_pool.ttl = AsyncMock(return_value=-1)
        redis_pool.expire = AsyncMock(return_value=None)

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
            await custom_rate_limiter(mock_request, mock_credentials)
        self.assertEqual(exc.exception.status_code, 429)
        self.assertIn('Rate limit exceeded', exc.exception.detail)

        # TTL was -1, so we set expire
        redis_pool.expire.assert_awaited_once_with(
            'rate_limit:guest:test_user:/rate_limit_test', 86400,
        )

    @patch(
        'examples.auth.cache.get_user_data',
        return_value={'jti_list': ['test_jti']},
    )
    async def test_rate_limiter_guest_role_within_limit(
        self,
        mock_get_user_data,
    ):
        """
        Test rate limiter for a guest role within limit (24/day).
        """
        redis_pool = AsyncMock(spec=Redis)
        redis_pool.incr = AsyncMock(return_value=5)
        redis_pool.ttl = AsyncMock(return_value=100)
        redis_pool.expire = AsyncMock()

        mock_request = MagicMock(spec=Request)
        mock_request.app.state.redis_client.client = redis_pool
        mock_request.url.path = '/rate_limit_test'

        mock_credentials = MagicMock()
        mock_credentials.subject = {
            'role': 'guest',
            'username': 'test_user',
            'jti': 'test_jti',
        }

        remaining = await custom_rate_limiter(mock_request, mock_credentials)
        self.assertEqual(remaining, 24 - 5)

        # TTL is 100, not -1 => no expire call
        redis_pool.expire.assert_not_awaited()
        # incr call
        redis_pool.incr.assert_awaited_once_with(
            'rate_limit:guest:test_user:/rate_limit_test',
        )

    @patch(
        'examples.auth.cache.get_user_data',
        return_value={'jti_list': ['test_jti']},
    )
    async def test_rate_limiter_user_role_within_limit(
        self,
        mock_get_user_data,
    ):
        """
        Test rate limiter for user role within limit (3000/min).
        """
        redis_pool = AsyncMock(spec=Redis)
        redis_pool.incr = AsyncMock(return_value=500)
        redis_pool.ttl = AsyncMock(return_value=45)
        redis_pool.expire = AsyncMock()

        mock_request = MagicMock(spec=Request)
        mock_request.app.state.redis_client.client = redis_pool
        mock_request.url.path = '/user_endpoint'

        mock_credentials = MagicMock()
        mock_credentials.subject = {
            'role': 'user',
            'username': 'test_user',
            'jti': 'test_jti',
        }

        remaining = await custom_rate_limiter(mock_request, mock_credentials)
        self.assertEqual(remaining, 3000 - 500)

        # TTL is 45, not -1 => no expire
        redis_pool.expire.assert_not_awaited()

    @patch(
        'examples.auth.cache.get_user_data',
        return_value={'jti_list': ['test_jti']},
    )
    async def test_rate_limiter_user_role_exceeds_limit(
        self,
        mock_get_user_data,
    ):
        """
        Test rate limiter for user role exceeding limit (3000/min).
        """
        redis_pool = AsyncMock(spec=Redis)
        redis_pool.incr = AsyncMock(return_value=3001)
        redis_pool.ttl = AsyncMock(return_value=60)
        redis_pool.expire = AsyncMock()

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
            await custom_rate_limiter(mock_request, mock_credentials)
        self.assertEqual(exc.exception.status_code, 429)
        self.assertEqual(exc.exception.detail, 'Rate limit exceeded')

    @patch(
        'examples.auth.cache.get_user_data',
        return_value={'jti_list': ['test_jti']},
    )
    async def test_rate_limiter_with_ttl_expiry(self, mock_get_user_data):
        """
        Test rate limiter TTL handling when ttl == -1 (no expiry set yet).
        """
        redis_pool = AsyncMock(spec=Redis)
        redis_pool.incr = AsyncMock(return_value=10)
        redis_pool.ttl = AsyncMock(return_value=-1)
        redis_pool.expire = AsyncMock()

        mock_request = MagicMock(spec=Request)
        mock_request.app.state.redis_client.client = redis_pool
        mock_request.url.path = '/rate_limit_test'

        mock_credentials = MagicMock()
        mock_credentials.subject = {
            'role': 'guest',
            'username': 'test_user',
            'jti': 'test_jti',
        }

        remaining = await custom_rate_limiter(mock_request, mock_credentials)
        self.assertEqual(remaining, 24 - 10)

        redis_pool.expire.assert_awaited_once_with(
            'rate_limit:guest:test_user:/rate_limit_test', 86400,
        )

    async def test_rate_limiter_invalid_jti(self):
        """
        Test rate limiter with an invalid token jti.
        """
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
            'examples.auth.cache.get_user_data',
            return_value={'jti_list': ['test_jti']},
        ):
            with self.assertRaises(HTTPException) as exc:
                await custom_rate_limiter(mock_request, mock_credentials)
            self.assertEqual(exc.exception.status_code, 401)
            self.assertEqual(
                exc.exception.detail,
                'Token jti is invalid or replaced',
            )

    async def test_rate_limiter_missing_or_invalid_fields(self):
        """
        Test rate limiter with missing/invalid username or jti fields.
        """
        redis_pool = AsyncMock(spec=Redis)
        mock_request = MagicMock(spec=Request)
        mock_request.app.state.redis_client.client = redis_pool
        mock_request.url.path = '/rate_limit_test'

        # Missing username
        mock_credentials = MagicMock()
        mock_credentials.subject = {'role': 'guest', 'jti': 'test_jti'}
        with self.assertRaises(HTTPException) as exc:
            await custom_rate_limiter(mock_request, mock_credentials)
        self.assertEqual(exc.exception.status_code, 401)
        self.assertIn('missing or invalid fields', exc.exception.detail)

        # Missing jti
        mock_credentials.subject = {'role': 'guest', 'username': 'test_user'}
        with self.assertRaises(HTTPException) as exc:
            await custom_rate_limiter(mock_request, mock_credentials)
        self.assertEqual(exc.exception.status_code, 401)
        self.assertIn('missing or invalid fields', exc.exception.detail)

        # Invalid data types
        mock_credentials.subject = {
            'role': 'guest',
            'username': 123, 'jti': ['bad_jti'],
        }
        with self.assertRaises(HTTPException) as exc:
            await custom_rate_limiter(mock_request, mock_credentials)
        self.assertEqual(exc.exception.status_code, 401)
        self.assertIn('missing or invalid fields', exc.exception.detail)

    async def test_rate_limiter_no_user_in_redis(self):
        """
        Test rate limiter when no user data is found in Redis.
        """
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

        with patch('examples.auth.cache.get_user_data', return_value=None):
            with self.assertRaises(HTTPException) as exc:
                await custom_rate_limiter(mock_request, mock_credentials)
            self.assertEqual(exc.exception.status_code, 401)
            self.assertEqual(exc.exception.detail, 'No such user in Redis')


if __name__ == '__main__':
    unittest.main()

'''
pytest \
    --cov=examples.auth.cache \
    --cov-report=term-missing tests/examples/auth/cache_test.py
'''
