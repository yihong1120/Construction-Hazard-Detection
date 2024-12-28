from __future__ import annotations

import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import HTTPException

from examples.YOLO_server_api.backend.cache import custom_rate_limiter
from examples.YOLO_server_api.backend.cache import get_user_data
from examples.YOLO_server_api.backend.cache import set_user_data


class CacheTestCase(unittest.IsolatedAsyncioTestCase):
    """
    Test cases for cache functionalities and rate limiter.
    """

    async def test_get_user_data(self):
        """
        Test retrieving user data from the Redis cache.
        """
        redis_pool = AsyncMock()
        redis_pool.get.return_value = (
            '{"username": "test_user", "role": "user"}'
        )

        user_data = await get_user_data(redis_pool, 'test_user')

        # Validate the user data retrieved from Redis
        self.assertIsInstance(user_data, dict)
        self.assertEqual(user_data['username'], 'test_user')
        self.assertEqual(user_data['role'], 'user')

        redis_pool.get.assert_called_once_with('user_cache:test_user')

    async def test_get_user_data_not_found(self):
        """
        Test retrieving user data when it does not exist in the Redis cache.
        """
        redis_pool = AsyncMock()
        redis_pool.get.return_value = None  # 模擬未找到用戶數據

        user_data = await get_user_data(redis_pool, 'nonexistent_user')

        # 驗證返回值為 None
        self.assertIsNone(user_data)
        redis_pool.get.assert_called_once_with('user_cache:nonexistent_user')

    async def test_set_user_data(self):
        """
        Test storing user data in the Redis cache.
        """
        redis_pool = AsyncMock()

        user_data = {'username': 'test_user', 'role': 'user'}
        await set_user_data(redis_pool, 'test_user', user_data)

        # Validate the user data stored in Redis
        redis_pool.set.assert_called_once_with(
            'user_cache:test_user',
            '{"username": "test_user", "role": "user"}',
        )

    async def test_rate_limiter_guest_role(self):
        """
        Test rate limiter functionality for a guest role that exceeds
        the limit.
        """
        # Mock Redis and request
        redis_pool = AsyncMock()
        redis_pool.incr.return_value = 25  # Exceeding limit
        redis_pool.ttl.return_value = -1

        mock_request = MagicMock()
        mock_request.app.state.redis_client.client = redis_pool
        mock_request.url.path = '/rate_limit_test'

        # Mock JWT credentials
        mock_jwt_access = MagicMock(
            subject={
                'role': 'guest',
                'username': 'test_user',
                'jti': 'test_jti',
            },
        )

        # Mock Redis user data
        with patch(
            'examples.YOLO_server_api.backend.cache.get_user_data',
            return_value={'jti_list': ['test_jti']},
        ):
            with self.assertRaises(HTTPException) as exc:
                await custom_rate_limiter(
                    mock_request,
                    mock_jwt_access,
                )
            self.assertEqual(exc.exception.status_code, 429)
            self.assertEqual(exc.exception.detail, 'Rate limit exceeded')

    async def test_rate_limiter_with_ttl_expiry(self):
        """
        Test rate limiter functionality where TTL is set because it was -1.
        """
        # Mock Redis and request
        redis_pool = AsyncMock()
        redis_pool.incr.return_value = 10  # Within limit
        redis_pool.ttl.return_value = -1  # TTL not set

        mock_request = MagicMock()
        mock_request.app.state.redis_client.client = redis_pool
        mock_request.url.path = '/rate_limit_test'

        # Mock JWT credentials
        mock_jwt_access = MagicMock(
            subject={
                'role': 'guest',
                'username': 'test_user',
                'jti': 'test_jti',
            },
        )

        # Mock Redis user data with jti
        with patch(
            'examples.YOLO_server_api.backend.cache.get_user_data',
            return_value={'jti_list': ['test_jti']},
        ):
            remaining_requests = await custom_rate_limiter(
                mock_request,
                mock_jwt_access,
            )

        # Assert remaining requests are calculated correctly
        self.assertEqual(remaining_requests, 24 - 10)

        # Verify Redis interactions
        redis_pool.incr.assert_called_once_with(
            'rate_limit:guest:test_user:/rate_limit_test',
        )
        redis_pool.ttl.assert_called_once_with(
            'rate_limit:guest:test_user:/rate_limit_test',
        )
        redis_pool.expire.assert_called_once_with(
            'rate_limit:guest:test_user:/rate_limit_test', 86400,
        )

    async def test_rate_limiter_invalid_jti(self):
        """
        Test rate limiter with an invalid token jti.
        """
        redis_pool = AsyncMock()

        mock_request = MagicMock()
        mock_request.app.state.redis_client.client = redis_pool
        mock_request.url.path = '/rate_limit_test'

        mock_jwt_access = MagicMock(
            subject={
                'role': 'user', 'username': 'test_user',
                'jti': 'invalid_jti',
            },
        )

        # Mock Redis user data with invalid jti
        with patch(
            'examples.YOLO_server_api.backend.cache.get_user_data',
            return_value={'jti_list': ['valid_jti']},
        ):
            with self.assertRaises(HTTPException) as exc:
                await custom_rate_limiter(
                    mock_request,
                    mock_jwt_access,
                )
            self.assertEqual(exc.exception.status_code, 401)
            self.assertEqual(
                exc.exception.detail,
                'Token jti is invalid or replaced',
            )

    async def test_rate_limiter_missing_or_invalid_fields(self):
        """
        Test rate limiter when the token has missing or invalid fields.
        """
        redis_pool = AsyncMock()

        mock_request = MagicMock()
        mock_request.app.state.redis_client.client = redis_pool
        mock_request.url.path = '/rate_limit_test'

        # Mock missing username field
        mock_jwt_access = MagicMock(
            subject={
                'role': 'guest',
                'jti': 'test_jti',  # Lack of username
            },
        )

        with self.assertRaises(HTTPException) as exc:
            await custom_rate_limiter(mock_request, mock_jwt_access)
        self.assertEqual(exc.exception.status_code, 401)
        self.assertEqual(
            exc.exception.detail,
            'Token is missing or invalid fields',
        )

        # Mock missing jti field
        mock_jwt_access = MagicMock(
            subject={
                'role': 'guest',
                'username': 'test_user',  # Lack of jti
            },
        )

        with self.assertRaises(HTTPException) as exc:
            await custom_rate_limiter(mock_request, mock_jwt_access)
        self.assertEqual(exc.exception.status_code, 401)
        self.assertEqual(
            exc.exception.detail,
            'Token is missing or invalid fields',
        )

        # Mock invalid data types for username and jti
        mock_jwt_access = MagicMock(
            subject={
                'role': 'guest',
                'username': 123,  # Non-string
                'jti': ['invalid_jti'],  # Non-string
            },
        )

        with self.assertRaises(HTTPException) as exc:
            await custom_rate_limiter(mock_request, mock_jwt_access)
        self.assertEqual(exc.exception.status_code, 401)
        self.assertEqual(
            exc.exception.detail,
            'Token is missing or invalid fields',
        )

    async def test_rate_limiter_no_user_in_redis(self):
        """
        Test rate limiter when no user data is found in Redis.
        """
        redis_pool = AsyncMock()

        mock_request = MagicMock()
        mock_request.app.state.redis_client.client = redis_pool
        mock_request.url.path = '/rate_limit_test'

        mock_jwt_access = MagicMock(
            subject={
                'role': 'guest',
                'username': 'test_user',
                'jti': 'test_jti',
            },
        )

        # Mock no user data in Redis
        with patch(
            'examples.YOLO_server_api.backend.cache.get_user_data',
            return_value=None,  # No such user in Redis
        ):
            with self.assertRaises(HTTPException) as exc:
                await custom_rate_limiter(mock_request, mock_jwt_access)
            self.assertEqual(exc.exception.status_code, 401)
            self.assertEqual(exc.exception.detail, 'No such user in Redis')


if __name__ == '__main__':
    unittest.main()
