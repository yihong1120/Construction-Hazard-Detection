from __future__ import annotations

import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

from fastapi import HTTPException

from examples.YOLO_server_api.backend.cache import custom_rate_limiter
from examples.YOLO_server_api.backend.cache import user_cache


class CacheTestCase(unittest.IsolatedAsyncioTestCase):
    """
    Test cases for cache functionalities and rate limiter.
    """

    def setUp(self):
        """
        Set up the test environment before each test.
        """
        # Clear the cache before each test to ensure a clean slate
        user_cache.clear()

    def tearDown(self):
        """
        Clean up after each test.
        """
        # Clear the cache after each test to ensure no residue data
        user_cache.clear()

    def test_add_to_cache(self):
        """
        Test adding a user to the cache.
        """
        user_cache['user1'] = 'data1'
        self.assertIn('user1', user_cache)
        self.assertEqual(user_cache['user1'], 'data1')

    def test_remove_from_cache(self):
        """
        Test removing a user from the cache.
        """
        user_cache['user1'] = 'data1'
        del user_cache['user1']
        self.assertNotIn('user1', user_cache)

    def test_update_cache(self):
        """
        Test updating a user in the cache.
        """
        user_cache['user1'] = 'data1'
        user_cache['user1'] = 'data2'
        self.assertEqual(user_cache['user1'], 'data2')

    def test_clear_cache(self):
        """
        Test clearing the entire cache.
        """
        user_cache['user1'] = 'data1'
        user_cache['user2'] = 'data2'
        user_cache.clear()
        self.assertEqual(len(user_cache), 0)

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
        mock_request.app.state.redis_pool = redis_pool
        mock_request.url.path = '/rate_limit_test'

        # Mock JWT credentials
        mock_jwt_access = MagicMock(
            subject={'role': 'guest', 'username': 'test_user'},
        )

        # Verify HTTPException is raised for exceeding rate limit
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
        mock_request.app.state.redis_pool = redis_pool
        mock_request.url.path = '/rate_limit_test'

        # Mock JWT credentials
        mock_jwt_access = MagicMock(
            subject={'role': 'guest', 'username': 'test_user'},
        )

        # Call the rate limiter
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


if __name__ == '__main__':
    unittest.main()
