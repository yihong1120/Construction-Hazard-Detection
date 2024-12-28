from __future__ import annotations

import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import Request

from examples.YOLO_server_api.backend.redis_pool import get_redis_pool
from examples.YOLO_server_api.backend.redis_pool import RedisClient


class TestRedisClient(unittest.IsolatedAsyncioTestCase):
    """
    Unit tests for the RedisClient class.
    """

    def setUp(self) -> None:
        """
        Set up the test environment before each test.
        """
        self.redis_url = 'redis://localhost:6379/0'
        self.redis_client = RedisClient(self.redis_url)

    async def test_initialisation(self) -> None:
        """
        Test that the RedisClient is initialised correctly.
        """
        self.assertEqual(self.redis_client.url, self.redis_url)
        self.assertIsNone(self.redis_client.client)

    @patch('redis.asyncio.from_url', new_callable=AsyncMock)
    async def test_connect(self, mock_from_url: AsyncMock) -> None:
        """
        Test the connect method of RedisClient.
        """
        mock_redis_instance = AsyncMock()
        mock_from_url.return_value = mock_redis_instance

        # Call the connect method
        client = await self.redis_client.connect()

        # Verify that redis.from_url was called with the correct arguments
        mock_from_url.assert_called_once_with(
            self.redis_url,
            encoding='utf-8',
            decode_responses=True,
        )

        # Verify the returned client is the mocked instance
        self.assertEqual(client, mock_redis_instance)
        self.assertEqual(self.redis_client.client, mock_redis_instance)

    @patch('redis.asyncio.from_url', new_callable=AsyncMock)
    async def test_connect_existing_client(
        self,
        mock_from_url: AsyncMock,
    ) -> None:
        """
        Test that connect does not reinitialise an existing client.
        """
        mock_redis_instance = AsyncMock()
        self.redis_client.client = mock_redis_instance

        # Call the connect method
        client = await self.redis_client.connect()

        # Ensure redis.from_url was not called again
        mock_from_url.assert_not_called()

        # Verify the client is still the existing instance
        self.assertEqual(client, mock_redis_instance)

    async def test_close(self) -> None:
        """
        Test the close method of RedisClient.
        """
        mock_redis_instance = AsyncMock()
        self.redis_client.client = mock_redis_instance

        # Call the close method
        await self.redis_client.close()

        # Verify that the close method of the Redis client was called
        mock_redis_instance.close.assert_awaited_once()

        # Verify that the client is set to None
        self.assertIsNone(self.redis_client.client)

    async def test_close_no_client(self) -> None:
        """
        Test the close method when no client is connected.
        """
        # Ensure no client is set
        self.redis_client.client = None

        # Call the close method
        await self.redis_client.close()

        # Verify that no exceptions are raised and client remains None
        self.assertIsNone(self.redis_client.client)


class TestGetRedisPool(unittest.IsolatedAsyncioTestCase):
    """
    Unit tests for the get_redis_pool function.
    """

    @patch('redis.asyncio.from_url', new_callable=AsyncMock)
    async def test_get_redis_pool(self, mock_from_url: AsyncMock) -> None:
        # Build a mock Redis instance
        mock_redis_instance = AsyncMock()
        # Mock the from_url function to return the mock instance
        mock_from_url.return_value = mock_redis_instance

        # Build a mock RedisClient instance
        mock_redis_client = RedisClient(url='redis://localhost')

        # Ensure the client is not connected
        self.assertIsNone(mock_redis_client.client)

        # Build a mock request with the RedisClient instance
        mock_request = MagicMock(spec=Request)
        mock_request.app.state.redis_client = mock_redis_client

        # Call the get_redis_pool function
        client = await get_redis_pool(mock_request)

        # Verify that the connect method was called
        mock_from_url.assert_awaited_once()

        # Verify the returned client is the mocked instance
        self.assertIsNotNone(client)
        self.assertIs(client, mock_redis_instance)
        self.assertIs(mock_redis_client.client, mock_redis_instance)

    async def test_get_redis_pool_existing_client(self) -> None:
        """
        Test that get_redis_pool does not reconnect if a client exists.
        """
        # Mock the Redis client and request
        mock_redis_instance = AsyncMock()

        mock_redis_client = MagicMock(spec=RedisClient)
        mock_redis_client.client = mock_redis_instance

        mock_request = MagicMock(spec=Request)
        mock_request.app.state.redis_client = mock_redis_client

        # Call the get_redis_pool function
        client = await get_redis_pool(mock_request)

        # Verify that the connect method was not called
        mock_redis_client.connect.assert_not_called()

        # Verify the returned client is the mocked instance
        self.assertEqual(client, mock_redis_instance)

    @patch('redis.asyncio.from_url', new_callable=AsyncMock)
    async def test_get_redis_pool_raises_runtime_error(
        self,
        mock_from_url: AsyncMock,
    ) -> None:
        """
        Test that get_redis_pool raises RuntimeError
        when the client is not connected.

        Args:
            mock_from_url (AsyncMock): Mock for the from_url function.
        """
        # Build a mock RedisClient instance
        mock_redis_client = RedisClient(url='redis://localhost')

        # Simulate that client is not created
        mock_from_url.return_value = None

        # Build a mock request with the RedisClient instance
        mock_request = MagicMock(spec=Request)
        mock_request.app.state.redis_client = mock_redis_client

        # Call the get_redis_pool function and verify the exception
        with self.assertRaises(RuntimeError) as context:
            await get_redis_pool(mock_request)

        self.assertEqual(
            str(context.exception),
            'Redis client is not connected.',
        )


if __name__ == '__main__':
    unittest.main()
