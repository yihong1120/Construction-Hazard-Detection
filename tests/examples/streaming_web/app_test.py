from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import uvicorn
from fastapi.testclient import TestClient

import examples.streaming_web.app as app_module


class TestStreamingWebApp(unittest.IsolatedAsyncioTestCase):
    """
    Test suite for the streaming_web FastAPI app.
    """

    def setUp(self) -> None:
        """
        Set up the test environment before each test.
        """
        self.app = app_module.app
        self.client = TestClient(self.app)

    @patch('examples.streaming_web.app.redis_manager', new_callable=AsyncMock)
    async def test_redis_connection(self, mock_redis: AsyncMock) -> None:
        """
        Test that the Redis connection is properly established.
        """
        mock_redis.return_value = AsyncMock()
        r = await mock_redis(
            host='localhost', port=6379,
            password='passcode', decode_responses=False,
        )
        self.assertIsInstance(r, AsyncMock)
        mock_redis.assert_awaited_once_with(
            host='localhost', port=6379,
            password='passcode', decode_responses=False,
        )

    @patch('examples.streaming_web.app.CORSMiddleware')
    def test_cors_initialization(self, mock_cors: MagicMock) -> None:
        """
        Test that CORS is properly initialized for the FastAPI app.
        """
        cors = mock_cors(
            self.app, allow_origins=['*'],
            allow_credentials=True, allow_methods=['*'], allow_headers=['*'],
        )
        self.assertIsInstance(cors, MagicMock)
        mock_cors.assert_called_once_with(
            self.app, allow_origins=['*'],
            allow_credentials=True, allow_methods=['*'], allow_headers=['*'],
        )

    @patch('examples.streaming_web.app.FastAPILimiter.init', new_callable=AsyncMock)
    async def test_rate_limiter_initialization(self, mock_limiter_init: AsyncMock) -> None:
        """
        Test that the rate limiter is properly initialized.
        """
        await mock_limiter_init(app_module.redis_manager)
        mock_limiter_init.assert_awaited_once_with(app_module.redis_manager)

    @patch('uvicorn.run')
    def test_app_running_configuration(self, mock_uvicorn_run: MagicMock) -> None:
        """
        Test that the application runs with the expected configurations.
        """
        uvicorn.run(
            'examples.streaming_web.app:sio_app',
            host='127.0.0.1', port=8000, log_level='info',
        )
        mock_uvicorn_run.assert_called_once_with(
            'examples.streaming_web.app:sio_app',
            host='127.0.0.1', port=8000, log_level='info',
        )

    def tearDown(self) -> None:
        """
        Clean up after each test.
        """
        del self.client


if __name__ == '__main__':
    unittest.main()
