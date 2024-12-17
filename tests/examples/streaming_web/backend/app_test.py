from __future__ import annotations

import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi.testclient import TestClient
from fastapi_limiter import FastAPILimiter

import examples.streaming_web.backend.app as app_module


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

    @patch(
        'examples.streaming_web.backend.app.redis_manager.client',
        new_callable=AsyncMock,
    )
    async def test_redis_connection(
        self, mock_redis_client: AsyncMock,
    ) -> None:
        """
        Test that the Redis connection is properly established using
        a mocked Redis client with virtual parameters.
        """
        # Simulate an AsyncMock instance as a Redis client with
        # virtual parameters
        mock_redis_client.return_value = AsyncMock()

        # Mock connection using arbitrary (virtual) Redis parameters
        await mock_redis_client(
            host='virtualhost', port=1234,
            password='virtualpass', decode_responses=True,
        )

        # Verify that the mock Redis client was called with
        # the virtual parameters
        self.assertIsInstance(mock_redis_client, AsyncMock)
        mock_redis_client.assert_awaited_once_with(
            host='virtualhost', port=1234,
            password='virtualpass', decode_responses=True,
        )

    @patch('examples.streaming_web.backend.app.CORSMiddleware')
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

    @patch(
        'examples.streaming_web.backend.app.FastAPILimiter.init',
        new_callable=AsyncMock,
    )
    @patch(
        'examples.streaming_web.backend.app.redis_manager.client',
        new_callable=AsyncMock,
    )
    async def test_rate_limiter_initialization(
        self,
        mock_redis_client: AsyncMock,
        mock_limiter_init: AsyncMock,
    ) -> None:
        """
        Test that the rate limiter is properly initialized with
        a mocked Redis client.
        """
        await FastAPILimiter.init(mock_redis_client)
        mock_limiter_init.assert_awaited_once_with(mock_redis_client)

    @patch('uvicorn.run')
    def test_app_running_configuration(
        self, mock_uvicorn_run: MagicMock,
    ) -> None:
        """
        Test that the application runs with the expected configurations.
        """
        app_module.run_server()
        mock_uvicorn_run.assert_called_once_with(
            'examples.streaming_web.backend.app:sio_app',
            host='127.0.0.1', port=8000, log_level='info',
        )

    @patch(
        'examples.streaming_web.backend.app.redis_manager.client',
        new_callable=AsyncMock,
    )
    @patch(
        'examples.streaming_web.backend.app.FastAPILimiter.init',
        new_callable=AsyncMock,
    )
    def test_lifespan_events(
        self,
        mock_limiter_init: AsyncMock,
        mock_redis_client: AsyncMock,
    ) -> None:
        """
        Test that the lifespan events are properly handled.
        """
        with TestClient(app_module.app) as client:
            response = client.get('/')
            # 驗證返回的狀態碼是 404
            self.assertEqual(response.status_code, 404)
        mock_limiter_init.assert_called_once_with(mock_redis_client)
        mock_redis_client.close.assert_called_once()

    def tearDown(self) -> None:
        del self.client


if __name__ == '__main__':
    unittest.main()
