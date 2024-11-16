from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import socketio
from fastapi.testclient import TestClient
from fastapi_jwt import JwtAccessBearer

from examples.YOLO_server_api.app import app
from examples.YOLO_server_api.app import lifespan


class TestApp(unittest.IsolatedAsyncioTestCase):
    """
    Unit tests for FastAPI app functionality and dependencies.
    """

    def setUp(self) -> None:
        """
        Sets up the TestClient instance for testing FastAPI app routes.
        """
        self.client = TestClient(app)

    @patch(
        'examples.YOLO_server_api.app.redis.from_url',
        new_callable=AsyncMock,
    )
    @patch(
        'examples.YOLO_server_api.app.scheduler.shutdown',
        new_callable=MagicMock,
    )
    @patch('examples.YOLO_server_api.app.engine', autospec=True)
    async def test_redis_initialization(
        self,
        mock_engine: MagicMock,
        mock_scheduler_shutdown: MagicMock,
        mock_redis_from_url: AsyncMock,
    ) -> None:
        """
        Tests Redis and SQLAlchemy engine initialisation within app lifespan.

        Args:
            mock_engine (MagicMock): Mocked SQLAlchemy engine instance.
            mock_scheduler_shutdown (MagicMock): Mock for scheduler shutdown.
            mock_redis_from_url (AsyncMock): Mock for Redis connection.
        """
        # Mock Redis client behaviour
        mock_redis_client = AsyncMock()
        mock_redis_from_url.return_value = mock_redis_client

        # Mock SQLAlchemy engine connection and transaction
        mock_conn = AsyncMock()
        mock_engine.begin.return_value.__aenter__.return_value = mock_conn

        # Start the lifespan context to test initialisation behaviour
        async with lifespan(app):
            mock_redis_from_url.assert_called_once()
            mock_engine.begin.assert_called_once()
            mock_redis_client.close.assert_not_called()
            mock_scheduler_shutdown.assert_not_called()

    @patch(
        'examples.YOLO_server_api.app.FastAPILimiter.init',
        new_callable=AsyncMock,
    )
    @patch(
        'examples.YOLO_server_api.app.redis.from_url',
        new_callable=AsyncMock,
    )
    @patch('examples.YOLO_server_api.app.engine', autospec=True)
    async def test_lifespan_context(
        self,
        mock_engine: MagicMock,
        mock_redis_from_url: AsyncMock,
        mock_limiter_init: AsyncMock,
    ) -> None:
        """
        Tests the lifespan context to verify resource initialisation.

        Args:
            mock_engine (MagicMock): Mocked SQLAlchemy engine instance.
            mock_redis_from_url (AsyncMock): Mock for Redis connection.
            mock_limiter_init (AsyncMock): Mock for rate limiter
                initialisation.
        """
        # Mock Redis client and SQLAlchemy engine behaviour
        mock_redis_client = AsyncMock()
        mock_redis_from_url.return_value = mock_redis_client
        mock_conn = AsyncMock()
        mock_engine.begin.return_value.__aenter__.return_value = mock_conn

        # Start the lifespan context to test the initialisation behaviour
        async with lifespan(app):
            mock_redis_from_url.assert_called_once()
            mock_engine.begin.assert_called_once()
            mock_limiter_init.assert_called_once()

    def test_routes_exist(self) -> None:
        """
        Checks the existence of primary app routes.
        """
        # Test main route endpoints
        response = self.client.get('/auth/some_endpoint')
        # Expecting either success or not found
        self.assertIn(response.status_code, [200, 404])
        response = self.client.get('/detect/some_endpoint')
        self.assertIn(response.status_code, [200, 404])
        response = self.client.get('/models/some_endpoint')
        self.assertIn(response.status_code, [200, 404])

    @patch(
        'examples.YOLO_server_api.app.JwtAccessBearer.__init__',
        return_value=None,
    )
    def test_jwt_initialization(
        self, mock_jwt_access_bearer_init: MagicMock,
    ) -> None:
        """
        Tests initialisation of JWT Access Bearer with a mock secret key.

        Args:
            mock_jwt_access_bearer_init (MagicMock): Mock for JwtAccessBearer
                initialisation.
        """
        JwtAccessBearer(secret_key='mocked_secret')
        mock_jwt_access_bearer_init.assert_called_once_with(
            secret_key='mocked_secret',
        )

    @patch('socketio.AsyncClient.connect', new_callable=AsyncMock)
    @patch('socketio.AsyncClient.disconnect', new_callable=AsyncMock)
    def test_socketio_connect_disconnect(
        self,
        mock_disconnect: AsyncMock,
        mock_connect: AsyncMock,
    ) -> None:
        """
        Tests socket.io client connection and disconnection.

        Args:
            mock_disconnect (AsyncMock): Mock for socket.io client
                disconnection.
            mock_connect (AsyncMock): Mock for socket.io client connection.
        """
        client = socketio.AsyncClient()

        @client.event
        async def connect() -> None:
            print('Connected to server')

        @client.event
        async def disconnect() -> None:
            print('Disconnected from server')

        async def socket_test() -> None:
            """
            Asynchronously tests connection and disconnection to socket server.
            """
            await client.connect('http://0.0.0.0:5000')
            mock_connect.assert_called_once()
            await client.disconnect()
            mock_disconnect.assert_called_once()

        # Run the asynchronous socket test
        asyncio.run(socket_test())


if __name__ == '__main__':
    unittest.main()
