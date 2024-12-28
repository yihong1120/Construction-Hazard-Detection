from __future__ import annotations

import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import socketio
from fastapi.testclient import TestClient

from examples.YOLO_server_api.backend.app import app
from examples.YOLO_server_api.backend.app import lifespan


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
        'examples.YOLO_server_api.backend.app.RedisClient.connect',
        new_callable=AsyncMock,
    )
    @patch(
        'examples.YOLO_server_api.backend.app.scheduler.shutdown',
        new_callable=MagicMock,
    )
    @patch('examples.YOLO_server_api.backend.app.engine', autospec=True)
    async def test_redis_initialization(
        self,
        mock_engine: MagicMock,
        mock_scheduler_shutdown: MagicMock,
        mock_redis_connect: AsyncMock,
    ) -> None:
        """
        Tests Redis and SQLAlchemy engine initialisation within app lifespan.

        Args:
            mock_engine (MagicMock): Mocked SQLAlchemy engine instance.
            mock_scheduler_shutdown (MagicMock): Mock for scheduler shutdown.
            mock_redis_connect (AsyncMock): Mock for Redis connection.
        """
        # Mock SQLAlchemy engine connection and transaction
        mock_conn = AsyncMock()
        mock_engine.begin.return_value.__aenter__.return_value = mock_conn

        # Start the lifespan context to test initialisation behaviour
        async with lifespan(app):
            mock_redis_connect.assert_called_once()
            mock_engine.begin.assert_called_once()
            mock_scheduler_shutdown.assert_not_called()

    @patch(
        'examples.YOLO_server_api.backend.app.FastAPILimiter.init',
        new_callable=AsyncMock,
    )
    @patch(
        'examples.YOLO_server_api.backend.app.RedisClient.connect',
        new_callable=AsyncMock,
    )
    @patch('examples.YOLO_server_api.backend.app.engine', autospec=True)
    async def test_lifespan_context(
        self,
        mock_engine: MagicMock,
        mock_redis_connect: AsyncMock,
        mock_limiter_init: AsyncMock,
    ) -> None:
        """
        Tests the lifespan context to verify resource initialisation.

        Args:
            mock_engine (MagicMock): Mocked SQLAlchemy engine instance.
            mock_redis_connect (AsyncMock): Mock for Redis connection.
            mock_limiter_init (AsyncMock): Mock for rate limiter
                initialisation.
        """
        # Mock SQLAlchemy engine connection and transaction
        mock_conn = AsyncMock()
        mock_engine.begin.return_value.__aenter__.return_value = mock_conn

        # Start the lifespan context to test the initialisation behaviour
        async with lifespan(app):
            mock_redis_connect.assert_called_once()
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

    @patch('socketio.AsyncClient.emit', new_callable=AsyncMock)
    async def test_socketio_connect_disconnect(
        self,
        mock_client_emit: AsyncMock,
    ) -> None:
        """
        Tests socket.io client connection and disconnection.

        Args:
            mock_client_emit (AsyncMock): Mock for client emit method.
        """
        sio = socketio.AsyncClient()

        @sio.event
        async def connect() -> None:
            print('Connected to server')
            await sio.emit('connect_event', {'message': 'Hello'})

        @sio.event
        async def disconnect() -> None:
            print('Disconnected from server')

        async def socket_test() -> None:
            """
            Asynchronously tests connection and disconnection to socket server.
            """
            await sio.connect('http://0.0.0.0:8000')
            # Expecting a connection event to be emitted
            mock_client_emit.assert_called_once_with(
                'connect_event', {'message': 'Hello'},
            )
            await sio.disconnect()

        await socket_test()


if __name__ == '__main__':
    unittest.main()
