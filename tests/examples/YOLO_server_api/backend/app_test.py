from __future__ import annotations

import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import socketio
from fastapi.testclient import TestClient

from examples.YOLO_server_api.backend.app import app
from examples.YOLO_server_api.backend.app import lifespan
from examples.YOLO_server_api.backend.app import run_uvicorn_app
from examples.YOLO_server_api.backend.app import sio
from examples.YOLO_server_api.backend.app import sio_app


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
        Verifies the existence and behaviour of primary application routes.
        """
        # Test the /api/token endpoint; it should
        # return 405 (method not allowed for GET).
        response = self.client.get('/api/token')
        self.assertEqual(response.status_code, 405)

        # Test the /api/detect endpoint; it should
        # return 405 (method not allowed for GET).
        response = self.client.get('/api/detect')
        self.assertEqual(response.status_code, 405)

        # Test the /api/model_file_update endpoint; it should
        # return 405 (method not allowed for GET).
        response = self.client.get('/api/model_file_update')
        self.assertEqual(response.status_code, 405)

        # Test the /api/get_new_model endpoint; it should
        # return 405 (method not allowed for GET).
        response = self.client.get('/api/get_new_model')
        self.assertEqual(response.status_code, 405)

        # Test the /api/add_user endpoint; it should
        # return 405 (method not allowed for GET).
        response = self.client.get('/api/add_user')
        self.assertEqual(response.status_code, 405)

    @patch('builtins.print')
    @patch.object(socketio.AsyncClient, 'disconnect', new_callable=AsyncMock)
    @patch.object(socketio.AsyncClient, 'connect', new_callable=AsyncMock)
    async def test_socketio_connect_disconnect(
        self,
        mock_connect: AsyncMock,
        mock_disconnect: AsyncMock,
        mock_print: MagicMock,
    ) -> None:
        """
        Tests the Socket.IO client connection and disconnection.

        Args:
            mock_connect (AsyncMock): Mocked connect() method of the client.
            mock_disconnect (AsyncMock): Mocked disconnect() method of
                the client.
            mock_print (MagicMock): Mocked print function for logging.
        """
        # 1) Mock the connect() / disconnect() so no real network calls occur
        mock_connect.return_value = None
        mock_disconnect.return_value = None

        # 2) Create a client and "connect" / "disconnect" (all mock)
        client = socketio.AsyncClient()
        await client.connect('http://fake-host:9999', wait=True)
        await client.disconnect()

        # 3) Manually trigger the server's 'connect' event
        sid = '123'
        environ = {'REMOTE_ADDR': '127.0.0.1'}
        await sio._trigger_event('connect', '/', sid, environ)

        # 4) Manually trigger the server's 'disconnect' event (without reason)
        await sio._trigger_event('disconnect', '/', sid)

        # 5) Check the print calls for server side
        mock_print.assert_any_call('Client connected:', sid)
        mock_print.assert_any_call('Client disconnected:', sid)

    @patch('examples.YOLO_server_api.backend.app.uvicorn.run')
    def test_main_entry_uvicorn_run(self, mock_uvicorn_run: MagicMock) -> None:
        """
        Test the run_uvicorn_app function to ensure
        uvicorn.run is called with correct parameters.

        Args:
            mock_uvicorn_run (MagicMock): Mocked uvicorn.run function.
        """
        run_uvicorn_app()
        mock_uvicorn_run.assert_called_once_with(
            sio_app,
            host='0.0.0.0',
            port=8000,
            workers=2,
        )


if __name__ == '__main__':
    unittest.main()
