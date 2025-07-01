from __future__ import annotations

import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi.testclient import TestClient

from examples.YOLO_server_api.backend.app import app
from examples.YOLO_server_api.backend.app import main


class TestApp(unittest.IsolatedAsyncioTestCase):
    """
    Unit tests for the FastAPI application.
    """

    def setUp(self) -> None:
        """
        Initialises the test client for the FastAPI application.
        """
        self.client = TestClient(app)

    @patch(
        'examples.auth.redis_pool.RedisClient.connect',
        new_callable=AsyncMock,
    )
    @patch(
        'examples.auth.jwt_scheduler.BackgroundScheduler.shutdown',
        new_callable=MagicMock,
    )
    @patch('fastapi_limiter.FastAPILimiter.init', new_callable=AsyncMock)
    @patch(
        'examples.auth.database.AsyncSessionLocal',
        new_callable=MagicMock,
    )
    @patch(
        'examples.auth.database.engine',
        new_callable=MagicMock,
    )
    async def test_lifespan_context(
        self,
        mock_engine: MagicMock,
        mock_async_session: MagicMock,
        mock_limiter_init: AsyncMock,
        mock_scheduler_shutdown: MagicMock,
        mock_redis_connect: AsyncMock,
    ) -> None:
        """
        Tests the lifespan context of the FastAPI application.

        Args:
            mock_engine (MagicMock): Mocked SQLAlchemy engine.
            mock_async_session (MagicMock): Mocked AsyncSession.
            mock_limiter_init (AsyncMock):
                A mock for FastAPILimiter's init method.
            mock_scheduler_shutdown (MagicMock):
                A mock for the background scheduler's shutdown method.
            mock_redis_connect (AsyncMock):
                A mock for the Redis client's connect method.
        """
        # Startup phase
        async with app.router.lifespan_context(app):
            mock_redis_connect.assert_awaited_once()
            mock_limiter_init.assert_awaited_once()
            mock_scheduler_shutdown.assert_not_called()

        # Teardown phase => scheduler.shutdown called
        mock_scheduler_shutdown.assert_called_once()

    @patch('uvicorn.run')
    def test_main(self, mock_uvicorn_run: MagicMock) -> None:
        """
        Tests the main function that starts the FastAPI application.

        Args:
            mock_uvicorn_run (MagicMock): A mock for the uvicorn.run function.
        """
        main()
        mock_uvicorn_run.assert_called_once_with(
            app,
            host='127.0.0.1',
            port=8000,
            workers=2,
        )


if __name__ == '__main__':
    unittest.main()


'''
pytest \
    --cov=examples.YOLO_server_api.backend.app \
    --cov-report=term-missing \
    tests/examples/YOLO_server_api/backend/app_test.py
'''
