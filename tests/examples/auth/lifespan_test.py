from __future__ import annotations

import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import FastAPI

from examples.auth.lifespan import global_lifespan


class TestGlobalLifespan(unittest.IsolatedAsyncioTestCase):
    """
    Test suite for the global_lifespan async context manager,
    ensuring startup and shutdown logic is executed correctly.
    """

    @patch('examples.auth.lifespan.engine')
    @patch('examples.auth.lifespan.start_jwt_scheduler')
    @patch('examples.auth.lifespan.FastAPILimiter.init')
    @patch('examples.auth.lifespan.RedisClient')
    async def test_global_lifespan(
        self,
        mock_redis_client_cls: MagicMock,
        mock_limiter_init: MagicMock,
        mock_start_scheduler: MagicMock,
        mock_engine_obj: MagicMock,
    ) -> None:
        """
        Test the global_lifespan async context manager.

        Args:
            mock_redis_client_cls (MagicMock): Patches the RedisClient class.
            mock_limiter_init (MagicMock): Patches FastAPILimiter.init
                to avoid real initialisation.
            mock_start_scheduler (MagicMock): Patches start_jwt_scheduler to
                avoid real scheduling.
        """
        # (A) Mock the returned scheduler from start_jwt_scheduler.
        mock_scheduler: MagicMock = MagicMock()
        mock_start_scheduler.return_value = mock_scheduler

        # (B) Mock the RedisClient class/methods.
        mock_redis_client: MagicMock = MagicMock()
        mock_redis_client.connect = AsyncMock()
        mock_redis_client.close = AsyncMock()
        mock_redis_client_cls.return_value = mock_redis_client

        # Mock engine.begin() async context manager and run_sync
        mock_conn_ctx = AsyncMock()
        mock_conn = MagicMock()
        mock_conn.run_sync = AsyncMock()
        mock_conn_ctx.__aenter__.return_value = mock_conn
        mock_engine_obj.begin.return_value = mock_conn_ctx
        mock_engine_obj.dispose = AsyncMock()

        # (D) Instantiate a FastAPI app to pass into global_lifespan.
        app: FastAPI = FastAPI()

        # Enter the async context manager.
        async with global_lifespan(app):
            # "Startup" logic should have completed by now.
            mock_redis_client.connect.assert_awaited_once()

            # The app.state.redis_client should be our mock_redis_client.
            self.assertIs(
                app.state.redis_client,
                mock_redis_client,
                "Expected the app's redis_client to "
                'be set to our mock object.',
            )

            # FastAPILimiter should be initialised
            # with the mock_redis_client's connection.
            mock_limiter_init.assert_called_once_with(
                mock_redis_client.connect.return_value,
            )

            # The scheduler should not be shut down
            # while we are in the context.
            mock_scheduler.shutdown.assert_not_called()

            mock_conn.run_sync.assert_awaited_once()
        # Once we exit the context => "shutdown" logic runs.
        mock_scheduler.shutdown.assert_called_once()
        mock_redis_client.close.assert_awaited_once()
        mock_engine_obj.dispose.assert_awaited_once()


if __name__ == '__main__':
    unittest.main()

'''
pytest \
    --cov=examples.auth.lifespan \
    --cov-report=term-missing tests/examples/auth/lifespan_test.py
'''
