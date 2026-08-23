from __future__ import annotations

import os
import unittest
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from typing import ClassVar
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from examples.local_notification_server.app import app


class TestLocalNotificationServer(unittest.TestCase):
    """Test suite for the local notification server's FastAPI application."""

    client: ClassVar[TestClient]

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the test client for the FastAPI application."""
        cls.client = TestClient(app)

    def test_swagger_ui(self) -> None:
        """Verify that the Swagger UI (docs) endpoint is reachable."""
        response = self.client.get('/docs')
        self.assertEqual(response.status_code, 200)
        self.assertIn('swagger', response.text.lower())

    def test_redoc_ui(self) -> None:
        """Verify that the Redoc endpoint is reachable."""
        response = self.client.get('/redoc')
        self.assertEqual(response.status_code, 200)
        self.assertIn('redoc', response.text.lower())

    def test_lifespan_init(self) -> None:
        """Test the lifespan logic to ensure database initialisation is
        triggered.

        Patch Firebase credential and init to avoid real file access.
        """
        flag = False

        @asynccontextmanager
        async def fake_begin() -> AsyncIterator[SimpleNamespace]:
            """Support fake_begin."""

            async def run_sync(fn: Any, *args: Any, **kwargs: Any) -> None:
                """Support run_sync.

                Args:
                    fn: Test helper value.
                """
                nonlocal flag
                flag = True

            # Yield a simple object exposing run_sync
            yield SimpleNamespace(run_sync=run_sync)

        fake_engine = MagicMock()
        fake_engine.begin = lambda: fake_begin()
        fake_engine.dispose = AsyncMock()  # Make dispose awaitable

        fernet_key = Fernet.generate_key().decode('utf-8')
        with patch.dict(
            os.environ,
            {
                'FCM_TOKEN_ENCRYPTION_KEY': fernet_key,
                'FIREBASE_CRED_PATH': '/tmp/test-firebase-credentials.json',
                'FIREBASE_PROJECT_ID': 'test-project',
                'AUTO_CREATE_SCHEMA': 'true',
            },
        ):
            with patch('examples.auth.lifespan.engine', fake_engine):
                with patch(
                    'examples.auth.lifespan.drain_site_media_cleanup_jobs',
                    new_callable=AsyncMock,
                ):
                    with patch(
                        'firebase_admin.credentials.Certificate',
                        return_value=MagicMock(),
                    ):
                        with patch(
                            'firebase_admin.initialize_app',
                            return_value=MagicMock(),
                        ):
                            with patch(
                                'examples.auth.redis_pool.RedisClient.connect',
                                new_callable=AsyncMock,
                            ):
                                # Using TestClient triggers the lifespan context.
                                with TestClient(app):
                                    pass

        self.assertTrue(
            flag,
            'Database initialisation logic was not triggered.',
        )
        fake_engine.dispose.assert_awaited_once()


if __name__ == '__main__':
    unittest.main()
\
"""Pytest \

--cov=examples.local_notification_server.app \
--cov-report=term-missing \
tests/examples/local_notification_server/app_test.py
"""
