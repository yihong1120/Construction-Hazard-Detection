from __future__ import annotations

import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from typing import ClassVar
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import uvicorn
from fastapi.testclient import TestClient

from examples.local_notification_server.app import _redact_sensitive_payload
from examples.local_notification_server.app import app
from examples.local_notification_server.app import main


class TestLocalNotificationServer(unittest.TestCase):
    """
    Test suite for the local notification server's FastAPI application.
    """

    client: ClassVar[TestClient]

    @classmethod
    def setUpClass(cls) -> None:
        """
        Set up the test client for the FastAPI application.
        """
        cls.client = TestClient(app)

    def test_swagger_ui(self) -> None:
        """
        Verify that the Swagger UI (docs) endpoint is reachable.
        """
        response = self.client.get('/docs')
        self.assertEqual(response.status_code, 200)
        self.assertIn('swagger', response.text.lower())

    def test_redoc_ui(self) -> None:
        """
        Verify that the Redoc endpoint is reachable.
        """
        response = self.client.get('/redoc')
        self.assertEqual(response.status_code, 200)
        self.assertIn('redoc', response.text.lower())

    def test_redact_sensitive_payload_handles_camel_case(self) -> None:
        """Request logging redacts token fields regardless of casing style."""
        redacted = _redact_sensitive_payload({
            'deviceToken': 'raw-device-token',
            'fcm_token': 'raw-fcm-token',
            'nested': {
                'accessToken': 'raw-access-token',
                'safe': 'value',
            },
        })

        self.assertEqual(redacted['deviceToken'], '<redacted>')
        self.assertEqual(redacted['fcm_token'], '<redacted>')
        self.assertEqual(redacted['nested']['accessToken'], '<redacted>')
        self.assertEqual(redacted['nested']['safe'], 'value')

    def test_main(self) -> None:
        """
        Test the main() function to ensure uvicorn.run is called
        with the correct parameters.
        """
        called = False

        def fake_run(app_obj: Any, host: str, port: int) -> None:
            """Support fake_run.

            Args:
                app_obj: Test helper value.
            """
            nonlocal called
            called = True
            # Check parameters
            self.assertEqual(host, '127.0.0.1')
            self.assertEqual(port, 8003)

        with patch.object(uvicorn, 'run', fake_run):
            main()
            self.assertTrue(called)

    def test_lifespan_init(self) -> None:
        """
        Test the lifespan logic to ensure database initialisation is triggered.
        Patch Firebase credential and init to avoid real file access.
        """
        flag = False

        @asynccontextmanager
        async def fake_begin() -> None:
            """Support fake_begin."""
            async def run_sync(fn: Any, *args, **kwargs) -> None:
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

        with patch('examples.auth.lifespan.engine', fake_engine):
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
                        # Using TestClient triggers the lifespan context
                        with TestClient(app):
                            pass

        self.assertTrue(
            flag, 'Database initialisation logic was not triggered.',
        )
        fake_engine.dispose.assert_awaited_once()


if __name__ == '__main__':
    unittest.main()

"""
pytest \
    --cov=examples.local_notification_server.app \
    --cov-report=term-missing \
    tests/examples/local_notification_server/app_test.py
"""
