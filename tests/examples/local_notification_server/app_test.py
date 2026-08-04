from __future__ import annotations

import asyncio
import unittest
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from typing import ClassVar
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import uvicorn
from fastapi.testclient import TestClient

from examples.local_notification_server.app import _is_sensitive_payload_key
from examples.local_notification_server.app import _redact_sensitive_payload
from examples.local_notification_server.app import _safe_body_preview
from examples.local_notification_server.app import app
from examples.local_notification_server.app import main
from examples.local_notification_server.app import validation_exception_handler


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

    def test_redact_sensitive_payload_handles_camel_case(self) -> None:
        """Request logging redacts token fields regardless of casing style."""
        redacted = _redact_sensitive_payload(
            {
                'deviceToken': 'raw-device-token',
                'fcm_token': 'raw-fcm-token',
                'nested': {
                    'accessToken': 'raw-access-token',
                    'safe': 'value',
                },
            },
        )

        assert isinstance(redacted, dict)
        nested = redacted['nested']
        assert isinstance(nested, dict)
        self.assertEqual(redacted['deviceToken'], '<redacted>')
        self.assertEqual(redacted['fcm_token'], '<redacted>')
        self.assertEqual(nested['accessToken'], '<redacted>')
        self.assertEqual(nested['safe'], 'value')

    def test_redaction_handles_lists_non_string_keys_and_invalid_json(
        self,
    ) -> None:
        """Logging helpers redact nested lists without assuming JSON input."""
        self.assertFalse(_is_sensitive_payload_key(123))
        self.assertEqual(
            _redact_sensitive_payload(
                [
                    {'refresh-token': 'secret'},
                    'safe-value',
                ],
            ),
            [{'refresh-token': '<redacted>'}, 'safe-value'],
        )
        self.assertEqual(
            _safe_body_preview(b'not-json'),
            'not-json',
        )
        self.assertEqual(
            _safe_body_preview(b'{"token":"secret","nested":[1]}'),
            '{"token": "<redacted>", "nested": [1]}',
        )

    def test_validation_handler_redacts_body_before_logging(self) -> None:
        """Validation errors return FastAPI details without logging tokens."""
        request = MagicMock()
        request.body = AsyncMock(return_value=b'{"access_token":"secret"}')
        request.url.path = '/notifications'
        errors = [{'loc': ['body', 'title'], 'msg': 'required'}]
        exc = MagicMock()
        exc.errors.return_value = errors

        with patch(
            'examples.local_notification_server.app.logger.warning',
        ) as warning:
            response = asyncio.run(
                validation_exception_handler(request, exc),
            )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(
            response.body,
            b'{"detail":[{"loc":["body","title"],"msg":"required"}]}',
        )
        self.assertIn('<redacted>', warning.call_args.args[-1])

    def test_main(self) -> None:
        """Test the main() function to ensure uvicorn.run is called with the
        correct parameters."""
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
        """Test the lifespan logic to ensure database initialisation is
        triggered.

        Patch Firebase credential and init to avoid real file access.
        """
        flag = False

        @asynccontextmanager
        async def fake_begin() -> AsyncIterator[SimpleNamespace]:
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
