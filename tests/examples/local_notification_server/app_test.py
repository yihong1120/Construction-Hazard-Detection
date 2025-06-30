from __future__ import annotations

import unittest
from typing import ClassVar
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import uvicorn
from fastapi.testclient import TestClient

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

    def test_cors_headers(self) -> None:
        """
        Check that CORS headers are returned for OPTIONS requests.
        """
        headers = {
            'Origin': 'http://example.com',
            'Access-Control-Request-Method': 'GET',
        }
        response = self.client.options('/openapi.json', headers=headers)
        self.assertEqual(response.status_code, 200)
        # Check CORS headers
        self.assertEqual(
            response.headers.get('access-control-allow-origin'),
            'http://example.com',
        )
        self.assertIn('access-control-allow-methods', response.headers)

    def test_main(self) -> None:
        """
        Test the main() function to ensure uvicorn.run is called
        with the correct parameters.
        """
        called = False

        def fake_run(app_obj, host: str, port: int) -> None:
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

        class FakeConn:
            """
            Fake connection class to simulate the database connection.
            """

            async def run_sync(self, fn, *args, **kwargs) -> None:
                nonlocal flag
                flag = True

            async def __aenter__(self) -> FakeConn:
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
                pass

        fake_engine = MagicMock()
        fake_engine.begin = lambda: FakeConn()
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
                        'examples.auth.redis_pool.get_redis_pool',
                        return_value=MagicMock(),
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
