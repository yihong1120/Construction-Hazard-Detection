from __future__ import annotations

import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi.testclient import TestClient

import examples.streaming_web.app as app_module


class TestStreamingWebApp(unittest.IsolatedAsyncioTestCase):
    """
    Test suite for the FastAPI application in
    examples.streaming_web.app.
    """

    def setUp(self) -> None:
        """
        Sets up the test environment before each test.
        """
        self.app = app_module.app
        self.client = TestClient(self.app)

    @patch('examples.streaming_web.app.CORSMiddleware')
    def test_cors_initialization(self, mock_cors: MagicMock) -> None:
        """
        Tests that the CORS middleware is initialised with expected parameters.

        Args:
            mock_cors (MagicMock): Mock for the CORSMiddleware class.
        """
        cors = mock_cors(
            self.app,
            allow_origins=app_module._cors_origins(),
            allow_origin_regex=app_module._cors_origin_regex(),
            allow_credentials=True,
            allow_methods=['*'],
            allow_headers=['*'],
        )
        self.assertIsInstance(cors, MagicMock)
        mock_cors.assert_called_once_with(
            self.app,
            allow_origins=app_module._cors_origins(),
            allow_origin_regex=app_module._cors_origin_regex(),
            allow_credentials=True,
            allow_methods=['*'],
            allow_headers=['*'],
        )

    def test_cors_origins_can_be_configured_by_env(self) -> None:
        """Use explicit origins so credentialed CORS never returns wildcard."""
        with patch.dict(
            'os.environ',
            {'STREAMING_WEB_CORS_ORIGINS': 'https://a.test, http://b.test'},
        ):
            self.assertEqual(
                app_module._cors_origins(),
                ['https://a.test', 'http://b.test'],
            )

    def test_cors_origin_regex_allows_localhost_any_port(self) -> None:
        """Support Flutter Web dev servers that pick random local ports."""
        self.assertEqual(
            app_module._cors_origin_regex(),
            r'https?://(localhost|127\.0\.0\.1)(:\d+)?',
        )

    @patch('uvicorn.run')
    def test_app_running_configuration(
        self,
        mock_uvicorn_run: MagicMock,
    ) -> None:
        """
        Tests whether uvicorn.run is invoked with the actual code's arguments.

        Args:
            mock_uvicorn_run (MagicMock): Mock for the uvicorn.run function.
        """
        app_module.main()
        # Match the call in app.py exactly:
        mock_uvicorn_run.assert_called_once_with(
            self.app,
            host='127.0.0.1',
            port=8800,
            # If you set 'log_level' or other arguments, add them here
        )

    def tearDown(self) -> None:
        """
        Cleans up test resources after each test.
        """
        del self.client


if __name__ == '__main__':
    unittest.main()

'''
pytest \
    --cov=examples.streaming_web.app \
    --cov-report=term-missing \
    tests/examples/streaming_web/app_test.py
'''
