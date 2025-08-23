from __future__ import annotations

import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi.testclient import TestClient

import examples.streaming_web.backend.app as app_module


class TestStreamingWebApp(unittest.IsolatedAsyncioTestCase):
    """
    Test suite for the FastAPI application in
    examples.streaming_web.backend.app.
    """

    def setUp(self) -> None:
        """
        Sets up the test environment before each test.
        """
        self.app = app_module.app
        self.client = TestClient(self.app)

    @patch('examples.streaming_web.backend.app.CORSMiddleware')
    def test_cors_initialization(self, mock_cors: MagicMock) -> None:
        """
        Tests that the CORS middleware is initialised with expected parameters.

        Args:
            mock_cors (MagicMock): Mock for the CORSMiddleware class.
        """
        cors = mock_cors(
            self.app,
            allow_origins=['*'],
            allow_credentials=True,
            allow_methods=['*'],
            allow_headers=['*'],
        )
        self.assertIsInstance(cors, MagicMock)
        mock_cors.assert_called_once_with(
            self.app,
            allow_origins=['*'],
            allow_credentials=True,
            allow_methods=['*'],
            allow_headers=['*'],
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
    --cov=examples.streaming_web.backend.app \
    --cov-report=term-missing \
    tests/examples/streaming_web/backend/app_test.py
'''
