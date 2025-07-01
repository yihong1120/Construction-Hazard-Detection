from __future__ import annotations

import unittest
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
