from __future__ import annotations

import unittest
from contextlib import asynccontextmanager
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi.testclient import TestClient

from examples.YOLO_server_api import app as yolo_app

app = yolo_app.app
main = yolo_app.main


class TestApp(unittest.IsolatedAsyncioTestCase):
    """Unit tests for the FastAPI application."""

    def setUp(self) -> None:
        """Initialises the test client for the FastAPI application."""
        self.client = TestClient(app)

    @patch('uvicorn.run')
    def test_main(self, mock_uvicorn_run: MagicMock) -> None:
        """Tests the main function that starts the FastAPI application.

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

    async def test_lifespan_logs_then_uses_the_shared_lifecycle(self) -> None:
        """YOLO workers initialise through the shared application lifecycle."""

        @asynccontextmanager
        async def shared_lifespan(_app: object):
            """Provide an isolated shared lifespan context for the test."""
            yield

        with (
            patch.object(yolo_app, 'log_configuration') as log_configuration,
            patch.object(yolo_app, 'global_lifespan', shared_lifespan),
        ):
            async with yolo_app._lifespan(MagicMock()):
                log_configuration.assert_called_once()


if __name__ == '__main__':
    unittest.main()
