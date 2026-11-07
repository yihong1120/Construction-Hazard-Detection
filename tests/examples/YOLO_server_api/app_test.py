from __future__ import annotations

import os
import unittest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi.testclient import TestClient

from examples.YOLO_server_api import app as yolo_app

app = yolo_app.app
main = yolo_app.main


async def _lifespan_logs_then_uses_the_shared_lifespan(_app: object):
    """Provide an isolated shared lifespan context for the test."""
    yield


async def _overlay_close_failure_still_closes_shared(_):
    yield


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
        with patch('sys.argv', ['yolo-api']):
            main()
        mock_uvicorn_run.assert_called_once_with(
            'examples.YOLO_server_api.app:app',
            host='127.0.0.1',
            port=8000,
            workers=2,
            ws_max_size=yolo_app.ApiInferenceSettings().max_upload_bytes,
        )

    @patch('uvicorn.run')
    def test_transport_uses_configured_upload_limit(self, run):
        with (
            patch.dict(os.environ, {'DETECT_MAX_UPLOAD_BYTES': '1234'}),
            patch(
                'sys.argv',
                [
                    'yolo-api',
                    '--host',
                    '127.0.0.2',
                    '--port',
                    '6000',
                    '--workers',
                    '1',
                ],
            ),
        ):
            main()
        self.assertEqual(
            run.call_args.kwargs,
            {
                'host': '127.0.0.2',
                'port': 6000,
                'workers': 1,
                'ws_max_size': 1234,
            },
        )

    async def test_lifespan_logs_then_uses_the_shared_lifecycle(self) -> None:
        """YOLO workers initialise through the shared application lifecycle."""

        shared_lifespan = asynccontextmanager(
            _lifespan_logs_then_uses_the_shared_lifespan,
        )

        with (
            patch.object(yolo_app, 'log_configuration') as log_configuration,
            patch.object(yolo_app, 'global_lifespan', shared_lifespan),
        ):
            async with yolo_app._lifespan(MagicMock()):
                log_configuration.assert_called_once()

    async def test_overlay_close_failure_still_closes_inference(self):
        shared = asynccontextmanager(
            _overlay_close_failure_still_closes_shared,
        )

        inference = MagicMock(close=AsyncMock())
        overlays = MagicMock(
            close=AsyncMock(side_effect=RuntimeError('close')),
        )
        with (
            patch.object(
                yolo_app,
                'get_inference_runtime',
                return_value=inference,
            ),
            patch.object(
                yolo_app,
                'get_overlay_runtime',
                return_value=overlays,
            ),
            patch.object(yolo_app, 'global_lifespan', shared),
        ):
            with self.assertRaises(RuntimeError):
                async with yolo_app._lifespan(MagicMock()):
                    pass
        inference.close.assert_awaited_once()


if __name__ == '__main__':
    unittest.main()
