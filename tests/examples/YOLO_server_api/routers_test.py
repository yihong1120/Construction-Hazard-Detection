from __future__ import annotations

import asyncio
import unittest
from collections.abc import Callable
from functools import partial
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
from typing import ClassVar
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
from fastapi import FastAPI
from fastapi import UploadFile
from fastapi.testclient import TestClient

from examples.YOLO_server_api import routers as routers_mod
from examples.YOLO_server_api.routers import detection_router
from examples.YOLO_server_api.routers import jwt_access
from examples.YOLO_server_api.routers import model_management_router
from examples.YOLO_server_api.routers import rate_limiter_service
from examples.YOLO_server_api.routers import websocket_detect

"""Tests for FastAPI routers layer.

This module exercises the thin routing layer in
``examples.YOLO_server_api.routers``. It focuses on:

- REST endpoints happy paths and error handling.
- WebSocket endpoint delegating to the handler function.

The business logic is mocked; these tests verify HTTP contract,
authorisation, and delegation behaviour.
"""


def _setUp_zeros():
    return np.zeros((2, 2, 3), dtype=np.uint8)


def _setUpClass_callback():
    return 999


def _override_jwt_role_factory_override(
    role: Any,
    /,
) -> SimpleNamespace:
    """Support _override."""
    return SimpleNamespace(subject={'role': role, 'user': 'tester'})


class TestRouters(unittest.IsolatedAsyncioTestCase):
    """Unit tests for ``routers.py`` focused on routing and delegation."""

    app: ClassVar[FastAPI]
    client: ClassVar[TestClient]

    def setUp(self):
        decode = patch(
            'examples.YOLO_server_api.inference_runtime.prepare_image',
            return_value=SimpleNamespace(
                decoded_bytes=12,
                decode=_setUp_zeros,
            ),
        )
        decode.start()
        self.addCleanup(decode.stop)
        self.app.dependency_overrides[jwt_access] = (
            self._override_jwt_role_factory('admin')
        )
        patcher = patch(
            'examples.YOLO_server_api.routers.select_model_async',
            return_value=SimpleNamespace(
                id='yolo26n',
                version='commit',
                sha256='a' * 64,
                classes=[],
            ),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    @classmethod
    def setUpClass(cls) -> None:
        """Initialise shared FastAPI app and client for the test suite."""
        cls.app = FastAPI()
        # Mount routers under /api to match tests
        cls.app.include_router(detection_router, prefix='/api')
        cls.app.include_router(model_management_router, prefix='/api')
        # Default dependency overrides
        cls.app.dependency_overrides[rate_limiter_service] = (
            _setUpClass_callback
        )
        # Default role = admin unless a test overrides it
        cls.app.dependency_overrides[jwt_access] = (
            cls._override_jwt_role_factory('admin')
        )
        cls.client = TestClient(cls.app)

    @staticmethod
    def _override_jwt_role_factory(role: str) -> Callable[[], SimpleNamespace]:
        """Create a dependency override that injects a static JWT payload.

        Args:
            role: The role to inject into the JWT subject.

        Returns:
            A zero-argument callable returning a ``SimpleNamespace`` with a
            ``subject`` dict containing the provided role and a fixed user.
        """

        _override = partial(_override_jwt_role_factory_override, role)

        return _override

    def test_detect_endpoint_success(self):
        from examples.YOLO_server_api import (
            inference_runtime as runtime_module,
        )

        runtime = runtime_module.InferenceRuntime()
        with patch.object(
            routers_mod,
            'get_inference_runtime',
            return_value=runtime,
        ), patch.object(
            runtime.models,
            '_load_registry_model',
            return_value=Mock(),
        ), patch.object(
            runtime_module,
            'run_detection',
            return_value=(
                [[1, 2, 3, 4, 0.9, 0]],
                {'inference': 0.01, 'post': 0.005},
            ),
        ) as infer:
            response = self.client.post(
                '/api/detect',
                files={'image': ('test.jpg', b'123')},
                data={'model': 'yolo26n'},
            )
        asyncio.run(runtime.close())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [[1, 2, 3, 4, 0.9, 0]])
        infer.assert_called_once()

    def test_detect_endpoint_model_not_found(self):
        from fastapi import HTTPException

        with patch(
            'examples.YOLO_server_api.routers.select_model_async',
            side_effect=HTTPException(
                409,
                detail={'code': 'MODEL_UNAVAILABLE'},
            ),
        ):
            response = self.client.post(
                '/api/detect',
                data={'model': 'nope'},
                files={'image': ('x.jpg', b'image')},
            )
        self.assertEqual(response.status_code, 409)

    def test_model_file_update_requires_huggingface_publication(self):
        response = self.client.post(
            '/api/model_file_update',
            data={'model': 'yolo26n'},
            files={'file': ('model.pt', b'model')},
        )
        self.assertEqual(response.status_code, 409)

    def test_model_file_update_forbidden_role(self) -> None:
        """Verify non-privileged role gets 403 for model update."""
        # Downgrade role to user
        self.app.dependency_overrides[jwt_access] = (
            self._override_jwt_role_factory('user')
        )
        files = {
            'file': (
                'model.pt',
                b'model content',
                'application/octet-stream',
            ),
        }
        data = {'model': 'yolo26n'}
        resp = self.client.post(
            '/api/model_file_update',
            data=data,
            files=files,
        )
        self.assertEqual(resp.status_code, 403)
        self.assertIn("Need 'admin' or 'model_manage' role", resp.text)
        # revert
        self.app.dependency_overrides[jwt_access] = (
            self._override_jwt_role_factory('admin')
        )

    @patch(
        'examples.YOLO_server_api.routers.verified_artifact',
    )
    @patch('examples.YOLO_server_api.routers.logger')
    def test_get_new_model_updated(
        self,
        mock_logger: MagicMock,
        mock_get_file: AsyncMock,
    ) -> None:
        """Verify an updated model is returned as a binary stream."""
        # Ensure privileged role
        self.app.dependency_overrides[jwt_access] = (
            self._override_jwt_role_factory('admin')
        )
        from examples.YOLO_server_api.model_registry import VerifiedArtifact
        from examples.YOLO_server_api.model_files import model_file_checksum

        mock_get_file.return_value = VerifiedArtifact(
            Path(__file__),
            model_file_checksum(Path(__file__)),
            'yolo26n',
            'v1',
        )
        payload = {
            'model': 'yolo26n',
            'last_update_time': '2023-10-01T12:30:00',
        }
        resp = self.client.post('/api/get_new_model', json=payload)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(
            resp.headers['content-type'],
            'application/octet-stream',
        )
        self.assertIn('x-model-sha256', resp.headers)
        self.assertTrue(resp.content)
        mock_logger.info.assert_called()

    @patch(
        'examples.YOLO_server_api.routers.verified_artifact',
    )
    @patch('examples.YOLO_server_api.routers.logger')
    def test_get_new_model_up_to_date(
        self,
        _mock_logger: MagicMock,
        mock_get_file: AsyncMock,
    ) -> None:
        """Verify up-to-date model returns no response body.

        Args:
            _mock_logger: Mocked logger instance.
            mock_get_file: Mocked get new model file function.
        """
        from examples.YOLO_server_api.model_registry import VerifiedArtifact
        from examples.YOLO_server_api.model_files import model_file_checksum

        mock_get_file.return_value = VerifiedArtifact(
            Path(__file__),
            model_file_checksum(Path(__file__)),
            'yolo26n',
            'v1',
        )
        payload = {
            'model': 'yolo26n',
            'last_update_time': '2023-10-01T12:30:00',
        }
        from examples.YOLO_server_api.model_files import model_file_checksum

        etag = '"' + model_file_checksum(Path(__file__)) + '"'
        resp = self.client.post(
            '/api/get_new_model',
            json=payload,
            headers={
                'If-None-Match': etag,
            },
        )
        self.assertEqual(resp.status_code, 304)
        self.assertEqual(resp.content, b'')

    def test_get_new_model_invalid_datetime(self) -> None:
        """Verify invalid datetime yields 400 Bad Request."""
        payload = {'model': 'yolo26n', 'last_update_time': 'invalid_datetime'}
        resp = self.client.post('/api/get_new_model', json=payload)
        self.assertEqual(resp.status_code, 400)

    @patch(
        'examples.YOLO_server_api.routers.verified_artifact',
        side_effect=Exception('Some error'),
    )
    @patch('examples.YOLO_server_api.routers.logger')
    def test_get_new_model_exception(
        self,
        mock_logger: MagicMock,
        _patch_file: MagicMock,
    ) -> None:
        """Verify unexpected exception yields 500 and is logged.

        Args:
            mock_logger: Mocked logger instance.
            _patch_file: Mocked get new model file function.
        """
        payload = {
            'model': 'yolo26n',
            'last_update_time': '2023-10-01T12:30:00',
        }
        resp = self.client.post('/api/get_new_model', json=payload)
        self.assertEqual(resp.status_code, 500)
        self.assertIn('Failed to retrieve model.', resp.text)
        mock_logger.exception.assert_called()

    def test_get_new_model_forbidden_guest(self) -> None:
        """Verify guest role is forbidden to retrieve model artefacts.

        Args:
            mock_logger: Mocked logger instance.
            _patch_file: Mocked get new model file function.
        """
        self.app.dependency_overrides[jwt_access] = (
            self._override_jwt_role_factory('guest')
        )
        payload = {
            'model': 'yolo26n',
            'last_update_time': '2023-10-01T12:30:00',
        }
        resp = self.client.post('/api/get_new_model', json=payload)
        self.assertEqual(resp.status_code, 403)
        self.assertIn("Need 'admin' or 'model_manage' role", resp.text)
        # revert
        self.app.dependency_overrides[jwt_access] = (
            self._override_jwt_role_factory('admin')
        )

    async def test_stream_upload_rejects_empty_files(self) -> None:
        """Model uploads must contain at least one byte."""
        upload = UploadFile(filename='empty.pt', file=BytesIO())

        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'Empty upload file'):
                await routers_mod._stream_upload_to_path(
                    upload,
                    Path(directory) / 'model.pt',
                )

    async def test_websocket_route_delegates_to_handler(self) -> None:
        """Verify WebSocket route delegates to ``handle_websocket_detect``."""
        # Prepare fake websocket and settings object
        mock_ws: AsyncMock = AsyncMock()
        mock_rds: Mock = Mock()
        with patch.object(
            routers_mod,
            'handle_websocket_detect',
            new_callable=AsyncMock,
        ) as mock_handler:
            await websocket_detect(mock_ws, mock_rds)
            mock_handler.assert_awaited_once()
            call = getattr(mock_handler, 'await_args', None)
            self.assertIsNotNone(call)
            if call is not None:
                kwargs = getattr(call, 'kwargs', {})
                self.assertIs(kwargs.get('websocket'), mock_ws)
                self.assertIs(kwargs.get('rds'), mock_rds)
                self.assertIs(kwargs.get('settings'), routers_mod.settings)
                self.assertIs(
                    kwargs.get('runtime'),
                    routers_mod.get_inference_runtime(),
                )


if __name__ == '__main__':
    unittest.main()
