from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Callable
from typing import ClassVar
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from examples.YOLO_server_api.backend import routers as routers_mod
from examples.YOLO_server_api.backend.routers import custom_rate_limiter
from examples.YOLO_server_api.backend.routers import detection_router
from examples.YOLO_server_api.backend.routers import jwt_access
from examples.YOLO_server_api.backend.routers import model_loader
from examples.YOLO_server_api.backend.routers import model_management_router
from examples.YOLO_server_api.backend.routers import websocket_detect
"""Tests for FastAPI routers layer.

This module exercises the thin routing layer in
``examples.YOLO_server_api.backend.routers``. It focuses on:

- REST endpoints happy paths and error handling.
- WebSocket endpoint delegating to the handler function.

The business logic is mocked; these tests verify HTTP contract,
authorisation, and delegation behaviour.
"""


class TestRouters(unittest.IsolatedAsyncioTestCase):
    """
    Unit tests for ``routers.py`` focused on routing and delegation.
    """

    app: ClassVar[FastAPI]
    client: ClassVar[TestClient]

    @classmethod
    def setUpClass(cls) -> None:
        """
        Initialise shared FastAPI app and client for the test suite.
        """
        cls.app = FastAPI()
        # Mount routers under /api to match tests
        cls.app.include_router(detection_router, prefix='/api')
        cls.app.include_router(model_management_router, prefix='/api')
        # Default dependency overrides
        cls.app.dependency_overrides[custom_rate_limiter] = lambda: 999
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

        def _override() -> SimpleNamespace:
            return SimpleNamespace(subject={'role': role, 'user': 'tester'})

        return _override

    @patch.object(model_loader, 'get_model')
    @patch(
        'examples.YOLO_server_api.backend.routers.run_detection_from_bytes',
        new_callable=AsyncMock,
    )
    def test_detect_endpoint_success(
        self,
        mock_run_det: AsyncMock,
        mock_get_model: MagicMock,
    ) -> None:
        """Verify POST /api/detect returns detections on success.

        Args:
            mock_run_det: Mocked async detection function.
            mock_get_model: Mocked model loader function.
        """
        # Prepare mocks
        mock_get_model.return_value = Mock()
        mock_run_det.return_value = (
            [[1, 2, 3, 4, 0.9, 0]],
            {'inference': 0.01, 'post': 0.005},
        )
        # Issue request with image file and model name
        files = {'image': ('test.jpg', b'123', 'image/jpeg')}
        data = {'model': 'yolo11n'}
        # Exercise endpoint
        resp = self.client.post('/api/detect', files=files, data=data)
        # Verify response and delegation
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), [[1, 2, 3, 4, 0.9, 0]])
        mock_run_det.assert_awaited_once()

    @patch.object(model_loader, 'get_model', return_value=None)
    def test_detect_endpoint_model_not_found(self, _: MagicMock) -> None:
        """Verify 404 is returned when the named model cannot be found."""
        files = {'image': ('test.jpg', b'123', 'image/jpeg')}
        data = {'model': 'nope'}
        resp = self.client.post('/api/detect', files=files, data=data)
        self.assertEqual(resp.status_code, 404)
        self.assertIn('Model not found', resp.text)

    @patch(
        'examples.YOLO_server_api.backend.routers.update_model_file',
        new_callable=AsyncMock,
    )
    @patch('examples.YOLO_server_api.backend.routers.logger')
    def test_model_file_update_success(
        self,
        mock_logger: MagicMock,
        mock_update_file: AsyncMock,
    ) -> None:
        """Verify POST /api/model_file_update accepts admin role and logs.

        Args:
            mock_logger: Mocked logger instance.
            mock_update_file: Mocked update model file function.
        """
        # Ensure privileged role
        self.app.dependency_overrides[jwt_access] = (
            self._override_jwt_role_factory('admin')
        )
        # Prepare upload payload
        files = {
            'file': (
                'model.pt', b'model content', 'application/octet-stream',
            ),
        }
        data = {'model': 'yolo11n'}
        # Exercise endpoint
        resp = self.client.post(
            '/api/model_file_update', data=data, files=files,
        )
        # Verify outcome and side effects
        self.assertEqual(resp.status_code, 200)
        self.assertIn('updated successfully', resp.text)
        mock_update_file.assert_awaited()
        mock_logger.info.assert_called()

    def test_model_file_update_forbidden_role(self) -> None:
        """Verify non-privileged role gets 403 for model update."""
        # Downgrade role to user
        self.app.dependency_overrides[jwt_access] = (
            self._override_jwt_role_factory('user')
        )
        files = {
            'file': (
                'model.pt', b'model content', 'application/octet-stream',
            ),
        }
        data = {'model': 'yolo11n'}
        resp = self.client.post(
            '/api/model_file_update', data=data, files=files,
        )
        self.assertEqual(resp.status_code, 403)
        self.assertIn("Need 'admin' or 'model_manage' role", resp.text)
        # revert
        self.app.dependency_overrides[jwt_access] = (
            self._override_jwt_role_factory('admin')
        )

    @patch(
        'examples.YOLO_server_api.backend.routers.update_model_file',
        side_effect=ValueError('Invalid model'),
    )
    @patch('examples.YOLO_server_api.backend.routers.logger')
    def test_model_file_update_value_error(
        self, mock_logger: MagicMock, _patch_update: MagicMock,
    ) -> None:
        """Verify ValueError becomes 400 with an error log entry.

        Args:
            mock_logger: Mocked logger instance.
            _patch_update: Mocked update model file function.
        """
        files = {
            'file': (
                'bad.pt', b'xxx', 'application/octet-stream',
            ),
        }
        data = {'model': 'bad'}
        resp = self.client.post(
            '/api/model_file_update', data=data, files=files,
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn('Invalid model', resp.text)
        mock_logger.error.assert_called()

    @patch(
        'examples.YOLO_server_api.backend.routers.update_model_file',
        side_effect=OSError('Disk error'),
    )
    @patch('examples.YOLO_server_api.backend.routers.logger')
    def test_model_file_update_os_error(
        self, mock_logger: MagicMock, _patch_update: MagicMock,
    ) -> None:
        """Verify OSError becomes 500 with an error log entry.

        Args:
            mock_logger: Mocked logger instance.
            _patch_update: Mocked update model file function.
        """
        files = {
            'file': (
                'model.pt', b'xxx', 'application/octet-stream',
            ),
        }
        data = {'model': 'yolo11n'}
        resp = self.client.post(
            '/api/model_file_update', data=data, files=files,
        )
        self.assertEqual(resp.status_code, 500)
        self.assertIn('Disk error', resp.text)
        mock_logger.error.assert_called()

    @patch(
        'examples.YOLO_server_api.backend.routers.get_new_model_file',
        new_callable=AsyncMock,
    )
    @patch('examples.YOLO_server_api.backend.routers.logger')
    def test_get_new_model_updated(
        self, mock_logger: MagicMock, mock_get_file: AsyncMock,
    ) -> None:
        """Verify updated model returns base64 content and logs info."""
        # Ensure privileged role
        self.app.dependency_overrides[jwt_access] = (
            self._override_jwt_role_factory('admin')
        )
        mock_get_file.return_value = b'data'
        payload = {
            'model': 'yolo11n',
            'last_update_time': '2023-10-01T12:30:00',
        }
        resp = self.client.post('/api/get_new_model', json=payload)
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body['message'], 'Model yolo11n is updated.')
        self.assertEqual(body['model_file'], 'ZGF0YQ==')  # base64 of b'data'
        mock_logger.info.assert_called()

    @patch(
        'examples.YOLO_server_api.backend.routers.get_new_model_file',
        new_callable=AsyncMock,
    )
    @patch('examples.YOLO_server_api.backend.routers.logger')
    def test_get_new_model_up_to_date(
        self, _mock_logger: MagicMock, mock_get_file: AsyncMock,
    ) -> None:
        """Verify up-to-date model returns a simple message.

        Args:
            _mock_logger: Mocked logger instance.
            mock_get_file: Mocked get new model file function.
        """
        mock_get_file.return_value = None
        payload = {
            'model': 'yolo11n',
            'last_update_time': '2023-10-01T12:30:00',
        }
        resp = self.client.post('/api/get_new_model', json=payload)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(
            resp.json(), {'message': 'Model yolo11n is up to date.'},
        )

    def test_get_new_model_invalid_datetime(self) -> None:
        """Verify invalid datetime yields 400 Bad Request."""
        payload = {'model': 'yolo11n', 'last_update_time': 'invalid_datetime'}
        resp = self.client.post('/api/get_new_model', json=payload)
        self.assertEqual(resp.status_code, 400)

    @patch(
        'examples.YOLO_server_api.backend.routers.get_new_model_file',
        side_effect=Exception('Some error'),
    )
    @patch('examples.YOLO_server_api.backend.routers.logger')
    def test_get_new_model_exception(
        self, mock_logger: MagicMock, _patch_file: MagicMock,
    ) -> None:
        """Verify unexpected exception yields 500 and is logged.

        Args:
            mock_logger: Mocked logger instance.
            _patch_file: Mocked get new model file function.
        """
        payload = {
            'model': 'yolo11n',
            'last_update_time': '2023-10-01T12:30:00',
        }
        resp = self.client.post('/api/get_new_model', json=payload)
        self.assertEqual(resp.status_code, 500)
        self.assertIn('Failed to retrieve model.', resp.text)
        mock_logger.error.assert_called()

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
            'model': 'yolo11n',
            'last_update_time': '2023-10-01T12:30:00',
        }
        resp = self.client.post('/api/get_new_model', json=payload)
        self.assertEqual(resp.status_code, 403)
        self.assertIn("Need 'admin' or 'model_manage' role", resp.text)
        # revert
        self.app.dependency_overrides[jwt_access] = (
            self._override_jwt_role_factory('admin')
        )

    async def test_websocket_route_delegates_to_handler(self) -> None:
        """
        Verify WebSocket route delegates to ``handle_websocket_detect``.
        """
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
                    kwargs.get('model_loader'),
                    routers_mod.model_loader,
                )


if __name__ == '__main__':
    unittest.main()

'''
pytest \
    --cov=examples.YOLO_server_api.backend.routers \
    --cov-report=term-missing \
    tests/examples/YOLO_server_api/backend/routers_test.py
'''
