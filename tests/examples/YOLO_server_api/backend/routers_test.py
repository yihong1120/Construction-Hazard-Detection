from __future__ import annotations

import base64
import datetime
import unittest
from typing import ClassVar
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

from fastapi import FastAPI
from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient
from fastapi_jwt import JwtAuthorizationCredentials
from jwt import InvalidTokenError

from examples.YOLO_server_api.backend.routers import custom_rate_limiter
from examples.YOLO_server_api.backend.routers import detection_router
from examples.YOLO_server_api.backend.routers import jwt_access
from examples.YOLO_server_api.backend.routers import model_loader
from examples.YOLO_server_api.backend.routers import model_management_router
from examples.YOLO_server_api.backend.routers import websocket_detect


class TestRouters(unittest.IsolatedAsyncioTestCase):
    """
    Tests the detection and model management endpoints in routers.py,
    ensuring 100% coverage of success, error, and role-based conditions.
    """
    app: ClassVar[FastAPI]
    client: ClassVar[TestClient]

    @classmethod
    def setUpClass(cls) -> None:
        """
        Sets up the FastAPI application and test client for testing.
        """
        cls.app = FastAPI()

        # Include the detection and model management routers under /api prefix
        cls.app.include_router(detection_router, prefix='/api')
        cls.app.include_router(model_management_router, prefix='/api')

        def override_jwt_access() -> JwtAuthorizationCredentials:
            """
            Provides default 'admin' role in JWT credentials.
            """
            return JwtAuthorizationCredentials(
                subject={
                    'username': 'test_admin',
                    'role': 'admin',
                    'jti': 'some_jti',
                },
            )

        def override_custom_rate_limiter() -> int:
            """
            Returns a fixed integer for rate-limiting checks.
            """
            return 10

        # Override the original dependencies
        cls.app.dependency_overrides[jwt_access] = override_jwt_access
        cls.app.dependency_overrides[custom_rate_limiter] = (
            override_custom_rate_limiter
        )

        # Create a TestClient for making HTTP requests
        cls.client = TestClient(cls.app)

    def override_jwt_role(self, role: str):
        """
        Returns a function to override jwt_access with a specific role.

        Args:
            role (str): The user role to inject into JWT credentials.

        Returns:
            function: A function that returns JwtAuthorizationCredentials
                with the specified role.
        """
        def _override() -> JwtAuthorizationCredentials:
            return JwtAuthorizationCredentials(
                subject={
                    'username': 'test_user',
                    'role': role,
                    'jti': 'some_jti',
                },
            )
        return _override

    # ------------------------------------------------------------------------
    # TEST: /api/detect
    # ------------------------------------------------------------------------
    @patch.object(model_loader, 'get_model')
    @patch(
        'examples.YOLO_server_api.backend.routers.convert_to_image',
        new_callable=AsyncMock,
    )
    @patch(
        'examples.YOLO_server_api.backend.routers.get_prediction_result',
        new_callable=AsyncMock,
    )
    @patch('examples.YOLO_server_api.backend.routers.compile_detection_data')
    @patch(
        'examples.YOLO_server_api.backend.routers.process_labels',
        new_callable=AsyncMock,
    )
    def test_detect_endpoint_success(
        self,
        mock_process_labels: AsyncMock,
        mock_compile_detection_data: MagicMock,
        mock_get_prediction_result: AsyncMock,
        mock_convert_to_image: AsyncMock,
        mock_get_model: MagicMock,
    ) -> None:
        """
        Verifies /api/detect returns 200 and the detection results
        when a valid model and image are provided.

        Args:
            mock_process_labels (AsyncMock):
                Mock for the process_labels function to simulate processing.
            mock_compile_detection_data (MagicMock):
                Mock for the compile_detection_data function to simulate
                data compilation.
            mock_get_prediction_result (AsyncMock):
                Mock for the get_prediction_result function to simulate
                prediction results.
            mock_convert_to_image (AsyncMock):
                Mock for the convert_to_image function to simulate image
                conversion.
            mock_get_model (MagicMock):
                Mock for the get_model function to simulate model retrieval.
        """
        mock_get_model.return_value = 'mock_model_instance'
        mock_convert_to_image.return_value = 'mock_image'
        mock_get_prediction_result.return_value = 'mock_result'
        mock_compile_detection_data.return_value = [[1.0, 2.0, 3.0, 4.0]]
        mock_process_labels.return_value = [[1.0, 2.0, 3.0, 4.0]]

        files = {'image': ('test.jpg', b'fake_image_data', 'image/jpeg')}
        data = {'model': 'yolo11n'}

        resp = self.client.post('/api/detect', data=data, files=files)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), [[1.0, 2.0, 3.0, 4.0]])

    @patch.object(model_loader, 'get_model', return_value=None)
    def test_detect_endpoint_model_not_found(self, _) -> None:
        """
        Verifies /api/detect returns 404 if the specified model is not found.
        """
        files = {'image': ('test.jpg', b'fake_image_data', 'image/jpeg')}
        data = {'model': 'unknown_model'}

        resp = self.client.post('/api/detect', data=data, files=files)
        self.assertEqual(resp.status_code, 404)
        self.assertIn('Model not found', resp.text)

    # ------------------------------------------------------------------------
    # TEST: /api/model_file_update
    # ------------------------------------------------------------------------
    @patch(
        'examples.YOLO_server_api.backend.routers.update_model_file',
        new_callable=AsyncMock,
    )
    @patch('examples.YOLO_server_api.backend.routers.logger')
    def test_model_file_update_success(
        self,
        mock_logger: MagicMock,
        mock_update_func: AsyncMock,
    ) -> None:
        """
        Verifies /api/model_file_update returns 200 for an admin user
        and that the model is updated successfully.

        Args:
            mock_logger (MagicMock):
                Mock for the logger to capture info messages.
            mock_update_func (AsyncMock):
                Mock for the update_model_file function to simulate success.
        """
        mock_update_func.return_value = None  # simulate success

        files = {
            'file': ('model.pt', b'model content', 'application/octet-stream'),
        }
        data = {'model': 'yolo11n'}

        resp = self.client.post(
            '/api/model_file_update', data=data, files=files,
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn('yolo11n updated successfully', resp.text)
        mock_logger.info.assert_any_call('Model yolo11n updated successfully.')

    def test_model_file_update_forbidden_role(self) -> None:
        """
        Verifies /api/model_file_update => 403 if the role is neither
        'admin' nor 'model_manage'.
        """
        self.app.dependency_overrides[jwt_access] = self.override_jwt_role(
            'user',
        )

        files = {
            'file': (
                'model.pt', b'model content',
                'application/octet-stream',
            ),
        }
        data = {'model': 'yolo11n'}

        resp = self.client.post(
            '/api/model_file_update', data=data, files=files,
        )
        self.assertEqual(resp.status_code, 403)
        self.assertIn("Need 'admin' or 'model_manage' role", resp.text)

        # revert to admin
        self.app.dependency_overrides[jwt_access] = self.override_jwt_role(
            'admin',
        )

    @patch(
        'examples.YOLO_server_api.backend.routers.update_model_file',
        side_effect=ValueError('Invalid model'),
    )
    @patch('examples.YOLO_server_api.backend.routers.logger')
    def test_model_file_update_value_error(
        self,
        mock_logger: MagicMock, _,
    ) -> None:
        """
        Verifies /api/model_file_update => 400 if a ValueError occurs
        during model update.

        Args:
            mock_logger (MagicMock):
                Mock for the logger to capture error messages.
        """
        files = {
            'file': (
                'model.pt', b'model content',
                'application/octet-stream',
            ),
        }
        data = {'model': 'yolo11n'}

        resp = self.client.post(
            '/api/model_file_update', data=data, files=files,
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn('Invalid model', resp.text)
        mock_logger.error.assert_any_call(
            'Model update validation error: Invalid model',
        )

    @patch(
        'examples.YOLO_server_api.backend.routers.update_model_file',
        side_effect=OSError('Disk error'),
    )
    @patch('examples.YOLO_server_api.backend.routers.logger')
    def test_model_file_update_os_error(
        self, mock_logger: MagicMock,
        mock_update_func: MagicMock,
    ) -> None:
        """
        Verifies /api/model_file_update => 500 if an OSError occurs
        during model update.

        Args:
            mock_logger (MagicMock):
                Mock for the logger to capture error messages.
            mock_update_func (MagicMock):
                Mock for the update_model_file function to simulate an OSError.
        """
        files = {
            'file': (
                'model.pt', b'model content',
                'application/octet-stream',
            ),
        }
        data = {'model': 'yolo11n'}

        resp = self.client.post(
            '/api/model_file_update', data=data, files=files,
        )
        self.assertEqual(resp.status_code, 500)
        self.assertIn('Disk error', resp.text)
        mock_logger.error.assert_any_call(
            'Model update I/O error: Disk error',
        )

    # ------------------------------------------------------------------------
    # TEST: /api/get_new_model
    # ------------------------------------------------------------------------
    @patch(
        'examples.YOLO_server_api.backend.routers.get_new_model_file',
        new_callable=AsyncMock,
    )
    @patch('examples.YOLO_server_api.backend.routers.logger')
    def test_get_new_model_updated(
        self,
        mock_logger: MagicMock,
        mock_get_file: AsyncMock,
    ) -> None:
        """
        Verifies /api/get_new_model => 200 if a newer model file is available,
        returning a base64-encoded model_file.

        Args:
            mock_logger (MagicMock):
                Mock for the logger to capture info messages.
            mock_get_file (AsyncMock):
                Mock for the get_new_model_file function to
                simulate a new file.
        """
        mock_get_file.return_value = b'new_model_content'

        payload = {
            'model': 'yolo11n',
            'last_update_time': datetime.datetime.now().isoformat(),
        }
        resp = self.client.post('/api/get_new_model', json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn('Model yolo11n is updated.', data['message'])
        decoded = base64.b64decode(data['model_file'].encode())
        self.assertEqual(decoded, b'new_model_content')

        mock_logger.info.assert_any_call(
            'Newer model file for yolo11n retrieved.',
        )

    @patch(
        'examples.YOLO_server_api.backend.routers.get_new_model_file',
        new_callable=AsyncMock,
    )
    @patch('examples.YOLO_server_api.backend.routers.logger')
    def test_get_new_model_up_to_date(
        self,
        mock_logger: MagicMock,
        mock_get_file: AsyncMock,
    ) -> None:
        """
        Verifies /api/get_new_model => 200 (no new file) if the server's model
        is not newer than the client's.

        Args:
            mock_logger (MagicMock):
                Mock for the logger to capture info messages.
            mock_get_file (AsyncMock):
                Mock for the get_new_model_file function to simulate no update.
        """
        mock_get_file.return_value = None

        payload = {
            'model': 'yolo11n',
            'last_update_time': datetime.datetime.now().isoformat(),
        }
        resp = self.client.post('/api/get_new_model', json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn('up to date', data['message'])

        # When content is None, no logger.info call is made

    def test_get_new_model_invalid_datetime(self) -> None:
        """
        Verifies /api/get_new_model => 400 if `last_update_time` is invalid.
        """
        payload = {
            'model': 'yolo11n',
            'last_update_time': 'invalid_datetime',
        }
        resp = self.client.post('/api/get_new_model', json=payload)
        self.assertEqual(resp.status_code, 400)

    @patch(
        'examples.YOLO_server_api.backend.routers.get_new_model_file',
        side_effect=Exception('Some error'),
    )
    @patch('examples.YOLO_server_api.backend.routers.logger')
    def test_get_new_model_exception(self, mock_logger: MagicMock, _) -> None:
        """
        Verifies /api/get_new_model => 500 if an unexpected exception occurs
        during file retrieval.

        Args:
            mock_logger (MagicMock):
                Mock for the logger to capture error messages.
        """
        payload = {
            'model': 'yolo11n',
            'last_update_time': datetime.datetime.now().isoformat(),
        }
        resp = self.client.post('/api/get_new_model', json=payload)
        self.assertEqual(resp.status_code, 500)
        self.assertIn('Failed to retrieve model', resp.text)
        mock_logger.error.assert_any_call('Error retrieving model: Some error')

    def test_get_new_model_forbidden_guest(self) -> None:
        """
        Verifies /api/get_new_model => 403 if the user role is 'guest'.
        This triggers the 'guest' in the endpoints' role check.
        """
        self.app.dependency_overrides[jwt_access] = self.override_jwt_role(
            'guest',
        )

        payload = {
            'model': 'yolo11n',
            'last_update_time': '2023-10-01T12:30:00',
        }
        resp = self.client.post('/api/get_new_model', json=payload)
        self.assertEqual(resp.status_code, 403)
        self.assertIn("Need 'admin' or 'model_manage' role", resp.text)

        # Revert role to admin to avoid affecting other tests
        self.app.dependency_overrides[jwt_access] = self.override_jwt_role(
            'admin',
        )

    # ------------------------------------------------------------------------
    # TEST: WebSocket endpoint /ws/detect - Integration Tests
    # ------------------------------------------------------------------------
    async def test_websocket_no_client_header_integration(self) -> None:
        """
        Integration test for WebSocket with missing client info
        (simulates unknown client).
        """
        # This tests the line: client_ip = websocket.client.host
        # if websocket.client else "unknown"
        mock_websocket = AsyncMock()
        mock_websocket.client = None  # Simulate missing client
        mock_websocket.accept = AsyncMock()
        mock_websocket.close = AsyncMock()
        # Normal dict, not async
        mock_websocket.headers = {'authorization': None}
        mock_websocket.query_params = {}

        await websocket_detect(mock_websocket, Mock())

        # Should close due to missing token
        mock_websocket.close.assert_called_with(
            code=1008, reason='Missing authentication token',
        )

    async def test_websocket_token_from_header_integration(self) -> None:
        """
        Integration test for WebSocket token extraction from header.
        """
        mock_websocket = AsyncMock()
        mock_websocket.client = Mock()
        mock_websocket.client.host = '192.168.1.1'
        mock_websocket.accept = AsyncMock()
        mock_websocket.close = AsyncMock()
        mock_websocket.headers = {'authorization': 'Bearer test_token'}

        # Mock JWT decode to fail with InvalidTokenError
        with patch(
            'examples.YOLO_server_api.backend.routers.jwt.decode',
            side_effect=InvalidTokenError('Invalid token'),
        ):

            await websocket_detect(mock_websocket, Mock())

        # Should close due to invalid token
        mock_websocket.close.assert_called_with(
            code=1008, reason='Invalid token',
        )

    async def test_websocket_token_from_query_integration(self) -> None:
        """
        Integration test for WebSocket token extraction from query parameter.
        """
        mock_websocket = AsyncMock()
        mock_websocket.client = Mock()
        mock_websocket.client.host = '192.168.1.1'
        mock_websocket.accept = AsyncMock()
        mock_websocket.close = AsyncMock()
        mock_websocket.headers = {'authorization': None}  # No header token
        mock_websocket.query_params = {'token': 'query_token'}

        # Mock JWT decode to fail with InvalidTokenError
        with patch(
            'examples.YOLO_server_api.backend.routers.jwt.decode',
            side_effect=InvalidTokenError('Invalid token'),
        ):

            await websocket_detect(mock_websocket, Mock())

        # Should close due to invalid token
        mock_websocket.close.assert_called_with(
            code=1008, reason='Invalid token',
        )

    async def test_websocket_empty_payload_integration(self) -> None:
        """
        Integration test for WebSocket with empty JWT payload.
        """
        mock_websocket = AsyncMock()
        mock_websocket.client = Mock()
        mock_websocket.client.host = '192.168.1.1'
        mock_websocket.accept = AsyncMock()
        mock_websocket.close = AsyncMock()
        mock_websocket.headers = {'authorization': 'Bearer valid_token'}

        with (
            patch(
                'examples.YOLO_server_api.backend.routers.jwt.decode',
                return_value={},
            ),
            patch('examples.YOLO_server_api.backend.routers.settings'),
        ):
            await websocket_detect(mock_websocket, Mock())

        mock_websocket.close.assert_called_with(
            code=1008, reason='Empty token payload',
        )

    async def test_websocket_missing_username_jti_integration(self) -> None:
        """
        Integration test for WebSocket with missing username/JTI.
        """
        mock_websocket = AsyncMock()
        mock_websocket.client = Mock()
        mock_websocket.client.host = '192.168.1.1'
        mock_websocket.accept = AsyncMock()
        mock_websocket.close = AsyncMock()
        mock_websocket.headers = {'authorization': 'Bearer valid_token'}

        payload: dict[str, object] = {  # Missing username and jti
            'subject': {},
        }

        with (
            patch(
                'examples.YOLO_server_api.backend.routers.jwt.decode',
                return_value=payload,
            ),
            patch('examples.YOLO_server_api.backend.routers.settings'),
        ):
            await websocket_detect(mock_websocket, Mock())

        mock_websocket.close.assert_called_with(
            code=1008, reason='Invalid token data',
        )

    async def test_websocket_jti_not_active_integration(self) -> None:
        """
        Integration test for WebSocket with inactive JTI.
        """
        mock_websocket = AsyncMock()
        mock_websocket.client = Mock()
        mock_websocket.client.host = '192.168.1.1'
        mock_websocket.accept = AsyncMock()
        mock_websocket.close = AsyncMock()
        mock_websocket.headers = {'authorization': 'Bearer valid_token'}

        payload = {'subject': {'username': 'testuser', 'jti': 'test_jti'}}

        with (
            patch(
                'examples.YOLO_server_api.backend.routers.jwt.decode',
                return_value=payload,
            ),
            patch('examples.YOLO_server_api.backend.routers.settings'),
            patch(
                'examples.YOLO_server_api.backend.routers.get_user_data',
                return_value={'jti_list': ['other_jti']},
            ),
        ):
            await websocket_detect(mock_websocket, Mock())

        mock_websocket.close.assert_called_with(
            code=1008, reason='Token not active',
        )

    async def test_websocket_model_not_found_integration(self) -> None:
        """
        Integration test for WebSocket with model not found.
        """
        mock_websocket = AsyncMock()
        mock_websocket.client = Mock()
        mock_websocket.client.host = '192.168.1.1'
        mock_websocket.accept = AsyncMock()
        mock_websocket.close = AsyncMock()
        mock_websocket.headers = {'authorization': 'Bearer valid_token'}
        mock_websocket.query_params = {'model': 'unknown_model'}

        payload = {'subject': {'username': 'testuser', 'jti': 'test_jti'}}

        with (
            patch(
                'examples.YOLO_server_api.backend.routers.jwt.decode',
                return_value=payload,
            ),
            patch('examples.YOLO_server_api.backend.routers.settings'),
            patch(
                'examples.YOLO_server_api.backend.routers.get_user_data',
                return_value={'jti_list': ['test_jti']},
            ),
            patch.object(model_loader, 'get_model', return_value=None),
        ):
            await websocket_detect(mock_websocket, Mock())

        mock_websocket.close.assert_called_with(
            code=1003, reason='Model not found',
        )

    async def test_websocket_successful_connection_and_cleanup(self) -> None:
        """
        Integration test for successful WebSocket connection
        and cleanup on disconnect.
        """
        mock_websocket = AsyncMock()
        mock_websocket.client = Mock()
        mock_websocket.client.host = '192.168.1.1'
        mock_websocket.accept = AsyncMock()
        mock_websocket.close = AsyncMock()
        mock_websocket.headers = {'authorization': 'Bearer valid_token'}
        mock_websocket.query_params = {'model': 'yolo_model'}

        payload = {'subject': {'username': 'testuser', 'jti': 'test_jti'}}
        mock_model = Mock()

        # Create mock boxes without using cls as parameter name
        mock_boxes = Mock()
        mock_boxes.xyxy = [[0, 0, 10, 10]]
        mock_boxes.cls = [0]  # Set as attribute after creation
        mock_boxes.conf = [0.9]

        mock_result = [Mock()]
        mock_result[0].boxes = mock_boxes

        # Mock the _safe_websocket_receive_bytes to return data once, then None
        with (
            patch(
                'examples.YOLO_server_api.backend.routers.jwt.decode',
                return_value=payload,
            ),
            patch('examples.YOLO_server_api.backend.routers.settings'),
            patch(
                'examples.YOLO_server_api.backend.routers.get_user_data',
                return_value={'jti_list': ['test_jti']},
            ),
            patch.object(model_loader, 'get_model', return_value=mock_model),
            patch(
                'examples.YOLO_server_api.backend.'
                'routers._safe_websocket_receive_bytes',
                side_effect=[b'fake_image_data', None],
            ),
            patch(
                'examples.YOLO_server_api.backend.'
                'routers._safe_websocket_send_json',
                return_value=True,
            ),
            patch(
                'examples.YOLO_server_api.backend.routers.convert_to_image',
                return_value=Mock(),
            ),
            patch(
                'examples.YOLO_server_api.backend.'
                'routers.get_prediction_result',
                new_callable=AsyncMock, return_value=mock_result,
            ),
            patch(
                'examples.YOLO_server_api.backend.'
                'routers.compile_detection_data',
                return_value=[],
            ),
            patch(
                'examples.YOLO_server_api.backend.routers.process_labels',
                new_callable=AsyncMock, return_value=[],
            ),
        ):
            # This will test the main loop and cleanup
            await websocket_detect(mock_websocket, Mock())

        # Verify connection was accepted
        mock_websocket.accept.assert_called_once()

    async def test_websocket_prediction_and_send_error(self) -> None:
        """
        Integration test for WebSocket prediction with send error handling.
        """
        mock_websocket = AsyncMock()
        mock_websocket.client = Mock()
        mock_websocket.client.host = '192.168.1.1'
        mock_websocket.accept = AsyncMock()
        mock_websocket.close = AsyncMock()
        mock_websocket.headers = {'authorization': 'Bearer valid_token'}
        mock_websocket.query_params = {'model': 'yolo_model'}

        payload = {'subject': {'username': 'testuser', 'jti': 'test_jti'}}
        mock_model = Mock()

        # Create mock boxes without using cls as parameter name
        mock_boxes = Mock()
        mock_boxes.xyxy = [[0, 0, 10, 10]]
        mock_boxes.cls = [0]  # Set as attribute after creation
        mock_boxes.conf = [0.9]

        mock_result = [Mock()]
        mock_result[0].boxes = mock_boxes

        with (
            patch(
                'examples.YOLO_server_api.backend.routers.jwt.decode',
                return_value=payload,
            ),
            patch('examples.YOLO_server_api.backend.routers.settings'),
            patch(
                'examples.YOLO_server_api.backend.routers.get_user_data',
                return_value={'jti_list': ['test_jti']},
            ),
            patch.object(model_loader, 'get_model', return_value=mock_model),
            patch(
                'examples.YOLO_server_api.backend.'
                'routers._safe_websocket_receive_bytes',
                side_effect=[b'fake_image_data', None],
            ),
            patch(
                'examples.YOLO_server_api.backend.'
                'routers._safe_websocket_send_json',
                side_effect=[True, False],
            ),
            patch(
                'examples.YOLO_server_api.backend.routers.convert_to_image',
                return_value=Mock(),
            ),
            patch(
                'examples.YOLO_server_api.backend.'
                'routers.get_prediction_result',
                new_callable=AsyncMock, return_value=mock_result,
            ),
            patch(
                'examples.YOLO_server_api.backend.'
                'routers.compile_detection_data',
                return_value=[],
            ),
            patch(
                'examples.YOLO_server_api.backend.routers.process_labels',
                new_callable=AsyncMock, return_value=[],
            ),
        ):
            await websocket_detect(mock_websocket, Mock())

        # Verify connection was accepted (prediction is internal detail)
        mock_websocket.accept.assert_called_once()

    async def test_websocket_model_key_from_header(self) -> None:
        """
        Integration test for WebSocket with model key from header.
        """
        mock_websocket = AsyncMock()
        mock_websocket.client = Mock()
        mock_websocket.client.host = '192.168.1.1'
        mock_websocket.accept = AsyncMock()
        mock_websocket.close = AsyncMock()
        mock_websocket.headers = {
            'authorization': 'Bearer valid_token',
            'x-model-key': 'header_model',
        }
        mock_websocket.query_params = {}

        payload = {'subject': {'username': 'testuser', 'jti': 'test_jti'}}

        with (
            patch(
                'examples.YOLO_server_api.backend.routers.jwt.decode',
                return_value=payload,
            ),
            patch('examples.YOLO_server_api.backend.routers.settings'),
            patch(
                'examples.YOLO_server_api.backend.routers.get_user_data',
                return_value={'jti_list': ['test_jti']},
            ),
            patch.object(model_loader, 'get_model', return_value=None),
        ):
            await websocket_detect(mock_websocket, Mock())

        mock_websocket.close.assert_called_with(
            code=1003, reason='Model not found',
        )

    async def test_websocket_model_key_from_first_message_valid(self) -> None:
        """
        Integration test for WebSocket with model key from first message.
        """
        mock_websocket = AsyncMock()
        mock_websocket.client = Mock()
        mock_websocket.client.host = '192.168.1.1'
        mock_websocket.accept = AsyncMock()
        mock_websocket.close = AsyncMock()
        mock_websocket.headers = {'authorization': 'Bearer valid_token'}
        mock_websocket.query_params = {}
        mock_websocket.receive_text.return_value = (
            '{"model_key": "message_model"}'
        )

        payload = {'subject': {'username': 'testuser', 'jti': 'test_jti'}}

        with (
            patch(
                'examples.YOLO_server_api.backend.routers.jwt.decode',
                return_value=payload,
            ),
            patch('examples.YOLO_server_api.backend.routers.settings'),
            patch(
                'examples.YOLO_server_api.backend.routers.get_user_data',
                return_value={'jti_list': ['test_jti']},
            ),
            patch.object(model_loader, 'get_model', return_value=None),
        ):
            await websocket_detect(mock_websocket, Mock())

        mock_websocket.close.assert_called_with(
            code=1003, reason='Model not found',
        )

    async def test_websocket_model_key_from_first_message_missing(
            self,
    ) -> None:
        """
        Integration test for WebSocket with missing model key in first message.
        """
        mock_websocket = AsyncMock()
        mock_websocket.client = Mock()
        mock_websocket.client.host = '192.168.1.1'
        mock_websocket.accept = AsyncMock()
        mock_websocket.close = AsyncMock()
        mock_websocket.headers = {'authorization': 'Bearer valid_token'}
        mock_websocket.query_params = {}
        mock_websocket.receive_text.return_value = '{"other_key": "value"}'

        payload = {'subject': {'username': 'testuser', 'jti': 'test_jti'}}

        with (
            patch(
                'examples.YOLO_server_api.backend.routers.jwt.decode',
                return_value=payload,
            ),
            patch('examples.YOLO_server_api.backend.routers.settings'),
            patch(
                'examples.YOLO_server_api.backend.routers.get_user_data',
                return_value={'jti_list': ['test_jti']},
            ),
        ):
            await websocket_detect(mock_websocket, Mock())

        mock_websocket.close.assert_called_with(
            code=1008, reason='Missing model_key in configuration',
        )

    async def test_websocket_model_key_from_first_message_invalid_json(
        self,
    ) -> None:
        """
        Integration test for WebSocket with invalid JSON in first message.
        """
        mock_websocket = AsyncMock()
        mock_websocket.client = Mock()
        mock_websocket.client.host = '192.168.1.1'
        mock_websocket.accept = AsyncMock()
        mock_websocket.close = AsyncMock()
        mock_websocket.headers = {'authorization': 'Bearer valid_token'}
        mock_websocket.query_params = {}
        mock_websocket.receive_text.return_value = 'invalid json'

        payload = {'subject': {'username': 'testuser', 'jti': 'test_jti'}}

        with (
            patch(
                'examples.YOLO_server_api.backend.routers.jwt.decode',
                return_value=payload,
            ),
            patch('examples.YOLO_server_api.backend.routers.settings'),
            patch(
                'examples.YOLO_server_api.backend.routers.get_user_data',
                return_value={'jti_list': ['test_jti']},
            ),
        ):
            await websocket_detect(mock_websocket, Mock())

        mock_websocket.close.assert_called_with(
            code=1008, reason='Invalid configuration message',
        )

    async def test_websocket_no_model_key_at_all(self) -> None:
        """
        Integration test for WebSocket with no model key provided anywhere.
        """
        mock_websocket = AsyncMock()
        mock_websocket.client = Mock()
        mock_websocket.client.host = '192.168.1.1'
        mock_websocket.accept = AsyncMock()
        mock_websocket.close = AsyncMock()
        mock_websocket.headers = {'authorization': 'Bearer valid_token'}
        mock_websocket.query_params = {}
        # Empty JSON, no model_key
        mock_websocket.receive_text.return_value = '{}'

        payload = {'subject': {'username': 'testuser', 'jti': 'test_jti'}}

        with (
            patch(
                'examples.YOLO_server_api.backend.routers.jwt.decode',
                return_value=payload,
            ),
            patch('examples.YOLO_server_api.backend.routers.settings'),
            patch(
                'examples.YOLO_server_api.backend.routers.get_user_data',
                return_value={'jti_list': ['test_jti']},
            ),
        ):
            await websocket_detect(mock_websocket, Mock())

        mock_websocket.close.assert_called_with(
            code=1008, reason='Missing model_key in configuration',
        )

    async def test_websocket_disconnect_exception(self) -> None:
        """
        Integration test for WebSocket with WebSocketDisconnect exception.
        """
        mock_websocket = AsyncMock()
        mock_websocket.client = Mock()
        mock_websocket.client.host = '192.168.1.1'
        mock_websocket.accept = AsyncMock()
        mock_websocket.close = AsyncMock()
        mock_websocket.headers = {'authorization': 'Bearer valid_token'}
        mock_websocket.query_params = {'model': 'yolo_model'}

        payload = {'subject': {'username': 'testuser', 'jti': 'test_jti'}}
        mock_model = Mock()

        with (
            patch(
                'examples.YOLO_server_api.backend.routers.jwt.decode',
                return_value=payload,
            ),
            patch('examples.YOLO_server_api.backend.routers.settings'),
            patch(
                'examples.YOLO_server_api.backend.routers.get_user_data',
                return_value={'jti_list': ['test_jti']},
            ),
            patch.object(model_loader, 'get_model', return_value=mock_model),
            patch(
                'examples.YOLO_server_api.backend.'
                'routers._safe_websocket_send_json',
                return_value=True,
            ),
            patch(
                'examples.YOLO_server_api.backend.'
                'routers._safe_websocket_receive_bytes',
                side_effect=WebSocketDisconnect,
            ),
        ):
            await websocket_detect(mock_websocket, Mock())

        # Should handle the disconnect gracefully
        mock_websocket.accept.assert_called_once()

    async def test_websocket_unexpected_exception(self) -> None:
        """
        Integration test for WebSocket with unexpected exception.
        """
        mock_websocket = AsyncMock()
        mock_websocket.client = Mock()
        mock_websocket.client.host = '192.168.1.1'
        mock_websocket.accept = AsyncMock()
        mock_websocket.close = AsyncMock()
        mock_websocket.headers = {'authorization': 'Bearer valid_token'}
        mock_websocket.query_params = {'model': 'yolo_model'}

        payload = {'subject': {'username': 'testuser', 'jti': 'test_jti'}}
        mock_model = Mock()

        with (
            patch(
                'examples.YOLO_server_api.backend.routers.jwt.decode',
                return_value=payload,
            ),
            patch('examples.YOLO_server_api.backend.routers.settings'),
            patch(
                'examples.YOLO_server_api.backend.routers.get_user_data',
                return_value={'jti_list': ['test_jti']},
            ),
            patch.object(model_loader, 'get_model', return_value=mock_model),
            patch(
                'examples.YOLO_server_api.backend.'
                'routers._safe_websocket_send_json',
                return_value=True,
            ),
            patch(
                'examples.YOLO_server_api.backend.'
                'routers._safe_websocket_receive_bytes',
                side_effect=RuntimeError('Unexpected error'),
            ),
        ):
            await websocket_detect(mock_websocket, Mock())

        # Should handle the unexpected error
        mock_websocket.accept.assert_called_once()
        mock_websocket.close.assert_called_with(
            code=1011, reason='Internal server error',
        )

    async def test_websocket_unexpected_exception_close_fails(self) -> None:
        """
        Integration test for WebSocket
        with unexpected exception and close failure.
        """
        mock_websocket = AsyncMock()
        mock_websocket.client = Mock()
        mock_websocket.client.host = '192.168.1.1'
        mock_websocket.accept = AsyncMock()
        mock_websocket.close = AsyncMock(
            side_effect=Exception('Close failed'),
        )  # Make close fail
        mock_websocket.headers = {'authorization': 'Bearer valid_token'}
        mock_websocket.query_params = {'model': 'yolo_model'}

        payload = {'subject': {'username': 'testuser', 'jti': 'test_jti'}}
        mock_model = Mock()

        with (
            patch(
                'examples.YOLO_server_api.backend.routers.jwt.decode',
                return_value=payload,
            ),
            patch('examples.YOLO_server_api.backend.routers.settings'),
            patch(
                'examples.YOLO_server_api.backend.routers.get_user_data',
                return_value={'jti_list': ['test_jti']},
            ),
            patch.object(model_loader, 'get_model', return_value=mock_model),
            patch(
                'examples.YOLO_server_api.backend.'
                'routers._safe_websocket_send_json',
                return_value=True,
            ),
            patch(
                'examples.YOLO_server_api.backend.'
                'routers._safe_websocket_receive_bytes',
                side_effect=RuntimeError('Unexpected error'),
            ),
        ):
            await websocket_detect(mock_websocket, Mock())

        # Handle the unexpected error and gracefully handle close failure
        mock_websocket.accept.assert_called_once()
        mock_websocket.close.assert_called_with(
            code=1011, reason='Internal server error',
        )

    async def test_websocket_configuration_send_failure(self) -> None:
        """
        Integration test for WebSocket where configuration response send fails.
        """
        mock_websocket = AsyncMock()
        mock_websocket.client = Mock()
        mock_websocket.client.host = '192.168.1.1'
        mock_websocket.accept = AsyncMock()
        mock_websocket.close = AsyncMock()
        mock_websocket.headers = {'authorization': 'Bearer valid_token'}
        mock_websocket.query_params = {'model': 'yolo_model'}

        payload = {'subject': {'username': 'testuser', 'jti': 'test_jti'}}
        mock_model = Mock()

        # Mock configuration send to fail
        with (
            patch(
                'examples.YOLO_server_api.backend.routers.jwt.decode',
                return_value=payload,
            ),
            patch('examples.YOLO_server_api.backend.routers.settings'),
            patch(
                'examples.YOLO_server_api.backend.routers.get_user_data',
                return_value={'jti_list': ['test_jti']},
            ),
            patch.object(model_loader, 'get_model', return_value=mock_model),
            patch(
                'examples.YOLO_server_api.backend.'
                'routers._safe_websocket_send_json',
                return_value=False,
            ),  # Fail
        ):
            await websocket_detect(mock_websocket, Mock())

        # Should return early due to failed configuration send (lines 272-273)
        mock_websocket.accept.assert_called_once()

    async def test_websocket_frame_count_logging(self) -> None:
        """
        Integration test for WebSocket frame count logging every 100 frames.
        """
        mock_websocket = AsyncMock()
        mock_websocket.client = Mock()
        mock_websocket.client.host = '192.168.1.1'
        mock_websocket.accept = AsyncMock()
        mock_websocket.close = AsyncMock()
        mock_websocket.headers = {'authorization': 'Bearer valid_token'}
        mock_websocket.query_params = {'model': 'yolo_model'}

        payload = {'subject': {'username': 'testuser', 'jti': 'test_jti'}}
        mock_model = Mock()

        # Create mock result
        mock_result = [Mock()]
        mock_result[0].boxes = Mock()
        mock_result[0].boxes.xyxy = [[0, 0, 10, 10]]
        mock_result[0].boxes.cls = [0]
        mock_result[0].boxes.conf = [0.9]

        # Simulate receiving 100 frames
        receive_calls = [b'fake_image_data'] * \
            100 + [None]  # 100 frames then stop

        with (
            patch(
                'examples.YOLO_server_api.backend.routers.jwt.decode',
                return_value=payload,
            ),
            patch('examples.YOLO_server_api.backend.routers.settings'),
            patch(
                'examples.YOLO_server_api.backend.routers.get_user_data',
                return_value={'jti_list': ['test_jti']},
            ),
            patch.object(model_loader, 'get_model', return_value=mock_model),
            patch(
                'examples.YOLO_server_api.backend.'
                'routers._safe_websocket_receive_bytes',
                side_effect=receive_calls,
            ),
            patch(
                'examples.YOLO_server_api.backend.'
                'routers._safe_websocket_send_json',
                return_value=True,
            ),
            patch(
                'examples.YOLO_server_api.backend.routers.convert_to_image',
                return_value=Mock(),
            ),
            patch(
                'examples.YOLO_server_api.backend.'
                'routers.get_prediction_result',
                new_callable=AsyncMock, return_value=mock_result,
            ),
            patch(
                'examples.YOLO_server_api.backend.'
                'routers.compile_detection_data',
                return_value=[],
            ),
            patch(
                'examples.YOLO_server_api.backend.routers.process_labels',
                new_callable=AsyncMock, return_value=[],
            ),
        ):
            await websocket_detect(mock_websocket, Mock())

        # Should have processed 100 frames and hit the logging line (line 307)
        mock_websocket.accept.assert_called_once()

    async def test_websocket_cv2_decode_failure(self) -> None:
        """
        Integration test for WebSocket with CV2 decode failure.
        """
        mock_websocket = AsyncMock()
        mock_websocket.client = Mock()
        mock_websocket.client.host = '192.168.1.1'
        mock_websocket.accept = AsyncMock()
        mock_websocket.close = AsyncMock()
        mock_websocket.headers = {'authorization': 'Bearer valid_token'}
        mock_websocket.query_params = {'model': 'yolo_model'}

        payload = {'subject': {'username': 'testuser', 'jti': 'test_jti'}}
        mock_model = Mock()

        # Setup patches for WebSocket CV2 decode failure test
        patches = [
            patch(
                'examples.YOLO_server_api.backend.routers.jwt.decode',
                return_value=payload,
            ),
            patch('examples.YOLO_server_api.backend.routers.settings'),
            patch(
                'examples.YOLO_server_api.backend.routers.get_user_data',
                return_value={'jti_list': ['test_jti']},
            ),
            patch.object(model_loader, 'get_model', return_value=mock_model),
            patch(
                'examples.YOLO_server_api.backend.'
                'routers._safe_websocket_receive_bytes',
                side_effect=[b'invalid_image_data', None],
            ),
            patch(
                'examples.YOLO_server_api.backend.'
                'routers._safe_websocket_send_json',
                return_value=True,
            ),
            patch(
                'examples.YOLO_server_api.backend.routers.convert_to_image',
                side_effect=Exception('Image decode failed'),
            ),
        ]

        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
        ):
            await websocket_detect(mock_websocket, Mock())

        # Verify no prediction was made due to image decode failure
        mock_model.predict.assert_not_called()

    # ------------------------------------------------------------------------
    # Additional Test Cases for Complete Coverage
    # ------------------------------------------------------------------------
    @patch(
        'examples.YOLO_server_api.backend.routers.update_model_file',
        new_callable=AsyncMock,
    )
    def test_model_file_update_model_manage_role(
        self, mock_update_func: AsyncMock,
    ) -> None:
        """
        Verifies /api/model_file_update works with 'model_manage' role.
        """
        # Override to use 'model_manage' role
        self.app.dependency_overrides[jwt_access] = self.override_jwt_role(
            'model_manage',
        )

        mock_update_func.return_value = None

        files = {
            'file': ('model.pt', b'model content', 'application/octet-stream'),
        }
        data = {'model': 'yolo11n'}

        resp = self.client.post(
            '/api/model_file_update', data=data, files=files,
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn('yolo11n updated successfully', resp.text)

        # Revert to admin
        self.app.dependency_overrides[jwt_access] = self.override_jwt_role(
            'admin',
        )

    @patch(
        'examples.YOLO_server_api.backend.routers.get_new_model_file',
        new_callable=AsyncMock,
    )
    def test_get_new_model_non_guest_role(
        self, mock_get_new_model_file: AsyncMock,
    ) -> None:
        """
        Verifies /api/get_new_model works with non-guest roles like 'user'.
        """
        # Override to use 'user' role (not guest)
        self.app.dependency_overrides[jwt_access] = self.override_jwt_role(
            'user',
        )

        mock_get_new_model_file.return_value = None  # Up to date

        payload = {
            'model': 'yolo11n',
            'last_update_time': '2023-10-01T12:30:00',
        }
        resp = self.client.post('/api/get_new_model', json=payload)
        self.assertEqual(resp.status_code, 200)
        self.assertIn('is up to date', resp.text)

        # Revert to admin
        self.app.dependency_overrides[jwt_access] = self.override_jwt_role(
            'admin',
        )

    def test_credentials_subject_access(self) -> None:
        """
        Tests that JWT credentials subject can be accessed properly.
        """
        files = {'image': ('test.jpg', b'fake_image_data', 'image/jpeg')}
        data = {'model': 'yolo11n'}

        patches = [
            patch.object(model_loader, 'get_model', return_value='mock_model'),
            patch(
                'examples.YOLO_server_api.backend.routers.convert_to_image',
                return_value='mock_image',
            ),
            patch(
                'examples.YOLO_server_api.backend.'
                'routers.get_prediction_result',
                new_callable=AsyncMock, return_value='mock_result',
            ),
            patch(
                'examples.YOLO_server_api.backend.'
                'routers.compile_detection_data',
                return_value=[[1.0, 2.0, 3.0, 4.0]],
            ),
            patch(
                'examples.YOLO_server_api.backend.routers.process_labels',
                new_callable=AsyncMock, return_value=[[1.0, 2.0, 3.0, 4.0]],
            ),
        ]
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            resp = self.client.post('/api/detect', data=data, files=files)
            self.assertEqual(resp.status_code, 200)

    @patch(
        'examples.YOLO_server_api.backend.routers.update_model_file',
        new_callable=AsyncMock,
    )
    def test_model_file_update_file_cleanup(
        self, mock_update_func,
    ) -> None:
        """
        Tests that model update function is called correctly.
        """
        mock_update_func.return_value = None

        files = {
            'file': ('model.pt', b'model content', 'application/octet-stream'),
        }
        data = {'model': 'yolo11n'}

        resp = self.client.post(
            '/api/model_file_update', data=data, files=files,
        )
        self.assertEqual(resp.status_code, 200)

        # Verify update function was called
        mock_update_func.assert_called_once()


if __name__ == '__main__':
    unittest.main()

'''
pytest \
    --cov=examples.YOLO_server_api.backend.routers \
    --cov-report=term-missing \
    tests/examples/YOLO_server_api/backend/routers_test.py
'''
