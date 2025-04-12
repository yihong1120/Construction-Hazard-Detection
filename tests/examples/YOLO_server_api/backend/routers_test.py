from __future__ import annotations

import base64
import datetime
import unittest
from typing import ClassVar
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi_jwt import JwtAuthorizationCredentials

from examples.YOLO_server_api.backend.routers import custom_rate_limiter
from examples.YOLO_server_api.backend.routers import detection_router
from examples.YOLO_server_api.backend.routers import jwt_access
from examples.YOLO_server_api.backend.routers import model_loader
from examples.YOLO_server_api.backend.routers import model_management_router


class TestRouters(unittest.TestCase):
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

    @patch('examples.YOLO_server_api.backend.routers.update_model_file')
    @patch('examples.YOLO_server_api.backend.routers.logger')
    def test_model_file_update_os_error(
        self, mock_logger: MagicMock,
        mock_update_func: MagicMock,
    ) -> None:
        """
        Verifies /api/model_file_update => 500 if an OSError occurs
        during file writing.

        Args:
            mock_logger (MagicMock):
                Mock for the logger to capture error messages.
            mock_update_func (MagicMock):
                Mock for the update_model_file function to simulate an OSError.
        """
        mock_update_func.return_value = None
        with patch('pathlib.Path.open', side_effect=OSError('Disk error')):
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

        mock_logger.info.assert_any_call(
            'No update required for model yolo11n.',
        )

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


if __name__ == '__main__':
    unittest.main()

'''
pytest \
    --cov=examples.YOLO_server_api.backend.routers \
    --cov-report=term-missing \
    tests/examples/YOLO_server_api/backend/routers_test.py
'''
