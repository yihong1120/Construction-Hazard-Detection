from __future__ import annotations

import base64
import datetime
import unittest
from collections.abc import AsyncGenerator
from contextlib import contextmanager
from unittest.mock import AsyncMock
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi_jwt import JwtAuthorizationCredentials

from examples.YOLO_server_api.backend.routers import custom_rate_limiter
from examples.YOLO_server_api.backend.routers import detection_router
from examples.YOLO_server_api.backend.routers import get_db
from examples.YOLO_server_api.backend.routers import jwt_access
from examples.YOLO_server_api.backend.routers import model_management_router
from examples.YOLO_server_api.backend.routers import user_management_router
# from fastapi import UploadFile


class TestRouters(unittest.TestCase):
    """
    Tests for the routers of the YOLO server API.
    """
    app: FastAPI  # Type hint for the app attribute
    client: TestClient  # Type hint for the client attribute

    @classmethod
    def setUpClass(cls) -> None:
        """
        Set up the FastAPI application and override dependencies.
        """
        app = FastAPI()
        app.include_router(detection_router)
        app.include_router(user_management_router)
        app.include_router(model_management_router)

        def override_jwt_access() -> JwtAuthorizationCredentials:
            """
            Override JWT access for testing with admin credentials.
            """
            return JwtAuthorizationCredentials(
                subject={'role': 'admin', 'id': 1},
            )

        def override_custom_rate_limiter() -> int:
            """
            Override the rate limiter for testing purposes.
            """
            return 10  # Or any other value you wish

        async def override_get_db() -> AsyncGenerator[str]:
            """
            Simulate a database session for testing.
            """
            yield 'mock_db_session'

        app.dependency_overrides[jwt_access] = override_jwt_access
        app.dependency_overrides[custom_rate_limiter] = (
            override_custom_rate_limiter
        )
        app.dependency_overrides[get_db] = override_get_db

        cls.app = app
        cls.client = TestClient(app)

    @contextmanager
    def override_jwt_credentials(self, subject: dict):
        """
        Context manager to temporarily override JWT credentials
        during tests.
        """
        original_jwt_access = self.app.dependency_overrides.get(jwt_access)
        self.app.dependency_overrides[jwt_access] = (
            lambda: JwtAuthorizationCredentials(
                subject=subject,
            )
        )
        try:
            yield
        finally:
            if original_jwt_access is not None:
                self.app.dependency_overrides[jwt_access] = original_jwt_access
            else:
                del self.app.dependency_overrides[jwt_access]

    @patch(
        'examples.YOLO_server_api.backend.routers.process_labels',
        new_callable=AsyncMock,
    )
    @patch('examples.YOLO_server_api.backend.routers.compile_detection_data')
    @patch(
        'examples.YOLO_server_api.backend.routers.get_prediction_result',
        new_callable=AsyncMock,
    )
    @patch(
        'examples.YOLO_server_api.backend.routers.convert_to_image',
        new_callable=AsyncMock,
    )
    @patch('examples.YOLO_server_api.backend.routers.model_loader.get_model')
    def test_detect_endpoint(
        self,
        mock_get_model,
        mock_convert_to_image,
        mock_get_prediction_result,
        mock_compile_detection_data,
        mock_process_labels,
    ):
        """
        Test the detection endpoint with mocked dependencies.
        """
        # Mock dependencies
        mock_get_model.return_value = 'mock_model_instance'
        mock_convert_to_image.return_value = 'mock_image'
        mock_get_prediction_result.return_value = 'mock_result'
        mock_compile_detection_data.return_value = [
            [1.0, 2.0, 3.0, 4.0],
        ]  # Return a list of the expected type
        # Return a list of the expected type
        mock_process_labels.return_value = [[1.0, 2.0, 3.0, 4.0]]

        # Create a test image file
        image_content = b'test_image_data'
        files = {'image': ('test_image.jpg', image_content, 'image/jpeg')}
        data = {'model': 'yolo11n'}

        # Successful case
        response = self.client.post('/api/detect', data=data, files=files)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [[1.0, 2.0, 3.0, 4.0]])

        # Model not found case
        mock_get_model.return_value = None
        response = self.client.post('/api/detect', data=data, files=files)
        self.assertEqual(response.status_code, 404)

    @patch(
        'examples.YOLO_server_api.backend.routers.add_user',
        new_callable=AsyncMock,
    )
    @patch('examples.YOLO_server_api.backend.routers.logger')
    def test_add_user(self, mock_logger, mock_add_user):
        """
        Test adding a user with mocked add_user function.
        """
        mock_add_user.return_value = {'success': True, 'message': 'User added'}

        user_data = {
            'username': 'testuser',
            'password': 'testpassword',
            'role': 'user',
        }

        response = self.client.post('/api/add_user', json=user_data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(), {
                'message': 'User added successfully.',
            },
        )
        mock_logger.info.assert_called_with('User added')

        # Non-admin role
        with self.override_jwt_credentials({'role': 'user', 'id': 1}):
            response = self.client.post('/api/add_user', json=user_data)
            self.assertEqual(response.status_code, 400)

        # IntegrityError case
        mock_add_user.return_value = {
            'success': False,
            'error': 'IntegrityError',
            'message': 'Duplicate',
        }
        with self.override_jwt_credentials({'role': 'admin', 'id': 1}):
            response = self.client.post('/api/add_user', json=user_data)
            self.assertEqual(response.status_code, 400)

    @patch(
        'examples.YOLO_server_api.backend.routers.delete_user',
        new_callable=AsyncMock,
    )
    @patch('examples.YOLO_server_api.backend.routers.logger')
    def test_delete_user(self, mock_logger, mock_delete_user):
        """
        Test deleting a user with mocked delete_user function.
        """
        mock_delete_user.return_value = {
            'success': True, 'message': 'User deleted',
        }

        user_data = {'username': 'testuser'}

        response = self.client.post('/api/delete_user', json=user_data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(), {
                'message': 'User deleted successfully.',
            },
        )
        mock_logger.info.assert_called_with('User deleted')

        # NotFound case
        mock_delete_user.return_value = {
            'success': False, 'error': 'NotFound', 'message': 'No user',
        }
        response = self.client.post('/api/delete_user', json=user_data)
        self.assertEqual(response.status_code, 404)

        # Other error case
        mock_delete_user.return_value = {
            'success': False, 'error': 'Other', 'message': 'Error',
        }
        response = self.client.post('/api/delete_user', json=user_data)
        self.assertEqual(response.status_code, 500)

        # Non-admin role
        self.app.dependency_overrides[jwt_access] = (
            lambda: JwtAuthorizationCredentials(
                subject={'role': 'user', 'id': 1},
            )
        )
        response = self.client.post('/api/delete_user', json=user_data)
        self.assertEqual(response.status_code, 403)

    @patch(
        'examples.YOLO_server_api.backend.routers.update_username',
        new_callable=AsyncMock,
    )
    @patch('examples.YOLO_server_api.backend.routers.logger')
    def test_update_username(self, mock_logger, mock_update_username):
        """
        Test updating a username with mocked update_username function.
        """
        async def success_func(old_username, new_username, db):
            return {'success': True, 'message': 'Username updated'}
        mock_update_username.side_effect = success_func

        user_data = {'old_username': 'olduser', 'new_username': 'newuser'}
        with self.override_jwt_credentials({'role': 'admin', 'id': 1}):
            response = self.client.put('/api/update_username', json=user_data)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.json(), {
                    'message': 'Username updated successfully.',
                },
            )
            mock_logger.info.assert_called_with('Username updated')

        # IntegrityError
        async def integrity_error_func(old_username, new_username, db):
            return {
                'success': False,
                'error': 'IntegrityError',
                'message': 'Duplicate',
            }
        mock_update_username.side_effect = integrity_error_func
        with self.override_jwt_credentials({'role': 'admin', 'id': 1}):
            response = self.client.put('/api/update_username', json=user_data)
            self.assertEqual(response.status_code, 400)

        # NotFound
        async def not_found_func(old_username, new_username, db):
            return {
                'success': False,
                'error': 'NotFound',
                'message': 'Not found',
            }
        mock_update_username.side_effect = not_found_func
        with self.override_jwt_credentials({'role': 'admin', 'id': 1}):
            response = self.client.put('/api/update_username', json=user_data)
            self.assertEqual(response.status_code, 404)

        # Non-admin role
        with self.override_jwt_credentials({'role': 'user', 'id': 1}):
            response = self.client.put('/api/update_username', json=user_data)
            self.assertEqual(response.status_code, 400)

    @patch(
        'examples.YOLO_server_api.backend.routers.update_password',
        new_callable=AsyncMock,
    )
    @patch('examples.YOLO_server_api.backend.routers.logger')
    def test_update_password(self, mock_logger, mock_update_password):
        """
        Test updating a password with mocked update_password function.
        """
        async def success_func(username, new_password, db):
            return {'success': True, 'message': 'Password updated'}
        mock_update_password.side_effect = success_func

        user_data = {'username': 'testuser', 'new_password': 'newpassword'}
        with self.override_jwt_credentials({'role': 'admin', 'id': 1}):
            response = self.client.put('/api/update_password', json=user_data)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.json(), {
                    'message': 'Password updated successfully.',
                },
            )
            mock_logger.info.assert_called_with('Password updated')

        # NotFound
        async def not_found_func(username, new_password, db):
            return {
                'success': False,
                'error': 'NotFound',
                'message': 'No user',
            }
        mock_update_password.side_effect = not_found_func
        with self.override_jwt_credentials({'role': 'admin', 'id': 1}):
            response = self.client.put('/api/update_password', json=user_data)
            self.assertEqual(response.status_code, 404)

        # Other error
        async def other_error_func(username, new_password, db):
            return {'success': False, 'error': 'Other', 'message': 'Error'}
        mock_update_password.side_effect = other_error_func
        with self.override_jwt_credentials({'role': 'admin', 'id': 1}):
            response = self.client.put('/api/update_password', json=user_data)
            self.assertEqual(response.status_code, 500)

        # Non-admin role
        with self.override_jwt_credentials({'role': 'user', 'id': 1}):
            response = self.client.put('/api/update_password', json=user_data)
            self.assertEqual(response.status_code, 400)

    @patch(
        'examples.YOLO_server_api.backend.routers.set_user_active_status',
        new_callable=AsyncMock,
    )
    @patch('examples.YOLO_server_api.backend.routers.logger')
    def test_set_user_active_status(
        self,
        mock_logger,
        mock_set_user_active_status,
    ):
        """
        Test setting a user's active status
        with mocked set_user_active_status function.
        """
        async def success_func(username, is_active, db):
            return {'success': True, 'message': 'Status updated'}
        mock_set_user_active_status.side_effect = success_func

        user_data = {'username': 'testuser', 'is_active': True}
        with self.override_jwt_credentials({'role': 'admin', 'id': 1}):
            response = self.client.put(
                '/api/set_user_active_status', json=user_data,
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.json(), {
                    'message': 'User active status updated successfully.',
                },
            )
            mock_logger.info.assert_called_with('Status updated')

        # NotFound
        async def not_found_func(username, is_active, db):
            return {
                'success': False,
                'error': 'NotFound',
                'message': 'No user',
            }
        mock_set_user_active_status.side_effect = not_found_func
        with self.override_jwt_credentials({'role': 'admin', 'id': 1}):
            response = self.client.put(
                '/api/set_user_active_status', json=user_data,
            )
            self.assertEqual(response.status_code, 404)

        # Other error
        async def other_error_func(username, is_active, db):
            return {'success': False, 'error': 'Other', 'message': 'Error'}
        mock_set_user_active_status.side_effect = other_error_func
        with self.override_jwt_credentials({'role': 'admin', 'id': 1}):
            response = self.client.put(
                '/api/set_user_active_status', json=user_data,
            )
            self.assertEqual(response.status_code, 500)

        # Non-admin role
        with self.override_jwt_credentials({'role': 'user', 'id': 1}):
            response = self.client.put(
                '/api/set_user_active_status', json=user_data,
            )
            self.assertEqual(response.status_code, 403)

    @patch(
        'examples.YOLO_server_api.backend.routers.update_model_file',
        new_callable=AsyncMock,
    )
    @patch('examples.YOLO_server_api.backend.routers.logger')
    def test_model_file_update(self, mock_logger, mock_update_model_file):
        """
        Test updating a model file with mocked update_model_file function.
        """
        async def mock_update_model_file_func(model, temp_path):
            pass  # Simulate successful execution
        mock_update_model_file.side_effect = mock_update_model_file_func

        model_file_content = b'model file content'
        files = {
            'file': (
                'model.pt', model_file_content,
                'application/octet-stream',
            ),
        }
        data = {'model': 'yolo11n'}

        # Admin role success
        with self.override_jwt_credentials({'role': 'admin', 'id': 1}):
            response = self.client.post(
                '/api/model_file_update', data=data, files=files,
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.json(), {
                    'message': 'Model yolo11n updated successfully.',
                },
            )
            mock_logger.info.assert_called_with(
                'Model yolo11n updated successfully.',
            )

        # Non-admin or model_manage role
        with self.override_jwt_credentials({'role': 'user', 'id': 1}):
            response = self.client.post(
                '/api/model_file_update', data=data, files=files,
            )
            self.assertEqual(response.status_code, 403)

        # Invalid file path scenario
        with patch(
            'examples.YOLO_server_api.backend.routers.secure_filename',
            return_value='../model.pt',
        ):
            with self.override_jwt_credentials({'role': 'admin', 'id': 1}):
                response = self.client.post(
                    '/api/model_file_update', data=data, files=files,
                )
                self.assertEqual(response.status_code, 400)

    @patch('examples.YOLO_server_api.backend.routers.get_new_model_file')
    @patch('examples.YOLO_server_api.backend.routers.logger')
    def test_get_new_model(self, mock_logger, mock_get_new_model_file):
        """
        Test retrieving a new model file
        with mocked get_new_model_file function.
        """
        async def mock_get_new_model_file_func(model, user_last_update):
            return b'new model content'  # Assume a new model is available
        mock_get_new_model_file.side_effect = mock_get_new_model_file_func

        request_data = {
            'model': 'yolo11n',
            'last_update_time': datetime.datetime.now().isoformat(),
        }

        response = self.client.post('/api/get_new_model', json=request_data)
        self.assertEqual(response.status_code, 200)
        expected_response = {
            'message': 'Model yolo11n is updated.',
            'model_file': base64.b64encode(b'new model content').decode(),
        }
        self.assertEqual(response.json(), expected_response)
        mock_logger.info.assert_called_with(
            'Newer model file for yolo11n retrieved.',
        )

        # No new model available
        async def mock_no_update(model, user_last_update):
            return None
        mock_get_new_model_file.side_effect = mock_no_update
        response = self.client.post('/api/get_new_model', json=request_data)
        self.assertEqual(response.status_code, 200)
        expected_response = {'message': 'Model yolo11n is up to date.'}
        self.assertEqual(response.json(), expected_response)
        mock_logger.info.assert_called_with(
            'No update required for model yolo11n.',
        )

        # Invalid datetime format
        invalid_request_data = {
            'model': 'yolo11n',
            'last_update_time': 'invalid_datetime',
        }
        response = self.client.post(
            '/api/get_new_model', json=invalid_request_data,
        )
        self.assertEqual(response.status_code, 400)

        # Exception scenario
        async def mock_exception(model, user_last_update):
            raise Exception('Some error')
        mock_get_new_model_file.side_effect = mock_exception
        response = self.client.post('/api/get_new_model', json=request_data)
        self.assertEqual(response.status_code, 500)


if __name__ == '__main__':
    unittest.main()
