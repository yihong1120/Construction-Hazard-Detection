from __future__ import annotations

import unittest
from collections.abc import AsyncIterator
from typing import Callable
from typing import ClassVar
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import jwt
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi_jwt import JwtAuthorizationCredentials

from examples.auth.routers import auth_router
from examples.auth.routers import get_db
from examples.auth.routers import get_redis_pool
from examples.auth.routers import jwt_access
from examples.auth.routers import user_management_router


class TestRouters(unittest.TestCase):
    """
    Test suite for various router endpoints
    related to authentication and user management.
    """

    app: ClassVar[FastAPI]
    client: ClassVar[TestClient]

    @classmethod
    def setUpClass(cls) -> None:
        """
        Set up the FastAPI application and test client once for all tests.
        """
        cls.app = FastAPI()
        cls.app.include_router(auth_router, prefix='/api')
        cls.app.include_router(user_management_router, prefix='/api')

        # Default overrides
        async def default_db_override() -> AsyncIterator[AsyncMock]:
            """
            Default DB override that
            returns an async generator yielding an AsyncMock.
            """
            yield AsyncMock()

        cls.app.dependency_overrides[get_db] = default_db_override

        async def default_redis_override() -> AsyncMock:
            """
            Default Redis override returning an AsyncMock object
            with no specialised behaviour.
            """
            return AsyncMock()

        cls.app.dependency_overrides[get_redis_pool] = default_redis_override

        def override_jwt_access() -> JwtAuthorizationCredentials:
            """
            Override JWT access dependency
            with a mock JWT containing admin role.
            """
            return JwtAuthorizationCredentials(
                subject={
                    'username': 'admin_user',
                    'role': 'admin',
                    'jti': 'mock_jti',
                },
            )

        cls.app.dependency_overrides[jwt_access] = override_jwt_access

        cls.client = TestClient(cls.app)

    def override_jwt_role(
        self,
        role: str,
        username: str = 'some_user',
    ) -> Callable[[], JwtAuthorizationCredentials]:
        """
        Helper method to override the JWT role for specific tests.

        Args:
            role (str): The role to be placed in the JWT.
            username (str): The username to be placed in the JWT.

        Returns:
            Callable[[], JwtAuthorizationCredentials]:
                A function that returns the overridden credentials.
        """
        def _override() -> JwtAuthorizationCredentials:
            return JwtAuthorizationCredentials(
                subject={
                    'username': username,
                    'role': role,
                    'jti': 'mock_jti',
                },
            )
        return _override

    # --------------------------------------------------
    # /api/login Tests
    # --------------------------------------------------

    @patch('examples.auth.routers.get_user_data', new_callable=AsyncMock)
    def test_login_success(self, mock_get_user_data: AsyncMock) -> None:
        """
        Test a successful login scenario
        where correct user credentials are provided.

        Args:
            mock_get_user_data (AsyncMock):
                Mock for the get_user_data function.
        """
        # Redis does not have data for this user
        mock_get_user_data.return_value = None

        db_sess = AsyncMock()

        async def override_db() -> AsyncIterator[AsyncMock]:
            """Override DB dependency to provide the mock DB session."""
            yield db_sess

        self.app.dependency_overrides[get_db] = override_db

        # Mock a valid user returned by the database
        mock_user = MagicMock()
        mock_user.id = 123
        mock_user.username = 'testuser'
        mock_user.role = 'user'
        mock_user.is_active = True
        mock_user.check_password = AsyncMock(return_value=True)

        # Simulate a database response
        mock_result = MagicMock()
        mock_result.scalar = MagicMock(return_value=mock_user)
        db_sess.execute.return_value = mock_result

        resp = self.client.post(
            '/api/login',
            json={'username': 'testuser', 'password': '12345'},
        )

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn('access_token', data)
        self.assertIn('refresh_token', data)
        self.assertEqual(data['username'], 'testuser')
        self.assertEqual(data['role'], 'user')

    @patch('examples.auth.routers.get_user_data', new_callable=AsyncMock)
    def test_login_user_not_found(self, mock_get_user_data: AsyncMock) -> None:
        """
        Test login failure scenario
        where the provided username does not exist in the database.

        Args:
            mock_get_user_data (AsyncMock):
                Mock for the get_user_data function.
        """
        mock_get_user_data.return_value = None
        db_sess = AsyncMock()

        async def override_db() -> AsyncIterator[AsyncMock]:
            """Override DB dependency to provide the mock DB session."""
            yield db_sess

        self.app.dependency_overrides[get_db] = override_db

        # DB returns None for the user
        mock_result = MagicMock()
        mock_result.scalar.return_value = None
        db_sess.execute.return_value = mock_result

        resp = self.client.post(
            '/api/login', json={'username': 'no_such_user', 'password': 'xxx'},
        )
        self.assertEqual(resp.status_code, 401)
        self.assertIn('Wrong username or password', resp.text)

    @patch('examples.auth.routers.get_user_data', new_callable=AsyncMock)
    def test_login_wrong_password(self, mock_get_user_data: AsyncMock) -> None:
        """
        Test login failure scenario due to incorrect password.

        Args:
            mock_get_user_data (AsyncMock):
                Mock for the get_user_data function.
        """
        mock_get_user_data.return_value = None
        db_sess = AsyncMock()

        async def override_db() -> AsyncIterator[AsyncMock]:
            """
            Override DB dependency to provide the mock DB session.
            """
            yield db_sess

        self.app.dependency_overrides[get_db] = override_db

        mock_user = MagicMock()
        mock_user.is_active = True
        mock_user.role = 'user'
        mock_user.check_password = AsyncMock(return_value=False)
        mock_result = MagicMock()
        mock_result.scalar.return_value = mock_user
        db_sess.execute.return_value = mock_result

        resp = self.client.post(
            '/api/login', json={'username': 'testuser', 'password': 'bad'},
        )
        self.assertEqual(resp.status_code, 401)
        self.assertIn('Wrong username or password', resp.text)

    @patch('examples.auth.routers.get_user_data', new_callable=AsyncMock)
    def test_login_inactive(self, mock_get_user_data: AsyncMock) -> None:
        """
        Test login failure scenario where the user is marked inactive.

        Args:
            mock_get_user_data (AsyncMock):
                Mock for the get_user_data function.
        """
        mock_get_user_data.return_value = None
        db_sess = AsyncMock()

        async def override_db() -> AsyncIterator[AsyncMock]:
            """
            Override DB dependency to provide the mock DB session.
            """
            yield db_sess

        self.app.dependency_overrides[get_db] = override_db

        mock_user = MagicMock()
        mock_user.is_active = False
        mock_user.role = 'user'
        mock_user.check_password = AsyncMock(return_value=True)
        mock_result = MagicMock()
        mock_result.scalar.return_value = mock_user
        db_sess.execute.return_value = mock_result

        resp = self.client.post(
            '/api/login', json={'username': 'inactive', 'password': '123'},
        )
        self.assertEqual(resp.status_code, 403)
        self.assertIn('inactive', resp.text)

    @patch('examples.auth.routers.get_user_data', new_callable=AsyncMock)
    def test_login_invalid_role(self, mock_get_user_data: AsyncMock) -> None:
        """
        Test login failure where the user's role is invalid
        (not among the recognised roles).

        Args:
            mock_get_user_data (AsyncMock):
                Mock for the get_user_data function.
        """
        mock_get_user_data.return_value = None
        db_sess = AsyncMock()

        async def override_db() -> AsyncIterator[AsyncMock]:
            """
            Override DB dependency to provide the mock DB session.
            """
            yield db_sess

        self.app.dependency_overrides[get_db] = override_db

        mock_user = MagicMock()
        mock_user.is_active = True
        mock_user.role = 'superhacker'
        mock_user.check_password = AsyncMock(return_value=True)
        mock_result = MagicMock()
        mock_result.scalar.return_value = mock_user
        db_sess.execute.return_value = mock_result

        resp = self.client.post(
            '/api/login', json={'username': 'x', 'password': 'y'},
        )
        self.assertEqual(resp.status_code, 403)
        self.assertIn('required role', resp.text)

    # --------------------------------------------------
    # /api/logout Tests
    # --------------------------------------------------

    @patch(
        'examples.auth.routers.jwt.decode',
        return_value={'username': 'some_user', 'jti': 'mock_jti'},
    )
    @patch(
        'examples.auth.routers.get_user_data',
        new_callable=AsyncMock,
    )
    @patch(
        'examples.auth.routers.set_user_data',
        new_callable=AsyncMock,
    )
    def test_logout_success(
        self,
        mock_set_user_data: AsyncMock,
        mock_get_user_data: AsyncMock,
        mock_jwt_decode: MagicMock,
    ) -> None:
        """
        Test a successful logout scenario.

        Args:
            mock_set_user_data (AsyncMock):
                Mock for the set_user_data function.
            mock_get_user_data (AsyncMock):
                Mock for the get_user_data function.
            mock_jwt_decode (MagicMock):
                Mock for the jwt.decode function.
        """
        mock_get_user_data.return_value = {
            'refresh_tokens': ['some_refresh_token', 'logout_refresh_token'],
            'jti_list': ['mock_jti', 'other_jti'],
        }

        headers = {'Authorization': 'Bearer valid_access_token'}
        body = {'refresh_token': 'logout_refresh_token'}
        resp = self.client.post('/api/logout', headers=headers, json=body)
        self.assertEqual(resp.status_code, 200)
        self.assertIn('Logged out successfully', resp.text)

        # Validate that the updated data no longer has the token and jti
        call_args = mock_set_user_data.await_args
        # The set_user_data call is set_user_data
        # (redis_pool, username, updated_data)
        updated_data = call_args[0][2] if call_args and call_args[0] else {}
        self.assertNotIn(
            'logout_refresh_token',
            updated_data.get('refresh_tokens', []),
        )
        self.assertNotIn('mock_jti', updated_data.get('jti_list', []))

    @patch(
        'examples.auth.routers.jwt.decode',
        side_effect=jwt.PyJWTError('Invalid token'),
    )
    def test_logout_invalid_token(self, _) -> None:
        """
        Test logout scenario where the provided access token is invalid.

        Args:
            _: Mock for the jwt.decode function.
        """
        headers = {'Authorization': 'Bearer invalid_token'}
        body = {'refresh_token': 'some_refresh_token'}
        resp = self.client.post('/api/logout', headers=headers, json=body)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(
            'Invalid token, but local logout can proceed.', resp.text,
        )

    def test_logout_no_auth_header(self) -> None:
        """Test logout scenario where no access token is provided."""
        body = {'refresh_token': 'some_refresh_token'}
        resp = self.client.post('/api/logout', json=body)
        self.assertEqual(resp.status_code, 200)
        self.assertIn('No access token provided', resp.text)

    @patch(
        'examples.auth.routers.jwt.decode',
        return_value={'username': 'some_user', 'jti': 'mock_jti'},
    )
    @patch('examples.auth.routers.get_user_data', return_value=None)
    def test_logout_no_user_data(
        self,
        mock_get_user_data: MagicMock,
        mock_jwt_decode: MagicMock,
    ) -> None:
        """
        Test logout scenario where user data is not found in Redis.

        Args:
            mock_get_user_data (MagicMock):
                Mock for the get_user_data function.
            mock_jwt_decode (MagicMock):
                Mock for the jwt.decode function.
        """
        headers = {'Authorization': 'Bearer some_access_token'}
        body = {'refresh_token': 'any_refresh'}
        resp = self.client.post('/api/logout', headers=headers, json=body)
        self.assertEqual(resp.status_code, 200)
        self.assertIn('User data not found in Redis', resp.text)

    @patch(
        'examples.auth.routers.jwt.decode',
        return_value={'jti': 'mock_jti'},
    )
    def test_logout_missing_username(self, mock_jwt_decode: MagicMock) -> None:
        """
        Test logout scenario where the JWT payload
        does not contain a username field.

        Args:
            mock_jwt_decode (MagicMock):
                Mock for the jwt.decode function.
        """
        headers = {'Authorization': 'Bearer valid_token'}
        body = {'refresh_token': 'test_refresh'}
        resp = self.client.post('/api/logout', headers=headers, json=body)
        self.assertEqual(resp.status_code, 200)
        self.assertIn('No username in token payload', resp.text)

    def test_logout_invalid_header_format(self) -> None:
        """
        Test logout scenario where the Authorisation header format is invalid.
        """
        headers = {'Authorization': 'InvalidHeader'}
        body = {'refresh_token': 'whatever'}
        resp = self.client.post('/api/logout', headers=headers, json=body)
        self.assertEqual(resp.status_code, 200)
        self.assertIn('Invalid Authorization header format.', resp.text)

    # --------------------------------------------------
    # /api/refresh Tests
    # --------------------------------------------------

    @patch('examples.auth.auth.jwt.decode')
    @patch(
        'examples.auth.routers.jwt_access.create_access_token',
        return_value='new_access',
    )
    @patch(
        'examples.auth.routers.jwt_refresh.create_access_token',
        return_value='new_refresh',
    )
    @patch('examples.auth.routers.set_user_data', new_callable=AsyncMock)
    def test_refresh_success(
        self,
        mock_set_user_data: AsyncMock,
        mock_create_refresh: MagicMock,
        mock_create_access: MagicMock,
        mock_jwt_decode: MagicMock,
    ) -> None:
        """
        Test refresh scenario where the provided token is valid.

        Args:
            mock_set_user_data (AsyncMock):
                Mock for the set_user_data function.
            mock_create_refresh (MagicMock):
                Mock for creating a new refresh token.
            mock_create_access (MagicMock):
                Mock for creating a new access token.
            mock_jwt_decode (MagicMock):
                Mock for the jwt.decode function.
        """
        mock_jwt_decode.return_value = {'subject': {'username': 'testuser'}}
        redis_mock = AsyncMock()
        redis_mock.get.return_value = (
            '{"db_user": {"username": "testuser", "role": "user"},'
            ' "refresh_tokens": ["valid_refresh"], "jti_list": []}'
        )
        self.app.dependency_overrides[get_redis_pool] = lambda: redis_mock

        body = {'refresh_token': 'valid_refresh'}
        resp = self.client.post('/api/refresh', json=body)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data['access_token'], 'new_access')
        self.assertEqual(data['refresh_token'], 'new_refresh')
        self.assertIn('Token refreshed successfully', data['message'])

    @patch(
        'examples.auth.auth.jwt.decode',
        side_effect=jwt.InvalidTokenError('Invalid refresh'),
    )
    def test_refresh_invalid_token(self, _) -> None:
        """
        Test refresh scenario where the provided token is invalid.
        """
        body = {'refresh_token': 'really_bad_token'}
        resp = self.client.post('/api/refresh', json=body)
        self.assertEqual(resp.status_code, 401)
        self.assertIn('Invalid refresh token', resp.text)

    @patch(
        'examples.auth.auth.jwt.decode',
        side_effect=jwt.ExpiredSignatureError('Token expired'),
    )
    def test_refresh_expired_token(self, _) -> None:
        """
        Test refresh scenario where the provided token has expired.
        """
        body = {'refresh_token': 'expired_refresh'}
        resp = self.client.post('/api/refresh', json=body)
        self.assertEqual(resp.status_code, 401)
        self.assertIn('Refresh token has expired', resp.text)

    def test_refresh_missing(self) -> None:
        """
        Test refresh scenario where no token is provided in the request body.
        """
        body = {'refresh_token': ''}
        resp = self.client.post('/api/refresh', json=body)
        self.assertEqual(resp.status_code, 401)
        self.assertIn('Refresh token is missing', resp.text)

    @patch('examples.auth.auth.jwt.decode', return_value={'subject': {}})
    def test_refresh_no_username_in_payload(self, _) -> None:
        """
        Test refresh scenario where the JWT payload
        does not contain a username.
        """
        body = {'refresh_token': 'valid_but_missing_username'}
        resp = self.client.post('/api/refresh', json=body)
        self.assertEqual(resp.status_code, 401)
        self.assertIn('Invalid refresh token payload', resp.text)

    @patch(
        'examples.auth.auth.jwt.decode',
        return_value={'subject': {'username': 'testuser'}},
    )
    def test_refresh_no_user_data_in_redis(
        self,
        mock_jwt_decode: MagicMock,
    ) -> None:
        """
        Test refresh scenario where no user data is found in Redis.

        Args:
            mock_jwt_decode (MagicMock):
                Mock for the jwt.decode function.
        """
        redis_mock = AsyncMock()
        redis_mock.get.return_value = None
        self.app.dependency_overrides[get_redis_pool] = lambda: redis_mock

        body = {'refresh_token': 'some_refresh'}
        resp = self.client.post('/api/refresh', json=body)
        self.assertEqual(resp.status_code, 401)
        self.assertIn('No user data in Redis', resp.text)

    @patch(
        'examples.auth.auth.jwt.decode',
        return_value={'subject': {'username': 'testuser'}},
    )
    def test_refresh_not_in_refresh_tokens(
        self,
        mock_jwt_decode: MagicMock,
    ) -> None:
        """
        Test refresh scenario where the provided token
        does not match the stored list of valid tokens.

        Args:
            mock_jwt_decode (MagicMock):
                Mock for the jwt.decode function.
        """
        redis_mock = AsyncMock()
        redis_mock.get.return_value = (
            '{"db_user": {"username": "testuser", "role": "user"},'
            ' "refresh_tokens": ["some_other_token"], "jti_list": []}'
        )
        self.app.dependency_overrides[get_redis_pool] = lambda: redis_mock

        body = {'refresh_token': 'not_in_list'}
        resp = self.client.post('/api/refresh', json=body)
        self.assertEqual(resp.status_code, 401)
        self.assertIn('Refresh token not recognised', resp.text)

    # --------------------------------------------------
    # User Management Tests
    # --------------------------------------------------

    @patch('examples.auth.routers.add_user', new_callable=AsyncMock)
    def test_add_user_success(self, mock_add_user: AsyncMock) -> None:
        """
        Test successful addition of a new user via the /api/add_user endpoint.

        Args:
            mock_add_user (AsyncMock):
                Mock for the add_user function.
        """
        mock_add_user.return_value = {'success': True}
        payload = {'username': 'u1', 'password': 'p1', 'role': 'user'}
        resp = self.client.post('/api/add_user', json=payload)
        self.assertEqual(resp.status_code, 200)
        self.assertIn('User added successfully', resp.text)

    @patch('examples.auth.routers.add_user', new_callable=AsyncMock)
    def test_add_user_error(self, mock_add_user: AsyncMock) -> None:
        """
        Test various error scenarios during user creation.

        Args:
            mock_add_user (AsyncMock):
                Mock for the add_user function.
        """
        mock_add_user.return_value = {
            'success': False, 'error': 'IntegrityError',
        }
        payload = {'username': 'u1', 'password': 'p1', 'role': 'user'}
        resp = self.client.post('/api/add_user', json=payload)
        self.assertEqual(resp.status_code, 400)

        mock_add_user.return_value = {'success': False, 'error': 'Other'}
        resp = self.client.post('/api/add_user', json=payload)
        self.assertEqual(resp.status_code, 500)

    def test_add_user_not_admin(self) -> None:
        """
        Test that non-admin roles are forbidden from adding users.
        """
        self.app.dependency_overrides[jwt_access] = self.override_jwt_role(
            'user',
        )
        payload = {'username': 'u1', 'password': 'p1', 'role': 'user'}
        resp = self.client.post('/api/add_user', json=payload)
        self.assertEqual(resp.status_code, 403)

        # Revert to admin user for subsequent tests
        self.app.dependency_overrides[jwt_access] = self.override_jwt_role(
            'admin',
        )

    @patch('examples.auth.routers.delete_user', new_callable=AsyncMock)
    def test_delete_user(self, mock_delete_user: AsyncMock) -> None:
        """
        Test the deletion of an existing user
        via the /api/delete_user endpoint.

        Args:
            mock_delete_user (AsyncMock):
                Mock for the delete_user function.
        """
        mock_delete_user.return_value = {'success': True}
        payload = {'username': 'u1'}
        resp = self.client.post('/api/delete_user', json=payload)
        self.assertEqual(resp.status_code, 200)
        self.assertIn('User deleted successfully', resp.text)

        mock_delete_user.return_value = {'success': False, 'error': 'NotFound'}
        resp = self.client.post('/api/delete_user', json=payload)
        self.assertEqual(resp.status_code, 404)

        mock_delete_user.return_value = {'success': False, 'error': 'Other'}
        resp = self.client.post('/api/delete_user', json=payload)
        self.assertEqual(resp.status_code, 500)

    def test_delete_user_not_admin(self) -> None:
        """
        Test that non-admin roles are forbidden from deleting users.
        """
        self.app.dependency_overrides[jwt_access] = self.override_jwt_role(
            'user',
        )
        resp = self.client.post('/api/delete_user', json={'username': 'u1'})
        self.assertEqual(resp.status_code, 403)

        # Revert to admin user for subsequent tests
        self.app.dependency_overrides[jwt_access] = self.override_jwt_role(
            'admin',
        )

    @patch('examples.auth.routers.update_username', new_callable=AsyncMock)
    def test_update_username(self, mock_update_username: AsyncMock) -> None:
        """
        Test the /api/update_username endpoint.

        Args:
            mock_update_username (AsyncMock):
                Mock for the update_username function.
        """
        mock_update_username.return_value = {'success': True}
        payload = {'old_username': 'x', 'new_username': 'y'}
        resp = self.client.put('/api/update_username', json=payload)
        self.assertEqual(resp.status_code, 200)

        mock_update_username.return_value = {
            'success': False, 'error': 'IntegrityError',
        }
        resp = self.client.put('/api/update_username', json=payload)
        self.assertEqual(resp.status_code, 400)

        mock_update_username.return_value = {
            'success': False, 'error': 'NotFound',
        }
        resp = self.client.put('/api/update_username', json=payload)
        self.assertEqual(resp.status_code, 404)

    def test_update_username_not_admin(self) -> None:
        """
        Test that non-admin roles are forbidden from updating a username.
        """
        self.app.dependency_overrides[jwt_access] = self.override_jwt_role(
            'user',
        )
        payload = {'old_username': 'x', 'new_username': 'y'}
        resp = self.client.put('/api/update_username', json=payload)
        self.assertEqual(resp.status_code, 403)

        # Revert to admin user for subsequent tests
        self.app.dependency_overrides[jwt_access] = self.override_jwt_role(
            'admin',
        )

    @patch('examples.auth.routers.update_password', new_callable=AsyncMock)
    def test_update_password(self, mock_update_password: AsyncMock) -> None:
        """
        Test the /api/update_password endpoint.

        Args:
            mock_update_password (AsyncMock):
                Mock for the update_password function.
        """
        mock_update_password.return_value = {'success': True}
        payload = {'username': 'x', 'new_password': 'z'}
        resp = self.client.put('/api/update_password', json=payload)
        self.assertEqual(resp.status_code, 200)

        mock_update_password.return_value = {
            'success': False, 'error': 'NotFound',
        }
        resp = self.client.put('/api/update_password', json=payload)
        self.assertEqual(resp.status_code, 404)

        mock_update_password.return_value = {
            'success': False, 'error': 'Other',
        }
        resp = self.client.put('/api/update_password', json=payload)
        self.assertEqual(resp.status_code, 500)

    def test_update_password_not_admin(self) -> None:
        """
        Test that non-admin roles are forbidden
        from updating a user's password.
        """
        self.app.dependency_overrides[jwt_access] = self.override_jwt_role(
            'user',
        )
        payload = {'username': 'x', 'new_password': 'z'}
        resp = self.client.put('/api/update_password', json=payload)
        self.assertEqual(resp.status_code, 403)

        # Revert to admin user
        self.app.dependency_overrides[jwt_access] = self.override_jwt_role(
            'admin',
        )

    @patch(
        'examples.auth.routers.set_user_active_status',
        new_callable=AsyncMock,
    )
    def test_set_user_active_status(self, mock_set_active: AsyncMock) -> None:
        """
        Test the /api/set_user_active_status endpoint.

        Args:
            mock_set_active (AsyncMock):
                Mock for the set_user_active_status function.
        """
        mock_set_active.return_value = {'success': True}
        payload = {'username': 'x', 'is_active': True}
        resp = self.client.put('/api/set_user_active_status', json=payload)
        self.assertEqual(resp.status_code, 200)

        mock_set_active.return_value = {'success': False, 'error': 'NotFound'}
        resp = self.client.put('/api/set_user_active_status', json=payload)
        self.assertEqual(resp.status_code, 404)

        mock_set_active.return_value = {'success': False, 'error': 'Other'}
        resp = self.client.put('/api/set_user_active_status', json=payload)
        self.assertEqual(resp.status_code, 500)

    def test_set_user_active_status_not_admin(self) -> None:
        """
        Test that non-admin roles are forbidden
        from setting a user's active status.
        """
        self.app.dependency_overrides[jwt_access] = self.override_jwt_role(
            'user',
        )
        payload = {'username': 'x', 'is_active': True}
        resp = self.client.put('/api/set_user_active_status', json=payload)
        self.assertEqual(resp.status_code, 403)

        # Revert to admin user
        self.app.dependency_overrides[jwt_access] = self.override_jwt_role(
            'admin',
        )


if __name__ == '__main__':
    unittest.main()


'''
pytest \
    --cov=examples.auth.routers \
    --cov-report=term-missing tests/examples/auth/routers_test.py
'''
