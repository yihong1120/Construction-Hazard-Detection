from __future__ import annotations

import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import jwt
from fastapi import HTTPException

from examples.db_management.schemas.auth import RefreshRequest
from examples.db_management.schemas.auth import UserLogin
from examples.db_management.services import auth_services


class TestAuthServices(unittest.IsolatedAsyncioTestCase):
    """Unit tests for auth_services module using asynchronous mocks."""

    def setUp(self) -> None:
        """Set up common mock objects for each test."""
        self.db: AsyncMock = AsyncMock()
        self.redis_pool: AsyncMock = AsyncMock()

    @patch('examples.db_management.services.auth_services._authenticate')
    @patch('examples.db_management.services.auth_services._load_feature_names')
    @patch('examples.db_management.services.auth_services.jwt_access')
    @patch('examples.db_management.services.auth_services.jwt_refresh')
    @patch('examples.db_management.services.auth_services.set_user_data')
    async def test_login_user_success(
        self,
        mock_set_user_data,
        mock_jwt_refresh,
        mock_jwt_access,
        mock_load_feature_names,
        mock_authenticate,
    ) -> None:
        """Test successful user login."""
        user_mock = AsyncMock(
            id=1,
            username='user',
            role='user',
            group_id=1,
            is_active=True,
        )
        mock_authenticate.return_value = user_mock
        mock_load_feature_names.return_value = ['feature1', 'feature2']
        mock_jwt_access.create_access_token.return_value = 'access_token'
        mock_jwt_refresh.create_access_token.return_value = 'refresh_token'

        mock_redis_data = (
            '{"db_user": {"id": 1, "username": "user", "role": "user", '
            '"group_id": 1, "is_active": true}, "jti_list": [], '
            '"refresh_tokens": []}'
        )
        self.redis_pool.get = AsyncMock(return_value=mock_redis_data)

        payload = UserLogin(username='user', password='pass')
        result = await auth_services.login_user(
            payload, self.db, self.redis_pool,
        )

        self.assertEqual(result['access_token'], 'access_token')
        self.assertEqual(result['refresh_token'], 'refresh_token')
        self.assertEqual(result['username'], 'user')
        self.assertEqual(result['feature_names'], ['feature1', 'feature2'])
        mock_set_user_data.assert_awaited()

    async def test_authenticate_invalid_credentials(self) -> None:
        """Test authentication with wrong credentials."""
        self.db.scalar = AsyncMock(return_value=None)
        with self.assertRaises(HTTPException) as ctx:
            await auth_services._authenticate(
                self.db, 'wronguser', 'wrongpass',
            )
        self.assertEqual(ctx.exception.status_code, 401)

    async def test_authenticate_inactive_user(self) -> None:
        """Test authentication with inactive user."""
        mock_user = AsyncMock()
        mock_user.check_password = AsyncMock(return_value=True)
        mock_user.is_active = False
        self.db.scalar = AsyncMock(return_value=mock_user)

        with self.assertRaises(HTTPException) as ctx:
            await auth_services._authenticate(self.db, 'user', 'pass')
        self.assertEqual(ctx.exception.status_code, 403)

    async def test_load_feature_names_none_group(self) -> None:
        """Test _load_feature_names returns empty list if group_id is None."""
        result = await auth_services._load_feature_names(self.db, None)
        self.assertEqual(result, [])

    @patch('examples.db_management.services.auth_services.jwt.decode')
    async def test_verify_refresh_token_expired(self, mock_decode) -> None:
        """Test expired refresh token raises HTTPException."""
        mock_decode.side_effect = jwt.ExpiredSignatureError()
        with self.assertRaises(HTTPException) as ctx:
            await auth_services.verify_refresh_token(
                'expired', self.redis_pool,
            )
        self.assertEqual(ctx.exception.status_code, 401)

    @patch('examples.db_management.services.auth_services.jwt.decode')
    async def test_verify_refresh_token_invalid(self, mock_decode) -> None:
        """Test invalid refresh token raises HTTPException."""
        mock_decode.side_effect = jwt.InvalidTokenError()
        with self.assertRaises(HTTPException) as ctx:
            await auth_services.verify_refresh_token(
                'invalid', self.redis_pool,
            )
        self.assertEqual(ctx.exception.status_code, 401)

    @patch('examples.db_management.services.auth_services.jwt')
    async def test_verify_refresh_token_missing_username(
        self, mock_jwt,
    ) -> None:
        """Test missing username in payload raises HTTPException."""
        mock_jwt.decode.return_value = {'subject': {}}
        with self.assertRaises(HTTPException) as ctx:
            await auth_services.verify_refresh_token('token', self.redis_pool)
        self.assertEqual(ctx.exception.status_code, 401)

    @patch('examples.db_management.services.auth_services.jwt')
    @patch('examples.db_management.services.auth_services.get_user_data')
    async def test_verify_refresh_token_not_recognised(
        self, mock_get_user_data, mock_jwt,
    ) -> None:
        """Test unrecognised refresh token raises HTTPException."""
        mock_jwt.decode.return_value = {'subject': {'username': 'user'}}
        mock_get_user_data.return_value = {'refresh_tokens': []}
        with self.assertRaises(HTTPException) as ctx:
            await auth_services.verify_refresh_token(
                'unknown', self.redis_pool,
            )
        self.assertEqual(ctx.exception.status_code, 401)

    async def test_logout_user_no_auth(self) -> None:
        """Test logout_user with missing authorization header returns early."""
        await auth_services.logout_user('token', None, self.redis_pool)

    async def test_logout_user_bad_auth_format(self) -> None:
        """
        Test logout_user with malformed authorization header returns early.
        """
        await auth_services.logout_user(
            'token', 'invalidtoken', self.redis_pool,
        )

    @patch('examples.db_management.services.auth_services.jwt.decode')
    async def test_logout_user_invalid_jwt(self, mock_decode) -> None:
        """Test logout_user with invalid JWT returns early."""
        mock_decode.side_effect = jwt.PyJWTError()
        await auth_services.logout_user(
            'token', 'Bearer abc.def.ghi', self.redis_pool,
        )

    @patch(
        'examples.db_management.services.auth_services.get_user_data',
    )
    @patch('examples.db_management.services.auth_services.jwt')
    async def test_logout_user_no_cache(
        self, mock_jwt, mock_get_user_data,
    ) -> None:
        """Test logout_user with no cache found returns early."""
        mock_jwt.decode.return_value = {'username': 'user', 'jti': 'id'}
        mock_get_user_data.return_value = None
        await auth_services.logout_user(
            'token', 'Bearer valid.token', self.redis_pool,
        )

    async def test_refresh_tokens_missing_token(self) -> None:
        """Test refresh_tokens raises if refresh token is missing."""
        with self.assertRaises(HTTPException) as ctx:
            await auth_services.refresh_tokens(
                RefreshRequest(refresh_token=''),
                self.redis_pool,
            )
        self.assertEqual(ctx.exception.status_code, 401)

    @patch('examples.db_management.services.auth_services.get_user_data')
    @patch(
        'examples.db_management.services.auth_services.verify_refresh_token',
    )
    async def test_refresh_tokens_invalid_cache(
        self, mock_verify, mock_get_user_data,
    ) -> None:
        """Test refresh_tokens raises if cache is invalid or missing token."""
        mock_verify.return_value = {'subject': {'username': 'user'}}
        mock_get_user_data.return_value = {'refresh_tokens': []}
        with self.assertRaises(HTTPException) as ctx:
            await auth_services.refresh_tokens(
                RefreshRequest(refresh_token='bad'),
                self.redis_pool,
            )
        self.assertEqual(ctx.exception.status_code, 401)

    async def test_load_feature_names_valid_group(self) -> None:
        """
        Test _load_feature_names returns correct features for a valid group_id.
        """
        mock_result = MagicMock()
        mock_result.__iter__.return_value = [
            MagicMock(feature_name='feature1'), MagicMock(
                feature_name='feature2',
            ),
        ]
        self.db.execute = AsyncMock(return_value=mock_result)

        features = await auth_services._load_feature_names(self.db, group_id=1)
        self.assertEqual(features, ['feature1', 'feature2'])

    async def test_authenticate_success(self) -> None:
        """
        Test _authenticate returns user object when credentials are valid.
        """
        mock_user = MagicMock(is_active=True)
        mock_user.check_password = AsyncMock(return_value=True)
        self.db.scalar = AsyncMock(return_value=mock_user)

        user = await auth_services._authenticate(
            self.db, 'valid_user', 'valid_password',
        )
        self.assertEqual(user, mock_user)

    @patch('examples.db_management.services.auth_services.jwt.decode')
    async def test_verify_refresh_token_success(self, mock_decode) -> None:
        """Test verify_refresh_token returns payload correctly."""
        mock_decode.return_value = {'subject': {'username': 'user'}}
        mock_cache_data = '{"refresh_tokens": ["valid_token"]}'
        self.redis_pool.get = AsyncMock(return_value=mock_cache_data)

        payload = await auth_services.verify_refresh_token(
            'valid_token', self.redis_pool,
        )
        self.assertEqual(payload, {'subject': {'username': 'user'}})

    @patch('examples.db_management.services.auth_services.set_user_data')
    @patch('examples.db_management.services.auth_services.get_user_data')
    @patch('examples.db_management.services.auth_services.jwt.decode')
    async def test_logout_user_success(
        self, mock_decode, mock_get_user_data, mock_set_user_data,
    ) -> None:
        """Test logout_user properly updates cache."""
        mock_decode.return_value = {'username': 'user', 'jti': 'jti123'}
        mock_get_user_data.return_value = {
            'jti_list': ['jti123', 'jti456'],
            'refresh_tokens': ['token123', 'token456'],
        }

        await auth_services.logout_user(
            'token123', 'Bearer jwt.token.here', self.redis_pool,
        )

        mock_set_user_data.assert_awaited_with(
            self.redis_pool, 'user',
            {'jti_list': ['jti456'], 'refresh_tokens': ['token456']},
        )

    @patch('examples.db_management.services.auth_services.set_user_data')
    @patch('examples.db_management.services.auth_services.jwt_refresh')
    @patch('examples.db_management.services.auth_services.jwt_access')
    @patch('examples.db_management.services.auth_services.get_user_data')
    @patch(
        'examples.db_management.services.'
        'auth_services.verify_refresh_token',
    )
    async def test_refresh_tokens_success(
        self,
        mock_verify_refresh_token,
        mock_get_user_data,
        mock_jwt_access,
        mock_jwt_refresh,
        mock_set_user_data,
    ) -> None:
        """Test refresh_tokens generates tokens and updates cache."""
        mock_verify_refresh_token.return_value = {
            'subject': {'username': 'user'},
        }
        mock_get_user_data.return_value = {
            'db_user': {'id': 1, 'role': 'user'},
            'refresh_tokens': ['old_refresh'],
            'feature_names': ['feature1'],
            'jti_list': [],
        }
        mock_jwt_access.create_access_token.return_value = 'new_access'
        mock_jwt_refresh.create_access_token.return_value = 'new_refresh'

        payload = RefreshRequest(refresh_token='old_refresh')
        result = await auth_services.refresh_tokens(payload, self.redis_pool)

        self.assertEqual(
            result, {
                'access_token': 'new_access',
                'refresh_token': 'new_refresh',
                'feature_names': ['feature1'],
            },
        )
        mock_set_user_data.assert_awaited()


if __name__ == '__main__':
    unittest.main()


'''
pytest --cov=examples.db_management.services.auth_services\
    --cov-report=term-missing\
        tests/examples/db_management/services/auth_services_test.py
'''
