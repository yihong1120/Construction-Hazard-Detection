from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import httpx
import jwt
from fastapi import HTTPException

from examples.db_management.schemas.auth import RefreshRequest
from examples.db_management.schemas.auth import RefreshTokenPayload
from examples.db_management.schemas.auth import UserLogin
from examples.db_management.services import auth_services
from examples.db_management.services import auth_services as svc


class TestAuthServices(unittest.IsolatedAsyncioTestCase):
    """
    Unit tests for auth_services module using asynchronous mocks.
    """

    def setUp(self) -> None:
        """Set up common mock objects for each test.

        This method initialises mock database and Redis pool objects for
        use in each test case.
        """
        self.db: AsyncMock = AsyncMock()
        self.redis_pool: AsyncMock = AsyncMock()
        self._original_hcaptcha_enabled = (
            auth_services.settings.hcaptcha_enabled
        )
        auth_services.settings.hcaptcha_enabled = True

    def tearDown(self) -> None:
        """Restore module-level settings changed during tests."""
        auth_services.settings.hcaptcha_enabled = (
            self._original_hcaptcha_enabled
        )

    @patch('examples.db_management.services.auth_services._authenticate')
    @patch('examples.db_management.services.auth_services._load_feature_names')
    @patch('examples.db_management.services.auth_services.jwt_access')
    @patch('examples.db_management.services.auth_services.jwt_refresh')
    @patch('examples.db_management.services.auth_services.set_user_data')
    @patch('examples.db_management.services.auth_services._verify_hcaptcha')
    async def test_login_user_success(
        self,
        mock_verify_hcaptcha: AsyncMock,
        mock_set_user_data: AsyncMock,
        mock_jwt_refresh: MagicMock,
        mock_jwt_access: MagicMock,
        mock_load_feature_names: MagicMock,
        mock_authenticate: AsyncMock,
    ) -> None:
        """Test successful user login.

        Verifies that a user can log in and receives correct tokens and
        feature names.
        """
        user_mock: AsyncMock = AsyncMock(
            id=1,
            username='user',
            role='user',
            group_id=1,
            status='active',
        )
        mock_authenticate.return_value = user_mock
        mock_load_feature_names.return_value = ['feature1', 'feature2']
        mock_jwt_access.create_access_token.return_value = 'access_token'
        mock_jwt_refresh.create_access_token.return_value = 'refresh_token'

        mock_redis_data: str = (
            '{"db_user": {"id": 1, "username": "user", "role": "user", '
            '"group_id": 1, "status": "active"}, "jti_list": [], '
            '"refresh_tokens": []}'
        )
        self.redis_pool.get = AsyncMock(
            side_effect=[None, None, mock_redis_data, mock_redis_data],
        )

        payload: UserLogin = UserLogin(
            identifier='user',
            password='pass',
            hcaptcha_token='captcha-token',
        )
        result = await auth_services.login_user(
            payload,
            self.db,
            self.redis_pool,
        )

        self.assertEqual(result['access_token'], 'access_token')
        self.assertEqual(result['refresh_token'], 'refresh_token')
        self.assertEqual(result['username'], 'user')
        self.assertEqual(result['feature_names'], ['feature1', 'feature2'])
        mock_verify_hcaptcha.assert_awaited_once_with('captcha-token', None)
        mock_set_user_data.assert_awaited()

    @patch('examples.db_management.services.auth_services._authenticate')
    @patch('examples.db_management.services.auth_services._verify_hcaptcha')
    async def test_login_user_invalid_credentials_reports_remaining_attempts(
        self,
        mock_verify_hcaptcha: AsyncMock,
        mock_authenticate: AsyncMock,
    ) -> None:
        """Test wrong credentials return structured remaining attempts."""
        self.redis_pool.get = AsyncMock(side_effect=[None, None])
        self.redis_pool.incr = AsyncMock(side_effect=[1, 1])
        self.redis_pool.sadd = AsyncMock()
        self.redis_pool.expire = AsyncMock()
        mock_authenticate.side_effect = HTTPException(
            status_code=401,
            detail='Wrong username/e-mail or password',
        )
        payload = UserLogin(
            identifier='user',
            password='bad',
            hcaptcha_token='captcha-token',
        )

        with self.assertRaises(HTTPException) as ctx:
            await auth_services.login_user(
                payload,
                self.db,
                self.redis_pool,
                client_ip='127.0.0.1',
            )

        self.assertEqual(ctx.exception.status_code, 401)
        self.assertEqual(
            ctx.exception.detail,
            {'code': 'invalid_credentials', 'remaining_attempts': 4},
        )
        mock_verify_hcaptcha.assert_awaited_once_with('captcha-token', None)
        self.redis_pool.sadd.assert_awaited_once()
        self.assertEqual(self.redis_pool.expire.await_count, 3)

    @patch('examples.db_management.services.auth_services.settings')
    @patch('examples.db_management.services.auth_services._authenticate')
    @patch('examples.db_management.services.auth_services._verify_hcaptcha')
    async def test_login_user_enters_cooldown(
        self,
        mock_verify_hcaptcha: AsyncMock,
        mock_authenticate: AsyncMock,
        mock_settings: MagicMock,
    ) -> None:
        """Test hitting cooldown threshold returns HTTP 429."""
        mock_settings.login_failure_window_seconds = 1800
        mock_settings.login_cooldown_threshold = 5
        mock_settings.login_cooldown_seconds = 300
        mock_settings.login_lock_threshold = 10
        mock_settings.login_lock_seconds = 1800
        self.redis_pool.get = AsyncMock(side_effect=[None, None])
        self.redis_pool.incr = AsyncMock(side_effect=[5, 5])
        self.redis_pool.sadd = AsyncMock()
        self.redis_pool.expire = AsyncMock()
        self.redis_pool.set = AsyncMock()
        mock_authenticate.side_effect = HTTPException(
            status_code=401,
            detail='Wrong username/e-mail or password',
        )
        payload = UserLogin(
            identifier='user',
            password='bad',
            hcaptcha_token='captcha-token',
        )

        with self.assertRaises(HTTPException) as ctx:
            await auth_services.login_user(
                payload,
                self.db,
                self.redis_pool,
                client_ip='127.0.0.1',
            )

        self.assertEqual(ctx.exception.status_code, 429)
        self.assertEqual(
            ctx.exception.detail,
            {'code': 'login_cooldown', 'retry_after_seconds': 300},
        )
        self.assertEqual(ctx.exception.headers, {'Retry-After': '300'})
        self.redis_pool.set.assert_awaited_once()
        self.redis_pool.sadd.assert_awaited_once()
        mock_verify_hcaptcha.assert_awaited_once_with('captcha-token', None)

    @patch('examples.db_management.services.auth_services.settings')
    @patch('examples.db_management.services.auth_services._verify_hcaptcha')
    async def test_login_user_rejects_existing_cooldown(
        self,
        mock_verify_hcaptcha: AsyncMock,
        mock_settings: MagicMock,
    ) -> None:
        """Test existing cooldown is rejected before credential check."""
        mock_settings.login_cooldown_seconds = 300
        self.redis_pool.get = AsyncMock(side_effect=[None, '1'])
        self.redis_pool.ttl = AsyncMock(return_value=123)
        payload = UserLogin(
            identifier='user',
            password='pw',
            hcaptcha_token='captcha-token',
        )

        with self.assertRaises(HTTPException) as ctx:
            await auth_services.login_user(
                payload,
                self.db,
                self.redis_pool,
                client_ip='127.0.0.1',
            )

        self.assertEqual(ctx.exception.status_code, 429)
        self.assertEqual(
            ctx.exception.detail,
            {'code': 'login_cooldown', 'retry_after_seconds': 123},
        )
        self.redis_pool.ttl.assert_awaited_once()
        mock_verify_hcaptcha.assert_awaited_once_with('captcha-token', None)

    @patch('examples.db_management.services.auth_services._verify_hcaptcha')
    async def test_login_user_rejects_existing_lock(
        self,
        mock_verify_hcaptcha: AsyncMock,
    ) -> None:
        """Test existing account lock returns HTTP 423."""
        locked_until = '2026-06-19T12:30:00Z'
        self.redis_pool.get = AsyncMock(return_value=locked_until)
        payload = UserLogin(
            identifier='user',
            password='pw',
            hcaptcha_token='captcha-token',
        )

        with self.assertRaises(HTTPException) as ctx:
            await auth_services.login_user(
                payload,
                self.db,
                self.redis_pool,
                client_ip='127.0.0.1',
            )

        self.assertEqual(ctx.exception.status_code, 423)
        self.assertEqual(
            ctx.exception.detail,
            {'code': 'account_locked', 'locked_until': locked_until},
        )
        mock_verify_hcaptcha.assert_awaited_once_with('captcha-token', None)

    async def test_verify_hcaptcha_missing_token(self) -> None:
        """Test missing hCaptcha token raises HTTP 400."""
        with self.assertRaises(HTTPException) as ctx:
            await auth_services._verify_hcaptcha('')

        self.assertEqual(ctx.exception.status_code, 400)

    @patch(
        'examples.db_management.services.auth_services.HCAPTCHA_BYPASS_KEY',
        'server-only-key',
    )
    @patch('examples.db_management.services.auth_services.httpx.AsyncClient')
    async def test_verify_hcaptcha_backend_bypass(
        self,
        mock_async_client: MagicMock,
    ) -> None:
        """Test trusted backend bypass skips external hCaptcha call."""
        await auth_services._verify_hcaptcha(None, 'server-only-key')

        mock_async_client.assert_not_called()

    @patch(
        'examples.db_management.services.auth_services.HCAPTCHA_SITE_KEY',
        'site-key',
    )
    @patch(
        'examples.db_management.services.auth_services.HCAPTCHA_SECRET_KEY',
        'secret-key',
    )
    @patch('examples.db_management.services.auth_services.httpx.AsyncClient')
    async def test_verify_hcaptcha_success(
        self,
        mock_async_client: MagicMock,
    ) -> None:
        """Test successful hCaptcha verification returns without error."""
        mock_response: MagicMock = MagicMock()
        mock_response.json.return_value = {'success': True}
        mock_client: AsyncMock = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_async_client.return_value.__aenter__.return_value = mock_client

        await auth_services._verify_hcaptcha('captcha-token')

        mock_client.post.assert_awaited_once_with(
            auth_services.HCAPTCHA_VERIFY_URL,
            data={
                'secret': 'secret-key',
                'response': 'captcha-token',
                'sitekey': 'site-key',
            },
        )
        mock_response.raise_for_status.assert_called_once()

    @patch(
        'examples.db_management.services.auth_services.HCAPTCHA_SITE_KEY',
        'site-key',
    )
    @patch(
        'examples.db_management.services.auth_services.HCAPTCHA_SECRET_KEY',
        'secret-key',
    )
    @patch('examples.db_management.services.auth_services.httpx.AsyncClient')
    async def test_verify_hcaptcha_failure(
        self,
        mock_async_client: MagicMock,
    ) -> None:
        """Test failed hCaptcha verification raises HTTP 403."""
        mock_response: MagicMock = MagicMock()
        mock_response.json.return_value = {'success': False}
        mock_client: AsyncMock = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_async_client.return_value.__aenter__.return_value = mock_client

        with self.assertRaises(HTTPException) as ctx:
            await auth_services._verify_hcaptcha('captcha-token')

        self.assertEqual(ctx.exception.status_code, 403)

    async def test_authenticate_invalid_credentials(self) -> None:
        """Test authentication with wrong credentials.

        Ensures that an HTTPException with status 401 is raised if the
        credentials are invalid.
        """
        self.db.scalar = AsyncMock(return_value=None)
        with self.assertRaises(HTTPException) as ctx:
            await auth_services._authenticate(
                self.db,
                'wronguser',
                'wrongpass',
            )
        self.assertEqual(ctx.exception.status_code, 401)

    async def test_authenticate_with_email(self) -> None:
        """Test authentication falls back to profile e-mail lookup."""
        mock_user: AsyncMock = AsyncMock()
        mock_user.check_password = AsyncMock(return_value=True)
        mock_user.status = 'active'
        self.db.scalar = AsyncMock(side_effect=[None, mock_user])

        result = await auth_services._authenticate(
            self.db,
            'USER@example.com',
            'pass',
        )

        self.assertIs(result, mock_user)
        self.assertEqual(self.db.scalar.await_count, 2)

    async def test_authenticate_inactive_user(self) -> None:
        """Test authentication with inactive user.

        Ensures that an HTTPException with status 403 is raised if the
        user is inactive.
        """
        mock_user: AsyncMock = AsyncMock()
        mock_user.check_password = AsyncMock(return_value=True)
        mock_user.status = 'inactive'
        self.db.scalar = AsyncMock(return_value=mock_user)

        with self.assertRaises(HTTPException) as ctx:
            await auth_services._authenticate(self.db, 'user', 'pass')
        self.assertEqual(ctx.exception.status_code, 403)

    async def test_load_feature_names_none_group(self) -> None:
        """Test _load_feature_names returns empty list if group_id is None.

        Ensures that an empty list is returned if no group_id is
        provided.
        """
        result: list = await auth_services._load_feature_names(self.db, None)
        self.assertEqual(result, [])

    @patch(
        'examples.db_management.services.auth_services.'
        'jwt_refresh.decode_token',
    )
    async def test_verify_refresh_token_expired(
        self,
        mock_decode: MagicMock,
    ) -> None:
        """Test expired refresh token raises HTTPException.

        Ensures that an HTTPException with status 401 is raised if the
        refresh token is expired.
        """
        mock_decode.side_effect = jwt.ExpiredSignatureError()
        with self.assertRaises(HTTPException) as ctx:
            await auth_services.verify_refresh_token(
                'expired',
                self.redis_pool,
            )
        self.assertEqual(ctx.exception.status_code, 401)

    @patch(
        'examples.db_management.services.auth_services.'
        'jwt_refresh.decode_token',
    )
    async def test_verify_refresh_token_invalid(
        self,
        mock_decode: MagicMock,
    ) -> None:
        """Test invalid refresh token raises HTTPException.

        Ensures that an HTTPException with status 401 is raised if the
        refresh token is invalid.
        """
        mock_decode.side_effect = jwt.InvalidTokenError()
        with self.assertRaises(HTTPException) as ctx:
            await auth_services.verify_refresh_token(
                'invalid',
                self.redis_pool,
            )
        self.assertEqual(ctx.exception.status_code, 401)

    @patch('examples.db_management.services.auth_services.jwt_refresh')
    async def test_verify_refresh_token_missing_username(
        self,
        mock_jwt: MagicMock,
    ) -> None:
        """Test missing username in payload raises HTTPException.

        Ensures that an HTTPException with status 401 is raised if the
        username is missing from the token payload.
        """
        mock_jwt.decode_token.return_value = {'subject': {}}
        with self.assertRaises(HTTPException) as ctx:
            await auth_services.verify_refresh_token('token', self.redis_pool)
        self.assertEqual(ctx.exception.status_code, 401)

    @patch('examples.db_management.services.auth_services.jwt_refresh')
    @patch('examples.db_management.services.auth_services.get_user_data')
    async def test_verify_refresh_token_not_recognised(
        self,
        mock_get_user_data: MagicMock,
        mock_jwt: MagicMock,
    ) -> None:
        """Test unrecognised refresh token raises HTTPException.

        Ensures that an HTTPException with status 401 is raised if the
        refresh token is not recognised in the cache.
        """
        mock_jwt.decode_token.return_value = {'subject': {'username': 'user'}}
        mock_get_user_data.return_value = {'refresh_tokens': []}
        with self.assertRaises(HTTPException) as ctx:
            await auth_services.verify_refresh_token(
                'unknown',
                self.redis_pool,
            )
        self.assertEqual(ctx.exception.status_code, 401)

    async def test_logout_user_no_auth(self) -> None:
        """Test logout_user with missing authorisation header returns early.

        Ensures that logout_user returns early if the authorisation
        header is missing.
        """
        await auth_services.logout_user('token', None, self.redis_pool)

    async def test_logout_user_bad_auth_format(self) -> None:
        """
        Test logout_user with malformed authorisation header returns early.

        Ensures that logout_user returns early if the authorisation
        header is malformed.
        """
        await auth_services.logout_user(
            'token',
            'invalidtoken',
            self.redis_pool,
        )

    @patch(
        'examples.db_management.services.auth_services.'
        'jwt_access.decode_token',
    )
    async def test_logout_user_invalid_jwt(
        self,
        mock_decode: MagicMock,
    ) -> None:
        """Test logout_user with invalid JWT returns early.

        Ensures that logout_user returns early if the JWT is invalid.
        """
        mock_decode.side_effect = jwt.PyJWTError()
        await auth_services.logout_user(
            'token',
            'Bearer abc.def.ghi',
            self.redis_pool,
        )

    @patch(
        'examples.db_management.services.auth_services.get_user_data',
    )
    @patch('examples.db_management.services.auth_services.jwt_access')
    async def test_logout_user_no_cache(
        self,
        mock_jwt: MagicMock,
        mock_get_user_data: MagicMock,
    ) -> None:
        """Test logout_user with no cache found returns early.

        Ensures that logout_user returns early if no cache is found for
        the user.
        """
        mock_jwt.decode_token.return_value = {
            'username': 'user',
            'jti': 'id',
        }
        mock_get_user_data.return_value = None
        await auth_services.logout_user(
            'token',
            'Bearer valid.token',
            self.redis_pool,
        )

    async def test_refresh_tokens_missing_token(self) -> None:
        """Test refresh_tokens raises if refresh token is missing.

        Ensures that an HTTPException with status 401 is raised if the
        refresh token is missing from the request.
        """
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
        self,
        mock_verify: MagicMock,
        mock_get_user_data: MagicMock,
    ) -> None:
        """Test refresh_tokens raises if cache is invalid or missing token.

        Ensures that an HTTPException with status 401 is raised if the
        cache is invalid or the token is missing.
        """
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

        Ensures that the correct feature names are returned for a valid
        group_id.
        """
        mock_result: MagicMock = MagicMock()
        mock_result.__iter__.return_value = [
            MagicMock(feature_name='feature1'),
            MagicMock(
                feature_name='feature2',
            ),
        ]
        self.db.execute = AsyncMock(return_value=mock_result)

        features: list[str] = await auth_services._load_feature_names(
            self.db,
            group_id=1,
        )
        self.assertEqual(features, ['feature1', 'feature2'])

    async def test_authenticate_success(self) -> None:
        """
        Test _authenticate returns user object when credentials are valid.

        Ensures that the user object is returned if the credentials are
        valid and the user is active.
        """
        mock_user: MagicMock = MagicMock(status='active')
        mock_user.check_password = AsyncMock(return_value=True)
        self.db.scalar = AsyncMock(return_value=mock_user)

        user = await auth_services._authenticate(
            self.db,
            'valid_user',
            'valid_password',
        )
        self.assertEqual(user, mock_user)

    @patch(
        'examples.db_management.services.auth_services.'
        'jwt_refresh.decode_token',
    )
    async def test_verify_refresh_token_success(
        self,
        mock_decode: MagicMock,
    ) -> None:
        """Test verify_refresh_token returns payload correctly.

        Ensures that the payload is returned correctly if the refresh
        token is valid.
        """
        mock_decode.return_value = {'subject': {'username': 'user'}}
        mock_cache_data: str = '{"refresh_tokens": ["valid_token"]}'
        self.redis_pool.get = AsyncMock(return_value=mock_cache_data)
        payload: RefreshTokenPayload = (
            await auth_services.verify_refresh_token(
                'valid_token',
                self.redis_pool,
            )
        )
        self.assertEqual(
            payload,
            {'subject': {'username': 'user'}},
        )

    @patch('examples.db_management.services.auth_services.set_user_data')
    @patch('examples.db_management.services.auth_services.get_user_data')
    @patch(
        'examples.db_management.services.auth_services.'
        'jwt_access.decode_token',
    )
    async def test_logout_user_success(
        self,
        mock_decode: MagicMock,
        mock_get_user_data: MagicMock,
        mock_set_user_data: AsyncMock,
    ) -> None:
        """Test logout_user properly updates cache.

        Ensures that logout_user updates the cache correctly when a user
        logs out.
        """
        mock_decode.return_value = {'username': 'user', 'jti': 'jti123'}
        mock_get_user_data.return_value = {
            'jti_list': ['jti123', 'jti456'],
            'refresh_tokens': ['token123', 'token456'],
        }

        await auth_services.logout_user(
            'token123',
            'Bearer jwt.token.here',
            self.redis_pool,
        )

        mock_set_user_data.assert_awaited_with(
            self.redis_pool,
            'user',
            {
                'jti_list': ['jti456'],
                'refresh_tokens': ['token456'],
                'jti_meta': {},
                'refresh_token_hashes': [],
            },
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
        mock_verify_refresh_token: MagicMock,
        mock_get_user_data: MagicMock,
        mock_jwt_access: MagicMock,
        mock_jwt_refresh: MagicMock,
        mock_set_user_data: AsyncMock,
    ) -> None:
        """Test refresh_tokens generates tokens and updates cache.

        Verifies that new tokens are generated and the cache is updated
        correctly when refreshing tokens.
        """
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

        # Simulate decode success with known exp
        mock_jwt_access.decode_token.return_value = {'exp': 456}

        payload: RefreshRequest = RefreshRequest(refresh_token='old_refresh')
        result = await auth_services.refresh_tokens(
            payload,
            self.redis_pool,
        )

        self.assertEqual(
            result,
            {
                'access_token': 'new_access',
                'refresh_token': 'new_refresh',
                'feature_names': ['feature1'],
            },
        )
        mock_set_user_data.assert_awaited()
        await_call = mock_set_user_data.await_args
        assert await_call is not None
        cache_arg = await_call.args[2]
        assert isinstance(cache_arg, dict)
        assert 'jti_meta' in cache_arg
        assert 456 in cache_arg['jti_meta'].values()

    @patch('examples.db_management.services.auth_services.set_user_data')
    @patch('examples.db_management.services.auth_services.jwt_refresh')
    @patch('examples.db_management.services.auth_services.jwt_access')
    @patch('examples.db_management.services.auth_services.get_user_data')
    @patch(
        'examples.db_management.services.'
        'auth_services.verify_refresh_token',
    )
    async def test_refresh_tokens_hashes_web_refresh_token(
        self,
        mock_verify_refresh_token: MagicMock,
        mock_get_user_data: MagicMock,
        mock_jwt_access: MagicMock,
        mock_jwt_refresh: MagicMock,
        mock_set_user_data: AsyncMock,
    ) -> None:
        """Web refresh rotation stores only refresh token hashes."""
        old_hash = auth_services._hash_refresh_token('old_refresh')
        mock_verify_refresh_token.return_value = {
            'subject': {'username': 'user'},
        }
        mock_get_user_data.return_value = {
            'db_user': {'id': 1, 'role': 'user'},
            'refresh_tokens': [],
            'refresh_token_hashes': [old_hash],
            'feature_names': ['feature1'],
            'jti_list': [],
        }
        mock_jwt_access.create_access_token.return_value = 'new_access'
        mock_jwt_refresh.create_access_token.return_value = 'new_refresh'
        mock_jwt_access.decode_token.return_value = {'exp': 456}

        result = await auth_services.refresh_tokens(
            RefreshRequest(refresh_token='old_refresh'),
            self.redis_pool,
            hash_refresh_token=True,
        )

        self.assertEqual(result['refresh_token'], 'new_refresh')
        await_call = mock_set_user_data.await_args
        assert await_call is not None
        cache_arg = await_call.args[2]
        self.assertEqual(cache_arg['refresh_tokens'], [])
        self.assertNotIn(old_hash, cache_arg['refresh_token_hashes'])
        self.assertIn(
            auth_services._hash_refresh_token('new_refresh'),
            cache_arg['refresh_token_hashes'],
        )

    @patch('examples.db_management.services.auth_services._load_feature_names')
    @patch('examples.db_management.services.auth_services.jwt_access')
    @patch('examples.db_management.services.auth_services.jwt_refresh')
    @patch('examples.db_management.services.auth_services.set_user_data')
    @patch('examples.db_management.services.auth_services._authenticate')
    @patch('examples.db_management.services.auth_services._verify_hcaptcha')
    async def test_login_user_jti_meta_decode_success(
        self,
        mock_verify_hcaptcha: AsyncMock,
        mock_authenticate: AsyncMock,
        mock_set_user_data: AsyncMock,
        mock_jwt_refresh: MagicMock,
        mock_jwt_access: MagicMock,
        mock_load_features: MagicMock,
    ) -> None:
        """Cover success path when decoding access token for jti_meta.

        Ensures that jti expiry is stored into ``jti_meta`` when decode
        succeeds.
        """
        user_mock: AsyncMock = AsyncMock(
            id=1,
            username='user',
            role='user',
            group_id=1,
            status='active',
        )
        mock_authenticate.return_value = user_mock
        mock_load_features.return_value = ['f1']
        mock_jwt_access.create_access_token.return_value = 'acc'
        mock_jwt_refresh.create_access_token.return_value = 'ref'
        mock_jwt_access.decode_token.return_value = {'exp': 123}
        self.redis_pool.get = AsyncMock(return_value=None)
        payload: UserLogin = UserLogin(
            identifier='user',
            password='pw',
            hcaptcha_token='captcha-token',
        )
        await auth_services.login_user(payload, self.db, self.redis_pool)

        mock_verify_hcaptcha.assert_awaited_once_with('captcha-token', None)
        mock_set_user_data.assert_awaited()
        await_call = mock_set_user_data.await_args
        assert await_call is not None
        cache_arg = await_call.args[2]
        assert isinstance(cache_arg, dict)
        assert 'jti_meta' in cache_arg
        # Should contain exactly one JTI mapped to the exp 123
        assert 123 in cache_arg['jti_meta'].values()

    @patch('examples.db_management.services.auth_services._load_feature_names')
    @patch('examples.db_management.services.auth_services.jwt_access')
    @patch('examples.db_management.services.auth_services.jwt_refresh')
    @patch('examples.db_management.services.auth_services.set_user_data')
    @patch('examples.db_management.services.auth_services._authenticate')
    @patch('examples.db_management.services.auth_services._verify_hcaptcha')
    async def test_login_user_jti_meta_decode_failure(
        self,
        mock_verify_hcaptcha: AsyncMock,
        mock_authenticate: AsyncMock,
        mock_set_user_data: AsyncMock,
        mock_jwt_refresh: MagicMock,
        mock_jwt_access: MagicMock,
        mock_load_features: MagicMock,
    ) -> None:
        """Cover exception path when decoding access token for jti_meta.

        Ensures the function continues gracefully when ``jwt.decode`` raises
        during access-token decoding used only for jti expiry bookkeeping.
        """
        user_mock: AsyncMock = AsyncMock(
            id=1,
            username='user',
            role='user',
            group_id=1,
            status='active',
        )
        mock_authenticate.return_value = user_mock
        mock_load_features.return_value = ['f1']
        mock_jwt_access.create_access_token.return_value = 'acc'
        mock_jwt_refresh.create_access_token.return_value = 'ref'
        # Force decode failure inside login_user jti_meta block
        mock_jwt_access.decode_token.side_effect = Exception('decode-fail')
        self.redis_pool.get = AsyncMock(return_value=None)

        payload: UserLogin = UserLogin(
            identifier='user',
            password='pw',
            hcaptcha_token='captcha-token',
        )
        result = await auth_services.login_user(
            payload,
            self.db,
            self.redis_pool,
        )

        self.assertEqual(result['access_token'], 'acc')
        self.assertEqual(result['refresh_token'], 'ref')
        self.assertEqual(result['feature_names'], ['f1'])
        mock_verify_hcaptcha.assert_awaited_once_with('captcha-token', None)
        mock_set_user_data.assert_awaited()

    @patch('examples.db_management.services.auth_services.set_user_data')
    @patch('examples.db_management.services.auth_services.get_user_data')
    @patch(
        'examples.db_management.services.auth_services.'
        'jwt_access.decode_token',
    )
    async def test_logout_user_jti_meta_pop_failure(
        self,
        mock_jwt_decode: MagicMock,
        mock_get_user_data: MagicMock,
        mock_set_user_data: AsyncMock,
    ) -> None:
        """Cover exception path when popping jti_meta fails.

        Sets ``jti_meta`` to a non-mapping so that ``pop`` raises, ensuring
        the broad ``except`` is exercised.
        """
        mock_jwt_decode.return_value = {
            'username': 'user',
            'jti': 'abc',
        }
        mock_get_user_data.return_value = {
            'jti_list': ['abc'],
            'refresh_tokens': ['tok'],
            'jti_meta': 123,  # not a mapping → AttributeError on pop
        }

        await auth_services.logout_user(
            'tok',
            'Bearer x.y.z',
            self.redis_pool,
        )

        mock_set_user_data.assert_awaited()

    @patch('examples.db_management.services.auth_services.set_user_data')
    @patch('examples.db_management.services.auth_services.jwt_refresh')
    @patch('examples.db_management.services.auth_services.jwt_access')
    @patch('examples.db_management.services.auth_services.get_user_data')
    @patch(
        'examples.db_management.services.auth_services.verify_refresh_token',
    )
    async def test_refresh_tokens_jti_meta_decode_failure(
        self,
        mock_verify: MagicMock,
        mock_get_user_data: MagicMock,
        mock_jwt_access: MagicMock,
        mock_jwt_refresh: MagicMock,
        mock_set_user_data: AsyncMock,
    ) -> None:
        """Cover exception path when decoding access token in refresh fails.

        Ensures that token refresh still succeeds even if jti expiry decoding
        fails; jti bookkeeping is best-effort.
        """
        mock_verify.return_value = {'subject': {'username': 'user'}}
        mock_get_user_data.return_value = {
            'db_user': {'id': 1, 'role': 'user'},
            'refresh_tokens': ['old'],
            'feature_names': ['f1'],
            'jti_list': [],
        }
        mock_jwt_access.create_access_token.return_value = 'new_acc'
        mock_jwt_refresh.create_access_token.return_value = 'new_ref'
        # Force decode failure inside refresh_tokens jti_meta block
        mock_jwt_access.decode_token.side_effect = Exception('decode-fail')

        req = RefreshRequest(refresh_token='old')
        result = await auth_services.refresh_tokens(req, self.redis_pool)

        self.assertEqual(result['access_token'], 'new_acc')
        self.assertEqual(result['refresh_token'], 'new_ref')
        self.assertEqual(result['feature_names'], ['f1'])
        mock_set_user_data.assert_awaited()


if __name__ == '__main__':
    unittest.main()

"""
pytest --cov=examples.db_management.services.auth_services\
    --cov-report=term-missing\
        tests/examples/db_management/services/auth_services_test.py
"""


class TestAuthServicesCoverage(unittest.IsolatedAsyncioTestCase):
    async def test_login_guard_helpers_and_identifier_cleanup(self) -> None:
        self.assertTrue(svc._login_fail_pair_key('pair').endswith('pair'))
        self.assertTrue(svc._login_cooldown_pair_key('pair').endswith('pair'))
        self.assertEqual(svc._decode_redis_value(None), None)
        self.assertEqual(svc._decode_redis_value(b'value'), 'value')
        self.assertEqual(svc._decode_redis_value('value'), 'value')
        self.assertEqual(svc._decode_redis_value(42), '42')
        self.assertEqual(svc._decode_redis_members(object()), [])
        self.assertEqual(
            svc._decode_redis_members([b'one', 'two', None]),
            ['one', 'two'],
        )

        redis = AsyncMock()
        redis.smembers.return_value = [b'first', 'second']
        await svc.clear_login_guard_for_identifier(redis, 'alice')
        deleted = redis.delete.await_args.args
        self.assertIn(svc._login_fail_pair_key('first'), deleted)
        self.assertIn(svc._login_cooldown_pair_key('second'), deleted)

        with patch.object(
            svc, 'clear_login_guard_for_identifier', AsyncMock(),
        ) as clear:
            await svc.clear_login_guard_for_identifiers(
                redis,
                [' Alice ', 'alice', '', 'BOB'],
            )
        self.assertEqual(clear.await_args_list[0].args[1], 'alice')
        self.assertEqual(clear.await_args_list[1].args[1], 'bob')

    async def test_logout_revokes_refresh_family_without_user_cache(
        self,
    ) -> None:
        """Refresh-only logout revokes its family before an evicted cache.

        The refresh family must be revoked before cache eviction can return.
        """
        redis = AsyncMock()
        with (
            patch.object(
                svc.jwt_refresh,
                'decode_token',
                return_value={
                    'subject': {
                        'username': 'alice',
                        'family_id': 'family-1',
                    },
                },
            ),
            patch.object(svc, 'prune_user_cache', AsyncMock()),
            patch.object(svc, 'get_user_data', AsyncMock(return_value=None)),
            patch.object(svc, '_revoke_refresh_family', AsyncMock()) as revoke,
        ):
            await svc.logout_user('refresh-token', None, redis)

        revoke.assert_awaited_once_with(redis, 'family-1')

    async def test_revoke_user_access_tokens_ignores_invalid_metadata(
        self,
    ) -> None:
        """Broken legacy cache metadata cannot be sent to Redis revocation."""
        redis = AsyncMock()
        with patch.object(
            svc,
            'get_user_data',
            AsyncMock(return_value={'jti_meta': ['not-a-mapping']}),
        ):
            assert await svc._revoke_user_access_tokens(redis, 'alice') == 0

    async def test_failed_login_locks_account(self) -> None:
        redis = AsyncMock()
        redis.incr.side_effect = [1, 3]
        settings = SimpleNamespace(
            login_failure_window_seconds=60,
            login_cooldown_seconds=30,
            login_lock_seconds=120,
            login_cooldown_threshold=5,
            login_lock_threshold=3,
        )
        with patch.object(svc, 'settings', settings):
            with self.assertRaises(HTTPException) as raised:
                await svc._record_failed_login(redis, 'alice', '127.0.0.1')
        self.assertEqual(raised.exception.status_code, 423)
        self.assertEqual(raised.exception.detail['code'], 'account_locked')
        self.assertTrue(raised.exception.detail['locked_until'].endswith('Z'))
        redis.delete.assert_awaited_once()

    async def test_authenticate_rejects_each_nonactive_status(self) -> None:
        for status, code in [
            (svc.USER_STATUS_EMAIL_UNVERIFIED, 'email_unverified'),
            (svc.USER_STATUS_PENDING, 'pending_admin_approval'),
            (svc.USER_STATUS_REJECTED, 'account_rejected'),
            (svc.USER_STATUS_SUSPENDED, 'account_suspended'),
        ]:
            user = SimpleNamespace(
                status=status,
                check_password=AsyncMock(return_value=True),
            )
            db = AsyncMock()
            db.scalar.return_value = user
            with self.assertRaises(HTTPException) as raised:
                await svc._authenticate(db, 'alice', 'password')
            self.assertEqual(raised.exception.detail['code'], code)

    async def test_hcaptcha_disabled_unconfigured_and_transport_failure(
        self,
    ) -> None:
        with patch.object(svc.settings, 'hcaptcha_enabled', False):
            await svc._verify_hcaptcha(None)

        with patch.object(svc.settings, 'hcaptcha_enabled', True):
            with patch.object(svc, 'HCAPTCHA_SECRET_KEY', ''):
                with patch.object(svc, 'HCAPTCHA_SITE_KEY', ''):
                    with self.assertRaises(HTTPException) as raised:
                        await svc._verify_hcaptcha('token')
        self.assertEqual(raised.exception.status_code, 500)

        client = AsyncMock()
        client.post.side_effect = httpx.ConnectError('unavailable')
        context = MagicMock()
        context.__aenter__ = AsyncMock(return_value=client)
        context.__aexit__ = AsyncMock(return_value=None)
        with patch.object(svc.settings, 'hcaptcha_enabled', True):
            with patch.object(svc, 'HCAPTCHA_SECRET_KEY', 'secret'):
                with patch.object(svc, 'HCAPTCHA_SITE_KEY', 'site'):
                    with patch.object(
                        svc.httpx, 'AsyncClient', return_value=context,
                    ):
                        with self.assertRaises(HTTPException) as raised:
                            await svc._verify_hcaptcha('token')
        self.assertEqual(raised.exception.status_code, 403)

    async def test_refresh_token_reuse_protection(self) -> None:
        redis = AsyncMock()
        with patch.object(
            svc.jwt_refresh,
            'decode_token',
            return_value={
                'subject': {'username': 'alice', 'family_id': 'family'},
            },
        ):
            redis.get.return_value = '1'
            with self.assertRaises(HTTPException) as raised:
                await svc.verify_refresh_token('refresh', redis)
        self.assertEqual(raised.exception.detail, 'Refresh token reused')

        redis.get.return_value = None
        with patch.object(
            svc.jwt_refresh,
            'decode_token',
            return_value={
                'subject': {'username': 'alice', 'family_id': 'family'},
            },
        ):
            with patch.object(svc, 'prune_user_cache', AsyncMock()):
                with patch.object(
                    svc,
                    'get_user_data',
                    AsyncMock(return_value={'refresh_tokens': []}),
                ):
                    with patch.object(
                        svc, '_revoke_refresh_family', AsyncMock(),
                    ) as revoke:
                        with self.assertRaises(HTTPException) as raised:
                            await svc.verify_refresh_token('refresh', redis)
        self.assertEqual(raised.exception.detail, 'Refresh token reused')
        revoke.assert_awaited_once_with(redis, 'family')

    async def test_refresh_state_registration_and_consumption(self) -> None:
        redis = AsyncMock()
        redis.get.return_value = '1'
        with self.assertRaises(HTTPException) as raised:
            await svc._register_refresh_token_state(
                redis,
                'refresh',
                'alice',
                'family',
                enforce_family_active=True,
            )
        self.assertEqual(raised.exception.detail, 'Refresh token reused')

        await svc._revoke_refresh_family(redis, 'family')
        self.assertEqual(
            redis.set.await_args.args[0],
            svc._refresh_family_revoked_key(
                'family',
            ),
        )

        redis.get.return_value = None
        redis.set.return_value = False
        with patch.object(
            svc, '_revoke_refresh_family', AsyncMock(),
        ) as revoke:
            with self.assertRaises(HTTPException) as raised:
                await svc._consume_refresh_token_state(
                    redis, 'refresh', 'family', 'alice',
                )
        self.assertEqual(raised.exception.detail, 'Refresh token reused')
        revoke.assert_awaited_once_with(redis, 'family')

        redis.get.return_value = '1'
        with patch.object(svc, '_revoke_user_access_tokens', AsyncMock()):
            with self.assertRaises(HTTPException) as raised:
                await svc._consume_refresh_token_state(
                    redis,
                    'refresh',
                    'family',
                    'alice',
                )
        self.assertEqual(raised.exception.detail, 'Refresh token reused')

        redis.set.return_value = True
        redis.get.side_effect = [None, b'not-json']
        with (
            patch.object(svc, '_revoke_user_access_tokens', AsyncMock()),
            patch.object(
                svc,
                '_revoke_refresh_family',
                AsyncMock(),
            ) as revoke,
        ):
            with self.assertRaises(HTTPException) as raised:
                await svc._consume_refresh_token_state(
                    redis, 'refresh', 'family', 'alice',
                )
        self.assertEqual(raised.exception.detail, 'Refresh token reused')
        revoke.assert_awaited_once_with(redis, 'family')

        redis.get.side_effect = [
            None,
            '{"status":"active","family_id":"family"}',
        ]
        await svc._consume_refresh_token_state(
            redis,
            'refresh',
            'family',
            'alice',
        )
        self.assertIn('"status":"used"', redis.set.await_args.args[1])

    async def test_login_and_logout_fallback_paths(self) -> None:
        payload = UserLogin(
            hcaptcha_token=None,
            identifier='alice',
            password='pw',
        )
        with patch.object(svc, '_verify_hcaptcha', AsyncMock()):
            with patch.object(svc, '_check_login_guard', AsyncMock()):
                with patch.object(
                    svc,
                    '_authenticate',
                    AsyncMock(
                        side_effect=HTTPException(
                            status_code=403,
                            detail='denied',
                        ),
                    ),
                ):
                    with self.assertRaises(HTTPException) as raised:
                        await svc.login_user(payload, AsyncMock(), AsyncMock())
        self.assertEqual(raised.exception.status_code, 403)

        redis = AsyncMock()
        cache = {'jti_list': ['jti'], 'refresh_tokens': ['refresh']}
        with (
            patch.object(
                svc.jwt_access,
                'decode_token',
                side_effect=jwt.PyJWTError(),
            ),
            patch.object(
                svc.jwt_refresh,
                'decode_token',
                return_value={'subject': {'username': 'alice'}},
            ),
        ):
            with patch.object(svc, 'prune_user_cache', AsyncMock()):
                with patch.object(
                    svc, 'get_user_data', AsyncMock(return_value=cache),
                ):
                    with patch.object(
                        svc, 'set_user_data', AsyncMock(),
                    ) as store:
                        await svc.logout_user(
                            'refresh', 'Bearer invalid', redis,
                        )
        assert store.await_args is not None
        self.assertEqual(store.await_args.args[1], 'alice')

        with patch.object(
            svc.jwt_refresh,
            'decode_token',
            return_value={'subject': {}},
        ):
            await svc.logout_user('refresh', None, redis)

    async def test_refresh_cache_reuse_with_family(self) -> None:
        redis = AsyncMock()
        with patch.object(
            svc,
            'verify_refresh_token',
            AsyncMock(
                return_value={
                    'subject': {
                        'username': 'alice',
                        'family_id': 'family',
                    },
                },
            ),
        ):
            with patch.object(svc, 'prune_user_cache', AsyncMock()):
                with patch.object(
                    svc,
                    'get_user_data',
                    AsyncMock(return_value={'refresh_tokens': []}),
                ):
                    with patch.object(
                        svc, '_revoke_refresh_family', AsyncMock(),
                    ) as revoke:
                        with self.assertRaises(HTTPException) as raised:
                            await svc.refresh_tokens(
                                RefreshRequest(refresh_token='old'), redis,
                            )
        self.assertEqual(raised.exception.detail, 'Refresh token reused')
        revoke.assert_awaited_once_with(redis, 'family')

        cache = {
            'db_user': {'id': 1, 'role': 'user'},
            'refresh_tokens': ['old'],
            'feature_names': [],
            'jti_list': [],
        }
        with patch.object(
            svc,
            'verify_refresh_token',
            AsyncMock(
                return_value={
                    'subject': {
                        'username': 'alice',
                        'family_id': 'family',
                    },
                },
            ),
        ):
            with patch.object(svc, 'prune_user_cache', AsyncMock()):
                with patch.object(
                    svc, 'get_user_data', AsyncMock(return_value=cache),
                ):
                    with patch.object(
                        svc, '_consume_refresh_token_state', AsyncMock(),
                    ) as consume:
                        with patch.object(svc, 'set_user_data', AsyncMock()):
                            with patch.object(
                                svc,
                                '_register_refresh_token_state',
                                AsyncMock(),
                            ):
                                with patch.object(
                                    svc.jwt_access,
                                    'create_access_token',
                                    return_value='access',
                                ):
                                    with patch.object(
                                        svc.jwt_refresh,
                                        'create_access_token',
                                        return_value='new-refresh',
                                    ):
                                        await svc.refresh_tokens(
                                            RefreshRequest(
                                                refresh_token='old',
                                            ),
                                            redis,
                                        )
        consume.assert_awaited_once_with(redis, 'old', 'family', 'alice')


if __name__ == '__main__':
    unittest.main()
