from __future__ import annotations

import unittest
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.testclient import TestClient

from examples.auth.database import get_db
from examples.auth.models import User
from examples.auth.redis_pool import get_redis_pool
from examples.db_management.deps import get_current_user
from examples.db_management.routers import auth
from examples.db_management.schemas.auth import LogoutRequest
from examples.db_management.schemas.auth import RefreshRequest
from examples.db_management.schemas.auth import UserLogin


class TestAuthRouter(unittest.IsolatedAsyncioTestCase):
    """Unit tests for the authentication router endpoints using FastAPI and
    unittest."""

    def setUp(self) -> None:
        """Set up the FastAPI app, TestClient, and dependency overrides for
        each test."""
        # Create FastAPI app and mount the auth router
        self.app: FastAPI = FastAPI()
        self.app.include_router(auth.router, prefix='/auth')
        # Mock Redis client in app state
        self.app.state.redis_client = MagicMock()
        self.client: TestClient = TestClient(self.app)

        async def override_get_db() -> AsyncGenerator[MagicMock]:
            """Override for get_db dependency, yields a mock DB session."""
            db_mock: MagicMock = MagicMock()
            # Avoid await db.scalar errors
            db_mock.scalar = AsyncMock(return_value=None)
            yield db_mock

        async def override_get_redis_pool() -> MagicMock:
            """Override for get_redis_pool dependency, returns a mock Redis
            pool."""
            redis_mock: MagicMock = MagicMock()
            # Avoid await redis.get errors
            redis_mock.get = AsyncMock(return_value=None)
            return redis_mock

        # Apply dependency overrides
        self.app.dependency_overrides[get_db] = override_get_db
        self.app.dependency_overrides[get_redis_pool] = override_get_redis_pool
        self.current_user = MagicMock(spec=User)
        self.current_user.id = 1
        self.current_user.username = 'testuser'
        self.current_user.password_hash = 'argon2-hash'

        async def override_get_current_user() -> MagicMock:
            """Perform override get current user.

            Returns:
                The callable result.
            """
            return self.current_user

        self.app.dependency_overrides[get_current_user] = (
            override_get_current_user
        )

    @patch(
        'examples.db_management.routers.auth.login_user',
        new_callable=AsyncMock,
    )
    async def test_login_success(self, mock_login_user: AsyncMock) -> None:
        """Test login success.

        Args:
            mock_login_user: Value used by this callable.
        """
        mock_login_user.return_value = {
            'access_token': 'access123',
            'refresh_token': 'refresh123',
            'username': 'testuser',
            'role': 'user',
            'user_id': 1,
            'group_id': 2,
            'feature_names': ['f1', 'f2'],
        }
        payload: UserLogin = UserLogin(
            identifier='testuser',
            password='pw',
            hcaptcha_token='captcha-token',
        )
        response = self.client.post('/auth/login', json=payload.model_dump())
        self.assertEqual(response.status_code, 200)
        data: dict = response.json()
        self.assertIn('access_token', data)
        self.assertIn('refresh_token', data)
        self.assertEqual(data['username'], 'testuser')
        self.assertIsNone(
            mock_login_user.call_args.kwargs['hcaptcha_bypass_key'],
        )
        self.assertIsNotNone(mock_login_user.call_args.kwargs['client_ip'])

    @patch(
        'examples.db_management.routers.auth.login_user',
        new_callable=AsyncMock,
    )
    async def test_login_forwards_hcaptcha_bypass_header(
        self,
        mock_login_user: AsyncMock,
    ) -> None:
        """Test login forwards backend-only hCaptcha bypass header."""
        mock_login_user.return_value = {
            'access_token': 'access123',
            'refresh_token': 'refresh123',
        }
        payload: UserLogin = UserLogin(identifier='script', password='pw')

        response = self.client.post(
            '/auth/login',
            json=payload.model_dump(),
            headers={'X-HCaptcha-Bypass-Key': 'server-only-key'},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            mock_login_user.call_args.kwargs['hcaptcha_bypass_key'],
            'server-only-key',
        )

    @patch(
        'examples.db_management.routers.auth.login_user',
        new_callable=AsyncMock,
    )
    async def test_web_login_requires_bff(
        self,
        mock_login_user: AsyncMock,
    ) -> None:
        """The legacy token endpoint never returns a JWT to browser code."""
        mock_login_user.return_value = {
            'access_token': 'access123',
            'refresh_token': 'refresh123',
            'username': 'testuser',
            'role': 'user',
            'user_id': 1,
            'group_id': 2,
            'feature_names': [],
        }
        payload = UserLogin(identifier='testuser', password='pw')

        response = self.client.post(
            '/auth/login',
            json=payload.model_dump(),
            headers={'Origin': 'https://changdar-server.mooo.com'},
        )

        self.assertEqual(response.status_code, 410)
        self.assertEqual(response.json()['detail'], 'use_bff_auth_endpoint')
        mock_login_user.assert_not_awaited()

    @patch(
        'examples.db_management.routers.auth.verify_email_token',
        new_callable=AsyncMock,
    )
    async def test_verify_email_success(
        self,
        mock_verify_email_token: AsyncMock,
    ) -> None:
        """Test email verification endpoint returns message payload."""
        mock_verify_email_token.return_value = {
            'message': 'Email verified successfully.',
            'code': 'email_verified',
            'status': 'pending_admin_approval',
        }

        response = self.client.post(
            '/auth/auth/verify-email',
            json={'token': 'raw-token'},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['code'], 'email_verified')
        mock_verify_email_token.assert_awaited_once()

    @patch(
        'examples.db_management.routers.auth.resend_verification_email',
        new_callable=AsyncMock,
    )
    async def test_resend_verification_success(
        self,
        mock_resend_verification: AsyncMock,
    ) -> None:
        """Test resend verification endpoint accepts an email."""
        mock_resend_verification.return_value = {
            'message': (
                'If the account requires verification, a verification '
                'email has been sent.'
            ),
            'code': 'verification_email_sent',
        }

        response = self.client.post(
            '/auth/auth/resend-verification',
            json={'email': 'user@example.com'},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['code'], 'verification_email_sent')
        mock_resend_verification.assert_awaited_once()

    @patch(
        'examples.db_management.routers.auth.login_with_google',
        new_callable=AsyncMock,
    )
    async def test_google_login_success(
        self,
        mock_login_with_google: AsyncMock,
    ) -> None:
        """Test Google provider login returns the shared token response."""
        mock_login_with_google.return_value = {
            'access_token': 'access123',
            'refresh_token': 'refresh123',
            'username': 'testuser',
            'role': 'user',
            'user_id': 1,
            'group_id': None,
            'feature_names': [],
        }

        response = self.client.post(
            '/auth/auth/google',
            json={
                'id_token': 'google-id-token',
                'email': 'user@example.com',
                'display_name': 'Test User',
                'device_lang': 'zh-TW',
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['access_token'], 'access123')
        self.assertEqual(
            mock_login_with_google.call_args.args[0],
            'google-id-token',
        )
        self.assertEqual(
            mock_login_with_google.call_args.kwargs['display_name'],
            'Test User',
        )
        self.assertEqual(
            mock_login_with_google.call_args.kwargs['device_lang'],
            'zh-TW',
        )

    @patch(
        'examples.db_management.routers.auth.login_with_apple',
        new_callable=AsyncMock,
    )
    async def test_apple_login_success(
        self,
        mock_login_with_apple: AsyncMock,
    ) -> None:
        """Test Apple provider login returns the shared token response."""
        mock_login_with_apple.return_value = {
            'access_token': 'access123',
            'refresh_token': 'refresh123',
            'username': 'testuser',
            'role': 'user',
            'user_id': 1,
            'group_id': None,
            'feature_names': [],
        }

        response = self.client.post(
            '/auth/auth/apple',
            json={
                'identity_token': 'apple-identity-token',
                'authorization_code': 'apple-code',
                'email': 'apple@example.com',
                'given_name': 'Given',
                'family_name': 'Family',
                'nonce': 'nonce',
                'device_lang': 'zh-TW',
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['refresh_token'], 'refresh123')
        self.assertEqual(
            mock_login_with_apple.call_args.args[:2],
            ('apple-identity-token', 'apple-code'),
        )
        self.assertEqual(
            mock_login_with_apple.call_args.kwargs['given_name'],
            'Given',
        )

    async def test_apple_callback_redirects_to_android_intent(self) -> None:
        """Test Apple callback preserves parameters in Android intent URL."""
        response = self.client.get(
            '/auth/auth/apple/callback',
            params={'code': 'apple-code', 'state': 'state value'},
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 302)
        location = response.headers['location']
        self.assertTrue(location.startswith('intent://callback?'))
        self.assertIn('code=apple-code', location)
        self.assertIn('state=state+value', location)
        self.assertIn('scheme=signinwithapple', location)
        self.assertIn('package=com.changdar.visionnaire', location)

    @patch(
        'examples.db_management.routers.auth.list_user_identities',
        new_callable=AsyncMock,
    )
    async def test_get_identities_success(
        self,
        mock_list_user_identities: AsyncMock,
    ) -> None:
        """Test get identities success.

        Args:
            mock_list_user_identities: Value used by this callable.
        """
        mock_list_user_identities.return_value = {
            'identities': [
                {
                    'id': 12,
                    'provider': 'google',
                    'email': 'user@example.com',
                    'display_name': 'User',
                    'linked_at': '2026-06-21T10:00:00Z',
                },
            ],
            'has_password': True,
        }

        response = self.client.get('/auth/auth/identities')

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()['has_password'])
        mock_list_user_identities.assert_awaited_once()

    @patch(
        'examples.db_management.routers.auth.link_google_identity',
        new_callable=AsyncMock,
    )
    async def test_link_google_success(
        self,
        mock_link_google_identity: AsyncMock,
    ) -> None:
        """Test link google success.

        Args:
            mock_link_google_identity: Value used by this callable.
        """
        mock_link_google_identity.return_value = {
            'id': 12,
            'provider': 'google',
            'email': 'user@example.com',
            'display_name': 'User',
            'linked_at': '2026-06-21T10:00:00Z',
        }

        response = self.client.post(
            '/auth/auth/identities/google/link',
            json={'id_token': 'google-id-token'},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['provider'], 'google')
        self.assertEqual(
            mock_link_google_identity.call_args.args[1],
            'google-id-token',
        )

    @patch(
        'examples.db_management.routers.auth.link_apple_identity',
        new_callable=AsyncMock,
    )
    async def test_link_apple_forwards_nonce(
        self,
        mock_link_apple_identity: AsyncMock,
    ) -> None:
        """Test link apple forwards nonce.

        Args:
            mock_link_apple_identity: Value used by this callable.
        """
        mock_link_apple_identity.return_value = {
            'id': 13,
            'provider': 'apple',
            'email': 'apple@example.com',
            'display_name': 'Apple User',
            'linked_at': '2026-06-21T10:00:00Z',
        }

        response = self.client.post(
            '/auth/auth/identities/apple/link',
            json={
                'identity_token': 'apple-identity-token',
                'authorization_code': 'apple-code',
                'nonce': 'nonce-from-client',
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['provider'], 'apple')
        self.assertEqual(
            mock_link_apple_identity.call_args.kwargs['nonce'],
            'nonce-from-client',
        )

    @patch(
        'examples.db_management.routers.auth.unlink_identity',
        new_callable=AsyncMock,
    )
    async def test_unlink_identity_success(
        self,
        mock_unlink_identity: AsyncMock,
    ) -> None:
        """Test unlink identity success.

        Args:
            mock_unlink_identity: Value used by this callable.
        """
        mock_unlink_identity.return_value = {
            'message': 'Identity unlinked successfully.',
        }

        response = self.client.delete('/auth/auth/identities/12')

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()['message'],
            'Identity unlinked successfully.',
        )
        self.assertEqual(mock_unlink_identity.call_args.args[1], 12)

    @patch(
        'examples.db_management.routers.auth.logout_user',
        new_callable=AsyncMock,
    )
    async def test_logout_success(self, mock_logout_user: AsyncMock) -> None:
        """Test logout success.

        Args:
            mock_logout_user: Value used by this callable.
        """
        payload: LogoutRequest = LogoutRequest(refresh_token='refresh123')
        headers: dict[str, str] = {'Authorization': 'Bearer access123'}
        response = self.client.post(
            '/auth/logout',
            json=payload.model_dump(),
            headers=headers,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()['message'],
            'Logged out successfully.',
        )

    @patch(
        'examples.db_management.routers.auth.refresh_tokens',
        new_callable=AsyncMock,
    )
    async def test_refresh_success(
        self,
        mock_refresh_tokens: AsyncMock,
    ) -> None:
        """Test refresh success.

        Args:
            mock_refresh_tokens: Value used by this callable.
        """
        mock_refresh_tokens.return_value = {
            'access_token': 'new_access',
            'refresh_token': 'new_refresh',
            'feature_names': ['f1'],
        }
        payload: RefreshRequest = RefreshRequest(refresh_token='refresh123')
        response = self.client.post('/auth/refresh', json=payload.model_dump())
        self.assertEqual(response.status_code, 200)
        data: dict = response.json()
        self.assertIn('access_token', data)
        self.assertIn('refresh_token', data)

    @patch(
        'examples.db_management.routers.auth.refresh_tokens',
        new_callable=AsyncMock,
    )
    async def test_web_refresh_requires_bff(
        self,
        mock_refresh_tokens: AsyncMock,
    ) -> None:
        """Web refresh is internal to the BFF and not callable by Flutter."""
        mock_refresh_tokens.return_value = {
            'access_token': 'new_access',
            'refresh_token': 'new_refresh',
            'feature_names': ['f1'],
        }

        response = self.client.post(
            '/auth/refresh',
            headers={
                'Origin': 'https://changdar-server.mooo.com',
                'Cookie': 'refresh_session=old_refresh',
            },
        )

        self.assertEqual(response.status_code, 410)
        self.assertEqual(response.json()['detail'], 'use_bff_auth_endpoint')
        mock_refresh_tokens.assert_not_awaited()

    @patch(
        'examples.db_management.routers.auth.login_user',
        new_callable=AsyncMock,
    )
    async def test_login_fail(self, mock_login_user: AsyncMock) -> None:
        """Test login fail.

        Args:
            mock_login_user: Value used by this callable.
        """
        mock_login_user.side_effect = HTTPException(
            status_code=401,
            detail='fail',
        )
        payload: UserLogin = UserLogin(
            identifier='bad',
            password='bad',
            hcaptcha_token='captcha-token',
        )
        response = self.client.post('/auth/login', json=payload.model_dump())
        self.assertEqual(response.status_code, 401)
        self.assertIn('detail', response.json())

    @patch(
        'examples.db_management.routers.auth.logout_user',
        new_callable=AsyncMock,
    )
    async def test_logout_fail(self, mock_logout_user: AsyncMock) -> None:
        """Test logout fail.

        Args:
            mock_logout_user: Value used by this callable.
        """
        mock_logout_user.side_effect = HTTPException(
            status_code=401,
            detail='fail',
        )
        payload: LogoutRequest = LogoutRequest(refresh_token='bad')
        headers: dict[str, str] = {
            'Authorization': 'Bearer access123',
        }  # Ensure mock is called
        response = self.client.post(
            '/auth/logout',
            json=payload.model_dump(),
            headers=headers,
        )
        self.assertEqual(response.status_code, 401)
        self.assertIn('detail', response.json())

    @patch(
        'examples.db_management.routers.auth.logout_user',
        new_callable=AsyncMock,
    )
    async def test_web_logout_uses_cookie_and_clears_cookie(
        self,
        mock_logout_user: AsyncMock,
    ) -> None:
        """Web logout can revoke refresh token from cookie without body."""
        response = self.client.post(
            '/auth/logout',
            headers={
                'Authorization': 'Bearer access123',
                'Origin': 'https://changdar-server.mooo.com',
                'Cookie': 'refresh_session=old_refresh',
            },
        )

        self.assertEqual(response.status_code, 200)
        mock_logout_user.assert_awaited_once()
        self.assertEqual(mock_logout_user.call_args.args[0], 'old_refresh')
        self.assertEqual(
            mock_logout_user.call_args.args[1],
            'Bearer access123',
        )
        set_cookie = response.headers['set-cookie']
        self.assertIn('refresh_session=', set_cookie)
        self.assertIn('Max-Age=0', set_cookie)

    @patch(
        'examples.db_management.routers.auth.refresh_tokens',
        new_callable=AsyncMock,
    )
    async def test_refresh_fail(self, mock_refresh_tokens: AsyncMock) -> None:
        """Test refresh fail.

        Args:
            mock_refresh_tokens: Value used by this callable.
        """
        mock_refresh_tokens.side_effect = HTTPException(
            status_code=401,
            detail='fail',
        )
        payload: RefreshRequest = RefreshRequest(refresh_token='bad')
        response = self.client.post(
            '/auth/refresh',
            json=payload.model_dump(),
        )
        self.assertEqual(response.status_code, 401)
        self.assertIn('detail', response.json())


if __name__ == '__main__':
    unittest.main()
