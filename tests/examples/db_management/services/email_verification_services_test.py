from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import call
from unittest.mock import MagicMock
from unittest.mock import patch

import httpx
from fastapi import HTTPException

from examples.auth.models import USER_STATUS_EMAIL_UNVERIFIED
from examples.auth.models import USER_STATUS_PENDING_ADMIN_APPROVAL
from examples.auth.models import USER_STATUS_REJECTED
from examples.db_management.services import email_verification_services as svc


class TestEmailVerificationServices(unittest.IsolatedAsyncioTestCase):
    """Unit tests for email verification token lifecycle."""

    def setUp(self) -> None:
        """Perform setUp."""
        self.db: AsyncMock = AsyncMock()
        self.redis: AsyncMock = AsyncMock()

    async def test_verify_email_token_success_advances_to_pending_admin(
        self,
    ) -> None:
        """Test verify email token success advances to pending admin."""
        user: MagicMock = MagicMock()
        user.id = 7
        user.status = USER_STATUS_EMAIL_UNVERIFIED
        user.email_verified_at = None
        self.db.get = AsyncMock(return_value=user)
        self.db.commit = AsyncMock()
        self.redis.getdel = AsyncMock(
            return_value=b'7',
        )
        self.redis.delete = AsyncMock()
        self.redis.set = AsyncMock()

        result = await svc.verify_email_token('raw-token', self.db, self.redis)

        self.assertEqual(result['code'], 'email_verified')
        self.assertEqual(result['status'], USER_STATUS_PENDING_ADMIN_APPROVAL)
        self.assertEqual(user.status, USER_STATUS_PENDING_ADMIN_APPROVAL)
        self.assertIsNotNone(user.email_verified_at)
        self.db.commit.assert_awaited_once()
        self.redis.getdel.assert_awaited_once()
        self.redis.delete.assert_awaited_once()
        self.redis.set.assert_awaited_once()

    async def test_verify_email_token_rejects_used_token(self) -> None:
        """Test verify email token rejects used token."""
        self.redis.getdel = AsyncMock(return_value=None)
        self.redis.get = AsyncMock(return_value=b'1')

        with self.assertRaises(HTTPException) as ctx:
            await svc.verify_email_token('raw-token', self.db, self.redis)

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.detail['code'], 'token_used')

    async def test_verify_email_token_rejects_invalid_or_expired_token(
        self,
    ) -> None:
        """Test verify email token rejects invalid or expired token."""
        self.redis.getdel = AsyncMock(return_value=None)
        self.redis.get = AsyncMock(return_value=None)

        with self.assertRaises(HTTPException) as ctx:
            await svc.verify_email_token('raw-token', self.db, self.redis)

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(
            ctx.exception.detail['code'],
            'invalid_or_expired_token',
        )

    @patch(
        (
            'examples.db_management.services.email_verification_services.'
            'httpx.AsyncClient'
        ),
    )
    @patch(
        'examples.db_management.services.email_verification_services.settings',
    )
    async def test_send_email_uses_brevo_template_when_configured(
        self,
        mock_settings: MagicMock,
        mock_async_client: MagicMock,
    ) -> None:
        """Test send email uses brevo template when configured.

        Args:
            mock_settings: Value used by this callable.
            mock_async_client: Value used by this callable.
        """
        mock_settings.brevo_api_key = 'brevo-key'
        mock_settings.mail_from = 'sender@example.com'
        mock_settings.mail_from_name = 'Visionnaire'
        mock_settings.brevo_email_verification_template_id = 123
        mock_settings.email_verification_token_ttl_seconds = 86400

        response: MagicMock = MagicMock()
        client: AsyncMock = AsyncMock()
        client.post.return_value = response
        mock_async_client.return_value.__aenter__.return_value = client

        await svc._send_email_verification_email(
            'user@example.com',
            'user',
            'https://example.com/verify-email?token=abc',
        )

        payload = client.post.call_args.kwargs['json']
        self.assertEqual(payload['templateId'], 123)
        self.assertEqual(
            payload['params']['VERIFY_URL'],
            payload['params']['verify_url'],
        )
        response.raise_for_status.assert_called_once()

    async def test_resend_verification_returns_generic_for_unknown_email(
        self,
    ) -> None:
        """Test resend verification returns generic for unknown email."""
        self.redis.incr = AsyncMock(return_value=1)
        self.redis.expire = AsyncMock()
        self.db.scalar = AsyncMock(return_value=None)

        result = await svc.resend_verification_email(
            'missing@example.com',
            self.db,
            self.redis,
        )

        self.assertEqual(result['code'], 'verification_email_sent')
        self.assertEqual(self.redis.expire.await_count, 2)

    async def test_resend_verification_rate_limit_returns_retry_after(
        self,
    ) -> None:
        """Test resend verification rate limit returns retry after."""
        self.redis.incr = AsyncMock(return_value=2)
        self.redis.expire = AsyncMock()
        self.redis.ttl = AsyncMock(return_value=42)

        with self.assertRaises(HTTPException) as ctx:
            await svc.resend_verification_email(
                'user@example.com',
                self.db,
                self.redis,
            )

        self.assertEqual(ctx.exception.status_code, 429)
        self.assertEqual(
            ctx.exception.detail,
            {
                'code': 'verification_resend_rate_limited',
                'retry_after_seconds': 42,
            },
        )
        self.assertEqual(ctx.exception.headers, {'Retry-After': '42'})

    @patch(
        'examples.db_management.services.email_verification_services.settings',
    )
    async def test_resend_verification_daily_limit_returns_retry_after(
        self,
        mock_settings: MagicMock,
    ) -> None:
        """Test resend verification daily limit returns retry after.

        Args:
            mock_settings: Value used by this callable.
        """
        mock_settings.email_verification_resend_rate_limit_seconds = 60
        mock_settings.email_verification_daily_limit = 5
        mock_settings.email_verification_daily_limit_window_seconds = 86400
        self.redis.incr = AsyncMock(side_effect=[1, 6])
        self.redis.expire = AsyncMock()
        self.redis.ttl = AsyncMock(return_value=3600)

        with self.assertRaises(HTTPException) as ctx:
            await svc.resend_verification_email(
                'user@example.com',
                self.db,
                self.redis,
            )

        self.assertEqual(ctx.exception.status_code, 429)
        self.assertEqual(
            ctx.exception.detail['code'],
            'verification_daily_limit_exceeded',
        )
        self.assertEqual(ctx.exception.detail['retry_after_seconds'], 3600)

    @patch(
        'examples.db_management.services.email_verification_services.secrets',
    )
    @patch(
        'examples.db_management.services.email_verification_services.settings',
    )
    async def test_create_token_invalidates_existing_user_token(
        self,
        mock_settings: MagicMock,
        mock_secrets: MagicMock,
    ) -> None:
        """Creating a new verification email immediately deletes the old
        link."""
        mock_settings.email_verification_token_ttl_seconds = 86400
        mock_secrets.token_urlsafe.return_value = 'new-raw-token'
        user: MagicMock = MagicMock(id=8)
        self.redis.get = AsyncMock(return_value=b'old-token-hash')
        self.redis.delete = AsyncMock()
        self.redis.set = AsyncMock()

        raw = await svc._create_email_verification_token(
            user,
            self.redis,
        )

        self.assertEqual(raw, 'new-raw-token')
        self.redis.delete.assert_any_await(
            'email_verification:old-token-hash',
        )
        self.redis.delete.assert_any_await('email_verification_user:8')
        self.redis.set.assert_has_awaits(
            [
                call(
                    'email_verification:' + svc._hash_token('new-raw-token'),
                    b'8',
                    ex=86400,
                ),
                call(
                    'email_verification_user:8',
                    svc._hash_token('new-raw-token').encode('ascii'),
                    ex=86400,
                ),
            ],
        )


if __name__ == '__main__':
    unittest.main()


def _settings(**values: object) -> SimpleNamespace:
    """Perform settings.

    Args:
        **values: Value used by this callable.

    Returns:
        The callable result.
    """
    defaults = {
        'app_public_url': 'https://app.example',
        'email_verification_resend_rate_limit_seconds': 60,
        'email_verification_daily_limit_window_seconds': 86400,
        'email_verification_daily_limit': 5,
        'email_verification_token_ttl_seconds': 3600,
        'brevo_api_key': 'brevo-key',
        'mail_from': 'sender@example.com',
        'mail_from_name': 'Visionnaire',
        'brevo_email_verification_template_id': 0,
    }
    defaults.update(values)
    return SimpleNamespace(**defaults)


def _http_client_context(post_result: object) -> tuple[MagicMock, MagicMock]:
    """Perform http client context.

    Args:
        post_result: Value used by this callable.

    Returns:
        The callable result.
    """
    client = MagicMock()
    client.post = AsyncMock(return_value=post_result)
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=client)
    context.__aexit__ = AsyncMock(return_value=None)
    return client, context


class TestEmailVerificationServiceCoverage(unittest.IsolatedAsyncioTestCase):
    """Provide TestEmailVerificationServiceCoverage."""

    def setUp(self) -> None:
        """Perform setUp."""
        self.db = MagicMock()
        self.db.get = AsyncMock()
        self.db.scalar = AsyncMock()
        self.db.commit = AsyncMock()
        self.redis = MagicMock()
        self.redis.get = AsyncMock()
        self.redis.getdel = AsyncMock()
        self.redis.set = AsyncMock()
        self.redis.delete = AsyncMock()
        self.redis.incr = AsyncMock(return_value=1)
        self.redis.expire = AsyncMock()
        self.redis.ttl = AsyncMock(return_value=0)

    def test_url_and_profile_helpers(self) -> None:
        """Test url and profile helpers."""
        with patch.object(
            svc,
            'settings',
            _settings(app_public_url='https://app.example/'),
        ):
            self.assertEqual(
                svc._build_verify_url('token'),
                'https://app.example/verify-email?token=token',
            )
        user = MagicMock(profile=MagicMock(email=' User@Example.COM '))
        self.assertEqual(svc._profile_email(user), 'user@example.com')
        with self.assertRaises(HTTPException) as missing_email:
            svc._profile_email(MagicMock(profile=None))
        self.assertEqual(missing_email.exception.status_code, 400)

    async def test_delete_existing_token_removes_the_stored_hash(
        self,
    ) -> None:
        """Test delete existing token removes the stored hash."""
        self.redis.get.return_value = b'old-hash'
        await svc._delete_existing_token_for_user(7, self.redis)
        self.redis.delete.assert_any_await('email_verification:old-hash')
        self.redis.delete.assert_any_await('email_verification_user:7')

        self.redis.delete.reset_mock()
        self.redis.get.return_value = None
        await svc._delete_existing_token_for_user(7, self.redis)
        self.redis.delete.assert_awaited_once_with('email_verification_user:7')

        self.redis.delete.reset_mock()
        await svc._delete_token_by_raw_token('raw-token', self.redis)
        self.redis.delete.assert_awaited_once_with(
            svc._email_verification_key(svc._hash_token('raw-token')),
        )

    async def test_send_email_handles_configuration_and_http_failures(
        self,
    ) -> None:
        """Test send email handles configuration and http failures."""
        with patch.object(svc, 'settings', _settings(brevo_api_key='')):
            with self.assertRaises(HTTPException) as unconfigured:
                await svc._send_email_verification_email(
                    'user@example.com',
                    'user',
                    'https://url',
                )
        self.assertEqual(unconfigured.exception.status_code, 500)

        response = MagicMock()
        client, context = _http_client_context(response)
        with patch.object(svc, 'settings', _settings()):
            with patch.object(svc.httpx, 'AsyncClient', return_value=context):
                await svc._send_email_verification_email(
                    'user@example.com',
                    'user',
                    'https://url',
                )
        payload = client.post.call_args.kwargs['json']
        self.assertNotIn('templateId', payload)
        self.assertIn('htmlContent', payload)
        response.raise_for_status.assert_called_once()

        request = httpx.Request('POST', svc.BREVO_SEND_EMAIL_URL)
        response = httpx.Response(400, request=request, text='rejected')
        error = httpx.HTTPStatusError(
            'rejected',
            request=request,
            response=response,
        )
        status_response = MagicMock()
        status_response.raise_for_status.side_effect = error
        _, status_context = _http_client_context(status_response)
        with patch.object(svc, 'settings', _settings()):
            with patch.object(
                svc.httpx,
                'AsyncClient',
                return_value=status_context,
            ):
                with self.assertRaises(HTTPException) as rejected:
                    await svc._send_email_verification_email(
                        'user@example.com',
                        'user',
                        'https://url',
                    )
        self.assertEqual(rejected.exception.status_code, 502)

        network_client, network_context = _http_client_context(MagicMock())
        network_client.post.side_effect = httpx.ConnectError('offline')
        with patch.object(svc, 'settings', _settings()):
            with patch.object(
                svc.httpx,
                'AsyncClient',
                return_value=network_context,
            ):
                with self.assertRaises(HTTPException) as unavailable:
                    await svc._send_email_verification_email(
                        'user@example.com',
                        'user',
                        'https://url',
                    )
        self.assertEqual(unavailable.exception.status_code, 502)

    async def test_signup_and_resend_email_paths(self) -> None:
        """Test signup and resend email paths."""
        user = MagicMock(
            id=7,
            username='alice',
            status=USER_STATUS_EMAIL_UNVERIFIED,
            profile=MagicMock(email='alice@example.com'),
            email_verified_at=None,
        )
        with patch.object(svc, 'settings', _settings()):
            with patch.object(
                svc,
                '_create_email_verification_token',
                new=AsyncMock(return_value='token'),
            ):
                with patch.object(
                    svc,
                    '_send_email_verification_email',
                    new=AsyncMock(),
                ) as send:
                    result = await svc.send_signup_verification_email(
                        user,
                        self.redis,
                    )
        self.assertEqual(result['code'], 'verification_email_sent')
        send.assert_awaited_once_with(
            'alice@example.com',
            'alice',
            'https://app.example/verify-email?token=token',
        )

        with patch.object(
            svc,
            '_create_email_verification_token',
            new=AsyncMock(return_value='token'),
        ):
            with patch.object(
                svc,
                '_send_email_verification_email',
                new=AsyncMock(side_effect=HTTPException(status_code=502)),
            ):
                with patch.object(
                    svc,
                    '_delete_token_by_raw_token',
                    new=AsyncMock(),
                ) as delete:
                    with self.assertRaises(HTTPException):
                        await svc.send_signup_verification_email(
                            user,
                            self.redis,
                        )
        delete.assert_awaited_once_with('token', self.redis)

        with patch.object(
            svc,
            '_find_user_by_email',
            new=AsyncMock(return_value=user),
        ):
            with patch.object(
                svc,
                'send_signup_verification_email',
                new=AsyncMock(
                    return_value={'code': 'verification_email_sent'},
                ),
            ) as send_signup:
                result = await svc.resend_verification_email(
                    ' ALICE@example.com ',
                    self.db,
                    self.redis,
                )
        self.assertEqual(result['code'], 'verification_email_sent')
        send_signup.assert_awaited_once_with(user, self.redis)

    async def test_verify_email_token_rejects_missing_user(
        self,
    ) -> None:
        """Test verify email token rejects missing user."""
        with self.assertRaises(HTTPException) as empty_token:
            await svc.verify_email_token(None, self.db, self.redis)
        self.assertEqual(empty_token.exception.status_code, 400)

        self.redis.getdel.return_value = b'9'
        self.db.get.return_value = None
        with self.assertRaises(HTTPException) as missing_user:
            await svc.verify_email_token('token', self.db, self.redis)
        self.assertEqual(
            missing_user.exception.detail['code'],
            'invalid_token',
        )

    async def test_verify_email_token_rejects_non_verifiable_statuses(
        self,
    ) -> None:
        """Test verify email token rejects non verifiable statuses."""
        for status, code in [
            (USER_STATUS_REJECTED, 'account_not_verifiable'),
            ('unexpected', 'account_not_active'),
        ]:
            user = MagicMock(id=7, status=status, email_verified_at=None)
            self.redis.getdel.return_value = b'7'
            self.db.get.return_value = user
            with self.assertRaises(HTTPException) as invalid_status:
                await svc.verify_email_token('token', self.db, self.redis)
            self.assertEqual(invalid_status.exception.status_code, 403)
            self.assertEqual(invalid_status.exception.detail['code'], code)


if __name__ == '__main__':
    unittest.main()
