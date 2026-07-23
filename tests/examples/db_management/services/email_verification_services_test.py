from __future__ import annotations

import json
import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import HTTPException

from examples.auth.models import USER_STATUS_EMAIL_UNVERIFIED
from examples.auth.models import USER_STATUS_PENDING
from examples.db_management.services import email_verification_services as svc


class TestEmailVerificationServices(unittest.IsolatedAsyncioTestCase):
    """Unit tests for email verification token lifecycle."""

    def setUp(self) -> None:
        self.db: AsyncMock = AsyncMock()
        self.redis: AsyncMock = AsyncMock()

    async def test_verify_email_token_success_advances_to_pending_admin(
        self,
    ) -> None:
        user: MagicMock = MagicMock()
        user.id = 7
        user.status = USER_STATUS_EMAIL_UNVERIFIED
        user.email_verified_at = None
        self.db.get = AsyncMock(return_value=user)
        self.db.commit = AsyncMock()
        self.redis.getdel = AsyncMock(
            return_value=json.dumps(
                {'user_id': 7, 'email': 'user@example.com'},
            ),
        )
        self.redis.delete = AsyncMock()
        self.redis.set = AsyncMock()

        result = await svc.verify_email_token('raw-token', self.db, self.redis)

        self.assertEqual(result['code'], 'email_verified')
        self.assertEqual(result['status'], USER_STATUS_PENDING)
        self.assertEqual(user.status, USER_STATUS_PENDING)
        self.assertIsNotNone(user.email_verified_at)
        self.db.commit.assert_awaited_once()
        self.redis.getdel.assert_awaited_once()
        self.redis.delete.assert_awaited_once()
        self.redis.set.assert_awaited_once()

    async def test_verify_email_token_rejects_used_token(self) -> None:
        self.redis.getdel = AsyncMock(return_value=None)
        self.redis.get = AsyncMock(return_value='1')

        with self.assertRaises(HTTPException) as ctx:
            await svc.verify_email_token('raw-token', self.db, self.redis)

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.detail['code'], 'token_used')

    async def test_verify_email_token_rejects_invalid_or_expired_token(
        self,
    ) -> None:
        self.redis.getdel = AsyncMock(return_value=None)
        self.redis.get = AsyncMock(return_value=None)

        with self.assertRaises(HTTPException) as ctx:
            await svc.verify_email_token('raw-token', self.db, self.redis)

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(
            ctx.exception.detail['code'],
            'invalid_or_expired_token',
        )

    @patch('examples.db_management.services.email_verification_services.httpx.AsyncClient')
    @patch('examples.db_management.services.email_verification_services.settings')
    async def test_send_email_uses_brevo_template_when_configured(
        self,
        mock_settings: MagicMock,
        mock_async_client: MagicMock,
    ) -> None:
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

    @patch('examples.db_management.services.email_verification_services.settings')
    async def test_resend_verification_daily_limit_returns_retry_after(
        self,
        mock_settings: MagicMock,
    ) -> None:
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

    @patch('examples.db_management.services.email_verification_services.secrets')
    @patch('examples.db_management.services.email_verification_services.settings')
    async def test_create_token_invalidates_existing_user_token(
        self,
        mock_settings: MagicMock,
        mock_secrets: MagicMock,
    ) -> None:
        """Creating a new verification email immediately deletes the old link."""
        mock_settings.email_verification_token_ttl_seconds = 86400
        mock_secrets.token_urlsafe.return_value = 'new-raw-token'
        user: MagicMock = MagicMock(id=8)
        self.redis.get = AsyncMock(return_value='old-token-hash')
        self.redis.delete = AsyncMock()
        self.redis.set = AsyncMock()

        raw = await svc._create_email_verification_token(
            user,
            'user@example.com',
            self.redis,
        )

        self.assertEqual(raw, 'new-raw-token')
        self.redis.delete.assert_any_await(
            'email_verification:old-token-hash',
        )
        self.redis.delete.assert_any_await('email_verification_user:8')
        self.assertEqual(self.redis.set.await_count, 2)


if __name__ == '__main__':
    unittest.main()
