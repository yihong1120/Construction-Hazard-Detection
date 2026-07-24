from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import httpx
from fastapi import HTTPException

from examples.db_management.services import password_reset_services as service
from examples.db_management.services import password_reset_services as svc


class TestPasswordResetServices(unittest.IsolatedAsyncioTestCase):
    """Tests for password reset service behaviour."""

    def setUp(self) -> None:
        self.db: AsyncMock = AsyncMock()
        self.redis: AsyncMock = AsyncMock()
        self.redis.incr = AsyncMock(side_effect=[1, 1])
        self.redis.expire = AsyncMock()
        self.redis.set = AsyncMock()
        self.redis.get = AsyncMock()
        self.redis.getdel = AsyncMock()
        self.redis.delete = AsyncMock()
        self.redis.smembers = AsyncMock(return_value=set())

    @patch.object(svc, '_send_password_reset_email', new_callable=AsyncMock)
    @patch.object(svc, '_find_user_by_email', new_callable=AsyncMock)
    @patch.object(svc.secrets, 'token_urlsafe', return_value='raw-token')
    async def test_request_password_reset_existing_email(
        self,
        mock_token_urlsafe: MagicMock,
        mock_find_user: AsyncMock,
        mock_send_email: AsyncMock,
    ) -> None:
        user = MagicMock(id=123)
        mock_find_user.return_value = user

        result = await svc.request_password_reset(
            'USER@example.com',
            self.db,
            self.redis,
            client_ip='127.0.0.1',
        )

        self.assertEqual(
            result['message'],
            svc.FORGOT_PASSWORD_RESPONSE,
        )
        token_hash = svc._hash_token('raw-token')
        self.redis.set.assert_awaited_once_with(
            f'password_reset:{token_hash}',
            json.dumps({'user_id': 123, 'email': 'user@example.com'}),
            ex=svc.settings.password_reset_token_ttl_seconds,
        )
        mock_send_email.assert_awaited_once_with(
            'user@example.com',
            'https://changdar-server.mooo.com/reset_password?token=raw-token',
        )
        mock_token_urlsafe.assert_called_once_with(48)

    @patch.object(svc, '_send_password_reset_email', new_callable=AsyncMock)
    @patch.object(svc, '_find_user_by_email', new_callable=AsyncMock)
    async def test_request_password_reset_unknown_email_does_not_send(
        self,
        mock_find_user: AsyncMock,
        mock_send_email: AsyncMock,
    ) -> None:
        mock_find_user.return_value = None

        result = await svc.request_password_reset(
            'missing@example.com',
            self.db,
            self.redis,
        )

        self.assertEqual(
            result['message'],
            svc.FORGOT_PASSWORD_RESPONSE,
        )
        self.redis.set.assert_not_awaited()
        mock_send_email.assert_not_awaited()

    @patch.object(svc, '_send_password_reset_email', new_callable=AsyncMock)
    @patch.object(svc, '_find_user_by_email', new_callable=AsyncMock)
    @patch.object(svc.secrets, 'token_urlsafe', return_value='raw-token')
    async def test_request_password_reset_deletes_token_on_mail_failure(
        self,
        _mock_token_urlsafe: MagicMock,
        mock_find_user: AsyncMock,
        mock_send_email: AsyncMock,
    ) -> None:
        mock_find_user.return_value = MagicMock(id=123)
        mock_send_email.side_effect = HTTPException(502, 'send failed')

        with self.assertRaises(HTTPException):
            await svc.request_password_reset(
                'user@example.com',
                self.db,
                self.redis,
            )

        self.redis.delete.assert_awaited_with(
            f"password_reset:{svc._hash_token('raw-token')}",
        )

    async def test_request_password_reset_email_rate_limited(self) -> None:
        self.redis.incr = AsyncMock(return_value=2)

        with self.assertRaises(HTTPException) as ctx:
            await svc.request_password_reset(
                'user@example.com',
                self.db,
                self.redis,
            )

        self.assertEqual(ctx.exception.status_code, 429)

    async def test_reset_password_invalid_or_expired_token(self) -> None:
        self.redis.getdel.return_value = None

        with self.assertRaises(HTTPException) as ctx:
            await svc.reset_password(
                'raw-token',
                'NewPass123',
                self.db,
                self.redis,
            )

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(
            ctx.exception.detail,
            svc.RESET_TOKEN_INVALID_RESPONSE,
        )

    async def test_reset_password_missing_token(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            await svc.reset_password(
                None,
                'NewPass123',
                self.db,
                self.redis,
            )

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(
            ctx.exception.detail,
            svc.RESET_TOKEN_INVALID_RESPONSE,
        )
        self.redis.getdel.assert_not_awaited()

    async def test_reset_password_missing_new_password(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            await svc.reset_password(
                'raw-token',
                None,
                self.db,
                self.redis,
            )

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(
            ctx.exception.detail,
            {'code': 'password_too_short', 'min_length': 8},
        )
        self.redis.getdel.assert_not_awaited()

    async def test_reset_password_rejects_short_password(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            await svc.reset_password(
                'raw-token',
                'short',
                self.db,
                self.redis,
            )

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(
            ctx.exception.detail,
            {'code': 'password_too_short', 'min_length': 8},
        )
        self.redis.getdel.assert_not_awaited()
        self.db.get.assert_not_awaited()

    async def test_reset_password_success_updates_password_and_revokes_cache(
        self,
    ) -> None:
        user = MagicMock(id=123, username='user')
        self.db.get = AsyncMock(return_value=user)
        self.db.commit = AsyncMock()
        self.db.rollback = AsyncMock()
        self.redis.getdel.return_value = json.dumps(
            {'user_id': 123, 'email': 'user@example.com'},
        )

        result = await svc.reset_password(
            'raw-token',
            'password',
            self.db,
            self.redis,
        )

        self.assertEqual(
            result['message'],
            svc.PASSWORD_RESET_SUCCESS_RESPONSE,
        )
        user.set_password.assert_called_once_with('password')
        self.db.commit.assert_awaited_once()
        user_cache_key = f'{svc.PROJECT_PREFIX}:user_cache:user'
        self.redis.getdel.assert_awaited_once_with(
            f"password_reset:{svc._hash_token('raw-token')}",
        )
        self.redis.delete.assert_any_await(user_cache_key)
        self.assertGreaterEqual(self.redis.smembers.await_count, 1)

    async def test_reset_password_rejects_corrupt_token_payload(self) -> None:
        self.redis.getdel.return_value = 'not-json'

        with self.assertRaises(HTTPException) as ctx:
            await svc.reset_password(
                'raw-token',
                'NewPass123',
                self.db,
                self.redis,
            )

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(
            ctx.exception.detail,
            svc.RESET_TOKEN_INVALID_RESPONSE,
        )
        self.redis.getdel.assert_awaited_once_with(
            f"password_reset:{svc._hash_token('raw-token')}",
        )


def _mail_settings() -> SimpleNamespace:
    """Return the minimum configuration used by the Brevo sender."""
    return SimpleNamespace(
        brevo_api_key='test-key',
        mail_from='no-reply@example.com',
        mail_from_name='Visionnaire',
    )


def _http_client(response: MagicMock) -> tuple[AsyncMock, AsyncMock]:
    """Build an async HTTP client context manager and its request mock."""
    client = AsyncMock()
    client.post.return_value = response
    context = AsyncMock()
    context.__aenter__.return_value = client
    context.__aexit__.return_value = False
    return context, client


class TestPasswordResetServiceCoverage(unittest.IsolatedAsyncioTestCase):
    """Exercise password-reset operational failures and validation guards."""

    def setUp(self) -> None:
        self.db = AsyncMock()
        self.redis = AsyncMock()
        self.redis.getdel = AsyncMock()

    async def test_send_reset_email_requires_configuration_and_sends_request(
        self,
    ) -> None:
        """Mail configuration is mandatory and valid mail builds a Brevo call."""
        with patch.object(
            service,
            'settings',
            SimpleNamespace(brevo_api_key='', mail_from='sender@example.com'),
        ):
            with self.assertRaisesRegex(HTTPException, 'not configured'):
                await service._send_password_reset_email(
                    'user@example.com',
                    'https://app.example/reset',
                )

        response = MagicMock()
        context, client = _http_client(response)
        with (
            patch.object(service, 'settings', _mail_settings()),
            patch.object(service.httpx, 'AsyncClient', return_value=context),
        ):
            await service._send_password_reset_email(
                'user@example.com',
                'https://app.example/reset',
            )

        client.post.assert_awaited_once()
        request = client.post.call_args
        self.assertEqual(request.args[0], service.BREVO_SEND_EMAIL_URL)
        self.assertEqual(
            request.kwargs['json']['to'], [
                {'email': 'user@example.com'},
            ],
        )
        response.raise_for_status.assert_called_once()

    async def test_send_reset_email_translates_brevo_failures(self) -> None:
        """HTTP status and connection failures become a stable API response."""
        response = MagicMock()
        response.text = 'recipient rejected'
        response.status_code = 422
        status_error = httpx.HTTPStatusError(
            'unprocessable',
            request=httpx.Request('POST', service.BREVO_SEND_EMAIL_URL),
            response=response,
        )
        response.raise_for_status.side_effect = status_error
        context, _client = _http_client(response)
        with (
            patch.object(service, 'settings', _mail_settings()),
            patch.object(service.httpx, 'AsyncClient', return_value=context),
        ):
            with self.assertRaisesRegex(HTTPException, 'Failed to send') as error:
                await service._send_password_reset_email(
                    'user@example.com',
                    'https://app.example/reset',
                )
        self.assertEqual(error.exception.status_code, 502)

        response = MagicMock()
        context, client = _http_client(response)
        client.post.side_effect = httpx.ConnectError('network unavailable')
        with (
            patch.object(service, 'settings', _mail_settings()),
            patch.object(service.httpx, 'AsyncClient', return_value=context),
        ):
            with self.assertRaisesRegex(HTTPException, 'Failed to send') as error:
                await service._send_password_reset_email(
                    'user@example.com',
                    'https://app.example/reset',
                )
        self.assertEqual(error.exception.status_code, 502)

    async def test_reset_password_rejects_blank_token_and_unknown_user(self) -> None:
        """Whitespace-only tokens and stale reset users cannot reset passwords."""
        with self.assertRaisesRegex(HTTPException, 'invalid or expired'):
            await service.reset_password('   ', 'password', self.db, self.redis)
        self.redis.getdel.assert_not_awaited()

        self.redis.getdel.return_value = json.dumps({
            'user_id': 77,
            'email': 'user@example.com',
        })
        self.db.get.return_value = None
        with self.assertRaisesRegex(HTTPException, 'invalid or expired'):
            await service.reset_password(
                'valid-token',
                'password',
                self.db,
                self.redis,
            )

    async def test_reset_password_rolls_back_a_failed_commit(self) -> None:
        """Database errors roll back the password update and return a 500."""
        user = MagicMock(username='user')
        self.redis.getdel.return_value = json.dumps({
            'user_id': 88,
            'email': 'user@example.com',
        })
        self.db.get.return_value = user
        self.db.commit.side_effect = RuntimeError('database unavailable')

        with self.assertRaisesRegex(HTTPException, 'Database error') as error:
            await service.reset_password(
                'valid-token',
                'password',
                self.db,
                self.redis,
            )

        self.assertEqual(error.exception.status_code, 500)
        user.set_password.assert_called_once_with('password')
        self.db.rollback.assert_awaited_once()

    async def test_ip_rate_limit_and_email_lookup_use_their_service_paths(
        self,
    ) -> None:
        """IP throttling and active-user lookup both delegate to storage."""
        redis = AsyncMock()
        redis.incr = AsyncMock(side_effect=[1, 3])
        settings = SimpleNamespace(
            password_reset_email_rate_limit_seconds=60,
            password_reset_ip_rate_limit_window_seconds=60,
            password_reset_ip_rate_limit_max=2,
        )
        with patch.object(service, 'settings', settings):
            with self.assertRaisesRegex(HTTPException, 'Too many requests'):
                await service._enforce_forgot_password_rate_limits(
                    'user@example.com',
                    '127.0.0.1',
                    redis,
                )

        user = MagicMock()
        self.db.scalar.return_value = user
        self.assertIs(
            await service._find_user_by_email('USER@example.com', self.db),
            user,
        )
        self.db.scalar.assert_awaited_once()


if __name__ == '__main__':
    unittest.main()
