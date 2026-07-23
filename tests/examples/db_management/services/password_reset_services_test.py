from __future__ import annotations

import json
import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import HTTPException

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
