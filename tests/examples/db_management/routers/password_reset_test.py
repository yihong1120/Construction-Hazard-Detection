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
from examples.auth.redis_pool import get_redis_pool
from examples.db_management.routers import password_reset


class TestPasswordResetRouter(unittest.IsolatedAsyncioTestCase):
    """Tests for password reset router endpoints."""

    def setUp(self) -> None:
        """Perform setUp."""
        self.app = FastAPI()
        self.app.include_router(password_reset.router)
        self.client = TestClient(self.app)

        async def override_get_db() -> AsyncGenerator[MagicMock]:
            """Perform override get db.

            Returns:
                The callable result.
            """
            yield MagicMock()

        async def override_get_redis_pool() -> MagicMock:
            """Perform override get redis pool.

            Returns:
                The callable result.
            """
            return MagicMock()

        self.app.dependency_overrides[get_db] = override_get_db
        self.app.dependency_overrides[get_redis_pool] = override_get_redis_pool

    @patch(
        'examples.db_management.routers.password_reset.request_password_reset',
        new_callable=AsyncMock,
    )
    async def test_forgot_password_success(
        self,
        mock_request_password_reset: AsyncMock,
    ) -> None:
        """Test forgot password success.

        Args:
            mock_request_password_reset: Value used by this callable.
        """
        mock_request_password_reset.return_value = {
            'message': 'If the email exists, a reset link has been sent.',
        }

        response = self.client.post(
            '/password/forgot',
            json={'email': 'user@example.com'},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()['message'],
            'If the email exists, a reset link has been sent.',
        )
        self.assertEqual(
            mock_request_password_reset.call_args.args[0],
            'user@example.com',
        )
        self.assertIn(
            'client_ip',
            mock_request_password_reset.call_args.kwargs,
        )

    @patch(
        'examples.db_management.routers.password_reset.reset_password',
        new_callable=AsyncMock,
    )
    async def test_reset_password_success(
        self,
        mock_reset_password: AsyncMock,
    ) -> None:
        """Test reset password success.

        Args:
            mock_reset_password: Value used by this callable.
        """
        mock_reset_password.return_value = {
            'message': 'Password reset successfully.',
        }

        response = self.client.post(
            '/password/reset',
            json={'token': 'raw-token', 'new_password': 'NewPass123'},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()['message'],
            'Password reset successfully.',
        )
        self.assertEqual(mock_reset_password.call_args.args[0], 'raw-token')
        self.assertEqual(mock_reset_password.call_args.args[1], 'NewPass123')

    @patch(
        'examples.db_management.routers.password_reset.reset_password',
        new_callable=AsyncMock,
    )
    async def test_reset_password_error_uses_message_field(
        self,
        mock_reset_password: AsyncMock,
    ) -> None:
        """Test reset password error uses message field.

        Args:
            mock_reset_password: Value used by this callable.
        """
        mock_reset_password.side_effect = HTTPException(
            status_code=400,
            detail={
                'code': 'reset_token_invalid',
                'message': 'Reset token is invalid or expired.',
            },
        )

        response = self.client.post(
            '/password/reset',
            json={'token': 'bad-token', 'new_password': 'NewPass123'},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json(),
            {
                'code': 'reset_token_invalid',
                'message': 'Reset token is invalid or expired.',
            },
        )

    async def test_reset_password_missing_fields_returns_400_message(
        self,
    ) -> None:
        """Test reset password missing fields returns 400 message."""
        response = self.client.post('/password/reset', json={})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json(),
            {
                'code': 'reset_token_invalid',
                'message': 'Reset token is invalid or expired.',
            },
        )

    @patch(
        'examples.db_management.routers.password_reset.reset_password',
        new_callable=AsyncMock,
    )
    async def test_reset_password_preserves_structured_error(
        self,
        mock_reset_password: AsyncMock,
    ) -> None:
        """Test reset password preserves structured error.

        Args:
            mock_reset_password: Value used by this callable.
        """
        mock_reset_password.side_effect = HTTPException(
            status_code=400,
            detail={
                'code': 'password_too_short',
                'message': 'Password is too short.',
                'min_length': 8,
            },
        )

        response = self.client.post(
            '/password/reset',
            json={'token': 'raw-token', 'new_password': 'short'},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json(),
            {
                'code': 'password_too_short',
                'message': 'Password is too short.',
                'min_length': 8,
            },
        )
