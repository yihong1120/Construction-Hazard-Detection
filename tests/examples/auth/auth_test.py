from __future__ import annotations

import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import jwt
from fastapi import HTTPException
from redis.asyncio import Redis

from examples.auth.auth import verify_refresh_token


class TestVerifyRefreshToken(unittest.IsolatedAsyncioTestCase):
    """
    Test suite for the verify_refresh_token function.

    Args:
        mock_redis (Redis): Mock Redis client for testing.
    """

    async def asyncSetUp(self) -> None:
        """
        Set up common mocks before each test.
        """
        self.mock_redis: Redis = AsyncMock(spec=Redis)

    @patch('examples.auth.auth.get_user_data', new_callable=AsyncMock)
    @patch('examples.auth.auth.jwt.decode')
    async def test_verify_refresh_token_success(
        self,
        mock_jwt_decode: MagicMock,
        mock_get_user_data: AsyncMock,
    ) -> None:
        """
        Test successful verification of a valid refresh token.

        Args:
            mock_jwt_decode (MagicMock):
                Mock for the jwt.decode function.
            mock_get_user_data (AsyncMock):
                Mock for the get_user_data function.
        """
        mock_jwt_decode.return_value = {'subject': {'username': 'testuser'}}
        mock_get_user_data.return_value = {
            'refresh_tokens': ['my_valid_refresh_token'],
        }

        payload = await verify_refresh_token(
            'my_valid_refresh_token',
            self.mock_redis,
        )
        self.assertEqual(payload, {'subject': {'username': 'testuser'}})

    @patch(
        'examples.auth.auth.jwt.decode',
        side_effect=jwt.ExpiredSignatureError(),
    )
    async def test_verify_refresh_token_expired(
        self,
        mock_jwt_decode: MagicMock,
    ) -> None:
        """
        Test that an expired refresh token raises a 401 HTTPException.

        Args:
            mock_jwt_decode (MagicMock):
                Mock for the jwt.decode function.
        """
        with self.assertRaises(HTTPException) as ctx:
            await verify_refresh_token('fake_token', self.mock_redis)
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn('expired', ctx.exception.detail)

    @patch(
        'examples.auth.auth.jwt.decode',
        side_effect=jwt.InvalidTokenError(),
    )
    async def test_verify_refresh_token_invalid(
        self,
        mock_jwt_decode: MagicMock,
    ) -> None:
        """
        Test that an invalid refresh token raises a 401 HTTPException.

        Args:
            mock_jwt_decode (MagicMock):
                Mock for the jwt.decode function.
        """
        with self.assertRaises(HTTPException) as ctx:
            await verify_refresh_token('fake_token', self.mock_redis)
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn('Invalid refresh token', ctx.exception.detail)

    @patch('examples.auth.auth.get_user_data', new_callable=AsyncMock)
    @patch('examples.auth.auth.jwt.decode', return_value={'subject': {}})
    async def test_verify_refresh_token_missing_username(
        self,
        mock_jwt_decode: MagicMock,
        mock_get_user_data: AsyncMock,
    ) -> None:
        """
        Test that a refresh token missing 'username' in the payload
        is considered invalid and returns None.

        Args:
            mock_jwt_decode (MagicMock):
                Mock for the jwt.decode function.
            mock_get_user_data (AsyncMock):
                Mock for the get_user_data function.
        """
        mock_jwt_decode.return_value = {'subject': {}}
        mock_get_user_data.return_value = {
            'refresh_tokens': ['fake_refresh_token'],
        }
        with self.assertRaises(HTTPException) as ctx:
            await verify_refresh_token('fake_refresh_token', self.mock_redis)

        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn('Invalid refresh token payload', ctx.exception.detail)

    @patch(
        'examples.auth.auth.jwt.decode',
        return_value={'subject': {'username': 'testuser'}},
    )
    @patch(
        'examples.auth.auth.get_user_data',
        return_value=None,
    )
    async def test_verify_refresh_token_user_not_in_cache(
        self,
        mock_get_user_data: AsyncMock,
        mock_jwt_decode: MagicMock,
    ) -> None:
        """
        Test that if the user data is not found in Redis,
        verify_refresh_token should raise HTTPException(401).

        Args:
            mock_get_user_data (AsyncMock):
                Mock for the get_user_data function.
            mock_jwt_decode (MagicMock):
                Mock for the jwt.decode function.
        """
        with self.assertRaises(HTTPException) as ctx:
            await verify_refresh_token(
                'fake_refresh_token',
                self.mock_redis,
            )
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn('No user data in Redis', ctx.exception.detail)

    @patch(
        'examples.auth.auth.jwt.decode',
        return_value={'subject': {'username': 'testuser'}},
    )
    @patch(
        'examples.auth.auth.get_user_data',
        return_value={'refresh_tokens': ['some_other_token']},
    )
    async def test_verify_refresh_token_not_in_user_tokens(
        self,
        mock_get_user_data: AsyncMock,
        mock_jwt_decode: MagicMock,
    ) -> None:
        """
        Test that if the refresh token is not in the user's
        refresh tokens, verify_refresh_token should raise
        HTTPException(401).

        Args:
            mock_get_user_data (AsyncMock):
                Mock for the get_user_data function.
            mock_jwt_decode (MagicMock):
                Mock for the jwt.decode function.
        """
        with self.assertRaises(HTTPException) as ctx:
            await verify_refresh_token(
                'fake_refresh_token',
                self.mock_redis,
            )
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn('Refresh token not recognised', ctx.exception.detail)


if __name__ == '__main__':
    unittest.main()

'''
pytest \
    --cov=examples.auth.auth \
    --cov-report=term-missing tests/examples/auth/auth_test.py
'''
