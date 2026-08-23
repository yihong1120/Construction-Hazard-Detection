from __future__ import annotations

import unittest
from datetime import timedelta
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import HTTPException
from jwt.exceptions import InvalidTokenError
from redis.exceptions import RedisError

from examples.auth.config import Settings
from examples.auth.jwt_config import jwt_access
from examples.auth.jwt_config import jwt_refresh
from examples.auth.jwt_config import JwtAuthorizationCredentials
from examples.auth.jwt_config import PyJWTBearer
from examples.db_management.schemas.auth import AccessTokenSubject


def _access_subject(
    username: str = 'alice',
    user_id: int = 1,
) -> AccessTokenSubject:
    """Perform access subject.

    Args:
        username: Value used by this callable.
        user_id: Value used by this callable.

    Returns:
        The callable result.
    """
    return {
        'username': username,
        'user_id': user_id,
        'role': 'user',
        'jti': 'access-jti',
        'features': [],
    }


def _refresh_subject(username: str = 'alice') -> dict[str, str]:
    """Perform refresh subject.

    Args:
        username: Value used by this callable.

    Returns:
        The callable result.
    """
    return {
        'username': username,
        'family_id': 'refresh-family',
        'token_id': 'refresh-token-id',
    }


class TestJwtConfig(unittest.TestCase):
    """Test suite for ensuring jwt_access and jwt_refresh in jwt_config.py are
    instantiated correctly and are able to produce valid tokens."""

    def setUp(self) -> None:
        """Set up the Settings object for reference in tests."""
        self.settings: Settings = Settings()

    def test_jwt_access_initialized(self) -> None:
        """Verify that jwt_access is initialised correctly with the expected
        secret key."""
        self.assertIsNotNone(jwt_access, 'jwt_access should not be None.')
        self.assertEqual(
            jwt_access.secret_key,
            self.settings.authjwt_secret_key,
            'jwt_access should use the same secret key as Settings.',
        )

        token: str = jwt_access.create_access_token(
            subject=_access_subject('access-user'),
        )
        self.assertIsInstance(
            token,
            str,
            'The created access token must be a string.',
        )
        self.assertGreater(
            len(token),
            0,
            'The token string should not be empty.',
        )

    def test_jwt_refresh_initialized(self) -> None:
        """Verify that jwt_refresh is initialised correctly with the expected
        secret key."""
        self.assertIsNotNone(jwt_refresh, 'jwt_refresh should not be None.')
        self.assertEqual(
            jwt_refresh.secret_key,
            self.settings.authjwt_secret_key,
            'jwt_refresh should use the same secret key as Settings.',
        )

        token: str = jwt_refresh.create_access_token(
            subject=_refresh_subject('refresh-user'),
        )
        self.assertIsInstance(
            token,
            str,
            'The created refresh token must be a string.',
        )
        self.assertGreater(
            len(token),
            0,
            'The token string should not be empty.',
        )

    def test_credentials_support_existing_mapping_access(self) -> None:
        """Existing handlers can read JWT claims as a mapping."""
        credentials = JwtAuthorizationCredentials(_access_subject())

        self.assertEqual(credentials['username'], 'alice')
        self.assertEqual(credentials.get('role'), 'user')
        self.assertEqual(credentials.get('role', 'viewer'), 'user')


if __name__ == '__main__':
    unittest.main()
\
"""Pytest \

--cov=examples.auth.jwt_config \
--cov-report=term-missing tests/examples/auth/jwt_config_test.py
"""


class TestPyJwtBearerAuthorization(unittest.IsolatedAsyncioTestCase):
    """Exercise JWT authorization subjects and invalid credential handling."""

    def setUp(self) -> None:
        """Perform setUp.
        """
        self.bearer = PyJWTBearer('test-secret-key-with-at-least-32-bytes')
        self.request = MagicMock()

    async def test_bearer_returns_full_subject(
        self,
    ) -> None:
        """Valid tokens expose their full subject payload."""
        with (
            patch.object(
                self.bearer,
                'oauth2_scheme',
                new=AsyncMock(return_value='current-token'),
            ),
            patch.object(
                self.bearer,
                'decode_token',
                new=MagicMock(
                    return_value={
                        'subject': {
                            **_access_subject(),
                            'role': 'admin',
                        },
                        'jti': 'access-jti',
                    },
                ),
            ),
            patch(
                'examples.auth.jwt_config.is_access_token_revoked',
                new=AsyncMock(return_value=False),
            ),
        ):
            credentials = await self.bearer(self.request)

        self.assertEqual(
            credentials.subject,
            {
                'username': 'alice',
                'role': 'admin',
                'user_id': 1,
                'jti': 'access-jti',
                'features': [],
            },
        )
        self.assertEqual(credentials.token, 'current-token')

    async def test_bearer_rejects_missing_invalid_or_subjectless_tokens(
        self,
    ) -> None:
        """Every malformed credential path returns the same 401 contract."""
        for token, decoded in [
            (None, None),
            ('invalid-token', InvalidTokenError('signature failed')),
            ('subjectless-token', {'subject': {}}),
        ]:
            if isinstance(decoded, Exception):
                decode_token = MagicMock(side_effect=decoded)
            else:
                decode_token = MagicMock(return_value=decoded)

            with (
                patch.object(
                    self.bearer,
                    'oauth2_scheme',
                    new=AsyncMock(return_value=token),
                ),
                patch.object(
                    self.bearer,
                    'decode_token',
                    new=decode_token,
                ),
                self.assertRaises(HTTPException) as error,
            ):
                await self.bearer(self.request)

            self.assertEqual(error.exception.status_code, 401)
            self.assertEqual(
                error.exception.headers,
                {'WWW-Authenticate': 'Bearer'},
            )

    async def test_bearer_round_trips_a_signed_access_token(self) -> None:
        """PyJWT decoding validates tokens created with the configured
        secret."""
        token = self.bearer.create_access_token(
            _access_subject(),
            expires_delta=timedelta(minutes=1),
        )

        payload = self.bearer.decode_token(token)

        self.assertEqual(payload['sub'], 'alice')
        self.assertEqual(payload['subject']['username'], 'alice')
        self.assertIsInstance(payload['subject']['jti'], str)

    def test_decode_rejects_token_for_another_use(self) -> None:
        """A refresh payload cannot be accepted by the access bearer."""
        with (
            patch(
                'examples.auth.jwt_config.jwt.decode',
                return_value={'token_use': 'refresh'},
            ),
            self.assertRaises(InvalidTokenError),
        ):
            self.bearer.decode_token('refresh-token')

    async def test_bearer_reports_unavailable_revocation_service(self) -> None:
        """Access tokens fail closed when their revocation store is absent."""
        self.request.app.state.redis_client = None
        with (
            patch.object(
                self.bearer,
                'oauth2_scheme',
                new=AsyncMock(return_value='access-token'),
            ),
            patch.object(
                self.bearer,
                'decode_token',
                new=MagicMock(
                    return_value={
                        'subject': _access_subject(),
                        'jti': 'access-jti',
                    },
                ),
            ),
            self.assertRaises(HTTPException) as error,
        ):
            await self.bearer(self.request)

        self.assertEqual(error.exception.status_code, 503)

    async def test_bearer_reports_revocation_store_errors(self) -> None:
        """Redis failures have the same service-unavailable response."""
        self.request.app.state.redis_client.client = MagicMock()
        with (
            patch.object(
                self.bearer,
                'oauth2_scheme',
                new=AsyncMock(return_value='access-token'),
            ),
            patch.object(
                self.bearer,
                'decode_token',
                new=MagicMock(
                    return_value={
                        'subject': _access_subject(),
                        'jti': 'access-jti',
                    },
                ),
            ),
            patch(
                'examples.auth.jwt_config.is_access_token_revoked',
                new=AsyncMock(side_effect=RedisError('offline')),
            ),
            self.assertRaises(HTTPException) as error,
        ):
            await self.bearer(self.request)

        self.assertEqual(error.exception.status_code, 503)

    async def test_bearer_rejects_non_mapping_subject(self) -> None:
        """A non-mapping subject cannot become an authenticated user."""
        self.request.app.state.redis_client.client = MagicMock()
        with (
            patch.object(
                self.bearer,
                'oauth2_scheme',
                new=AsyncMock(return_value='access-token'),
            ),
            patch.object(
                self.bearer,
                'decode_token',
                new=MagicMock(return_value={'subject': 'alice'}),
            ),
            patch(
                'examples.auth.jwt_config.is_access_token_revoked',
                new=AsyncMock(return_value=False),
            ),
            self.assertRaises(HTTPException) as error,
        ):
            await self.bearer(self.request)

        self.assertEqual(error.exception.status_code, 401)
