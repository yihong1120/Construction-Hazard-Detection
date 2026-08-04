from __future__ import annotations

import unittest
from datetime import timedelta
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import HTTPException
from jwt.exceptions import InvalidTokenError

from examples.auth.config import Settings
from examples.auth.jwt_config import jwt_access
from examples.auth.jwt_config import jwt_refresh
from examples.auth.jwt_config import PyJWTBearer


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

        token: str = jwt_access.create_access_token(subject={'foo': 'bar'})
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

        token: str = jwt_refresh.create_access_token(subject={'spam': 'ham'})
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
        self.bearer = PyJWTBearer('test-secret-key-with-at-least-32-bytes')
        self.request = MagicMock()

    async def test_bearer_returns_full_subject_and_legacy_sub_fallback(
        self,
    ) -> None:
        """Both current and legacy token payloads expose a username subject."""
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
                        'subject': {'username': 'alice', 'role': 'admin'},
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
            },
        )
        self.assertEqual(credentials.token, 'current-token')

        with (
            patch.object(
                self.bearer,
                'oauth2_scheme',
                new=AsyncMock(return_value='legacy-token'),
            ),
            patch.object(
                self.bearer,
                'decode_token',
                new=MagicMock(return_value={'sub': 'legacy'}),
            ),
            patch(
                'examples.auth.jwt_config.is_access_token_revoked',
                new=AsyncMock(return_value=False),
            ),
        ):
            credentials = await self.bearer(self.request)

        self.assertEqual(credentials.subject, {'username': 'legacy'})
        self.assertEqual(credentials.payload, {'sub': 'legacy'})

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
            {'username': 'alice'},
            expires_delta=timedelta(minutes=1),
        )

        payload = self.bearer.decode_token(token)

        self.assertEqual(payload['sub'], 'alice')
        self.assertEqual(payload['subject'], {'username': 'alice'})
