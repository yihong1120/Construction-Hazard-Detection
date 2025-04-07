from __future__ import annotations

import unittest

from examples.auth.config import Settings
from examples.auth.jwt_config import jwt_access
from examples.auth.jwt_config import jwt_refresh


class TestJwtConfig(unittest.TestCase):
    """
    Test suite for ensuring jwt_access and jwt_refresh in jwt_config.py
    are instantiated correctly and are able to produce valid tokens.
    """

    def setUp(self) -> None:
        """
        Set up the Settings object for reference in tests.
        """
        self.settings: Settings = Settings()

    def test_jwt_access_initialized(self) -> None:
        """
        Verify that jwt_access is initialised correctly
        with the expected secret key.
        """
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
        """
        Verify that jwt_refresh is initialised correctly with
        the expected secret key.
        """
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

'''
pytest \
    --cov=examples.auth.jwt_config \
    --cov-report=term-missing tests/examples/auth/jwt_config_test.py
'''
