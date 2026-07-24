from __future__ import annotations

import os
import unittest

from examples.auth.config import _env_bool
from examples.auth.config import _postgres_async_url
from examples.auth.config import Settings


class TestSettings(unittest.TestCase):
    """
    Unit tests for the FastAPI app settings configuration.
    """

    def setUp(self) -> None:
        """
        Backup environment variables
        """
        self.original_jwt_secret_key = os.getenv('JWT_SECRET_KEY')
        self.original_database_url = os.getenv('DATABASE_URL')

    def tearDown(self) -> None:
        """
        Restore environment variables after each test.
        """
        if self.original_jwt_secret_key is not None:
            os.environ['JWT_SECRET_KEY'] = self.original_jwt_secret_key
        if self.original_database_url is not None:
            os.environ['DATABASE_URL'] = self.original_database_url

    def test_settings_with_env_variables(self) -> None:
        """
        Test the settings configuration loads correctly.
        """
        # Instantiate the Settings class
        settings = Settings()

        # Assert that the settings are correctly loaded
        self.assertIsNotNone(settings.authjwt_secret_key)
        self.assertIsNotNone(settings.sqlalchemy_database_uri)

        # Assert that the database URL uses asyncpg driver
        self.assertIn('asyncpg', settings.sqlalchemy_database_uri)
        self.assertTrue(
            settings.sqlalchemy_database_uri.startswith(
                'postgresql+asyncpg://',
            ),
        )

        # Assert that the SQLAlchemy track modifications setting is
        # correctly loaded.
        self.assertFalse(settings.sqlalchemy_track_modifications)

    def test_settings_with_default_values(self) -> None:
        """
        Test the settings configuration reads from environment variables.
        """
        # Instantiate the Settings class
        settings = Settings()

        # Assert that the settings are correctly loaded
        # The values will come from environment variables,
        # possibly from the .env file.
        self.assertIsNotNone(settings.authjwt_secret_key)
        self.assertIsNotNone(settings.sqlalchemy_database_uri)

        # Assert that the database URL contains asyncpg driver
        self.assertIn('asyncpg', settings.sqlalchemy_database_uri)

        # Assert that the SQLAlchemy track modifications setting is
        # correctly loaded.
        self.assertFalse(settings.sqlalchemy_track_modifications)

    def test_configuration_helpers_normalise_urls_and_missing_booleans(self) -> None:
        """Low-level settings helpers keep legacy URLs and defaults predictable."""
        self.assertEqual(
            _postgres_async_url('postgres://user:password@host/database'),
            'postgresql+asyncpg://user:password@host/database',
        )
        with unittest.mock.patch.dict(os.environ, {}, clear=False):
            self.assertTrue(_env_bool('MISSING_BOOLEAN_SWITCH', True))

    def test_settings_requires_non_empty_jwt_secret(self) -> None:
        """An explicitly empty JWT secret prevents an unsafe application start."""
        field = Settings.model_fields['authjwt_secret_key']
        original_default = field.default
        try:
            field.default = ''
            Settings.model_rebuild(force=True)
            with self.assertRaisesRegex(RuntimeError, 'JWT_SECRET_KEY'):
                Settings()
        finally:
            field.default = original_default
            Settings.model_rebuild(force=True)


if __name__ == '__main__':
    unittest.main()

"""
pytest \
    --cov=examples.auth.config \
    --cov-report=term-missing tests/examples/auth/config_test.py
"""
