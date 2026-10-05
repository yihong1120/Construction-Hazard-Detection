from __future__ import annotations

import os
import unittest
from unittest import mock

from examples.auth.config import _env_bool
from examples.auth.config import Settings


class TestSettings(unittest.TestCase):
    """Unit tests for the FastAPI app settings configuration."""

    def setUp(self) -> None:
        """Backup environment variables."""
        self.original_database_url = os.getenv('DATABASE_URL')

    def tearDown(self) -> None:
        """Restore environment variables after each test."""
        if self.original_database_url is not None:
            os.environ['DATABASE_URL'] = self.original_database_url

    def test_settings_with_env_variables(self) -> None:
        """Test the settings configuration loads correctly."""
        # Instantiate the Settings class
        settings = Settings()

        # Assert that the settings are correctly loaded
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
        """Test the settings configuration reads from environment variables."""
        # Instantiate the Settings class
        settings = Settings()

        # Assert that the settings are correctly loaded
        # The values will come from environment variables,
        # possibly from the .env file.
        self.assertIsNotNone(settings.sqlalchemy_database_uri)

        # Assert that the database URL contains asyncpg driver
        self.assertIn('asyncpg', settings.sqlalchemy_database_uri)

        # Assert that the SQLAlchemy track modifications setting is
        # correctly loaded.
        self.assertFalse(settings.sqlalchemy_track_modifications)

    def test_configuration_boolean_helper(
        self,
    ) -> None:
        """Boolean settings retain their documented parsing behaviour."""
        with mock.patch.dict(os.environ, {}, clear=False):
            self.assertTrue(_env_bool('MISSING_BOOLEAN_SWITCH', True))
        with mock.patch.dict(
            os.environ,
            {'CONFIG_TEST_BOOLEAN_SWITCH': ' Yes '},
            clear=False,
        ):
            self.assertTrue(_env_bool('CONFIG_TEST_BOOLEAN_SWITCH'))


if __name__ == '__main__':
    unittest.main()
