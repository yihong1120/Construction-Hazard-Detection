from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from examples.YOLO_server_api.config import Settings


class TestSettings(unittest.TestCase):
    """
    Unit tests for the FastAPI app settings configuration.
    """

    def setUp(self):
        """
        Backup environment variables
        """
        self.original_jwt_secret_key = os.getenv('JWT_SECRET_KEY')
        self.original_database_url = os.getenv('DATABASE_URL')

    def tearDown(self):
        """
        Restore environment variables after each test.
        """
        if self.original_jwt_secret_key is not None:
            os.environ['JWT_SECRET_KEY'] = self.original_jwt_secret_key
        if self.original_database_url is not None:
            os.environ['DATABASE_URL'] = self.original_database_url

    @patch.dict(
        os.environ,
        {
            'JWT_SECRET_KEY': 'your_fallback_secret_key',
            'DATABASE_URL': (
                'mysql+asyncmy://test_user:test_password@localhost/'
                'test_db'
            ),
        },
    )
    def test_settings_with_env_variables(self):
        """
        Test the settings configuration with environment variables.
        """
        # Instantiate the Settings class with environment variables
        settings = Settings()

        # Assert that the settings are correctly loaded from
        # environment variables
        self.assertEqual(
            settings.authjwt_secret_key,
            'your_fallback_secret_key',
        )

        # Assert that the database URL is correctly loaded from
        # environment variables
        self.assertEqual(
            settings.sqlalchemy_database_uri,
            'mysql+asyncmy://user:passcode@localhost/'
            'construction_hazard_detection',
        )

        # Assert that the SQLAlchemy track modifications setting is correctly
        # loaded from environment variables
        self.assertFalse(settings.sqlalchemy_track_modifications)

    @patch.dict(os.environ, {}, clear=True)
    def test_settings_with_default_values(self):
        """
        Test the settings configuration with default values.
        """
        # Instantiate the Settings class with default values
        settings = Settings()

        # Assert that the settings are correctly loaded with default values
        self.assertEqual(
            settings.authjwt_secret_key,
            'your_fallback_secret_key',
        )

        # Assert that the database URL is correctly loaded with default values
        self.assertEqual(
            settings.sqlalchemy_database_uri,
            'mysql+asyncmy://user:passcode@localhost/'
            'construction_hazard_detection',
        )

        # Assert that the SQLAlchemy track modifications setting is correctly
        # loaded with default
        self.assertFalse(settings.sqlalchemy_track_modifications)


if __name__ == '__main__':
    unittest.main()
