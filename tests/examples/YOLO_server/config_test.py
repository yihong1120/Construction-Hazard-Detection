from __future__ import annotations

import unittest
from unittest.mock import patch
from examples.YOLO_server.config import Settings
import os

class TestSettings(unittest.TestCase):
    def setUp(self):
        # Backup environment variables
        self.original_jwt_secret_key = os.getenv('JWT_SECRET_KEY')
        self.original_database_url = os.getenv('DATABASE_URL')

    def tearDown(self):
        # Restore environment variables
        if self.original_jwt_secret_key is not None:
            os.environ['JWT_SECRET_KEY'] = self.original_jwt_secret_key
        if self.original_database_url is not None:
            os.environ['DATABASE_URL'] = self.original_database_url

    @patch.dict(os.environ, {'JWT_SECRET_KEY': 'test_secret_key', 'DATABASE_URL': 'mysql+pymysql://test_user:test_password@localhost/test_db'})
    def test_settings_with_env_variables(self):
        settings = Settings()  # Remove _env_file parameter to let pydantic read from environment variables
        self.assertEqual(settings.authjwt_secret_key, 'test_secret_key')
        self.assertEqual(settings.sqlalchemy_database_uri, 'mysql+pymysql://test_user:test_password@localhost/test_db')
        self.assertFalse(settings.sqlalchemy_track_modifications)

    @patch.dict(os.environ, {}, clear=True)
    def test_settings_with_default_values(self):
        settings = Settings()  # Remove _env_file parameter to let pydantic use default values
        self.assertEqual(settings.authjwt_secret_key, 'your_fallback_secret_key')
        self.assertEqual(settings.sqlalchemy_database_uri, 'mysql+pymysql://user:passcode@localhost/construction_hazard_detection')
        self.assertFalse(settings.sqlalchemy_track_modifications)

if __name__ == "__main__":
    unittest.main()
