import unittest
import os
from unittest.mock import patch
from importlib import reload
import examples.YOLOv8_server_api.config as config_module

class TestConfig(unittest.TestCase):

    @patch.dict(os.environ, {'JWT_SECRET_KEY': 'test_secret_key'}, clear=True)
    def test_jwt_secret_key_from_env(self):
        """
        Test that JWT_SECRET_KEY is fetched from environment variables.
        """
        reload(config_module)  # Reload the module to pick up the patched environment variable
        config = config_module.Config()
        self.assertEqual(config.JWT_SECRET_KEY, 'test_secret_key')

    @patch.dict(os.environ, {}, clear=True)
    def test_jwt_secret_key_fallback(self):
        """
        Test that JWT_SECRET_KEY uses the fallback value when not set in the environment.
        """
        reload(config_module)  # Reload the module to pick up the patched environment variable
        config = config_module.Config()
        self.assertEqual(config.JWT_SECRET_KEY, 'your_fallback_secret_key')

    @patch.dict(os.environ, {'DATABASE_URL': 'mysql://user:passcode@localhost/construction_hazard_detection'}, clear=True)
    def test_database_url_from_env(self):
        """
        Test that SQLALCHEMY_DATABASE_URI is fetched from environment variables.
        """
        reload(config_module)  # Reload the module to pick up the patched environment variable
        config = config_module.Config()
        self.assertEqual(config.SQLALCHEMY_DATABASE_URI, 'mysql://user:passcode@localhost/construction_hazard_detection')

    @patch.dict(os.environ, {}, clear=True)
    def test_database_url_fallback(self):
        """
        Test that SQLALCHEMY_DATABASE_URI uses the fallback value when not set in the environment.
        """
        reload(config_module)  # Reload the module to pick up the patched environment variable
        config = config_module.Config()
        self.assertEqual(config.SQLALCHEMY_DATABASE_URI, 'mysql://user:passcode@localhost/construction_hazard_detection')

    def test_sqlalchemy_track_modifications(self):
        """
        Test that SQLALCHEMY_TRACK_MODIFICATIONS is set to False.
        """
        config = config_module.Config()
        self.assertFalse(config.SQLALCHEMY_TRACK_MODIFICATIONS)


if __name__ == '__main__':
    unittest.main()