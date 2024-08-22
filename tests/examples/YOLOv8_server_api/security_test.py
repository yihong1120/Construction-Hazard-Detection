from __future__ import annotations

import unittest
from unittest.mock import patch

from flask import Flask

from examples.YOLOv8_server_api.security import update_secret_key


class TestUpdateSecretKey(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up the test environment before each test.
        """
        self.app = Flask(__name__)

    def tearDown(self) -> None:
        """
        Clean up after each test.
        """
        # Delete the Flask app instance
        del self.app

    @patch('examples.YOLOv8_server_api.security.secrets.token_urlsafe')
    def test_update_secret_key(self, mock_token_urlsafe):
        """
        Test that the JWT secret key is updated in the Flask app config.
        """
        # Define a mock secret key
        mock_secret_key = 'mocked_secret_key'
        mock_token_urlsafe.return_value = mock_secret_key

        # Call the function to update the secret key
        update_secret_key(self.app)

        # Assert that the secret key in app config has been updated correctly
        self.assertEqual(self.app.config['JWT_SECRET_KEY'], mock_secret_key)

        # Ensure that token_urlsafe was called once with the expected length
        mock_token_urlsafe.assert_called_once_with(16)


if __name__ == '__main__':
    unittest.main()
