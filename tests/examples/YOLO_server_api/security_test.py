from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi import FastAPI

from examples.YOLO_server_api.security import update_secret_key


class TestUpdateSecretKey(unittest.TestCase):
    """
    Unit tests for the `update_secret_key` function in FastAPI applications.
    """

    def setUp(self) -> None:
        """
        Initialises the FastAPI application instance before each test.
        """
        self.app: FastAPI = FastAPI()

    @patch('examples.YOLO_server_api.security.secrets.token_urlsafe')
    def test_update_secret_key(
        self, mock_token_urlsafe: unittest.mock.MagicMock,
    ) -> None:
        """
        Tests that `update_secret_key` updates the secret key correctly.

        Args:
            mock_token_urlsafe (unittest.mock.MagicMock): Mocked
                `token_urlsafe` function.
        """
        # Mock the return value of `token_urlsafe` to a specific key
        mock_token_urlsafe.return_value = 'mocked_secret_key'

        # Invoke `update_secret_key` function
        update_secret_key(self.app)

        # Verify if the `jwt_secret_key` has been set to the mocked key
        self.assertEqual(self.app.state.jwt_secret_key, 'mocked_secret_key')

        # Confirm that `token_urlsafe` was called exactly once with
        # an argument of 16
        mock_token_urlsafe.assert_called_once_with(16)

    def test_update_secret_key_different_keys(self) -> None:
        """Tests that `update_secret_key` generates unique keys each time.

        This test verifies that each invocation of `update_secret_key`
        generates a new and unique secret key.
        """
        # Generate the first secret key
        update_secret_key(self.app)
        first_key: str = self.app.state.jwt_secret_key

        # Generate the second secret key
        update_secret_key(self.app)
        second_key: str = self.app.state.jwt_secret_key

        # Assert that the two keys are different, are strings,
        # and the second key is non-empty
        self.assertNotEqual(first_key, second_key)
        self.assertIsInstance(second_key, str)
        self.assertGreater(len(second_key), 0)


if __name__ == '__main__':
    unittest.main()
