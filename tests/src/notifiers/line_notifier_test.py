from __future__ import annotations

import os
import subprocess
import unittest
from io import BytesIO
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.notifiers.line_notifier import LineNotifier
from src.notifiers.line_notifier import main


class TestLineNotifier(unittest.IsolatedAsyncioTestCase):
    """
    Unit tests for the LineNotifier class methods.
    """

    def setUp(self) -> None:
        """
        Sets up the test environment by initialising test variables
        and a LineNotifier instance.
        """
        # Mock LINE Notify token for testing
        self.line_token: str = 'test_token'

        # Mock message to send
        self.message: str = 'Test Message'
        self.image: np.ndarray = np.zeros(
            (100, 100, 3), dtype=np.uint8,
        )  # Mock image
        self.notifier: LineNotifier = LineNotifier()

    @patch.dict('os.environ', {'LINE_NOTIFY_TOKEN': 'test_env_token'})
    @patch('aiohttp.ClientSession.post')
    async def test_init_with_env_token(self, mock_post: MagicMock) -> None:
        """
        Tests the notification sending functionality
        using an environment variable token.

        Args:
            mock_post (MagicMock): Mock object for aiohttp.ClientSession.post.
        """
        mock_response: MagicMock = MagicMock()
        mock_response.status = 200  # Simulate a successful response
        mock_post.return_value.__aenter__.return_value = mock_response

        notifier: LineNotifier = LineNotifier()
        status_code: int = await notifier.send_notification(self.message)

        # Assert that the HTTP response status is 200
        self.assertEqual(status_code, 200)
        # Ensure that aiohttp's post method was called exactly once
        mock_post.assert_called_once()

    async def test_init_without_token(self) -> None:
        """
        Tests the behaviour when attempting to send a notification
        without providing a token.

        Raises:
            ValueError: Expected exception when no token is provided.
        """
        with self.assertRaises(ValueError):
            await self.notifier.send_notification(
                self.message, line_token=None,
            )

    @patch('aiohttp.ClientSession.post')
    async def test_send_notification_without_image(
        self, mock_post: MagicMock,
    ) -> None:
        """
        Tests sending a notification without an image.

        Args:
            mock_post (MagicMock): Mock object for aiohttp.ClientSession.post.
        """
        mock_response: MagicMock = MagicMock()
        mock_response.status = 200  # Simulate a successful response
        mock_post.return_value.__aenter__.return_value = mock_response

        status_code: int = await self.notifier.send_notification(
            self.message, line_token=self.line_token,
        )
        self.assertEqual(status_code, 200)
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertIn('data', kwargs)
        self._assert_field_exists(
            kwargs['data']._fields, 'message', self.message,
        )

    @patch('aiohttp.ClientSession.post')
    async def test_send_notification_with_image(
        self, mock_post: MagicMock,
    ) -> None:
        """
        Tests sending a notification with an image provided as a NumPy array.

        Args:
            mock_post (MagicMock): Mock object for aiohttp.ClientSession.post.
        """
        mock_response: MagicMock = MagicMock()
        mock_response.status = 200  # Simulate a successful response
        mock_post.return_value.__aenter__.return_value = mock_response

        status_code: int = await self.notifier.send_notification(
            self.message, self.image, line_token=self.line_token,
        )
        self.assertEqual(status_code, 200)
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertIn('data', kwargs)
        self._assert_field_exists(
            kwargs['data']._fields, 'message', self.message,
        )
        self._assert_field_exists(kwargs['data']._fields, 'imageFile', None)

    @patch('aiohttp.ClientSession.post')
    async def test_send_notification_with_bytes_image(
        self, mock_post: MagicMock,
    ) -> None:
        """
        Tests sending a notification with an image provided as bytes.

        Args:
            mock_post (MagicMock): Mock object for aiohttp.ClientSession.post.
        """
        mock_response: MagicMock = MagicMock()
        mock_response.status = 200  # Simulate a successful response
        mock_post.return_value.__aenter__.return_value = mock_response

        # Convert the NumPy array image to bytes
        buffer: BytesIO = BytesIO()
        Image.fromarray(self.image).save(buffer, format='PNG')
        buffer.seek(0)
        image_bytes: bytes = buffer.read()

        status_code: int = await self.notifier.send_notification(
            self.message, image_bytes, line_token=self.line_token,
        )
        self.assertEqual(status_code, 200)
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertIn('data', kwargs)
        self._assert_field_exists(
            kwargs['data']._fields, 'message', self.message,
        )
        self._assert_field_exists(kwargs['data']._fields, 'imageFile', None)

    @patch('aiohttp.ClientSession.post')
    async def test_main(self, mock_post: MagicMock) -> None:
        """
        Tests the main function to ensure the entire process works as expected.

        Args:
            mock_post (MagicMock): Mock object for aiohttp.ClientSession.post.
        """
        mock_response: MagicMock = MagicMock()
        mock_response.status = 200  # Simulate a successful response
        mock_post.return_value.__aenter__.return_value = mock_response

        with patch('builtins.print') as mock_print:
            await main()
            mock_print.assert_called_once_with('Response code: 200')

    def _assert_field_exists(
        self, fields: list, field_name: str, expected_value: str | None = None,
    ) -> None:
        """
        Helper method to assert that a field exists in FormData fields.

        Args:
            fields (list): List of FormData fields.
            field_name (str): The name of the field to check for.
            expected_value (str | None):
                The expected value of the field (optional).

        Raises:
            AssertionError: If the field is not found
            or does not match the expected value.
        """
        found = False
        for field, _, value in fields:
            if field.get('name') == field_name:
                found = True
                if expected_value is not None:
                    self.assertEqual(value, expected_value)
                break
        self.assertTrue(
            found, f"Field '{field_name}' not found in FormData fields.",
        )

    @patch.dict(os.environ, {'LINE_NOTIFY_TOKEN': 'test_token'})
    @patch('aiohttp.ClientSession.post')
    async def test_main_as_script(self, mock_post: MagicMock) -> None:
        """
        Tests running the main function as a standalone script.

        Args:
            mock_post (MagicMock): Mock object for aiohttp.ClientSession.post.
        """
        mock_response: MagicMock = MagicMock()
        mock_response.status = 200  # Simulate a successful response
        mock_post.return_value.__aenter__.return_value = mock_response

        # Get the absolute path to the script
        script_path = os.path.abspath(
            os.path.join(
                os.path.dirname(
                    __file__,
                ), '../../../src/notifiers/line_notifier.py',
            ),
        )

        # Run the script using subprocess
        result = subprocess.run(
            ['python', script_path], capture_output=True, text=True,
        )

        self.assertEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
