from __future__ import annotations

import unittest
from io import BytesIO
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.line_notifier import LineNotifier


class TestLineNotifier(unittest.TestCase):
    """
    Unit tests for the LineNotifier class methods.
    """

    def setUp(self) -> None:
        """
        Set up method to initialise test variables and LineNotifier instance.
        """
        self.line_token: str = 'test_token'
        self.message: str = 'Test Message'
        self.image: np.ndarray = np.zeros((100, 100, 3), dtype=np.uint8)
        self.notifier: LineNotifier = LineNotifier(line_token=self.line_token)

    @patch.dict('os.environ', {'LINE_NOTIFY_TOKEN': 'test_env_token'})
    def test_init_with_env_token(self) -> None:
        """
        Test case for initialising LineNotifier with environment token.
        """
        notifier: LineNotifier = LineNotifier()
        self.assertEqual(notifier.line_token, 'test_env_token')

    def test_init_with_provided_token(self) -> None:
        """
        Test case for initialising LineNotifier with provided token.
        """
        notifier: LineNotifier = LineNotifier(line_token='provided_token')
        self.assertEqual(notifier.line_token, 'provided_token')

    def test_init_without_token(self) -> None:
        """
        Test case for initialising LineNotifier
            without a token (expects ValueError).
        """
        with self.assertRaises(ValueError):
            LineNotifier(line_token=None)

    @patch('src.line_notifier.requests.post')
    def test_send_notification_without_image(
        self,
        mock_post: MagicMock,
    ) -> None:
        """
        Test case for sending notification without an image.
        """
        mock_response: MagicMock = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        status_code: int = self.notifier.send_notification(self.message)
        self.assertEqual(status_code, 200)
        mock_post.assert_called_once_with(
            'https://notify-api.line.me/api/notify',
            headers={'Authorization': f'Bearer {self.line_token}'},
            params={'message': self.message},
        )

    @patch('src.line_notifier.requests.post')
    def test_send_notification_with_image(self, mock_post: MagicMock) -> None:
        """
        Test case for sending notification with an image as a NumPy array.
        """
        mock_response: MagicMock = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        status_code: int = self.notifier.send_notification(
            self.message,
            self.image,
        )
        self.assertEqual(status_code, 200)
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertIn('files', kwargs)
        self.assertIn('imageFile', kwargs['files'])

        # Check if the image is correctly converted and sent
        image_file = kwargs['files']['imageFile']
        self.assertEqual(image_file[0], 'image.png')
        self.assertEqual(image_file[2], 'image/png')
        image: Image.Image = Image.open(image_file[1])
        self.assertTrue(np.array_equal(np.array(image), self.image))

    @patch('src.line_notifier.requests.post')
    def test_send_notification_with_bytes_image(
        self,
        mock_post: MagicMock,
    ) -> None:
        """
        Test case for sending notification
            with an image as bytes (e.g., BytesIO).
        """
        mock_response: MagicMock = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        buffer: BytesIO = BytesIO()
        Image.fromarray(self.image).save(buffer, format='PNG')
        buffer.seek(0)
        image_bytes: bytes = buffer.read()

        status_code: int = self.notifier.send_notification(
            self.message, np.array(Image.open(BytesIO(image_bytes))),
        )
        self.assertEqual(status_code, 200)
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertIn('files', kwargs)
        self.assertIn('imageFile', kwargs['files'])

        # Check if the image is correctly converted and sent
        image_file = kwargs['files']['imageFile']
        self.assertEqual(image_file[0], 'image.png')
        self.assertEqual(image_file[2], 'image/png')
        image: Image.Image = Image.open(image_file[1])
        self.assertTrue(np.array_equal(np.array(image), self.image))


if __name__ == '__main__':
    unittest.main()
