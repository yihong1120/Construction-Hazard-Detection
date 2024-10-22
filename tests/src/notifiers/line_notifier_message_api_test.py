from __future__ import annotations

import os
import unittest
from datetime import datetime
from datetime import timedelta
from unittest.mock import MagicMock, mock_open
from unittest.mock import patch

import cv2
import numpy as np

from src.notifiers.line_notifier_message_api import LineMessenger
from src.notifiers.line_notifier_message_api import main


class TestLineMessenger(unittest.TestCase):
    """
    Test cases for the LineMessenger class.
    """

    def setUp(self):
        """
        Set up mock data and environment for each test case.
        """
        self.channel_access_token = 'test_channel_access_token'
        self.messenger = LineMessenger(
            channel_access_token=self.channel_access_token,
        )
        self.message = 'Test message'
        self.recipient_id = 'test_recipient_id'
        self.image_bytes = b'test_image_bytes'
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f"Bearer {self.channel_access_token}",
        }

    @patch.dict(os.environ, {'LINE_CHANNEL_ACCESS_TOKEN': ''})
    def test_init_without_channel_access_token(self):
        """
        Test initializing LineMessenger without a channel access token.
        """
        with self.assertRaises(ValueError) as context:
            LineMessenger()
        self.assertEqual(
            str(context.exception),
            'LINE_CHANNEL_ACCESS_TOKEN not provided '
            'or in environment variables.',
        )

    @patch('src.notifiers.line_notifier_message_api.requests.post')
    @patch(
        'src.notifiers.line_notifier_message_api.LineMessenger.'
        'upload_image_to_cloudinary',
    )
    def test_push_message_with_image(
        self,
        mock_upload_image_to_cloudinary: unittest.mock.MagicMock,
        mock_post: unittest.mock.MagicMock,
    ):
        """
        Test sending a message with an image via LINE Messaging API.
        """
        mock_upload_image_to_cloudinary.return_value = (
            'http://mock_image_url', 'mock_public_id',
        )
        mock_post.return_value.status_code = 200

        response_code = self.messenger.push_message(
            recipient_id=self.recipient_id,
            message=self.message,
            image_bytes=self.image_bytes,
        )

        # Verify that the upload_image_to_cloudinary method was called
        mock_upload_image_to_cloudinary.assert_called_once_with(
            self.image_bytes,
        )

        # Verify that the LINE API was called with the correct data
        expected_data = {
            'to': self.recipient_id,
            'messages': [
                {
                    'type': 'text',
                    'text': self.message,
                },
                {
                    'type': 'image',
                    'originalContentUrl': 'http://mock_image_url',
                    'previewImageUrl': 'http://mock_image_url',
                },
            ],
        }
        mock_post.assert_called_once_with(
            'https://api.line.me/v2/bot/message/push',
            headers=self.headers,
            json=expected_data,
        )

        # Verify the response code is 200
        self.assertEqual(response_code, 200)

    @patch(
        'src.notifiers.line_notifier_message_api.LineMessenger.'
        'delete_old_images',
    )
    @patch(
        'src.notifiers.line_notifier_message_api.LineMessenger.'
        'save_image_records',
    )
    def test_delete_old_images_with_interval_no_last_checked(
        self,
        mock_save_image_records: unittest.mock.MagicMock,
        mock_delete_old_images: unittest.mock.MagicMock,
    ):
        """
        Test the delete_old_images_with_interval method
        when last_checked is None.
        """
        # Ensure last_checked is None
        self.messenger.image_records['last_checked'] = None

        # Call the method
        self.messenger.delete_old_images_with_interval()

        # Verify that delete_old_images
        # was called since last_checked is None
        mock_delete_old_images.assert_called_once()

        # Verify that the save_image_records
        # was called to save the new check time
        mock_save_image_records.assert_called_once()

        # Check that last_checked is now set to the current time
        self.assertIn('last_checked', self.messenger.image_records)

    @patch(
        'src.notifiers.line_notifier_message_api.LineMessenger.'
        'delete_old_images',
    )
    @patch(
        'src.notifiers.line_notifier_message_api.LineMessenger.'
        'save_image_records',
    )
    def test_delete_old_images_with_invalid_last_checked(
        self,
        mock_save_image_records: unittest.mock.MagicMock,
        mock_delete_old_images: unittest.mock.MagicMock,
    ):
        """
        Test the delete_old_images_with_interval method
        when last_checked is invalid.
        """
        # Set an invalid last_checked value to
        # simulate ValueError during parsing
        self.messenger.image_records['last_checked'] = (
            'invalid_datetime_format'
        )

        # Call the method
        self.messenger.delete_old_images_with_interval()

        # Verify that delete_old_images was called
        # since last_checked is invalid
        mock_delete_old_images.assert_called_once()

        # Verify that save_image_records was called to save the new check time
        mock_save_image_records.assert_called_once()

        # Check that last_checked is now set to the current time
        self.assertIn('last_checked', self.messenger.image_records)

    @patch('src.notifiers.line_notifier_message_api.requests.post')
    def test_push_message_without_image(self, mock_post):
        """
        Test sending a message without an image via LINE Messaging API.
        """
        mock_post.return_value.status_code = 200

        response_code = self.messenger.push_message(
            recipient_id=self.recipient_id,
            message=self.message,
        )

        # Verify that no image upload is attempted
        expected_data = {
            'to': self.recipient_id,
            'messages': [
                {
                    'type': 'text',
                    'text': self.message,
                },
            ],
        }
        mock_post.assert_called_once_with(
            'https://api.line.me/v2/bot/message/push',
            headers=self.headers,
            json=expected_data,
        )

        # Verify the response code is 200
        self.assertEqual(response_code, 200)

    @patch('cloudinary.uploader.upload')
    def test_upload_image_to_cloudinary_success(self, mock_upload):
        """
        Test successful image upload to Cloudinary.
        """
        mock_upload.return_value = {
            'secure_url': 'http://mock_image_url',
            'public_id': 'mock_public_id',
        }

        image_url, public_id = self.messenger.upload_image_to_cloudinary(
            self.image_bytes,
        )

        # Verify that the Cloudinary uploader
        # was called with the correct arguments
        mock_upload.assert_called_once_with(
            self.image_bytes, resource_type='image',
        )

        # Verify the return values
        self.assertEqual(image_url, 'http://mock_image_url')
        self.assertEqual(public_id, 'mock_public_id')

    @patch('cloudinary.uploader.upload')
    def test_upload_image_to_cloudinary_failure(self, mock_upload):
        """
        Test failure when uploading an image to Cloudinary.
        """
        mock_upload.side_effect = Exception('Cloudinary upload failed')

        image_url, public_id = self.messenger.upload_image_to_cloudinary(
            self.image_bytes,
        )

        # Verify the return values are empty strings on failure
        self.assertEqual(image_url, '')
        self.assertEqual(public_id, '')

    @patch('cloudinary.uploader.destroy')
    def test_delete_image_from_cloudinary_success(self, mock_destroy):
        """
        Test successful image deletion from Cloudinary.
        """
        mock_destroy.return_value = {'result': 'ok'}

        self.messenger.delete_image_from_cloudinary('mock_public_id')

        # Verify that Cloudinary's destroy method
        # was called with the correct public_id
        mock_destroy.assert_called_once_with('mock_public_id')

    @patch('cloudinary.uploader.destroy')
    def test_delete_image_from_cloudinary_failure(self, mock_destroy):
        """
        Test failure when deleting an image from Cloudinary.
        """
        mock_destroy.return_value = {'result': 'error'}

        self.messenger.delete_image_from_cloudinary('mock_public_id')

        # Verify that Cloudinary's destroy method
        # was called with the correct public_id
        mock_destroy.assert_called_once_with('mock_public_id')

    @patch('cloudinary.uploader.destroy')
    def test_delete_image_from_cloudinary_exception(self, mock_destroy):
        """
        Test exception handling when deleting an image from Cloudinary.
        """
        mock_destroy.side_effect = Exception('Mocked exception')

        with patch('builtins.print') as mock_print:
            self.messenger.delete_image_from_cloudinary('mock_public_id')
            mock_print.assert_called_once_with(
                'Error deleting image from Cloudinary: Mocked exception',
            )

    @patch(
        'src.notifiers.line_notifier_message_api.LineMessenger.'
        'delete_image_from_cloudinary',
    )
    @patch(
        'src.notifiers.line_notifier_message_api.LineMessenger.'
        'save_image_records',
    )
    def test_delete_old_images(
        self,
        mock_save_image_records: unittest.mock.MagicMock,
        mock_delete_image_from_cloudinary: unittest.mock.MagicMock,
    ):
        """
        Test the delete_old_images method.
        """
        now = datetime.now()
        old_time = (now - timedelta(days=8)).isoformat()
        recent_time = (now - timedelta(days=1)).isoformat()

        self.messenger.image_records = {
            'old_image_id': old_time,
            'recent_image_id': recent_time,
            'last_checked': now.isoformat(),
        }

        self.messenger.delete_old_images()

        # Verify that the old image was deleted
        mock_delete_image_from_cloudinary.assert_called_once_with(
            'old_image_id',
        )

        # Verify that the old image was removed from records
        self.assertNotIn('old_image_id', self.messenger.image_records)

        # Verify that the recent image was not deleted
        self.assertIn('recent_image_id', self.messenger.image_records)

        # Verify that the records were saved
        mock_save_image_records.assert_called_once()

    @patch(
        'src.notifiers.line_notifier_message_api.LineMessenger.'
        'delete_old_images',
    )
    @patch(
        'src.notifiers.line_notifier_message_api.LineMessenger.'
        'save_image_records',
    )
    @patch(
        'src.notifiers.line_notifier_message_api.datetime', wraps=datetime,
    )
    def test_delete_old_images_with_interval(
        self,
        mock_datetime: unittest.mock.MagicMock,
        mock_save_image_records: unittest.mock.MagicMock,
        mock_delete_old_images: unittest.mock.MagicMock,
    ):
        """
        Test the delete_old_images_with_interval method.
        """
        now = datetime(2023, 10, 1)
        mock_datetime.now.return_value = now

        # Set last_checked to 2 days ago
        self.messenger.image_records['last_checked'] = (
            now - timedelta(days=2)
        ).isoformat()

        self.messenger.delete_old_images_with_interval()

        # Verify that the check time was updated
        self.assertEqual(
            self.messenger.image_records['last_checked'], now.isoformat(),
        )

        # Verify that delete_old_images was called
        mock_delete_old_images.assert_called_once()

        # Verify that save_image_records was called
        mock_save_image_records.assert_called_once()

    @patch(
        'src.notifiers.line_notifier_message_api.'
        'LineMessenger.push_message',
    )
    @patch(
        'src.notifiers.line_notifier_message_api.LineMessenger.'
        'upload_image_to_cloudinary',
    )
    @patch(
        'src.notifiers.line_notifier_message_api.LineMessenger.__init__',
        return_value=None,
    )
    def test_main(
        self,
        mock_init: unittest.mock.MagicMock,
        mock_upload_image_to_cloudinary: unittest.mock.MagicMock,
        mock_push_message: unittest.mock.MagicMock,
    ):
        """
        Test the main function.
        """
        mock_push_message.return_value = 200
        mock_upload_image_to_cloudinary.return_value = (
            'http://mock_image_url', 'mock_public_id',
        )

        # Create a black image for testing
        height, width = 480, 640
        frame_with_detections = np.zeros((height, width, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.png', frame_with_detections)
        frame_bytes = buffer.tobytes()

        with patch('builtins.print') as mock_print:
            main()
            mock_print.assert_any_call('Response code: 200')

        # Verify that the LineMessenger
        # was initialised with the correct arguments
        mock_init.assert_called_once_with(
            channel_access_token='YOUR_LINE_CHANNEL',
            recipient_id='YOUR_RECIPIENT_ID',
        )

        # Verify that the push_message method
        # was called with the correct arguments
        mock_push_message.assert_called_once_with(
            'Hello, LINE Messaging API!', image_bytes=frame_bytes,
        )

    @patch('builtins.open', new_callable=mock_open)
    def test_load_image_records_failure(self, mock_file: MagicMock) -> None:
        """
        Test failure when loading image records from JSON file.
        """
        mock_file.side_effect = Exception('Mocked exception')

        with patch('builtins.print') as mock_print:
            records = self.messenger.load_image_records()
            mock_print.assert_called_once_with(
                'Failed to load image records: Mocked exception',
            )
            self.assertEqual(records, {})

    @patch('builtins.open', new_callable=mock_open)
    def test_save_image_records_failure(self, mock_file):
        """
        Test failure when saving image records to JSON file.
        """
        mock_file.side_effect = Exception('Mocked exception')

        with patch('builtins.print') as mock_print:
            self.messenger.save_image_records()
            mock_print.assert_called_once_with(
                'Failed to save image records: Mocked exception',
            )

    @patch(
        'src.notifiers.line_notifier_message_api.LineMessenger.'
        'upload_image_to_cloudinary',
    )
    def test_push_message_image_upload_failure(
        self,
        mock_upload_image_to_cloudinary: unittest.mock.MagicMock,
    ):
        """
        Test push_message method when image upload to Cloudinary fails.
        """
        mock_upload_image_to_cloudinary.return_value = ('', '')

        with self.assertRaises(ValueError) as context:
            self.messenger.push_message(
                recipient_id=self.recipient_id,
                message=self.message,
                image_bytes=self.image_bytes,
            )
        self.assertEqual(
            str(context.exception),
            'Failed to upload image to Cloudinary',
        )

    @patch('src.notifiers.line_notifier_message_api.requests.post')
    def test_push_message_api_error(self, mock_post):
        """
        Test push_message method when LINE Messaging API returns an error.
        """
        mock_post.return_value.status_code = 400
        mock_post.return_value.text = 'Bad Request'

        with patch('builtins.print') as mock_print:
            response_code = self.messenger.push_message(
                recipient_id=self.recipient_id,
                message=self.message,
            )
            mock_print.assert_called_once_with('Error: 400, Bad Request')
            self.assertEqual(response_code, 400)


if __name__ == '__main__':
    unittest.main()
