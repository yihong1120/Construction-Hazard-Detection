from __future__ import annotations

import io
import logging
import os
import unittest
from datetime import datetime
from datetime import timedelta
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.notifiers.line_notifier_message_api import LineMessenger
from src.notifiers.line_notifier_message_api import main


class TestLineMessenger(unittest.IsolatedAsyncioTestCase):
    """
    Test cases for the LineMessenger class.
    """

    def setUp(self) -> None:
        """
        Set up test fixtures for each test method.
        """
        logging.basicConfig(level=logging.ERROR)
        self.channel_access_token: str = 'test_channel_access_token'
        self.messenger: LineMessenger = LineMessenger(
            channel_access_token=self.channel_access_token,
        )
        self.message: str = 'Test message'
        self.recipient_id: str = 'test_recipient_id'
        self.image_bytes: bytes = b'test_image_bytes'
        self.headers: dict[str, str] = {
            'Content-Type': 'application/json',
            'Authorization': f"Bearer {self.channel_access_token}",
        }

    @patch.dict(os.environ, {'LINE_CHANNEL_ACCESS_TOKEN': ''})
    def test_init_without_channel_access_token(self) -> None:
        """
        Test initialisation without a channel access token raises ValueError.
        """
        with self.assertRaises(ValueError) as ctx:
            LineMessenger()
        self.assertEqual(
            str(ctx.exception),
            (
                'LINE_CHANNEL_ACCESS_TOKEN not provided or in environment '
                'variables.'
            ),
        )

    @patch.dict(
        os.environ, {'LINE_CHANNEL_ACCESS_TOKEN': 'test_channel_access_token'},
    )
    @patch('aiohttp.ClientSession.post')
    @patch(
        'src.notifiers.line_notifier_message_api.LineMessenger.'
        'upload_image_to_cloudinary',
    )
    async def test_push_message_with_image(
        self,
        mock_upload: MagicMock,
        mock_post: MagicMock,
    ) -> None:
        """
        Test push_message with an image attached.
        """
        self.messenger.channel_access_token = 'test_channel_access_token'
        mock_upload.return_value = ('http://mock_image_url', 'mock_public_id')
        mock_response = unittest.mock.AsyncMock()
        mock_response.status = 200
        mock_response.text = unittest.mock.AsyncMock(return_value='OK')
        mock_post.return_value.__aenter__.return_value = mock_response

        code = await self.messenger.push_message(
            recipient_id=self.recipient_id,
            message=self.message,
            image_bytes=self.image_bytes,
        )

        mock_upload.assert_called_once_with(self.image_bytes)

        expected_json = {
            'to': self.recipient_id,
            'messages': [
                {'type': 'text', 'text': self.message},
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
            json=expected_json,
        )
        self.assertEqual(code, 200)

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
        mock_save: MagicMock,
        mock_delete: MagicMock,
    ) -> None:
        """
        Test delete_old_images_with_interval when last_checked is None.
        """
        # Use empty string instead of None
        self.messenger.image_records['last_checked'] = ''
        self.messenger.delete_old_images_with_interval()
        mock_delete.assert_called_once()
        mock_save.assert_called_once()
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
        mock_save: MagicMock,
        mock_delete: MagicMock,
    ) -> None:
        """
        Test delete_old_images_with_interval with invalid last_checked value.
        """
        self.messenger.image_records['last_checked'] = 'invalid_datetime'
        self.messenger.delete_old_images_with_interval()
        mock_delete.assert_called_once()
        mock_save.assert_called_once()
        self.assertIn('last_checked', self.messenger.image_records)

    @patch.dict(
        os.environ, {'LINE_CHANNEL_ACCESS_TOKEN': 'test_channel_access_token'},
    )
    @patch('aiohttp.ClientSession.post')
    async def test_push_message_without_image(
        self,
        mock_post: MagicMock,
    ) -> None:
        """
        Test push_message without an image.
        """
        self.messenger.channel_access_token = 'test_channel_access_token'
        mock_response = unittest.mock.AsyncMock()
        mock_response.status = 200
        mock_response.text = unittest.mock.AsyncMock(return_value='OK')
        mock_post.return_value.__aenter__.return_value = mock_response
        code = await self.messenger.push_message(
            recipient_id=self.recipient_id,
            message=self.message,
        )
        expected_json = {
            'to': self.recipient_id,
            'messages': [{'type': 'text', 'text': self.message}],
        }
        mock_post.assert_called_once_with(
            'https://api.line.me/v2/bot/message/push',
            headers=self.headers,
            json=expected_json,
        )
        self.assertEqual(code, 200)

    @patch('cloudinary.uploader.upload')
    async def test_upload_image_success(self, mock_upload: MagicMock) -> None:
        """
        Test successful image upload to Cloudinary.
        """
        mock_upload.return_value = {
            'secure_url': 'http://mock_image_url',
            'public_id': 'mock_public_id',
        }
        url, pid = await self.messenger.upload_image_to_cloudinary(
            self.image_bytes,
        )
        self.assertEqual(url, 'http://mock_image_url')
        self.assertEqual(pid, 'mock_public_id')

    @patch('cloudinary.uploader.upload')
    async def test_upload_image_failure(self, mock_upload: MagicMock) -> None:
        """
        Test image upload failure to Cloudinary.
        """
        mock_upload.side_effect = Exception('upload error')
        url, pid = await self.messenger.upload_image_to_cloudinary(
            self.image_bytes,
        )
        self.assertEqual(url, '')
        self.assertEqual(pid, '')

    @patch('cloudinary.uploader.destroy')
    def test_delete_cloudinary_success(self, mock_destroy: MagicMock) -> None:
        """
        Test successful deletion of image from Cloudinary.
        """
        mock_destroy.return_value = {'result': 'ok'}
        self.messenger.delete_image_from_cloudinary('pid')
        mock_destroy.assert_called_once_with('pid')

    @patch('cloudinary.uploader.destroy')
    def test_delete_cloudinary_failure(self, mock_destroy: MagicMock) -> None:
        """
        Test failed deletion of image from Cloudinary.
        """
        mock_destroy.return_value = {'result': 'error'}
        self.messenger.delete_image_from_cloudinary('pid')
        mock_destroy.assert_called_once_with('pid')

    @patch('cloudinary.uploader.destroy')
    def test_delete_cloudinary_exception(
        self,
        mock_destroy: MagicMock,
    ) -> None:
        """
        Test exception during deletion of image from Cloudinary.
        """
        mock_destroy.side_effect = Exception('boom')
        with patch('builtins.print') as mock_print:
            self.messenger.delete_image_from_cloudinary('pid')
            mock_print.assert_called_once()

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
        mock_save: MagicMock,
        mock_del: MagicMock,
    ) -> None:
        """
        Test deletion of old images from records.
        """
        now = datetime.now()
        self.messenger.image_records = {
            'old_id': (now - timedelta(days=8)).isoformat(),
            'recent_id': (now - timedelta(days=1)).isoformat(),
            'last_checked': now.isoformat(),
        }
        self.messenger.delete_old_images()
        mock_del.assert_called_once_with('old_id')
        self.assertNotIn('old_id', self.messenger.image_records)
        self.assertIn('recent_id', self.messenger.image_records)
        mock_save.assert_called_once()

    @patch(
        'src.notifiers.line_notifier_message_api.LineMessenger.'
        'delete_old_images',
    )
    @patch(
        'src.notifiers.line_notifier_message_api.LineMessenger.'
        'save_image_records',
    )
    @patch(
        'src.notifiers.line_notifier_message_api.datetime',
        wraps=datetime,
    )
    def test_delete_old_images_interval(
        self,
        mock_datetime: MagicMock,
        mock_save: MagicMock,
        mock_delete: MagicMock,
    ) -> None:
        """
        Test delete_old_images_with_interval with timing logic.
        """
        now = datetime(2023, 10, 1)
        mock_datetime.now.return_value = now
        self.messenger.image_records['last_checked'] = (
            now - timedelta(days=2)
        ).isoformat()
        self.messenger.delete_old_images_with_interval()
        self.assertEqual(
            self.messenger.image_records['last_checked'],
            now.isoformat(),
        )
        mock_delete.assert_called_once()
        mock_save.assert_called_once()

    @patch(
        'src.notifiers.line_notifier_message_api.LineMessenger.'
        'push_message',
        new_callable=AsyncMock,
    )
    @patch(
        'src.notifiers.line_notifier_message_api.LineMessenger.'
        'upload_image_to_cloudinary',
        new_callable=AsyncMock,
    )
    @patch(
        'src.notifiers.line_notifier_message_api.LineMessenger.__init__',
        return_value=None,
    )
    async def test_main(
        self,
        mock_init: MagicMock,
        mock_upload: AsyncMock,
        mock_push: AsyncMock,
    ) -> None:
        """
        Smoke test for main() function.
        """
        mock_push.return_value = 200
        mock_upload.return_value = ('http://mock_image_url', 'mock_public_id')

        # Build sample PNG bytes (Pillow)
        h, w = 480, 640
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        frame_bytes = buf.getvalue()

        with patch('builtins.print') as mock_print:
            await main()
            mock_print.assert_any_call('Response code: 200')

        mock_init.assert_called_once()
        # Check push_message was awaited once,
        # and check message and image_bytes
        self.assertTrue(mock_push.await_count == 1)
        # Use call_args for robust and mypy-safe argument checking
        args, kwargs = mock_push.call_args
        self.assertEqual(kwargs['message'], 'Hello, LINE Messaging API!')
        self.assertEqual(kwargs['image_bytes'], frame_bytes)

    @patch('requests.post')
    async def test_push_message_api_error(self, mock_post: MagicMock) -> None:
        """
        Test push_message API error path (should print error and return code).
        """
        mock_post.return_value.status_code = 401
        mock_post.return_value.text = (
            '{"message":"Authentication failed. Confirm that '
            'the access token the authorization header is valid."}'
        )
        with patch('builtins.print') as mock_print:
            code = await self.messenger.push_message(
                recipient_id=self.recipient_id,
                message=self.message,
            )
            mock_print.assert_any_call(
                'Error: 401, '
                '{"message":"Authentication failed. Confirm that '
                'the access token in the authorization header is valid."}',
            )
            self.assertEqual(code, 401)

    @patch(
        'src.notifiers.line_notifier_message_api.LineMessenger.'
        'upload_image_to_cloudinary',
        new_callable=AsyncMock,
    )
    async def test_push_message_upload_fail(
        self,
        mock_upload: AsyncMock,
    ) -> None:
        """
        Test push_message when image upload fails (should raise ValueError).
        """
        mock_upload.return_value = ('', '')
        with self.assertRaises(ValueError):
            await self.messenger.push_message(
                recipient_id=self.recipient_id,
                message=self.message,
                image_bytes=self.image_bytes,
            )

    def test_load_image_records_exception(self) -> None:
        """
        Test load_image_records exception handling.
        """
        open_orig = open

        def open_side_effect(path, *args, **kwargs):
            if path == 'dummy.json':
                raise Exception('load error')
            return open_orig(path, *args, **kwargs)
        with patch('os.path.exists', return_value=True), \
                patch('builtins.open', side_effect=open_side_effect):
            messenger = LineMessenger(
                channel_access_token='x', image_record_file='dummy.json',
            )
            records = messenger.load_image_records()
            self.assertEqual(records, {})

    def test_save_image_records_exception(self) -> None:
        """
        Test save_image_records exception handling (should print error).
        """
        open_orig = open

        def open_side_effect(path, *args, **kwargs):
            if path == 'dummy.json':
                raise Exception('save error')
            return open_orig(path, *args, **kwargs)
        with patch('builtins.open', side_effect=open_side_effect):
            messenger = LineMessenger(
                channel_access_token='x', image_record_file='dummy.json',
            )
            with patch('builtins.print') as mock_print:
                messenger.save_image_records()
                mock_print.assert_called()

    @patch(
        'cloudinary.uploader.destroy',
        side_effect=Exception('delete error'),
    )
    def test_delete_image_from_cloudinary_exception(
        self,
        mock_destroy: MagicMock,
    ) -> None:
        """
        Test delete_image_from_cloudinary exception handling.
        """
        messenger = LineMessenger(channel_access_token='x')
        with patch('builtins.print') as mock_print:
            messenger.delete_image_from_cloudinary('pid')
            mock_print.assert_called()


if __name__ == '__main__':
    unittest.main()

'''
pytest --cov=src.notifiers.line_notifier_message_api\
    --cov-report=term-missing\
        tests/src/notifiers/line_notifier_message_api_test.py
'''
