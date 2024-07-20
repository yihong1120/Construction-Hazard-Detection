from __future__ import annotations

import unittest
from io import BytesIO
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.wechat_notifier import WeChatNotifier


class TestWeChatNotifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.wechat_notifier = WeChatNotifier(
            corp_id='test_corp_id',
            corp_secret='test_corp_secret',
            agent_id=1000002,
        )

    @patch('requests.get')
    def test_get_access_token(self, mock_get):
        """Test the get_access_token function."""
        mock_response = MagicMock()
        mock_response.json.return_value = {'access_token': 'test_access_token'}
        mock_get.return_value = mock_response
        access_token = self.wechat_notifier.get_access_token()
        self.assertEqual(access_token, 'test_access_token')
        mock_get.assert_called_once_with(
            'https://qyapi.weixin.qq.com/cgi-bin/gettoken?'
            'corpid=test_corp_id&corpsecret=test_corp_secret',
        )

    @patch('requests.post')
    def test_send_notification_no_image(self, mock_post):
        """Test sending a notification without an image."""
        mock_response = MagicMock()
        mock_response.json.return_value = {'errcode': 0, 'errmsg': 'ok'}
        mock_post.return_value = mock_response
        user_id = 'test_user_id'
        message = 'Hello, WeChat!'
        response = self.wechat_notifier.send_notification(user_id, message)
        self.assertEqual(response, {'errcode': 0, 'errmsg': 'ok'})
        url = (
            f"https://qyapi.weixin.qq.com/cgi-bin/message/send?"
            f"access_token={self.wechat_notifier.access_token}"
        )
        payload = {
            'touser': user_id,
            'msgtype': 'text',
            'agentid': self.wechat_notifier.agent_id,
            'text': {
                'content': message,
            },
            'safe': 0,
        }

        mock_post.assert_called_once_with(url, json=payload)

    @patch('requests.post')
    @patch.object(WeChatNotifier, 'upload_media')
    def test_send_notification_with_image(self, mock_upload_media, mock_post):
        """Test sending a notification with an image."""
        mock_post.return_value.json.return_value = {
            'errcode': 0, 'errmsg': 'ok',
        }
        mock_upload_media.return_value = 'test_media_id'
        user_id = 'test_user_id'
        message = 'Hello, WeChat!'
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        response = self.wechat_notifier.send_notification(
            user_id, message, image=image,
        )
        self.assertEqual(response, {'errcode': 0, 'errmsg': 'ok'})
        url = (
            f"https://qyapi.weixin.qq.com/cgi-bin/message/send?"
            f"access_token={self.wechat_notifier.access_token}"
        )
        payload = {
            'touser': user_id,
            'msgtype': 'image',
            'agentid': self.wechat_notifier.agent_id,
            'image': {
                'media_id': 'test_media_id',
            },
            'safe': 0,
        }

        mock_post.assert_called_once_with(url, json=payload)

    @patch('requests.post')
    def test_upload_media(self, mock_post):
        """Test the upload_media function."""
        mock_response = MagicMock()
        mock_response.json.return_value = {'media_id': 'test_media_id'}
        mock_post.return_value = mock_response
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        buffer = BytesIO()
        image_pil = Image.fromarray(image)
        image_pil.save(buffer, format='PNG')
        buffer.seek(0)
        media_id = self.wechat_notifier.upload_media(image)
        self.assertEqual(media_id, 'test_media_id')

        args, kwargs = mock_post.call_args
        self.assertEqual(kwargs['files']['media'][0], 'image.png')
        self.assertIsInstance(kwargs['files']['media'][1], BytesIO)
        self.assertEqual(kwargs['files']['media'][2], 'image/png')


if __name__ == '__main__':
    unittest.main()
