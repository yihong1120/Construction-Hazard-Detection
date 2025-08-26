from __future__ import annotations

import os
import subprocess
import unittest
from io import BytesIO
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.notifiers.wechat_notifier import main
from src.notifiers.wechat_notifier import WeChatNotifier


class TestWeChatNotifier(unittest.TestCase):
    """
    Unit tests for the WeChatNotifier class methods.
    """

    wechat_notifier: WeChatNotifier

    @classmethod
    def setUpClass(cls) -> None:
        """
        Set up the WeChatNotifier instance for tests.
        """
        cls.wechat_notifier = WeChatNotifier(
            corp_id='test_corp_id',
            corp_secret='test_corp_secret',
            agent_id=1000002,
        )

    @patch('requests.get')
    def test_get_access_token(self, mock_get: MagicMock) -> None:
        """
        Test the get_access_token method.
        """
        mock_response = MagicMock()
        mock_response.json.return_value = {'access_token': 'test_access_token'}
        mock_get.return_value = mock_response
        access_token: str = self.wechat_notifier.get_access_token()
        self.assertEqual(access_token, 'test_access_token')
        mock_get.assert_called_once_with(
            'https://qyapi.weixin.qq.com/cgi-bin/gettoken?'
            'corpid=test_corp_id&corpsecret=test_corp_secret',
        )

    @patch('requests.post')
    def test_send_notification_no_image(self, mock_post: MagicMock) -> None:
        """
        Test sending a notification without an image.
        """
        mock_response = MagicMock()
        mock_response.json.return_value = {'errcode': 0, 'errmsg': 'ok'}
        mock_post.return_value = mock_response
        user_id: str = 'test_user_id'
        message: str = 'Hello, WeChat!'
        response = self.wechat_notifier.send_notification(user_id, message)
        self.assertEqual(response, {'errcode': 0, 'errmsg': 'ok'})
        url: str = (
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
    def test_send_notification_with_image(
        self,
        mock_upload_media: MagicMock,
        mock_post: MagicMock,
    ) -> None:
        """
        Test sending a notification with an image.
        """
        mock_post.return_value.json.return_value = {
            'errcode': 0, 'errmsg': 'ok',
        }
        mock_upload_media.return_value = 'test_media_id'
        user_id: str = 'test_user_id'
        message: str = 'Hello, WeChat!'
        image: np.ndarray = np.zeros((100, 100, 3), dtype=np.uint8)
        response = self.wechat_notifier.send_notification(
            user_id, message, image=image,
        )
        self.assertEqual(response, {'errcode': 0, 'errmsg': 'ok'})
        url: str = (
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
    def test_upload_media(self, mock_post: MagicMock) -> None:
        """
        Test the upload_media method.
        """
        mock_response = MagicMock()
        mock_response.json.return_value = {'media_id': 'test_media_id'}
        mock_post.return_value = mock_response
        image: np.ndarray = np.zeros((100, 100, 3), dtype=np.uint8)
        buffer: BytesIO = BytesIO()
        image_pil: Image.Image = Image.fromarray(image)
        image_pil.save(buffer, format='PNG')
        buffer.seek(0)
        media_id: str = self.wechat_notifier.upload_media(image)
        self.assertEqual(media_id, 'test_media_id')

        args, kwargs = mock_post.call_args
        self.assertEqual(kwargs['files']['media'][0], 'image.png')
        self.assertIsInstance(kwargs['files']['media'][1], BytesIO)
        self.assertEqual(kwargs['files']['media'][2], 'image/png')

    @patch(
        'src.notifiers.wechat_notifier.WeChatNotifier.send_notification',
        return_value={'errcode': 0, 'errmsg': 'ok'},
    )
    @patch(
        'src.notifiers.wechat_notifier.WeChatNotifier.get_access_token',
        return_value='test_access_token',
    )
    @patch('src.notifiers.wechat_notifier.os.getenv')
    def test_main(
        self,
        mock_getenv: MagicMock,
        mock_get_access_token: MagicMock,
        mock_send_notification: MagicMock,
    ) -> None:
        """
        Test the main function.
        """
        mock_getenv.side_effect = lambda key: {
            'WECHAT_CORP_ID': 'test_corp_id',
            'WECHAT_CORP_SECRET': 'test_corp_secret',
            'WECHAT_AGENT_ID': '1000002',
        }.get(key, '')

        # Verify logging.info is called with a static, non-sensitive message
        with patch('src.notifiers.wechat_notifier.logging.info') as mock_log:
            main()
            mock_send_notification.assert_called_once()
            args, kwargs = mock_send_notification.call_args
            self.assertEqual(args[0], 'your_user_id_here')
            self.assertEqual(args[1], 'Hello, WeChat!')
            if len(args) > 2:
                self.assertIsInstance(args[2], np.ndarray)
                self.assertEqual(args[2].shape, (100, 100, 3))
            mock_log.assert_called_once_with(
                'WeChat send_notification completed',
            )

    @patch('requests.post')
    @patch.dict(
        os.environ, {
            'WECHAT_CORP_ID': 'test_corp_id',
            'WECHAT_CORP_SECRET': 'test_corp_secret',
            'WECHAT_AGENT_ID': '1000002',
        },
    )
    def test_main_as_script(self, mock_post: MagicMock) -> None:
        """
        Test running the wechat_notifier.py script as the main program.
        """
        mock_response: MagicMock = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        # Get the absolute path to the wechat_notifier.py script
        script_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                '../../../src/notifiers/wechat_notifier.py',
            ),
        )

        # Run the script using subprocess
        result = subprocess.run(
            ['python', script_path],
            capture_output=True, text=True,
        )

        # Print stdout and stderr for debugging
        print('STDOUT:', result.stdout)
        print('STDERR:', result.stderr)

        # Assert that the script runs without errors
        self.assertEqual(
            result.returncode, 0,
            'Script exited with a non-zero status.',
        )


if __name__ == '__main__':
    unittest.main()

'''
pytest \
    --cov=src.notifiers.wechat_notifier \
    --cov-report=term-missing \
    tests/src/notifiers/wechat_notifier_test.py
'''
