from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi import FastAPI

from examples.YOLO_server.security import update_secret_key


class TestUpdateSecretKey(unittest.TestCase):

    def setUp(self):
        # 初始化 FastAPI 應用程序
        self.app = FastAPI()

    @patch('examples.YOLO_server.security.secrets.token_urlsafe')
    def test_update_secret_key(self, mock_token_urlsafe):
        # Mock token_urlsafe 的返回值
        mock_token_urlsafe.return_value = 'mocked_secret_key'

        # 調用 update_secret_key 函數
        update_secret_key(self.app)

        # 檢查 jwt_secret_key 是否被更新為 mock 值
        self.assertEqual(self.app.state.jwt_secret_key, 'mocked_secret_key')

        # 驗證 token_urlsafe 已被調用一次，且參數為 16
        mock_token_urlsafe.assert_called_once_with(16)

    def test_update_secret_key_different_keys(self):
        # 生成兩次密鑰並檢查是否每次不同
        update_secret_key(self.app)
        first_key = self.app.state.jwt_secret_key

        update_secret_key(self.app)
        second_key = self.app.state.jwt_secret_key

        # 確保兩個密鑰不同
        self.assertNotEqual(first_key, second_key)
        self.assertIsInstance(second_key, str)
        self.assertGreater(len(second_key), 0)


if __name__ == '__main__':
    unittest.main()
