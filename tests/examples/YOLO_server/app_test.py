from __future__ import annotations

import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from fastapi_jwt import JwtAccessBearer
from examples.YOLO_server.app import app, sio, lifespan
from examples.YOLO_server.config import Settings
import socketio
import redis.asyncio as redis
from fastapi_limiter import FastAPILimiter
import asyncio

class TestApp(unittest.IsolatedAsyncioTestCase):

    @classmethod
    def setUpClass(cls):
        # 設置測試客戶端
        cls.client = TestClient(app)

    @patch("examples.YOLO_server.app.redis.from_url", new_callable=AsyncMock)
    @patch("examples.YOLO_server.app.scheduler.shutdown", new_callable=MagicMock)
    async def test_redis_initialization(self, mock_scheduler_shutdown, mock_redis_from_url):
        # 模擬 Redis 連接初始化
        mock_redis_client = AsyncMock()
        mock_redis_from_url.return_value = mock_redis_client

        async with lifespan(app):
            mock_redis_from_url.assert_called_once_with(
                "redis://localhost:6379",
                encoding="utf-8",
                decode_responses=True,
                password='_sua6oub4Ss'
            )
            mock_redis_client.close.assert_not_called()  # Redis 初始化期間不應關閉連接
            mock_scheduler_shutdown.assert_not_called()  # 確保 scheduler shutdown 被跳過

    @patch("examples.YOLO_server.app.FastAPILimiter.init")
    @patch("examples.YOLO_server.app.redis.from_url", new_callable=AsyncMock)
    async def test_lifespan_context(self, mock_redis_from_url, mock_limiter_init):
        # 測試 lifespan 事件中的資源初始化和釋放
        async with lifespan(app):
            mock_redis_from_url.assert_called_once()
            mock_limiter_init.assert_called_once()

    def test_routes_exist(self):
        # 測試應用程序的主要路由是否已正確註冊
        response = self.client.get("/auth/some_endpoint")
        self.assertIn(response.status_code, [200, 404])  # 路由應返回有效響應或 404
        response = self.client.get("/detect/some_endpoint")
        self.assertIn(response.status_code, [200, 404])
        response = self.client.get("/models/some_endpoint")
        self.assertIn(response.status_code, [200, 404])

    @patch("examples.YOLO_server.app.JwtAccessBearer.__init__", return_value=None)
    def test_jwt_initialization(self, mock_jwt_access_bearer_init):
        # 測試 JWT 初始化
        mock_secret_key = Settings().authjwt_secret_key
        JwtAccessBearer(secret_key=mock_secret_key)
        mock_jwt_access_bearer_init.assert_called_once_with(secret_key=mock_secret_key)

    @patch("socketio.AsyncClient.connect", new_callable=AsyncMock)
    @patch("socketio.AsyncClient.disconnect", new_callable=AsyncMock)
    def test_socketio_connect_disconnect(self, mock_disconnect, mock_connect):
        # 使用 Socket.IO 測試客戶端連接和斷開
        client = socketio.AsyncClient()

        @client.event
        async def connect():
            print('Connected to server')

        @client.event
        async def disconnect():
            print('Disconnected from server')

        async def socket_test():
            await client.connect('http://0.0.0.0:5000')
            mock_connect.assert_called_once()
            self.assertTrue(mock_connect.called, "Socket should be connected")
            await client.disconnect()
            mock_disconnect.assert_called_once()
            self.assertTrue(mock_disconnect.called, "Socket should be disconnected")

        # Run socket test in the loop
        asyncio.run(socket_test())

if __name__ == "__main__":
    unittest.main()
