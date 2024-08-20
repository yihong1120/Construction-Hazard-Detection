from __future__ import annotations

import unittest
from unittest.mock import patch, MagicMock
from flask import Flask
from flask_socketio import SocketIO
from flask_limiter import Limiter
from flask.testing import FlaskClient

from examples.streaming_web.app import app, socketio, register_routes, register_sockets

class TestApp(unittest.TestCase):
    def setUp(self):
        self.app = app
        self.client: FlaskClient = self.app.test_client()
        self.app.testing = True

    @patch('examples.streaming_web.app.r', new_callable=MagicMock)
    @patch('examples.streaming_web.app.redis.StrictRedis', autospec=True)
    def test_redis_connection(self, mock_redis, mock_r):
        mock_redis_instance = mock_redis.return_value
        mock_r.return_value = mock_redis_instance
        self.assertEqual(mock_r.return_value, mock_redis_instance)

    @patch('examples.streaming_web.app.r', new_callable=MagicMock)
    @patch('examples.streaming_web.app.register_routes')
    @patch('examples.streaming_web.app.register_sockets')
    @patch('examples.streaming_web.app.redis.StrictRedis', autospec=True)
    def test_routes_and_sockets_registration(self, mock_redis, mock_register_sockets, mock_register_routes, mock_r):
        mock_redis_instance = mock_redis.return_value
        limiter = Limiter(key_func=lambda: 'test')

        with self.app.app_context():
            register_routes(self.app, limiter, mock_redis_instance)
            register_sockets(socketio, mock_redis_instance)

        mock_register_routes.assert_called_once_with(self.app, limiter, mock_redis_instance)
        mock_register_sockets.assert_called_once_with(socketio, mock_redis_instance)

    @patch('examples.streaming_web.app.r', new_callable=MagicMock)
    @patch('examples.streaming_web.app.redis.StrictRedis', autospec=True)
    def test_index_route(self, mock_redis, mock_r):
        # Mock Redis 连接
        mock_redis_instance = mock_redis.return_value
        mock_r.return_value = mock_redis_instance

        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Hello, World!', response.data)

    @patch('examples.streaming_web.app.r', new_callable=MagicMock)
    @patch('examples.streaming_web.app.redis.StrictRedis', autospec=True)
    def test_custom_cors_headers(self, mock_redis, mock_r):
        mock_redis_instance = mock_redis.return_value
        mock_r.return_value = mock_redis_instance

        response = self.client.get('/')
        self.assertEqual(response.headers['Access-Control-Allow-Origin'], '*')
        self.assertEqual(response.headers['Access-Control-Allow-Methods'], 'GET, POST, PUT, DELETE, OPTIONS')
        self.assertEqual(response.headers['Access-Control-Allow-Headers'], 'Content-Type, Authorization')

    @patch('examples.streaming_web.app.r', new_callable=MagicMock)
    @patch('examples.streaming_web.app.redis.StrictRedis', autospec=True)
    def test_websocket_connection(self, mock_redis, mock_r):
        mock_redis_instance = mock_redis.return_value

        test_client = socketio.test_client(self.app)
        self.assertTrue(test_client.is_connected())

        test_client.emit('ping')
        received = test_client.get_received()
        self.assertEqual(received[0]['name'], 'message')
        test_client.disconnect()

    @patch('examples.streaming_web.app.socketio.run')
    def test_app_run(self, mock_run):
        from examples.streaming_web.app import __name__ as app_name
        if app_name == '__main__':
            socketio.run(app, host='127.0.0.1', port=8000, debug=False)
            mock_run.assert_called_once_with(app, host='127.0.0.1', port=8000, debug=False)

if __name__ == '__main__':
    unittest.main()
