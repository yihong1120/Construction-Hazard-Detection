from __future__ import annotations

import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

from flask import Flask
from flask_limiter.util import get_remote_address
from flask_socketio import SocketIO


class TestStreamingWebApp(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up the Flask test client and other necessary mocks.
        """
        self.app = Flask(__name__)
        self.app.testing = True
        self.client = self.app.test_client()

    @patch('examples.streaming_web.app.redis.StrictRedis')
    def test_redis_connection(self, mock_redis: MagicMock) -> None:
        """
        Test that the Redis connection is properly established.
        """
        mock_redis.return_value = MagicMock()
        r = mock_redis(
            host='localhost', port=6379,
            password='passcode', decode_responses=False,
        )
        self.assertIsInstance(r, MagicMock)
        mock_redis.assert_called_once_with(
            host='localhost', port=6379,
            password='passcode', decode_responses=False,
        )

    @patch('flask_cors.CORS')
    def test_cors_initialization(self, mock_cors: MagicMock) -> None:
        """
        Test that CORS is properly initialized for the Flask app.
        """
        cors = mock_cors(self.app, resources={r'/*': {'origins': '*'}})
        self.assertIsInstance(cors, MagicMock)
        mock_cors.assert_called_once_with(
            self.app, resources={r'/*': {'origins': '*'}},
        )

    @patch('examples.streaming_web.app.Limiter')
    def test_rate_limiter_initialization(
        self, mock_limiter: MagicMock,
    ) -> None:
        """
        Test that the rate limiter is properly initialized.
        """
        limiter = mock_limiter(key_func=get_remote_address)
        self.assertIsInstance(limiter, MagicMock)
        mock_limiter.assert_called_once_with(key_func=get_remote_address)

    @patch('examples.streaming_web.app.SocketIO.run')
    def test_app_running_configuration(self, mock_run: MagicMock) -> None:
        """
        Test that the application runs with the expected configurations.
        """
        socketio = SocketIO(self.app)
        socketio.run(self.app, host='127.0.0.1', port=8000, debug=False)
        mock_run.assert_called_once_with(
            self.app, host='127.0.0.1', port=8000, debug=False,
        )

    def tearDown(self) -> None:
        """
        Clean up after each test.
        """
        self.client = None


if __name__ == '__main__':
    unittest.main()
