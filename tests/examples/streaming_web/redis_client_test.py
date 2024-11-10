from __future__ import annotations
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import redis.asyncio as redis
import examples.streaming_web.redis_client as redis_client


class TestRedisClient(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up the test environment before each test.
        """
        self.mock_redis = MagicMock(spec=redis.StrictRedis)
        self.redis_patcher = patch('examples.streaming_web.redis_client.r', self.mock_redis)
        self.redis_patcher.start()

    def tearDown(self) -> None:
        """
        Clean up after each test.
        """
        self.redis_patcher.stop()

    @patch.dict('os.environ', {
        'redis_host': 'test_host',
        'redis_port': '1234',
        'redis_password': 'test_password'
    })
    @patch('redis.asyncio.StrictRedis')
    def test_redis_connection_config(self, mock_redis_class):
        """
        Test Redis connection configuration.
        """
        # Re-import the redis_client module to apply the patched environment variables
        patch('examples.streaming_web.redis_client.load_dotenv', return_value=True).start()
        import importlib
        importlib.reload(redis_client)

        # Verify Redis connection parameters
        mock_redis_class.assert_called_with(
            host='test_host',
            port=1234,
            password='test_password',
            decode_responses=False
        )

    def test_redis_get(self):
        """
        Test Redis get method.
        """
        # Set up mock return value
        self.mock_redis.get.return_value = b'test_value'

        # Call the actual method
        value = redis_client.r.get('test_key')

        # Verify that the Redis get method was called
        self.mock_redis.get.assert_called_once_with('test_key')
        self.assertEqual(value, b'test_value')

    def test_redis_set(self):
        """
        Test Redis set method.
        """
        # Set up mock return value
        self.mock_redis.set.return_value = True

        # Call the actual method
        result = redis_client.r.set('test_key', 'test_value')

        # Verify that the Redis set method was called
        self.mock_redis.set.assert_called_once_with('test_key', 'test_value')
        self.assertTrue(result)

    def test_redis_delete(self):
        """
        Test Redis delete method.
        """
        # Set up mock return value
        self.mock_redis.delete.return_value = 1

        # Call the actual method
        result = redis_client.r.delete('test_key')

        # Verify that the Redis delete method was called
        self.mock_redis.delete.assert_called_once_with('test_key')
        self.assertEqual(result, 1)

if __name__ == '__main__':
    unittest.main()
