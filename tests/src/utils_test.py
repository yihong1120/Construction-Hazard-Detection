from __future__ import annotations

import asyncio
import base64
import unittest
from datetime import datetime
from datetime import timedelta
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from watchdog.events import FileModifiedEvent

from src.utils import FileEventHandler
from src.utils import RedisManager
from src.utils import Utils


class TestUtils(unittest.IsolatedAsyncioTestCase):
    def test_is_expired_with_valid_date(self):
        # Test with a past date (should return True)
        past_date = (datetime.now() - timedelta(days=1)).isoformat()
        self.assertTrue(Utils.is_expired(past_date))

        # Test with a future date (should return False)
        future_date = (datetime.now() + timedelta(days=1)).isoformat()
        self.assertFalse(Utils.is_expired(future_date))

    def test_is_expired_with_invalid_date(self):
        # Test with an invalid ISO 8601 date (should return False)
        invalid_date = '2024-13-01T00:00:00'
        self.assertFalse(Utils.is_expired(invalid_date))

    def test_is_expired_with_none(self):
        # Test with None (should return False)
        self.assertFalse(Utils.is_expired(None))

    def test_encode(self):
        # Test encoding a string
        value = 'test_value'
        encoded_value = Utils.encode(value)
        expected_value = base64.urlsafe_b64encode(
            value.encode('utf-8'),
        ).decode('utf-8')
        self.assertEqual(encoded_value, expected_value)


class TestFileEventHandler(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    async def test_on_modified_triggers_callback(self):
        # Create a mock callback function
        mock_callback = AsyncMock()

        # File path to watch
        file_path = '/path/to/test/file.txt'

        # Create an instance of FileEventHandler
        event_handler = FileEventHandler(
            file_path=file_path,
            callback=mock_callback,
            loop=asyncio.get_running_loop(),
        )

        # Create a mock event for a file modification
        event = FileModifiedEvent(file_path)

        # Trigger the on_modified event
        event_handler.on_modified(event)

        # Assert that the callback was called
        mock_callback.assert_called_once()

    async def test_on_modified_different_file(self):
        # Create a mock callback function
        mock_callback = AsyncMock()

        # File path to watch
        file_path = '/path/to/test/file.txt'

        # Create an instance of FileEventHandler
        event_handler = FileEventHandler(
            file_path=file_path, callback=mock_callback, loop=self.loop,
        )

        # Create a mock event for a different file modification
        different_file_path = '/path/to/other/file.txt'
        event = FileModifiedEvent(different_file_path)

        # Trigger the on_modified event
        event_handler.on_modified(event)

        # Assert that the callback was not called
        mock_callback.assert_not_called()


class TestRedisManager(unittest.IsolatedAsyncioTestCase):
    """
    Test cases for the RedisManager class
    """

    @patch('src.utils.redis.Redis')
    def setUp(self, mock_redis):
        """
        Set up a RedisManager instance with a mocked Redis connection
        """
        # Mock Redis instance
        self.mock_redis_instance = MagicMock()
        self.mock_redis_instance.get = AsyncMock()
        self.mock_redis_instance.set = AsyncMock()
        self.mock_redis_instance.delete = AsyncMock()
        self.mock_redis_instance.xadd = AsyncMock()
        self.mock_redis_instance.xread = AsyncMock()
        self.mock_redis_instance.close = AsyncMock()
        self.mock_redis_instance.delete.side_effect = AsyncMock()

        mock_redis.return_value = self.mock_redis_instance

        # Initialize RedisManager
        self.redis_manager = RedisManager()

    async def test_set_success(self):
        """
        Test successful set operation
        """
        key = 'test_key'
        value = b'test_value'

        # Call the set method
        await self.redis_manager.set(key, value)

        # Assert that the Redis set method was called with correct parameters
        self.mock_redis_instance.set.assert_called_once_with(key, value)

    async def test_set_error(self):
        """
        Simulate an exception during the Redis set operation
        """
        key = 'test_key'
        value = b'test_value'
        self.mock_redis_instance.set.side_effect = Exception('Redis error')

        # Call the set method and verify it handles the exception
        with self.assertLogs(level='ERROR'):
            await self.redis_manager.set(key, value)

    async def test_get_success(self):
        """
        Mock the Redis get method to return a value
        """
        key = 'test_key'
        expected_value = b'test_value'
        self.mock_redis_instance.get.return_value = expected_value

        # Call the get method
        value = await self.redis_manager.get(key)

        # Assert that the Redis get method was called with correct parameters
        self.mock_redis_instance.get.assert_called_once_with(key)

        # Assert the value returned is correct
        self.assertEqual(value, expected_value)

    async def test_get_error(self):
        """
        Simulate an exception during the Redis get operation
        """
        key = 'test_key'
        self.mock_redis_instance.get.side_effect = Exception('Redis error')

        # Call the get method and verify it handles the exception
        with self.assertLogs(level='ERROR'):
            value = await self.redis_manager.get(key)
            self.assertIsNone(value)

    async def test_delete_success(self):
        """
        Test successful delete operation
        """
        key = 'test_key'

        # Call the delete method
        await self.redis_manager.delete(key)

        # Assert that the Redis delete method
        # was called with correct parameters
        self.mock_redis_instance.delete.assert_called_once_with(key)

    async def test_delete_error(self):
        """
        Simulate an exception during the Redis delete operation
        """
        key = 'test_key'
        self.mock_redis_instance.delete.side_effect = Exception('Redis error')

        # Call the delete method and verify it handles the exception
        with self.assertLogs(level='ERROR'):
            await self.redis_manager.delete(key)

    async def test_add_to_stream(self):
        """
        Test adding data to a Redis stream
        """
        stream_name = 'test_stream'
        data = {'field1': b'value1', 'field2': b'value2'}
        maxlen = 5

        # Call the add_to_stream method
        await self.redis_manager.add_to_stream(stream_name, data, maxlen)

        # Assert that the xadd method was called with the correct parameters
        self.mock_redis_instance.xadd.assert_called_once_with(
            stream_name, data, maxlen=maxlen,
        )

    async def test_add_to_stream_error(self):
        """
        Simulate an exception during adding data to a Redis stream
        """
        stream_name = 'test_stream'
        data = {'field1': b'value1', 'field2': b'value2'}
        maxlen = 5
        self.mock_redis_instance.xadd.side_effect = Exception('Redis error')

        # Call the add_to_stream method and verify it handles the exception
        with self.assertLogs(level='ERROR') as log:
            await self.redis_manager.add_to_stream(stream_name, data, maxlen)
            self.assertIn(
                f"Error adding to Redis stream {stream_name}: Redis error",
                log.output[0],
            )

    async def test_read_from_stream(self):
        """
        Test reading data from a Redis stream
        """
        stream_name = 'test_stream'
        last_id = '0'
        expected_messages = [
            ('1-0', {'field1': b'value1', 'field2': b'value2'}),
            ('2-0', {'field3': b'value3'}),
        ]
        self.mock_redis_instance.xread.return_value = expected_messages

        # Call the read_from_stream method
        messages = await self.redis_manager.read_from_stream(
            stream_name,
            last_id,
        )

        # Assert that the xread method was called with correct parameters
        self.mock_redis_instance.xread.assert_called_once_with(
            {stream_name: last_id},
        )

        # Assert that the returned messages match the expected messages
        self.assertEqual(messages, expected_messages)

    async def test_read_from_stream_error(self):
        """
        Simulate an exception during reading from a Redis stream
        """
        stream_name = 'test_stream'
        last_id = '0'
        self.mock_redis_instance.xread.side_effect = Exception('Redis error')

        # Call the read_from_stream method and verify it handles the exception
        with self.assertLogs(level='ERROR') as log:
            messages = await self.redis_manager.read_from_stream(
                stream_name, last_id,
            )
            self.assertEqual(messages, [])
            self.assertIn(
                f"Error reading from Redis stream {stream_name}: Redis error",
                log.output[0],
            )

    async def test_delete_stream_success(self):
        """
        Test successful deletion of a Redis stream
        """
        stream_name = 'test_stream'

        # Call the delete_stream method
        await self.redis_manager.delete_stream(stream_name)

        # Assert that the Redis delete method was called
        self.mock_redis_instance.delete.assert_called_once_with(stream_name)

    async def test_delete_stream_error(self):
        """
        Simulate an exception during deleting a Redis stream
        """
        stream_name = 'test_stream'
        self.mock_redis_instance.delete.side_effect = Exception('Redis error')

        # Call the delete_stream method and verify it handles the exception
        with self.assertLogs(level='ERROR') as log:
            await self.redis_manager.delete_stream(stream_name)
            self.assertIn(
                f"Error deleting Redis stream {stream_name}: Redis error",
                log.output[0],
            )

    async def test_close_connection(self):
        """
        Test closing the Redis connection
        """
        # Call the close_connection method
        await self.redis_manager.close_connection()

        # Assert that the Redis close method was called
        self.mock_redis_instance.close.assert_called_once()

    async def test_close_connection_error(self):
        """
        Simulate an exception during closing the Redis connection
        """
        self.mock_redis_instance.close.side_effect = Exception('Redis error')

        # Call the close_connection method and verify it handles the exception
        with self.assertLogs(level='ERROR') as log:
            await self.redis_manager.close_connection()
            self.assertIn(
                '[ERROR] Failed to close Redis connection: Redis error',
                log.output[0],
            )


if __name__ == '__main__':
    unittest.main()
