from __future__ import annotations

import unittest
from datetime import datetime
from datetime import timedelta
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from watchdog.events import FileModifiedEvent

from src.utils import FileEventHandler
from src.utils import RedisManager
from src.utils import Utils


class TestUtils(unittest.TestCase):
    def test_is_expired_with_valid_date(self):
        # Test with a past date (should return True)
        past_date = (datetime.now() - timedelta(days=1)).isoformat()
        self.assertTrue(Utils.is_expired(past_date))

        # Test with a future date (should return False)
        future_date = (datetime.now() + timedelta(days=1)).isoformat()
        self.assertFalse(Utils.is_expired(future_date))

    def test_is_expired_with_none(self):
        # Test with None (should return False)
        self.assertFalse(Utils.is_expired(None))


@pytest.mark.asyncio
class TestFileEventHandler(unittest.TestCase):
    async def test_on_modified_triggers_callback(self):
        # Create a mock callback function
        mock_callback = MagicMock()

        # File path to watch
        file_path = '/path/to/test/file.txt'

        # Create an instance of FileEventHandler
        event_handler = FileEventHandler(
            file_path=file_path, callback=mock_callback,
        )

        # Create a mock event for a file modification
        event = FileModifiedEvent(file_path)

        # Trigger the on_modified event
        await event_handler.on_modified(event)

        # Assert that the callback was called
        mock_callback.assert_called_once()

    async def test_on_modified_does_not_trigger_callback_for_different_file(
        self,
    ):
        # Create a mock callback function
        mock_callback = MagicMock()

        # File path to watch
        file_path = '/path/to/test/file.txt'

        # Create an instance of FileEventHandler
        event_handler = FileEventHandler(
            file_path=file_path, callback=mock_callback,
        )

        # Create a mock event for a different file modification
        different_file_path = '/path/to/other/file.txt'
        event = FileModifiedEvent(different_file_path)

        # Trigger the on_modified event
        await event_handler.on_modified(event)

        # Assert that the callback was not called
        mock_callback.assert_not_called()


@pytest.mark.asyncio
class TestRedisManager(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
