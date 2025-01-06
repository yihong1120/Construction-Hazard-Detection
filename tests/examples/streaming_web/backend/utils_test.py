from __future__ import annotations

import base64
import json
import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import mock_open
from unittest.mock import patch

import redis
from fastapi import HTTPException
from fastapi import WebSocket
from linebot import LineBotApi
from linebot.models import TextSendMessage

from examples.streaming_web.backend.utils import RedisManager
from examples.streaming_web.backend.utils import Utils
from examples.streaming_web.backend.utils import WebhookHandler


class TestRedisManager(unittest.IsolatedAsyncioTestCase):
    """
    Test suite for RedisManager methods.
    """

    def setUp(self) -> None:
        """
        Set up the test environment before each test.
        """
        self.redis_mock = MagicMock(spec=redis.Redis)
        self.redis_mock.xrevrange = AsyncMock()
        self.redis_mock.scan = AsyncMock()
        self.redis_mock.get = AsyncMock()
        self.redis_mock.set = AsyncMock()
        self.redis_mock.delete = AsyncMock()
        self.redis_mock.close = AsyncMock()

    async def test_fetch_latest_frame_for_key_with_data(self) -> None:
        """
        Test fetch_latest_frame_for_key method when Redis contains valid data.
        """
        redis_key = 'stream_frame:test_label_image1'
        last_id = '0-0'

        # Mock Redis response
        message_id = '1234-0'
        frame_data = b'sample_frame_data'
        warnings_data = b'Sample warning'
        self.redis_mock.xrevrange.return_value = [
            (
                message_id.encode('utf-8'),
                {b'frame': frame_data, b'warnings': warnings_data},
            ),
        ]

        redis_manager = RedisManager('localhost', 6379, 'password')
        redis_manager.client = self.redis_mock
        result = await redis_manager.fetch_latest_frame_for_key(
            redis_key,
            last_id,
        )

        expected_result = {
            'id': message_id,
            'image': base64.b64encode(frame_data).decode('utf-8'),
            'warnings': warnings_data.decode('utf-8'),
        }
        self.assertEqual(result, expected_result)

    async def test_fetch_latest_frame_for_key_no_data(self) -> None:
        """
        Test fetch_latest_frame_for_key method when Redis contains no new data.
        """
        redis_key = 'stream_frame:test_label_image1'
        last_id = '0-0'

        # Mock Redis response
        self.redis_mock.xrevrange.return_value = []

        redis_manager = RedisManager('localhost', 6379, 'password')
        redis_manager.client = self.redis_mock
        result = await redis_manager.fetch_latest_frame_for_key(
            redis_key,
            last_id,
        )

        self.assertIsNone(result)

    async def test_fetch_latest_frame_for_key_no_frame_data(self) -> None:
        """
        Test fetch_latest_frame_for_key method when Redis contains no new data.
        """
        redis_key = 'stream_frame:test_label_image1'
        last_id = '0-0'
        self.redis_mock.xrevrange.return_value = [
            (b'1234-0', {b'warnings': b'No frame here'}),
        ]

        redis_manager = RedisManager('localhost', 6379, 'password')
        redis_manager.client = self.redis_mock
        result = await redis_manager.fetch_latest_frame_for_key(
            redis_key,
            last_id,
        )

        self.assertIsNone(result)

    async def test_fetch_latest_frames(self) -> None:
        """
        Test the fetch_latest_frames method for multiple streams.
        """
        last_ids = {
            'stream_frame:test_label|image1': '0-0',
            'stream_frame:test_label|image2': '0-0',
        }

        # Mock Redis response
        self.redis_mock.xrevrange.side_effect = [
            [('1234-0', {b'frame': b'image1_frame'})],
            [('5678-0', {b'frame': b'image2_frame'})],
        ]

        redis_manager = RedisManager('localhost', 6379, 'password')
        redis_manager.client = self.redis_mock
        result = await redis_manager.fetch_latest_frames(last_ids)

        expected_result = [
            {
                'key': 'image1', 'image': base64.b64encode(
                    b'image1_frame',
                ).decode('utf-8'),
            },
            {
                'key': 'image2', 'image': base64.b64encode(
                    b'image2_frame',
                ).decode('utf-8'),
            },
        ]
        self.assertEqual(result, expected_result)

    async def test_fetch_latest_frames_empty(self) -> None:
        """
        Test fetch_latest_frames method when Redis contains no new data.
        """
        last_ids = {
            'stream_frame:test_label|image1': '0-0',
            'stream_frame:test_label|image2': '0-0',
        }

        self.redis_mock.xrevrange.side_effect = [
            [],  # 第一個key無回應
            [('5678-0', {b'frame': b'image2_frame'})],
        ]

        redis_manager = RedisManager('localhost', 6379, 'password')
        redis_manager.client = self.redis_mock
        result = await redis_manager.fetch_latest_frames(last_ids)

        expected_result = [
            {
                'key': 'image2', 'image': base64.b64encode(
                    b'image2_frame',
                ).decode('utf-8'),
            },
        ]
        self.assertEqual(result, expected_result)

    async def test_get_labels(self) -> None:
        """
        Test the get_labels function to ensure it returns expected labels.
        """
        # Mock the Redis scan method to return some keys
        self.redis_mock.scan.return_value = (
            0, [
                b'stream_frame:bGFiZWwx|image1',
                b'stream_frame:bGFiZWwx|image2',
                b'stream_frame:bGFiZWwy|image1',
                b'stream_frame:bGFiZWwz|image1',
                b'stream_frame:dGVzdA==|image',
                b'__invalid_key',
                b'_another_invalid_key',
            ],
        )

        # Call the function
        redis_manager = RedisManager('localhost', 6379, 'password')
        redis_manager.client = self.redis_mock
        result = await redis_manager.get_labels()

        # Check the expected result
        expected_result = ['label1', 'label2', 'label3']
        self.assertEqual(result, expected_result)

    async def test_get_keys_for_label(self) -> None:
        """
        Test the get_keys_for_label function to ensure it returns correct keys.
        """
        label = 'label1'
        encoded_label = Utils.encode(label)

        # Mock the Redis scan method to return keys matching the label
        self.redis_mock.scan.return_value = (
            0, [
                f'stream_frame:{encoded_label}|image1'.encode(),
                f'stream_frame:{encoded_label}|image2'.encode(),
            ],
        )

        # Call the function
        redis_manager = RedisManager('localhost', 6379, 'password')
        redis_manager.client = self.redis_mock
        result = await redis_manager.get_keys_for_label(label)

        # Check the expected result
        expected_result = [
            f'stream_frame:{encoded_label}|image1',
            f'stream_frame:{encoded_label}|image2',
        ]
        self.assertEqual(result, expected_result)

    async def test_update_partial_config(self) -> None:
        """
        Test the update_partial_config function to ensure it updates correctly.
        """
        key = 'new_key'
        value = 'new_value'
        cached_config = {'existing_key': 'existing_value'}

        # Mock the Redis get and set methods
        self.redis_mock.get.return_value = json.dumps(
            cached_config,
        ).encode('utf-8')

        redis_manager = RedisManager('localhost', 6379, 'password')
        redis_manager.client = self.redis_mock
        await redis_manager.update_partial_config(key, value)

        # Check if the set method was called with the updated config
        cached_config[key] = value
        self.redis_mock.set.assert_called_once_with(
            'config_cache', json.dumps(cached_config), ex=3600,
        )

    async def test_get_partial_config(self) -> None:
        """
        Test the get_partial_config function to ensure it retrieves correctly.
        """
        key = 'existing_key'
        cached_config = {'existing_key': 'existing_value'}

        # Mock the Redis get method
        self.redis_mock.get.return_value = json.dumps(
            cached_config,
        ).encode('utf-8')

        redis_manager = RedisManager('localhost', 6379, 'password')
        redis_manager.client = self.redis_mock
        result = await redis_manager.get_partial_config(key)

        # Check the expected result
        self.assertEqual(result, cached_config[key])

    async def test_delete_config_cache(self) -> None:
        """
        Test the delete_config_cache function to ensure it deletes correctly.
        """
        redis_manager = RedisManager('localhost', 6379, 'password')
        redis_manager.client = self.redis_mock
        await redis_manager.delete_config_cache()

        # Check if the delete method was called with the correct key
        self.redis_mock.delete.assert_called_once_with('config_cache')

    async def test_get_config_cache(self) -> None:
        """
        Test the get_config_cache function to ensure it retrieves correctly.
        """
        cached_config = {'existing_key': 'existing_value'}

        # Mock the Redis get method
        self.redis_mock.get.return_value = json.dumps(
            cached_config,
        ).encode('utf-8')

        redis_manager = RedisManager('localhost', 6379, 'password')
        redis_manager.client = self.redis_mock
        result = await redis_manager.get_config_cache()

        # Check the expected result
        self.assertEqual(result, cached_config)

    async def test_set_config_cache(self) -> None:
        """
        Test the set_config_cache function to ensure it sets correctly.
        """
        config = {'new_key': 'new_value'}

        redis_manager = RedisManager('localhost', 6379, 'password')
        redis_manager.client = self.redis_mock
        await redis_manager.set_config_cache(config)

        # Check if the set method was called with the correct config
        self.redis_mock.set.assert_called_once_with(
            'config_cache', json.dumps(config), ex=3600,
        )

    async def test_close(self) -> None:
        """
        Test the close method to ensure the Redis connection is closed.
        """
        redis_manager = RedisManager('localhost', 6379, 'password')
        redis_manager.client = self.redis_mock
        await redis_manager.close()
        self.redis_mock.close.assert_awaited_once()


class TestUtils(unittest.IsolatedAsyncioTestCase):
    """
    Test suite for utility functions in the streaming_web module.
    """

    def setUp(self) -> None:
        """
        Set up the test environment before each test.
        """
        self.redis_mock = MagicMock(spec=redis.Redis)
        self.redis_mock.scan = AsyncMock()
        self.redis_mock.mget = AsyncMock()
        self.redis_mock.get = AsyncMock()

    def tearDown(self) -> None:
        """
        Clean up after each test.
        """
        self.redis_mock.reset_mock()

    async def test_send_frames(self) -> None:
        """
        Test the send_frames function to ensure it sends data to
        WebSocket client.
        """
        # Mock the WebSocket send_json method
        websocket_mock = MagicMock(spec=WebSocket)
        websocket_mock.send_json = AsyncMock()

        # Prepare data
        label = 'label1'
        updated_data = [
            {'key': 'image1', 'image': 'encoded_image_data_1'},
            {'key': 'image2', 'image': 'encoded_image_data_2'},
        ]

        # Call the function
        await Utils.send_frames(websocket_mock, label, updated_data)

        # Check if send_json was called with correct data
        expected_data = {
            'label': label,
            'images': updated_data,
        }
        websocket_mock.send_json.assert_called_once_with(expected_data)

    async def test_is_base64(self) -> None:
        """
        Test the is_base64 function for different cases.
        """
        valid_base64 = 'QmFzZTY0U3RyaW5n'
        invalid_base64 = 'NotBase64@#%'
        empty_string = ''

        self.assertTrue(Utils.is_base64(valid_base64))
        self.assertFalse(Utils.is_base64(invalid_base64))
        self.assertFalse(Utils.is_base64(empty_string))

    async def test_encode(self) -> None:
        """
        Test the encode function to ensure it encodes correctly.
        """
        input_string = 'test_label'
        encoded_string = Utils.encode(input_string)

        # Check if encoding and underscore replacement work as expected
        expected_encoded = base64.urlsafe_b64encode(
            input_string.encode('utf-8'),
        ).decode('utf-8')
        self.assertEqual(encoded_string, expected_encoded)

    async def test_decode_valid_base64(self) -> None:
        """
        Test the decode function with valid Base64 input.
        """
        input_string = base64.urlsafe_b64encode(
            b'test_label',
        ).decode('utf-8')
        decoded_string = Utils.decode(input_string)
        self.assertEqual(decoded_string, 'test_label')

    async def test_decode_invalid_base64(self) -> None:
        """
        Test the decode function with invalid Base64 input.
        """
        input_string = 'Invalid_String!'
        decoded_string = Utils.decode(input_string)
        self.assertEqual(decoded_string, input_string)

    async def test_load_configuration(self) -> None:
        """
        Test the load_configuration function to ensure it loads correctly.
        """
        config_path = 'test_config.json'
        config_data = [{'key': 'value'}]

        # Mock the open function to return the config data
        with patch(
            'builtins.open',
            mock_open(read_data=json.dumps(config_data)),
        ):
            result = Utils.load_configuration(config_path)

        self.assertEqual(result, config_data)

    async def test_load_configuration_exception(self) -> None:
        """
        Test load_configuration when an error occurs while reading the file.
        """
        config_path = 'non_existent.json'
        with patch(
            'builtins.open',
            side_effect=FileNotFoundError('File not found'),
        ):
            result = Utils.load_configuration(config_path)
        self.assertEqual(result, [])

    async def test_save_configuration(self) -> None:
        """
        Test the save_configuration function to ensure it saves correctly.
        """
        config_path = 'test_config.json'
        config_data = [{'key': 'value'}]

        # Mock the open function
        with patch('builtins.open', mock_open()) as mock_file:
            Utils.save_configuration(config_path, config_data)

            # Check if the file was written with the correct data
            mock_file().write.assert_called_once_with(
                json.dumps(config_data, indent=4, ensure_ascii=False),
            )

    async def test_save_configuration_exception(self) -> None:
        """
        Test save_configuration when an error occurs while writing to the file.
        """
        config_path = 'test_config.json'
        config_data = [{'key': 'value'}]

        with patch('builtins.open', side_effect=OSError('Write error')):
            # This function should not raise an exception
            # even if an error occurs while writing to the file
            Utils.save_configuration(config_path, config_data)

    async def test_verify_localhost(self) -> None:
        """
        Test the verify_localhost function to ensure it verifies correctly.
        """
        request_mock = MagicMock()
        request_mock.client.host = '127.0.0.1'

        # Should not raise an exception
        Utils.verify_localhost(request_mock)

        request_mock.client.host = '192.168.1.1'
        with self.assertRaises(HTTPException):
            Utils.verify_localhost(request_mock)

    async def test_update_configuration(self) -> None:
        """
        Test the update_configuration function to ensure it updates correctly.
        """
        config_path = 'test_config.json'
        current_config = [{'video_url': 'url1', 'key': 'value1'}]
        new_config = [
            {'video_url': 'url1', 'key': 'new_value1'},
            {'video_url': 'url2', 'key': 'value2'},
        ]

        # Mock the load_configuration and save_configuration functions
        with patch(
            'examples.streaming_web.backend.utils.Utils.load_configuration',
            return_value=current_config,
        ):
            with patch(
                'examples.streaming_web.backend.utils.'
                'Utils.save_configuration',
            ) as mock_save:
                result = Utils.update_configuration(config_path, new_config)

                # Check the expected result
                expected_result = [
                    {'video_url': 'url1', 'key': 'new_value1'},
                    {'video_url': 'url2', 'key': 'value2'},
                ]
                self.assertEqual(result, expected_result)

                # Check if the save_configuration function
                # was called with the correct data
                mock_save.assert_called_once_with(config_path, expected_result)

    async def test_update_configuration_not_list(self) -> None:
        """
        Test update_configuration when the configuration is not a list.
        """
        config_path = 'test_config.json'
        not_a_list = {'video_url': 'url1', 'key': 'value1'}
        new_config = [{'video_url': 'url2', 'key': 'value2'}]

        with patch(
            'examples.streaming_web.backend.utils.Utils.load_configuration',
            return_value=not_a_list,
        ):
            with self.assertRaises(ValueError) as cm:
                Utils.update_configuration(config_path, new_config)
            self.assertIn('Invalid configuration format', str(cm.exception))

    async def test_get_config_cache_empty(self):
        """
        Test get_config_cache when Redis contains no configuration data.
        """
        # Mock Redis response
        self.redis_mock.get.return_value = None

        redis_manager = RedisManager('localhost', 6379, 'password')
        redis_manager.client = self.redis_mock
        result = await redis_manager.get_config_cache()

        # Check the expected result
        self.assertEqual(result, {})


class TestWebhookHandler(unittest.IsolatedAsyncioTestCase):
    """
    Test suite for the WebhookHandler class.
    """

    def setUp(self):
        """
        Set up the test environment before each test.
        """
        # Mock LineBotApi instance
        self.line_bot_api_mock = MagicMock(spec=LineBotApi)
        self.webhook_handler = WebhookHandler(
            line_bot_api=self.line_bot_api_mock,
        )

    async def test_process_webhook_events_success_group(self):
        """
        Test process_webhook_events with a group message containing 'token'.
        """
        body = {
            'events': [
                {
                    'type': 'message',
                    'message': {'type': 'text', 'text': 'token'},
                    'replyToken': 'test_reply_token',
                    'source': {
                        'type': 'group',
                        'groupId': 'test_group_id',
                        'userId': 'test_user_id',
                    },
                },
            ],
        }

        # Call the method under test
        responses = await self.webhook_handler.process_webhook_events(body)

        # Verify the push_message call
        self.line_bot_api_mock.push_message.assert_called_once_with(
            'test_group_id',
            TextSendMessage(
                text='group ID: test_group_id\nuser ID: test_user_id',
            ),
        )

        # Verify the response
        expected_responses = [
            {'status': 'success', 'target_id': 'test_group_id'},
        ]
        self.assertEqual(responses, expected_responses)

    async def test_process_webhook_events_success_user(self):
        """
        Test process_webhook_events with a user message containing 'token'.
        """
        body = {
            'events': [
                {
                    'type': 'message',
                    'message': {'type': 'text', 'text': 'token'},
                    'replyToken': 'test_reply_token',
                    'source': {'type': 'user', 'userId': 'test_user_id'},
                },
            ],
        }

        # Call the method under test
        responses = await self.webhook_handler.process_webhook_events(body)

        # Verify the push_message call
        self.line_bot_api_mock.push_message.assert_called_once_with(
            'test_user_id',
            TextSendMessage(
                text='group ID: not provided\nuser ID: test_user_id',
            ),
        )

        # Verify the response
        expected_responses = [
            {'status': 'success', 'target_id': 'test_user_id'},
        ]
        self.assertEqual(responses, expected_responses)

    async def test_process_webhook_events_redelivery(self):
        """
        Test process_webhook_events skips redelivery events.
        """
        body = {
            'events': [
                {
                    'type': 'message',
                    'deliveryContext': {'isRedelivery': True},
                    'message': {'type': 'text', 'text': 'token'},
                },
            ],
        }

        # Call the method under test
        responses = await self.webhook_handler.process_webhook_events(body)

        # Verify no push_message call
        self.line_bot_api_mock.push_message.assert_not_called()

        # Verify the response
        expected_responses = [{'status': 'skipped', 'reason': 'Redelivery'}]
        self.assertEqual(responses, expected_responses)

    async def test_process_webhook_events_unexpected_error(self):
        """
        Test process_webhook_events handles unexpected exceptions.
        """
        body = {
            'events': [
                {
                    'type': 'message',
                    'message': {'type': 'text', 'text': 'token'},
                    'replyToken': 'test_reply_token',
                    'source': {'type': 'user', 'userId': 'test_user_id'},
                },
            ],
        }

        # Simulate an unexpected error
        self.line_bot_api_mock.push_message.side_effect = Exception(
            'Unexpected error',
        )

        # Call the method under test
        responses = await self.webhook_handler.process_webhook_events(body)

        # Verify the response
        expected_responses = [
            {'status': 'error', 'message': 'Unexpected error'},
        ]
        self.assertEqual(responses, expected_responses)


if __name__ == '__main__':
    unittest.main()
