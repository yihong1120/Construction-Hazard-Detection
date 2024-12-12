from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import FastAPI
from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient
from fastapi_limiter import FastAPILimiter

from examples.streaming_web.backend.routes import rate_limiter_index
from examples.streaming_web.backend.routes import rate_limiter_label
from examples.streaming_web.backend.routes import register_routes
from examples.streaming_web.backend.utils import RedisManager


class TestRoutes(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        """
        Set up the test environment before each test.
        """
        self.app = FastAPI()

        # Mock Redis instance with AsyncMock
        self.redis_manager = RedisManager()
        RedisManager.client = AsyncMock()  # Mock Redis client

        # Register routes
        register_routes(self.app)

        # Initialise rate limiter with the Redis client
        asyncio.run(FastAPILimiter.init(RedisManager.client))

        # Mock rate limiter dependencies
        async def mock_rate_limiter():
            pass

        self.app.dependency_overrides[rate_limiter_index] = mock_rate_limiter
        self.app.dependency_overrides[rate_limiter_label] = mock_rate_limiter

        # Create a test client for the FastAPI app
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        """
        Clean up after each test.
        """
        self.app.dependency_overrides.clear()
        patch.stopall()

    @patch(
        'examples.streaming_web.backend.utils.'
        'RedisManager.get_labels',
        new_callable=AsyncMock,
    )
    def test_get_labels_success(self, mock_get_labels: AsyncMock):
        """
        Test the /api/labels route to ensure it returns
        correct labels on success.
        """
        mock_get_labels.return_value = ['label1', 'label2']
        response = self.client.get('/api/labels')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'labels': ['label1', 'label2']})

    @patch(
        'examples.streaming_web.backend.utils.RedisManager.get_labels',
        new_callable=AsyncMock,
    )
    def test_get_labels_value_error(self, mock_get_labels: AsyncMock):
        """
        Test the /api/labels route to ensure it handles ValueError.
        """
        mock_get_labels.side_effect = ValueError('Invalid data')
        response = self.client.get('/api/labels')
        self.assertEqual(response.status_code, 400)
        self.assertIn('Invalid data encountered', response.text)

    @patch(
        'examples.streaming_web.backend.utils.RedisManager.get_labels',
        new_callable=AsyncMock,
    )
    def test_get_labels_key_error(self, mock_get_labels: AsyncMock):
        """
        Test the /api/labels route to ensure it handles KeyError.
        """
        mock_get_labels.side_effect = KeyError('missing_key')
        response = self.client.get('/api/labels')
        self.assertEqual(response.status_code, 404)
        self.assertIn('Missing key encountered', response.text)

    @patch(
        'examples.streaming_web.backend.utils.RedisManager.get_labels',
        new_callable=AsyncMock,
    )
    def test_get_labels_connection_error(self, mock_get_labels: AsyncMock):
        """
        Test the /api/labels route to ensure it handles ConnectionError.
        """
        mock_get_labels.side_effect = ConnectionError('DB connection failed')
        response = self.client.get('/api/labels')
        self.assertEqual(response.status_code, 503)
        self.assertIn('Failed to connect to the database', response.text)

    @patch(
        'examples.streaming_web.backend.utils.RedisManager.get_labels',
        new_callable=AsyncMock,
    )
    def test_get_labels_timeout_error(self, mock_get_labels: AsyncMock):
        """
        Test the /api/labels route to ensure it handles TimeoutError.
        """
        mock_get_labels.side_effect = TimeoutError('Timeout!')
        response = self.client.get('/api/labels')
        self.assertEqual(response.status_code, 504)
        self.assertIn('Request timed out', response.text)

    @patch(
        'examples.streaming_web.backend.utils.RedisManager.get_labels',
        new_callable=AsyncMock,
    )
    def test_get_labels_generic_error(self, mock_get_labels: AsyncMock):
        """
        Test the /api/labels route to ensure it handles generic exceptions.
        """
        mock_get_labels.side_effect = Exception('Unknown error')
        response = self.client.get('/api/labels')
        self.assertEqual(response.status_code, 500)
        self.assertIn('Failed to fetch labels', response.text)

    @patch(
        'examples.streaming_web.backend.utils.RedisManager.get_labels',
        new_callable=AsyncMock,
    )
    @patch(
        'examples.streaming_web.backend.utils.RedisManager.get_keys_for_label',
        new_callable=AsyncMock,
    )
    def test_label_page_found(
        self, mock_get_keys: AsyncMock, mock_get_labels: AsyncMock,
    ):
        """
        Test the WebSocket route for an existing label.
        """
        mock_get_labels.return_value = ['label1', 'label2']
        mock_get_keys.return_value = ['stream_frame:label1_Cam0']

        # Call the API endpoint
        response = self.client.get('/api/labels')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'labels': ['label1', 'label2']})

    def test_webhook(self):
        """
        Test the webhook route to ensure it returns a successful response.
        """
        body = {'event': 'test_event'}
        response = self.client.post('/api/webhook', json=body)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'status': 'ok'})

    @patch('examples.streaming_web.backend.utils.Utils.verify_localhost')
    @patch('examples.streaming_web.backend.utils.Utils.load_configuration')
    def test_get_config(
        self,
        mock_load_config: MagicMock,
        mock_verify: MagicMock,
    ):
        """
        Test the GET /api/config endpoint.
        """
        mock_verify.return_value = True
        mock_load_config.return_value = {'setting': 'value'}
        response = self.client.get('/api/config')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'config': {'setting': 'value'}})
        mock_verify.assert_called_once()

    @patch('examples.streaming_web.backend.utils.Utils.verify_localhost')
    @patch('examples.streaming_web.backend.utils.Utils.update_configuration')
    async def test_update_config(
        self,
        mock_update_config: AsyncMock,
        mock_verify: MagicMock,
    ):
        """
        Test the POST /api/config endpoint.
        """
        mock_verify.return_value = True
        mock_update_config.return_value = {'setting': 'new_value'}

        response = self.client.post(
            '/api/config', json={'config': {'setting': 'new_value'}},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(), {
                'status': 'Configuration updated successfully.',
                'config': {'setting': 'new_value'},
            },
        )
        mock_verify.assert_called_once()

    @patch('examples.streaming_web.backend.utils.Utils.verify_localhost')
    @patch('examples.streaming_web.backend.utils.Utils.update_configuration')
    def test_update_config_failure(
        self,
        mock_update_config: MagicMock,
        mock_verify: MagicMock,
    ):
        """
        Test the POST /api/config endpoint when update fails.
        """
        mock_verify.return_value = True
        mock_update_config.side_effect = Exception('Update failed')

        response = self.client.post(
            '/api/config', json={'config': {'setting': 'fail'}},
        )
        self.assertEqual(response.status_code, 500)
        self.assertIn('Failed to update configuration', response.text)

    @patch(
        'examples.streaming_web.backend.utils.RedisManager.get_keys_for_label',
        new_callable=AsyncMock,
    )
    @patch(
        'examples.streaming_web.backend.utils.'
        'RedisManager.fetch_latest_frames',
        new_callable=AsyncMock,
    )
    def test_websocket_label_stream_no_keys(
        self,
        mock_fetch_frames: AsyncMock,
        mock_get_keys: AsyncMock,
    ):
        """
        Test the /api/ws/labels/{label} websocket when no keys are found.
        """
        mock_get_keys.return_value = []

        with self.client.websocket_connect(
            '/api/ws/labels/mylabel',
        ) as websocket:
            data = websocket.receive_json()
            self.assertIn('error', data)
            self.assertIn('No keys found for label', data['error'])

    @patch(
        'examples.streaming_web.backend.utils.RedisManager.get_keys_for_label',
        new_callable=AsyncMock,
    )
    @patch(
        'examples.streaming_web.backend.utils.'
        'RedisManager.fetch_latest_frames',
        new_callable=AsyncMock,
    )
    @patch(
        'examples.streaming_web.backend.utils.Utils.send_frames',
        new_callable=AsyncMock,
    )
    def test_websocket_label_stream_with_data(
        self,
        mock_send_frames: AsyncMock,
        mock_fetch_frames: AsyncMock,
        mock_get_keys: AsyncMock,
    ):
        """
        Test the /api/ws/labels/{label} websocket with some data.
        """
        mock_get_keys.return_value = ['stream_frame:label1_Cam0']
        mock_fetch_frames.side_effect = [
            [{'id': '1', 'data': 'frame1'}],
            WebSocketDisconnect(),
        ]

        # Test that we can connect and receive data until disconnect
        try:
            with self.client.websocket_connect('/api/ws/labels/label1'):
                # Mock the WebSocket connection
                pass

        except WebSocketDisconnect:
            pass

        mock_send_frames.assert_called()  # Ensure frames were sent

    @patch(
        'examples.streaming_web.backend.utils.Utils.encode',
        side_effect=lambda x: x,
    )
    @patch(
        (
            'examples.streaming_web.backend.utils.'
            'RedisManager.fetch_latest_frame_for_key'
        ),
        new_callable=AsyncMock,
    )
    def test_websocket_stream_with_data(
        self,
        mock_fetch_frame: AsyncMock,
        mock_encode: MagicMock,
    ):
        """
        Test the /api/ws/stream/{label}/{key} websocket with data.
        """
        # Mock the fetch_latest_frame_for_key method to return a frame
        # and then return None to simulate no new data available
        mock_fetch_frame.side_effect = [
            {'id': '1', 'frame': 'data'},
            None,
        ]

        with self.client.websocket_connect(
            '/api/ws/stream/label1/key1',
        ) as websocket:
            # First receive: should get the data frame
            data = websocket.receive_json()
            self.assertIn('id', data)
            self.assertEqual(data['id'], '1')
            self.assertEqual(data['frame'], 'data')

            # Second receive: should get an error message
            data = websocket.receive_json()
            self.assertIn('error', data)
            self.assertEqual(data['error'], 'No new data available')
            # Do not need to close the connection, the server will close it

    @patch(
        'examples.streaming_web.backend.utils.Utils.encode',
        side_effect=lambda x: x,
    )
    @patch(
        'examples.streaming_web.backend.utils.'
        'RedisManager.fetch_latest_frame_for_key',
        new_callable=AsyncMock,
    )
    def test_websocket_stream_error_handling(
        self,
        mock_fetch_frame: AsyncMock,
        mock_encode: MagicMock,
    ):
        """
        Test that /api/ws/stream/{label}/{key} handles unexpected exceptions.
        """
        mock_fetch_frame.side_effect = Exception('Some error')

        with self.assertRaises(WebSocketDisconnect):
            with self.client.websocket_connect(
                '/api/ws/stream/label1/key1',
            ) as websocket:
                # Once the connection is established, the server will close it
                websocket.receive_json()

    # def test_upload_file_successful(self):
    #     """
    #     Test the upload route to ensure it returns a successful response.
    #     """
    #     file_content = b'fake image data'
    #     files = {'file': ('test_image.png', file_content, 'image/png')}
    #     response = self.client.post('/api/upload', files=files)
    #     self.assertEqual(response.status_code, 200)
    #     self.assertIn('url', response.json())

    # def test_upload_file_missing_filename(self):
    #     """
    #     Test the upload route to ensure it returns a 422 error
    #     when the filename is missing.
    #     """
    #     # FastAPI automatically raises a 422 for missing file validation
    #     files = {'file': ('', b'data', 'image/png')}
    #     response = self.client.post('/api/upload', files=files)
    #     self.assertEqual(response.status_code, 422)


if __name__ == '__main__':
    unittest.main()
