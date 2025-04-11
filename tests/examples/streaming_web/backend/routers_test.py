from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import FastAPI
from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient
from fastapi_limiter import FastAPILimiter

from examples.auth.jwt_config import jwt_access
from examples.auth.redis_pool import get_redis_pool
from examples.auth.redis_pool import get_redis_pool_ws
from examples.streaming_web.backend.redis_service import DELIMITER
from examples.streaming_web.backend.routers import rate_limiter_index
from examples.streaming_web.backend.routers import rate_limiter_label
from examples.streaming_web.backend.routers import router


class TestRouters(unittest.IsolatedAsyncioTestCase):
    """
    Test suite for FastAPI routers.
    """

    def setUp(self) -> None:
        """
        Set up the FastAPI application and mock dependencies.
        """
        self.app = FastAPI()
        self.app.include_router(router, prefix='/api')

        async def mock_rate_limiter():
            pass

        # Override rate-limiters
        self.app.dependency_overrides[rate_limiter_index] = mock_rate_limiter
        self.app.dependency_overrides[rate_limiter_label] = mock_rate_limiter

        # Override Redis dependencies with a mock/async mock
        self.fake_redis = AsyncMock()
        self.app.dependency_overrides[get_redis_pool] = lambda: self.fake_redis
        self.app.dependency_overrides[get_redis_pool_ws] = (
            lambda: self.fake_redis
        )

        # Bypass JWT auth
        self.app.dependency_overrides[jwt_access] = lambda: None

        # Initialise FastAPILimiter with a mock
        asyncio.run(FastAPILimiter.init(AsyncMock()))
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.app.dependency_overrides.clear()

    # -----------------------------
    # Test GET /api/labels
    # -----------------------------
    @patch(
        'examples.streaming_web.backend.routers.scan_for_labels',
        new_callable=AsyncMock,
    )
    def test_get_labels_success(self, mock_scan_for_labels: AsyncMock):
        """
        Tests the GET /api/labels endpoint for successful label retrieval.

        Args:
            mock_scan_for_labels (AsyncMock): Mock for scan_for_labels.
        """
        mock_scan_for_labels.return_value = ['label1', 'label2']
        response = self.client.get('/api/labels')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'labels': ['label1', 'label2']})

    @patch(
        'examples.streaming_web.backend.routers.scan_for_labels',
        new_callable=AsyncMock,
    )
    def test_get_labels_value_error(self, mock_scan_for_labels: AsyncMock):
        """
        Tests the GET /api/labels endpoint for ValueError handling.

        Args:
            mock_scan_for_labels (AsyncMock): Mock for scan_for_labels.
        """
        mock_scan_for_labels.side_effect = ValueError('Invalid data')
        response = self.client.get('/api/labels')
        self.assertEqual(response.status_code, 500)
        self.assertIn('Invalid data', response.text)

    @patch(
        'examples.streaming_web.backend.routers.scan_for_labels',
        new_callable=AsyncMock,
    )
    def test_get_labels_key_error(self, mock_scan_for_labels: AsyncMock):
        """
        Tests the GET /api/labels endpoint for KeyError handling.

        Args:
            mock_scan_for_labels (AsyncMock): Mock for scan_for_labels.
        """
        mock_scan_for_labels.side_effect = KeyError('missing_key')
        response = self.client.get('/api/labels')
        self.assertEqual(response.status_code, 500)
        self.assertIn('missing_key', response.text)

    @patch(
        'examples.streaming_web.backend.routers.scan_for_labels',
        new_callable=AsyncMock,
    )
    def test_get_labels_connection_error(
        self,
        mock_scan_for_labels: AsyncMock,
    ):
        """
        Tests the GET /api/labels endpoint for connection error handling.

        Args:
            mock_scan_for_labels (AsyncMock): Mock for scan_for_labels.
        """
        mock_scan_for_labels.side_effect = ConnectionError(
            'DB connection failed',
        )
        response = self.client.get('/api/labels')
        self.assertEqual(response.status_code, 500)
        self.assertIn('DB connection failed', response.text)

    @patch(
        'examples.streaming_web.backend.routers.scan_for_labels',
        new_callable=AsyncMock,
    )
    def test_get_labels_timeout_error(self, mock_scan_for_labels: AsyncMock):
        """
        Tests the GET /api/labels endpoint for timeout error handling.

        Args:
            mock_scan_for_labels (AsyncMock): Mock for scan_for_labels.
        """
        mock_scan_for_labels.side_effect = TimeoutError('Timeout!')
        response = self.client.get('/api/labels')
        self.assertEqual(response.status_code, 500)
        self.assertIn('Timeout!', response.text)

    @patch(
        'examples.streaming_web.backend.routers.scan_for_labels',
        new_callable=AsyncMock,
    )
    def test_get_labels_generic_error(self, mock_scan_for_labels: AsyncMock):
        """
        Tests the GET /api/labels endpoint for generic error handling.

        Args:
            mock_scan_for_labels (AsyncMock): Mock for scan_for_labels.
        """
        mock_scan_for_labels.side_effect = Exception('Unknown error')
        response = self.client.get('/api/labels')
        self.assertEqual(response.status_code, 500)
        self.assertIn('Unknown error', response.text)

    # -----------------------------
    # Test POST /api/frames
    # -----------------------------
    @patch(
        'examples.streaming_web.backend.routers.store_to_redis',
        new_callable=AsyncMock,
    )
    def test_post_frame_success(self, mock_store_to_redis: AsyncMock):
        """
        Tests the POST /api/frames endpoint for successful frame storage.

        Args:
            mock_store_to_redis (AsyncMock): Mock for store_to_redis.
        """
        data = {
            'label': 'test_label',
            'key': 'test_key',
            'warnings_json': '',
            'cone_polygons_json': '',
            'pole_polygons_json': '',
            'detection_items_json': '',
            'width': '640',
            'height': '480',
        }
        files = {
            'file': ('test_image.png', b'dummy_image_data', 'image/png'),
        }
        response = self.client.post('/api/frames', data=data, files=files)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {'status': 'ok', 'message': 'Frame stored successfully.'},
        )
        mock_store_to_redis.assert_called_once()

    @patch(
        'examples.streaming_web.backend.routers.store_to_redis',
        new_callable=AsyncMock,
    )
    def test_post_frame_error(self, mock_store_to_redis: AsyncMock):
        """
        Tests the POST /api/frames endpoint for error handling.

        Args:
            mock_store_to_redis (AsyncMock): Mock for store_to_redis.
        """
        mock_store_to_redis.side_effect = Exception('Test error')
        data = {
            'label': 'test_label',
            'key': 'test_key',
            'warnings_json': '',
            'cone_polygons_json': '',
            'pole_polygons_json': '',
            'detection_items_json': '',
            'width': '640',
            'height': '480',
        }
        files = {
            'file': ('test_image.png', b'dummy_image_data', 'image/png'),
        }
        response = self.client.post('/api/frames', data=data, files=files)
        self.assertEqual(response.status_code, 500)
        self.assertIn('Test error', response.text)

    # -----------------------------
    # Test WebSocket /api/ws/labels/{label}
    # -----------------------------
    @patch(
        'examples.streaming_web.backend.routers.get_keys_for_label',
        new_callable=AsyncMock,
    )
    def test_websocket_label_stream_no_keys(
        self,
        mock_get_keys: AsyncMock,
    ):
        """
        Tests the WebSocket connection for label streaming
        when no keys are found.

        Args:
            mock_get_keys (AsyncMock): Mock for get_keys_for_label.
        """
        mock_get_keys.return_value = []
        with self.client.websocket_connect(
            '/api/ws/labels/mylabel',
        ) as websocket:
            try:
                data = websocket.receive_json()
                self.assertIn('No keys found for label', data.get('error', ''))
            except WebSocketDisconnect:
                pass

    @patch(
        'examples.streaming_web.backend.routers.get_keys_for_label',
        new_callable=AsyncMock,
    )
    @patch(
        'examples.streaming_web.backend.routers.fetch_latest_frames',
        new_callable=AsyncMock,
    )
    def test_websocket_label_stream_with_data(
        self,
        mock_fetch_frames: AsyncMock,
        mock_get_keys: AsyncMock,
    ):
        """
        Tests the WebSocket connection for label streaming.

        Args:
            mock_fetch_frames (AsyncMock): Mock for fetch_latest_frames.
            mock_get_keys (AsyncMock): Mock for get_keys_for_label.
        """
        mock_get_keys.return_value = ['stream_frame:label1_Cam0']
        mock_fetch_frames.side_effect = [
            [{
                'key': 'Cam0',
                'frame_bytes': b'frame_data',
                'warnings': '',
                'cone_polygons': '',
                'pole_polygons': '',
                'detection_items': '',
                'width': '640',
                'height': '480',
            }],
            WebSocketDisconnect(),  # Triggers the loop's exception path
        ]
        try:
            with self.client.websocket_connect(
                '/api/ws/labels/label1',
            ) as websocket:
                message_bytes = websocket.receive_bytes()
                header_json, frame_bytes = message_bytes.split(DELIMITER, 1)
                header = json.loads(header_json.decode('utf-8'))
                self.assertEqual(header.get('key'), 'Cam0')
                self.assertEqual(header.get('warnings'), '')
                self.assertEqual(header.get('width'), '640')
                self.assertEqual(frame_bytes, b'frame_data')
        except WebSocketDisconnect:
            pass

    @patch(
        'examples.streaming_web.backend.routers.get_keys_for_label',
        new_callable=AsyncMock,
    )
    @patch(
        'examples.streaming_web.backend.routers.fetch_latest_frames',
        new_callable=AsyncMock,
    )
    def test_websocket_label_stream_exception(
        self,
        mock_fetch_frames: AsyncMock,
        mock_get_keys: AsyncMock,
    ):
        """
        Causes fetch_latest_frames to raise an exception,
        triggering the except branch => WebSocketDisconnect.

        Args:
            mock_fetch_frames (AsyncMock): Mock for fetch_latest_frames.
            mock_get_keys (AsyncMock): Mock for get_keys_for_label.
        """
        mock_get_keys.return_value = ['stream_frame:label1_Cam0']
        mock_fetch_frames.side_effect = Exception('simulated error')
        with self.assertRaises(WebSocketDisconnect):
            with self.client.websocket_connect(
                '/api/ws/labels/label1',
            ) as websocket:
                websocket.receive_text()

    # -----------------------------
    # Test WebSocket /api/ws/stream/{label}/{key}
    # -----------------------------
    @patch(
        'examples.streaming_web.backend.routers.Utils.encode',
        side_effect=lambda x: x,
    )
    @patch(
        'examples.streaming_web.backend.routers.fetch_latest_frame_for_key',
        new_callable=AsyncMock,
    )
    def test_websocket_stream_with_data(
        self,
        mock_fetch_frame: AsyncMock,
        mock_encode: MagicMock,
    ):
        """
        Tests the pull action with two calls:
        - First call: returns a frame dict
        - Second call: returns None => no frame is sent

        Args:
            mock_fetch_frame (AsyncMock): Mock for fetch_latest_frame_for_key.
            mock_encode (MagicMock): Mock for Utils.encode.
        """
        mock_fetch_frame.side_effect = [
            {
                'id': '1',
                'frame_bytes': b'data',
                'warnings': 'warning',
                'cone_polygons': 'mycone',     # IMPORTANT: match the code
                'pole_polygons': 'mypole',     # If needed
                'detection_items': 'item',
                'width': '640',
                'height': '480',
            },
            None,
        ]
        with self.client.websocket_connect(
            '/api/ws/stream/label1/key1',
        ) as websocket:
            # First pull => expect frame data
            websocket.send_text(json.dumps({'action': 'pull'}))
            message_bytes = websocket.receive_bytes()

            from examples.streaming_web.backend.redis_service import DELIMITER
            header_json, frame_bytes = message_bytes.split(DELIMITER, 1)
            header = json.loads(header_json.decode('utf-8'))

            self.assertEqual(header.get('id'), '1')
            self.assertEqual(header.get('warnings'), 'warning')
            self.assertEqual(header.get('cone_polygons'), 'mycone')
            self.assertEqual(header.get('pole_polygons'), 'mypole')
            self.assertEqual(header.get('width'), '640')
            self.assertEqual(frame_bytes, b'data')

            # Second pull => returns None, so no message should arrive
            websocket.send_text(json.dumps({'action': 'pull'}))
            websocket.close()

    @patch(
        'examples.streaming_web.backend.routers.Utils.encode',
        side_effect=lambda x: x,
    )
    def test_websocket_stream_ping(self, mock_encode: MagicMock):
        """
        Sends a ping action and expects a pong response.

        Args:
            mock_encode (MagicMock): Mock for Utils.encode.
        """
        with self.client.websocket_connect(
            '/api/ws/stream/label1/key1',
        ) as websocket:
            websocket.send_text(json.dumps({'action': 'ping'}))
            response_text = websocket.receive_text()
            data = json.loads(response_text)
            self.assertEqual(data.get('action'), 'pong')

    @patch(
        'examples.streaming_web.backend.routers.Utils.encode',
        side_effect=lambda x: x,
    )
    def test_websocket_stream_unknown_action(self, mock_encode: MagicMock):
        """
        Sends an unrecognised action and expects an error response.

        Args:
            mock_encode (MagicMock): Mock for Utils.encode.
        """
        with self.client.websocket_connect(
            '/api/ws/stream/label1/key1',
        ) as websocket:
            websocket.send_text(json.dumps({'action': 'invalid_action'}))
            response_text = websocket.receive_text()
            data = json.loads(response_text)
            self.assertIn('error', data)
            self.assertEqual(data['error'], 'unknown action')

    @patch(
        'examples.streaming_web.backend.routers.Utils.encode',
        side_effect=lambda x: x,
    )
    @patch(
        'examples.streaming_web.backend.routers.fetch_latest_frame_for_key',
        new_callable=AsyncMock,
    )
    def test_websocket_stream_error_handling(
        self,
        mock_fetch_frame: AsyncMock,
        mock_encode: MagicMock,
    ):
        """
        Causes fetch_latest_frame_for_key to raise an exception,
        triggering the except branch => WebSocketDisconnect.

        Args:
            mock_fetch_frame (AsyncMock): Mock for fetch_latest_frame_for_key.
            mock_encode (MagicMock): Mock for Utils.encode.
        """
        mock_fetch_frame.side_effect = Exception('Some error')
        with self.assertRaises(WebSocketDisconnect):
            with self.client.websocket_connect(
                '/api/ws/stream/label1/key1',
            ) as websocket:
                websocket.send_text(json.dumps({'action': 'pull'}))
                websocket.receive_text()


if __name__ == '__main__':
    unittest.main()


'''
pytest \
    --cov=examples.streaming_web.backend.routers \
    --cov-report=term-missing \
    tests/examples/streaming_web/backend/routers_test.py
'''
