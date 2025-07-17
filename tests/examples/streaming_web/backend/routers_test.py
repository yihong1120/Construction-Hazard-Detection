from __future__ import annotations

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import FastAPI
from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient
from fastapi_limiter import FastAPILimiter
from jwt import InvalidTokenError

from examples.auth.database import get_db
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

    app: FastAPI
    fake_redis: AsyncMock
    mock_db_session: AsyncMock
    client: TestClient

    def setUp(self) -> None:
        """
        Set up the FastAPI app and mock dependencies for testing.
        """
        self.app: FastAPI = FastAPI()
        self.app.include_router(router, prefix='/api')

        async def mock_rate_limiter() -> None:
            """Mock rate limiter that does nothing."""
            pass

        # Override rate-limiters to avoid actual rate limiting during tests
        self.app.dependency_overrides[rate_limiter_index] = mock_rate_limiter
        self.app.dependency_overrides[rate_limiter_label] = mock_rate_limiter

        # Override Redis dependencies with an async mock
        self.fake_redis: AsyncMock = AsyncMock()
        self.app.dependency_overrides[get_redis_pool] = lambda: self.fake_redis
        self.app.dependency_overrides[get_redis_pool_ws] = (
            lambda: self.fake_redis
        )

        # Bypass JWT authentication with a mock credentials object
        mock_credentials: SimpleNamespace = SimpleNamespace(
            subject={'username': 'testuser'},
        )
        self.app.dependency_overrides[jwt_access] = lambda: mock_credentials

        # Mock the database session
        self.mock_db_session: AsyncMock = AsyncMock()
        self.app.dependency_overrides[get_db] = lambda: self.mock_db_session

        # Set up default mock user and result for database queries
        self.setup_default_db_mocks()

        # Initialise FastAPILimiter with a mock to avoid Redis dependency
        asyncio.run(FastAPILimiter.init(AsyncMock()))
        self.client: TestClient = TestClient(self.app)

    def setup_default_db_mocks(self) -> None:
        """
        Set up default mock user and site for database queries.
        """
        # Create default mock site and user
        mock_site: MagicMock = MagicMock()
        mock_site.name = 'label1'
        mock_user: MagicMock = MagicMock()
        mock_user.role = 'admin'
        mock_user.sites = [mock_site]

        # Create mock result for database query
        mock_result: MagicMock = MagicMock()
        mock_result.scalars.return_value.first.return_value = mock_user

        # Configure the mock session to return this result
        self.mock_db_session.execute.return_value = mock_result

    def tearDown(self) -> None:
        """Clear all dependency overrides after each test."""
        self.app.dependency_overrides.clear()

    # -----------------------------
    # Test GET /api/labels
    # -----------------------------
    @patch(
        'examples.streaming_web.backend.routers.scan_for_labels',
        new_callable=AsyncMock,
    )
    def test_get_labels_success(self, mock_scan_for_labels: AsyncMock) -> None:
        """
        Tests the GET /api/labels endpoint for successful label retrieval.

        Args:
            mock_scan_for_labels (AsyncMock): Mock for scan_for_labels.
        """
        # Ensure JWT subject is present
        self.app.dependency_overrides[jwt_access] = lambda: SimpleNamespace(
            subject={'username': 'testuser'},
        )
        # Arrange: Set the mock to return a known list of labels
        mock_scan_for_labels.return_value = ['label1', 'label2']

        # Act: Make a GET request to the endpoint
        response = self.client.get('/api/labels')

        # Assert: Check the response status and content
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
        self.app.dependency_overrides[jwt_access] = lambda: SimpleNamespace(
            subject={'username': 'testuser'},
        )
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
        self.app.dependency_overrides[jwt_access] = lambda: SimpleNamespace(
            subject={'username': 'testuser'},
        )
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
        self.app.dependency_overrides[jwt_access] = lambda: SimpleNamespace(
            subject={'username': 'testuser'},
        )
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
        self.app.dependency_overrides[jwt_access] = lambda: SimpleNamespace(
            subject={'username': 'testuser'},
        )
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
        self.app.dependency_overrides[jwt_access] = lambda: SimpleNamespace(
            subject={'username': 'testuser'},
        )
        mock_scan_for_labels.side_effect = Exception('Unknown error')
        response = self.client.get('/api/labels')
        self.assertEqual(response.status_code, 500)
        self.assertIn('Unknown error', response.text)

    @patch(
        'examples.streaming_web.backend.routers.scan_for_labels',
        new_callable=AsyncMock,
    )
    def test_get_labels_non_admin_filtering(
        self, mock_scan_for_labels: AsyncMock,
    ):
        """
        Tests /api/labels for non-admin user only getting allowed sites.

        Args:
            mock_scan_for_labels (AsyncMock): Mock for scan_for_labels.
        """
        # Setup user as non-admin with two sites
        mock_site1 = MagicMock()
        mock_site1.name = 'label1'
        mock_site2 = MagicMock()
        mock_site2.name = 'label2'
        mock_user = MagicMock()
        mock_user.role = 'user'
        mock_user.sites = [mock_site1, mock_site2]
        mock_result = MagicMock()
        mock_result.scalars.return_value.first.return_value = mock_user

        # Override the default mock for this test
        self.mock_db_session.execute.return_value = mock_result

        mock_scan_for_labels.return_value = ['label1', 'label2', 'label3']
        response = self.client.get('/api/labels')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'labels': ['label1', 'label2']})

    def test_get_labels_jwt_subject_missing(self):
        """
        Tests /api/labels for missing JWT subject.
        """
        self.app.dependency_overrides[jwt_access] = lambda: SimpleNamespace(
            subject={
            },
        )
        with patch(
            'examples.streaming_web.backend.routers.scan_for_labels',
            new_callable=AsyncMock,
        ) as mock_scan:
            mock_scan.return_value = ['label1']
            response = self.client.get('/api/labels')
            self.assertEqual(response.status_code, 500)
            self.assertIn('Invalid token', response.text)

    def test_get_labels_user_not_found(self):
        """
        Tests /api/labels for user not found in DB.
        """
        # Override the default mock to return None (user not found)
        mock_result = MagicMock()
        mock_result.scalars.return_value.first.return_value = None
        self.mock_db_session.execute.return_value = mock_result

        with patch(
            'examples.streaming_web.backend.routers.scan_for_labels',
            new_callable=AsyncMock,
        ) as mock_scan:
            mock_scan.return_value = ['label1']
            response = self.client.get('/api/labels')
            self.assertEqual(response.status_code, 500)
            self.assertIn('Invalid user', response.text)

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

    def test_websocket_frames_jwt_header_missing(self):
        with self.client.websocket_connect('/api/ws/frames') as websocket:
            websocket.send_bytes(b'test')
            with self.assertRaises(WebSocketDisconnect) as exc:
                websocket.receive_bytes()
            self.assertEqual(exc.exception.code, 1008)

    @patch(
        'examples.streaming_web.backend.routers.get_user_data',
        new_callable=AsyncMock,
    )
    def test_websocket_frames_jti_invalid(self, mock_get_user_data):
        with patch(
            'examples.streaming_web.backend.routers.jwt.decode',
            return_value={
                'subject': {'username': 'testuser', 'jti': 'badjti'},
                'username': 'testuser',
                'jti': 'badjti',
            },
        ):
            mock_get_user_data.return_value = {'jti_list': ['goodjti']}
            headers = {'authorization': 'Bearer faketoken'}
            with self.client.websocket_connect(
                '/api/ws/frames', headers=headers,
            ) as websocket:
                websocket.send_bytes(b'test')
                with self.assertRaises(WebSocketDisconnect) as exc:
                    websocket.receive_bytes()
                self.assertEqual(exc.exception.code, 1008)

    @patch(
        'examples.streaming_web.backend.routers.get_keys_for_label',
        new_callable=AsyncMock,
    )
    @patch(
        'examples.streaming_web.backend.routers.fetch_latest_frames',
        new_callable=AsyncMock,
    )
    def test_websocket_label_stream_asyncio_sleep(
        self, mock_fetch_frames, mock_get_keys,
    ):
        """
        Tests that asyncio.sleep is called in the websocket label stream loop.
        """
        mock_get_keys.return_value = ['stream_frame:label1_Cam0']
        mock_fetch_frames.side_effect = [[], WebSocketDisconnect()]

        with patch(
            'examples.streaming_web.backend.routers.asyncio.sleep',
            new_callable=AsyncMock,
        ) as mock_sleep:
            try:
                with self.client.websocket_connect(
                    '/api/ws/labels/label1',
                ) as _:
                    # This should trigger the loop that calls asyncio.sleep
                    pass
            except WebSocketDisconnect:
                pass
            # Verify asyncio.sleep was called
            mock_sleep.assert_called_with(0.1)

    def test_websocket_frames_jwt_token_invalid(self):
        """
        Tests /ws/frames for invalid JWT token.
        """
        with patch(
            'examples.streaming_web.backend.routers.jwt.decode',
            side_effect=InvalidTokenError('Invalid token'),
        ):
            headers = {'authorization': 'Bearer invalidtoken'}
            with self.client.websocket_connect(
                '/api/ws/frames', headers=headers,
            ) as websocket:
                websocket.send_bytes(b'test')
                with self.assertRaises(WebSocketDisconnect) as exc:
                    websocket.receive_bytes()
                self.assertEqual(exc.exception.code, 1008)

    def test_websocket_frames_jwt_payload_empty(self):
        """
        Tests /ws/frames for empty JWT payload.
        """
        with patch(
            'examples.streaming_web.backend.routers.jwt.decode',
            return_value=None,
        ):
            headers = {'authorization': 'Bearer faketoken'}
            with self.client.websocket_connect(
                '/api/ws/frames', headers=headers,
            ) as websocket:
                websocket.send_bytes(b'test')
                with self.assertRaises(WebSocketDisconnect) as exc:
                    websocket.receive_bytes()
                self.assertEqual(exc.exception.code, 1008)

    def test_websocket_frames_jwt_missing_username_or_jti(self):
        """
        Tests /ws/frames for missing username or JTI in JWT payload.
        """
        with patch(
            'examples.streaming_web.backend.routers.jwt.decode', return_value={
                'subject': {}, 'username': None, 'jti': None,
            },
        ):
            headers = {'authorization': 'Bearer faketoken'}
            with self.client.websocket_connect(
                '/api/ws/frames', headers=headers,
            ) as websocket:
                websocket.send_bytes(b'test')
                with self.assertRaises(WebSocketDisconnect) as exc:
                    websocket.receive_bytes()
                self.assertEqual(exc.exception.code, 1008)

    @patch(
        'examples.streaming_web.backend.routers.get_user_data',
        new_callable=AsyncMock,
    )
    @patch(
        'examples.streaming_web.backend.routers.store_to_redis',
        new_callable=AsyncMock,
    )
    def test_websocket_frames_successful_frame_upload(
        self, mock_store_to_redis, mock_get_user_data,
    ):
        """
        Tests successful frame upload via WebSocket.
        """
        with patch(
            'examples.streaming_web.backend.routers.jwt.decode',
            return_value={
                'subject': {'username': 'testuser', 'jti': 'validjti'},
                'username': 'testuser',
                'jti': 'validjti',
            },
        ):
            mock_get_user_data.return_value = {'jti_list': ['validjti']}
            headers = {'authorization': 'Bearer validtoken'}

            # Create test frame data
            header_data = {
                'label': 'test_label',
                'key': 'test_key',
                'warnings_json': '',
                'cone_polygons_json': '',
                'pole_polygons_json': '',
                'detection_items_json': '',
                'width': 640,
                'height': 480,
            }
            header_bytes = json.dumps(header_data).encode('utf-8')
            frame_bytes = b'test_frame_data'
            message = header_bytes + DELIMITER + frame_bytes

            with self.client.websocket_connect(
                '/api/ws/frames', headers=headers,
            ) as websocket:
                websocket.send_bytes(message)
                response = websocket.receive_json()
                self.assertEqual(response['status'], 'ok')
                self.assertEqual(
                    response['message'],
                    'Frame stored successfully.',
                )
                mock_store_to_redis.assert_called_once()

    @patch(
        'examples.streaming_web.backend.routers.get_user_data',
        new_callable=AsyncMock,
    )
    @patch(
        'examples.streaming_web.backend.routers.store_to_redis',
        new_callable=AsyncMock,
    )
    def test_websocket_frames_store_error(
        self, mock_store_to_redis, mock_get_user_data,
    ):
        """
        Tests error handling in frame upload when store_to_redis fails.
        """
        with patch(
            'examples.streaming_web.backend.routers.jwt.decode',
            return_value={
                'subject': {'username': 'testuser', 'jti': 'validjti'},
                'username': 'testuser',
                'jti': 'validjti',
            },
        ):
            mock_get_user_data.return_value = {'jti_list': ['validjti']}
            mock_store_to_redis.side_effect = Exception('Store failed')
            headers = {'authorization': 'Bearer validtoken'}

            # Create test frame data
            header_data = {
                'label': 'test_label',
                'key': 'test_key',
                'warnings_json': '',
                'cone_polygons_json': '',
                'pole_polygons_json': '',
                'detection_items_json': '',
                'width': 640,
                'height': 480,
            }
            header_bytes = json.dumps(header_data).encode('utf-8')
            frame_bytes = b'test_frame_data'
            message = header_bytes + DELIMITER + frame_bytes

            with self.client.websocket_connect(
                '/api/ws/frames', headers=headers,
            ) as websocket:
                websocket.send_bytes(message)
                response = websocket.receive_json()
                self.assertEqual(response['status'], 'error')
                self.assertIn('Store failed', response['message'])

    @patch(
        'examples.streaming_web.backend.routers.get_user_data',
        new_callable=AsyncMock,
    )
    def test_websocket_frames_invalid_data_format(self, mock_get_user_data):
        """
        Tests error handling when frame data format is invalid.
        """
        with patch(
            'examples.streaming_web.backend.routers.jwt.decode',
            return_value={
                'subject': {'username': 'testuser', 'jti': 'validjti'},
                'username': 'testuser',
                'jti': 'validjti',
            },
        ):
            mock_get_user_data.return_value = {'jti_list': ['validjti']}
            headers = {'authorization': 'Bearer validtoken'}

            # Send invalid data without DELIMITER
            invalid_data = b'invalid_data_without_delimiter'

            with self.client.websocket_connect(
                '/api/ws/frames', headers=headers,
            ) as websocket:
                websocket.send_bytes(invalid_data)
                response = websocket.receive_json()
                self.assertEqual(response['status'], 'error')
                self.assertIn('Failed to store frame', response['message'])

    @patch(
        'examples.streaming_web.backend.routers.get_user_data',
        new_callable=AsyncMock,
    )
    def test_websocket_frames_websocket_disconnect(self, mock_get_user_data):
        """
        Tests WebSocketDisconnect handling in /ws/frames endpoint.
        """
        with patch(
            'examples.streaming_web.backend.routers.jwt.decode',
            return_value={
                'subject': {'username': 'testuser', 'jti': 'validjti'},
                'username': 'testuser',
                'jti': 'validjti',
            },
        ):
            mock_get_user_data.return_value = {'jti_list': ['validjti']}
            headers = {'authorization': 'Bearer validtoken'}

            with self.client.websocket_connect(
                '/api/ws/frames', headers=headers,
            ) as websocket:
                # Close the websocket to trigger WebSocketDisconnect
                websocket.close()
                # The WebSocketDisconnect should be handled internally

    @patch(
        'examples.streaming_web.backend.routers.get_user_data',
        new_callable=AsyncMock,
    )
    @patch(
        'examples.streaming_web.backend.routers.store_to_redis',
        new_callable=AsyncMock,
    )
    def test_websocket_frames_outer_exception(
        self, mock_store_to_redis, mock_get_user_data,
    ):
        """
        Tests outer exception handling in /ws/frames endpoint.
        """
        mock_get_user_data.return_value = {'jti_list': ['validjti']}

        # Make store_to_redis raise an exception
        # that's not caught by inner try-except
        mock_store_to_redis.side_effect = RuntimeError('Outer exception')

        with patch(
            'examples.streaming_web.backend.routers.jwt.decode',
            return_value={
                'subject': {'username': 'testuser', 'jti': 'validjti'},
                'username': 'testuser',
                'jti': 'validjti',
            },
        ):
            headers = {'authorization': 'Bearer validtoken'}

            # Create a mock WebSocket that will raise an exception during
            # the main loop
            with patch(
                'fastapi.WebSocket.receive_bytes',
                new_callable=AsyncMock,
            ) as mock_receive:
                # First call succeeds,
                # second call raises RuntimeError to trigger outer exception
                mock_receive.side_effect = [
                    b'{"label":"test","key":"test"}' +
                    DELIMITER + b'framedata',
                    RuntimeError('Outer exception in receive_bytes'),
                ]

                with self.client.websocket_connect(
                    '/api/ws/frames', headers=headers,
                ) as websocket:
                    # This should trigger the outer exception handler
                    try:
                        websocket.receive_json()  # This might timeout or raise
                    except Exception:
                        pass  # Connection may be closed due to the exception

    @patch(
        'examples.streaming_web.backend.routers.get_user_data',
        new_callable=AsyncMock,
    )
    def test_websocket_frames_websocket_disconnect_outer(
        self, mock_get_user_data,
    ):
        """
        Tests WebSocketDisconnect handling in the outer
        exception handler of /ws/frames endpoint.
        """
        with patch(
            'examples.streaming_web.backend.routers.jwt.decode',
            return_value={
                'subject': {'username': 'testuser', 'jti': 'validjti'},
                'username': 'testuser',
                'jti': 'validjti',
            },
        ):
            mock_get_user_data.return_value = {'jti_list': ['validjti']}
            headers = {'authorization': 'Bearer validtoken'}

            # Mock the entire while loop to raise WebSocketDisconnect
            with patch(
                'fastapi.WebSocket.receive_bytes', new_callable=AsyncMock,
            ) as mock_receive:
                mock_receive.side_effect = WebSocketDisconnect()

                with self.client.websocket_connect(
                    '/api/ws/frames', headers=headers,
                ):
                    # The WebSocketDisconnect should be handled by the outer
                    # exception handler
                    pass


if __name__ == '__main__':
    unittest.main()


'''
pytest \
    --cov=examples.streaming_web.backend.routers \
    --cov-report=term-missing \
    tests/examples/streaming_web/backend/routers_test.py
'''
