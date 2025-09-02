from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi_limiter import FastAPILimiter

from examples.auth.database import get_db
from examples.auth.jwt_config import jwt_access
from examples.auth.redis_pool import get_redis_pool
from examples.auth.redis_pool import get_redis_pool_ws
from examples.streaming_web.backend.routers import rate_limiter_index
from examples.streaming_web.backend.routers import rate_limiter_label
from examples.streaming_web.backend.routers import router


class TestRouters(unittest.IsolatedAsyncioTestCase):
    """Test suite for FastAPI routers.

    This suite validates behaviour for the following endpoints:

    - GET /api/labels
    - POST /api/frames
    - WS  /api/ws/labels/{label}
    - WS  /api/ws/stream/{label}/{key}
    - WS  /api/ws/frames

    Attributes:
        app: The in-memory FastAPI application under test.
        fake_redis: Async mock used to stand in for the Redis pool.
        mock_db_session: Async mock simulating the database session.
        client: Test client for driving HTTP and WebSocket requests.
    """

    app: FastAPI
    fake_redis: AsyncMock
    mock_db_session: AsyncMock
    client: TestClient

    def setUp(self) -> None:
        """Initialise the app and mock dependencies.

        This method wires dependency overrides so that networked
        integrations (Redis, DB, rate limiting, and JWT credentials)
        are replaced with safe, deterministic test doubles.

        Returns:
            None
        """
        self.app: FastAPI = FastAPI()
        self.app.include_router(router, prefix='/api')

        async def mock_rate_limiter() -> None:
            return None

        # Override rate-limiters to avoid actual rate limiting during tests
        self.app.dependency_overrides[rate_limiter_index] = mock_rate_limiter
        self.app.dependency_overrides[rate_limiter_label] = mock_rate_limiter

        # Override Redis dependencies with an async mock
        self.fake_redis = AsyncMock()
        self.app.dependency_overrides[get_redis_pool] = lambda: self.fake_redis
        self.app.dependency_overrides[get_redis_pool_ws] = (
            lambda: self.fake_redis
        )

        # Bypass JWT authentication with a mock credentials object
        mock_credentials = SimpleNamespace(subject={'username': 'testuser'})
        self.app.dependency_overrides[jwt_access] = lambda: mock_credentials

        # Mock the database session
        self.mock_db_session = AsyncMock()
        self.app.dependency_overrides[get_db] = lambda: self.mock_db_session

        # Set up default mock user and result for database queries
        self.setup_default_db_mocks()

        # Initialise FastAPILimiter with a mock to avoid Redis dependency
        asyncio.run(FastAPILimiter.init(AsyncMock()))
        self.client = TestClient(self.app)

    def setup_default_db_mocks(self) -> None:
        """Set up default mock user and site for database queries.

        Prepares a default user in the mocked session so tests that do not
        explicitly tailor the user can proceed without additional setup.

        Returns:
            None
        """
        # Create default mock site and user
        mock_site = MagicMock()
        mock_site.name = 'label1'
        mock_user = MagicMock()
        mock_user.role = 'admin'
        mock_user.sites = [mock_site]

        # Create mock result for database query
        mock_result = MagicMock()
        mock_result.scalars.return_value.first.return_value = mock_user

        # Configure the mock session to return this result
        self.mock_db_session.execute.return_value = mock_result

    def tearDown(self) -> None:
        """Clear all dependency overrides after each test.

        Ensures state does not leak between test cases.

        Returns:
            None
        """
        self.app.dependency_overrides.clear()

    # -----------------------------
    # Test GET /api/labels
    # -----------------------------
    @patch(
        'examples.streaming_web.backend.routers.scan_for_labels',
        new_callable=AsyncMock,
    )
    def test_get_labels_success(self, mock_scan_for_labels: AsyncMock) -> None:
        """Test successful retrieval of labels."""
        mock_scan_for_labels.return_value = ['label1', 'label2']

        response = self.client.get('/api/labels')

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'labels': ['label1', 'label2']})

    @patch(
        'examples.streaming_web.backend.routers.scan_for_labels',
        new_callable=AsyncMock,
    )
    def test_get_labels_with_non_admin_user(
        self, mock_scan_for_labels: AsyncMock,
    ) -> None:
        """Test label filtering for non-admin users."""
        # Setup non-admin user
        mock_site = MagicMock()
        mock_site.name = 'label1'
        mock_user = MagicMock()
        mock_user.role = 'user'
        mock_user.sites = [mock_site]

        mock_result = MagicMock()
        mock_result.scalars.return_value.first.return_value = mock_user
        self.mock_db_session.execute.return_value = mock_result

        mock_scan_for_labels.return_value = ['label1', 'label2']

        response = self.client.get('/api/labels')

        self.assertEqual(response.status_code, 200)
        # Non-admin user should only see their allowed labels
        self.assertEqual(response.json(), {'labels': ['label1']})

    @patch(
        'examples.streaming_web.backend.routers.scan_for_labels',
        new_callable=AsyncMock,
    )
    def test_get_labels_error(self, mock_scan_for_labels: AsyncMock) -> None:
        """Test error handling in labels endpoint."""
        mock_scan_for_labels.side_effect = Exception('Redis error')

        response = self.client.get('/api/labels')

        self.assertEqual(response.status_code, 500)
        self.assertIn('Failed to fetch labels', response.json()['detail'])

    def test_get_labels_invalid_token(self) -> None:
        """Test labels endpoint with invalid token."""
        # Override JWT to return invalid credentials
        mock_credentials = SimpleNamespace(subject={})
        self.app.dependency_overrides[jwt_access] = lambda: mock_credentials

        response = self.client.get('/api/labels')

        # The endpoint catches the HTTPException and returns 500
        self.assertEqual(response.status_code, 500)
        self.assertIn('Invalid token', response.json()['detail'])

    # -----------------------------
    # Test POST /api/frames
    # -----------------------------
    @patch(
        'examples.streaming_web.backend.routers.store_to_redis',
        new_callable=AsyncMock,
    )
    def test_post_frame_success(self, mock_store_to_redis: AsyncMock) -> None:
        """Test successful frame upload."""
        mock_store_to_redis.return_value = None

        response = self.client.post(
            '/api/frames',
            data={
                'label': 'test_label',
                'key': 'test_key',
                'warnings_json': '[]',
                'cone_polygons_json': '[]',
                'pole_polygons_json': '[]',
                'detection_items_json': '[]',
                'width': '100',
                'height': '100',
            },
            files={'file': ('test.jpg', b'fake_image_data', 'image/jpeg')},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {'status': 'ok', 'message': 'Frame stored successfully.'},
        )

    @patch(
        'examples.streaming_web.backend.routers.store_to_redis',
        new_callable=AsyncMock,
    )
    def test_post_frame_error(self, mock_store_to_redis: AsyncMock) -> None:
        """Test error handling in frame upload."""
        mock_store_to_redis.side_effect = Exception('Redis error')

        response = self.client.post(
            '/api/frames',
            data={
                'label': 'test_label',
                'key': 'test_key',
                'warnings_json': '[]',
                'cone_polygons_json': '[]',
                'pole_polygons_json': '[]',
                'detection_items_json': '[]',
                'width': '100',
                'height': '100',
            },
            files={'file': ('test.jpg', b'fake_image_data', 'image/jpeg')},
        )

        self.assertEqual(response.status_code, 500)
        self.assertIn('Failed to store frame', response.json()['detail'])

    # -----------------------------
    # Test WebSocket endpoints (basic endpoint existence)
    # -----------------------------
    def test_websocket_endpoints_exist(self) -> None:
        """Test that WebSocket endpoints are properly defined in the router."""
        # Check that the router has the WebSocket routes
        routes = [route for route in self.app.routes]
        websocket_paths = [
            route.path for route in routes
            if hasattr(route, 'path') and route.path.startswith('/api/ws/')
        ]

        expected_paths = [
            '/api/ws/labels/{label}',
            '/api/ws/stream/{label}/{key}',
            '/api/ws/frames',
        ]

        for expected_path in expected_paths:
            self.assertIn(expected_path, websocket_paths)

    # -----------------------------
    # Test WebSocket handler calls
    # -----------------------------
    @patch(
        'examples.streaming_web.backend.routers.handle_label_stream_ws',
        new_callable=AsyncMock,
    )
    async def test_websocket_label_stream_handler_called(
        self, mock_handle: AsyncMock,
    ) -> None:
        """Test that label stream WebSocket calls the handler."""
        from examples.streaming_web.backend.routers import (
            websocket_label_stream,
        )

        mock_websocket = AsyncMock()
        mock_handle.return_value = None

        await websocket_label_stream(
            websocket=mock_websocket,
            label='test_label',
            rds=self.fake_redis,
        )

        mock_handle.assert_called_once()

    @patch(
        'examples.streaming_web.backend.routers.handle_stream_ws',
        new_callable=AsyncMock,
    )
    async def test_websocket_stream_handler_called(
        self, mock_handle: AsyncMock,
    ) -> None:
        """Test that stream WebSocket calls the handler."""
        from examples.streaming_web.backend.routers import websocket_stream

        mock_websocket = AsyncMock()
        mock_handle.return_value = None

        await websocket_stream(
            websocket=mock_websocket,
            label='test_label',
            key='test_key',
            rds=self.fake_redis,
        )

        mock_handle.assert_called_once()

    @patch(
        'examples.streaming_web.backend.routers.handle_frames_ws',
        new_callable=AsyncMock,
    )
    async def test_websocket_frames_handler_called(
        self, mock_handle: AsyncMock,
    ) -> None:
        """Test that frames WebSocket calls the handler."""
        from examples.streaming_web.backend.routers import websocket_frames

        mock_websocket = AsyncMock()
        mock_handle.return_value = None

        await websocket_frames(
            websocket=mock_websocket,
            rds=self.fake_redis,
        )

        mock_handle.assert_called_once()


if __name__ == '__main__':
    unittest.main()


'''
pytest \
    --cov=examples.streaming_web.backend.routers \
    --cov-report=term-missing \
    tests/examples/streaming_web/backend/routers_test_new.py
'''
