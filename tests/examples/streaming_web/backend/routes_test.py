from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi_limiter import FastAPILimiter

from examples.streaming_web.backend.routes import rate_limiter_index, rate_limiter_label, register_routes
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

    @patch('examples.streaming_web.backend.utils.RedisManager.get_labels', new_callable=AsyncMock)
    def test_index(self, mock_get_labels: AsyncMock):
        """
        Test the index route to ensure it renders the correct response.
        """
        mock_get_labels.return_value = ['label1', 'label2']
        response = self.client.get('/api/labels')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'labels': ['label1', 'label2']})

    @patch('examples.streaming_web.backend.utils.RedisManager.get_keys_for_label', new_callable=AsyncMock)
    def test_label_page_found(self, mock_get_keys: AsyncMock):
        """
        Test the WebSocket route for an existing label.
        """
        mock_get_keys.return_value = ['stream_frame:label1_Cam0']
        response = self.client.get('/api/labels')
        self.assertEqual(response.status_code, 200)

    def test_webhook(self):
        """
        Test the webhook route to ensure it returns a successful response.
        """
        body = {'event': 'test_event'}
        response = self.client.post('/api/webhook', json=body)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'status': 'ok'})

    def test_upload_file_successful(self):
        """
        Test the upload route to ensure it returns a successful response.
        """
        file_content = b'fake image data'
        files = {'file': ('test_image.png', file_content, 'image/png')}
        response = self.client.post('/api/upload', files=files)
        self.assertEqual(response.status_code, 200)
        self.assertIn('url', response.json())

    def test_upload_file_missing_filename(self):
        """
        Test the upload route to ensure it returns a 422 error
        when the filename is missing.
        """
        # FastAPI automatically raises a 422 for missing file validation
        files = {'file': ('', b'data', 'image/png')}
        response = self.client.post('/api/upload', files=files)
        self.assertEqual(response.status_code, 422)


if __name__ == '__main__':
    unittest.main()
