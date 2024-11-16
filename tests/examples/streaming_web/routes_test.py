from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi_limiter import FastAPILimiter

from examples.streaming_web.routes import rate_limiter_index
from examples.streaming_web.routes import rate_limiter_label
from examples.streaming_web.routes import register_routes
from examples.streaming_web.utils import redis_manager


class TestRoutes(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        """
        Set up the test environment before each test.
        """
        self.app = FastAPI()

        # Mock Redis instance with AsyncMock
        self.mock_redis_instance = AsyncMock()
        redis_manager.client = self.mock_redis_instance

        # Register routes
        register_routes(self.app)

        # Initialise rate limiter with the Redis client
        asyncio.run(FastAPILimiter.init(self.mock_redis_instance))

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
        'examples.streaming_web.utils.redis_manager.get_labels',
        new_callable=AsyncMock,
    )
    def test_index(self, mock_get_labels: AsyncMock):
        """
        Test the index route to ensure it renders the correct template
        and context.
        """
        # Mock the get_labels function to return a list of labels
        mock_get_labels.return_value = ['label1', 'label2']

        # Make a GET request to the index route
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn('label1', response.text)

    @patch(
        'examples.streaming_web.utils.redis_manager.get_labels',
        new_callable=AsyncMock,
    )
    def test_label_page_found(self, mock_get_labels: AsyncMock):
        """
        Test the label route to ensure it renders the correct template
        and context.

        Args:
            mock_get_labels (AsyncMock): Mocked get_labels function.
        """
        # Mock the get_labels function to return a list of labels
        mock_get_labels.return_value = ['test_label']

        # Make a GET request to the label route
        response = self.client.get('/label/test_label')
        self.assertEqual(response.status_code, 200)
        mock_get_labels.assert_called_once_with()

    @patch(
        'examples.streaming_web.utils.redis_manager.get_labels',
        new_callable=AsyncMock,
    )
    def test_label_page_not_found(self, mock_get_labels: AsyncMock):
        """
        Test the label route to ensure it returns a 404 error when the label
        is not found.

        Args:
            mock_get_labels (AsyncMock): Mocked get_labels function.
        """
        # Mock the get_labels function to return a different label
        mock_get_labels.return_value = ['another_label']

        # Make a GET request to the label route
        response = self.client.get('/label/test_label')
        self.assertEqual(response.status_code, 404)

    def test_webhook(self):
        """
        Test the webhook route to ensure it returns a successful response.
        """
        # Define the request body
        body = {'event': 'test_event'}

        # Make a POST request to the webhook route
        response = self.client.post('/webhook', json=body)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'status': 'ok'})

    def test_upload_file_successful(self):
        """
        Test the upload route to ensure it returns a successful response.
        """
        # Define the file content
        file_content = b'fake image data'

        # Create a file object with the content
        files = {'file': ('test_image.png', file_content, 'image/png')}
        response = self.client.post('/upload', files=files)
        self.assertEqual(response.status_code, 200)
        self.assertIn('url', response.json())

    def test_upload_file_missing_filename(self):
        """
        Test the upload route to ensure it returns a 422 error
        when the filename is missing.
        """
        files = {'file': ('', b'data', 'image/png')}
        response = self.client.post('/upload', files=files)
        self.assertEqual(response.status_code, 422)


if __name__ == '__main__':
    unittest.main()
