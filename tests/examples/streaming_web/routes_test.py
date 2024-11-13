from __future__ import annotations

import asyncio
import base64
import unittest
from unittest.mock import AsyncMock
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter

from examples.streaming_web.routes import register_routes


class TestRoutes(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        """
        Set up the test environment before each test.
        """
        self.app = FastAPI()

        # Mock Redis instance with AsyncMock
        self.mock_redis_instance = AsyncMock()
        self.mock_redis_instance.get.return_value = None
        self.mock_redis_instance.script_load.return_value = 'mock_lua_sha'

        # Register routes
        register_routes(self.app)

        # Mock Redis in the routes
        patcher_redis = patch(
            'examples.streaming_web.redis_client.r', self.mock_redis_instance,
        )
        patcher_redis.start()

        # Initialize FastAPILimiter with mocked Redis
        asyncio.run(FastAPILimiter.init(self.mock_redis_instance))

        # Disable rate limiter by overriding the Depends call
        self.app.dependency_overrides[RateLimiter] = lambda *args, **kwargs: None

        # Initialize the test client
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        """
        Clean up after each test.
        """
        patch.stopall()

    @patch('examples.streaming_web.utils.get_labels', new_callable=AsyncMock)
    def test_index(self, mock_get_labels: AsyncMock):
        """
        Test the index route to ensure it renders the correct template and context.
        """
        mock_get_labels.return_value = ['label1', 'label2']

        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        mock_get_labels.assert_called_once()
        self.assertIn('label1', response.text)

    @patch('examples.streaming_web.utils.get_image_data', new_callable=AsyncMock)
    def test_label_page(self, mock_get_image_data: AsyncMock):
        """
        Test the label page route to ensure it renders the correct template
        with image data.
        """
        mock_get_image_data.return_value = [
            ('image1', 'filename1'), ('image2', 'filename2'),
        ]

        response = self.client.get('/label/test_label')
        self.assertEqual(response.status_code, 200)
        mock_get_image_data.assert_called_once_with(
            self.mock_redis_instance, 'test_label',
        )
        self.assertIn('image1', response.text)
        self.assertIn('image2', response.text)

    def test_image_not_found(self):
        """
        Test the image route when image is not found in Redis,
        expecting a 404 error.
        """
        self.mock_redis_instance.get.return_value = None

        response = self.client.get('/image/test_label/test_image.png')
        self.assertEqual(response.status_code, 404)

    def test_image_found(self):
        """
        Test the image route when image is found in Redis.
        """
        img_data = base64.b64encode(b'test_image_data').decode('utf-8')
        self.mock_redis_instance.get.return_value = img_data

        response = self.client.get('/image/test_label/test_image.png')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['Content-Type'], 'image/png')
        self.assertEqual(response.content, b'test_image_data')

    def test_camera_page(self):
        """
        Test the camera page route to ensure it renders the correct template
        with camera data.
        """
        response = self.client.get('/camera/test_label/test_camera')
        self.assertEqual(response.status_code, 200)
        self.assertIn('test_label', response.text)
        self.assertIn('test_camera', response.text)

    def test_webhook(self):
        """
        Test the webhook endpoint to ensure it returns the correct response.
        """
        body = {'event': 'test_event'}
        response = self.client.post('/webhook', json=body)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'status': 'ok'})


if __name__ == '__main__':
    unittest.main()
