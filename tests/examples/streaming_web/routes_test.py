from __future__ import annotations

import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

from flask import Flask
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from examples.streaming_web.routes import register_routes


class TestRoutes(unittest.TestCase):
    """
    Test suite for the streaming_web routes.
    """

    @patch('redis.Redis')
    def setUp(self, mock_redis) -> None:
        """
        Set up the test environment before each test.

        This method creates a Flask app, sets up rate limiting with Limiter
        using a mocked Redis, registers the routes, and creates a test client.
        """
        self.app = Flask(__name__)

        # Mock Redis instance
        self.mock_redis_instance = mock_redis.return_value
        self.mock_redis_instance.get.return_value = None  # Mock default return

        # Use the mocked Redis instance in Limiter
        self.limiter = Limiter(
            get_remote_address,
            app=self.app,
            storage_uri='memory://',  # Avoid actual Redis
        )

        # Register the routes
        register_routes(self.app, self.limiter, self.mock_redis_instance)

        self.client = self.app.test_client()

    def tearDown(self) -> None:
        """
        Clean up after each test.
        """
        self.mock_redis_instance.reset_mock()

    @patch('examples.streaming_web.routes.get_labels')
    @patch('examples.streaming_web.routes.render_template')
    def test_index(
        self, mock_render_template: MagicMock, mock_get_labels: MagicMock,
    ) -> None:
        """
        Test the index route to ensure it renders
        the index.html template with the correct labels.
        """
        mock_get_labels.return_value = ['label1', 'label2']
        mock_render_template.return_value = 'rendered_template'

        response = self.client.get('/')

        mock_get_labels.assert_called_once_with(self.mock_redis_instance)
        mock_render_template.assert_called_once_with(
            'index.html', labels=['label1', 'label2'],
        )
        self.assertEqual(response.data.decode(), 'rendered_template')

    @patch('examples.streaming_web.routes.get_image_data')
    @patch('examples.streaming_web.routes.render_template')
    def test_label_page(
        self, mock_render_template: MagicMock, mock_get_image_data: MagicMock,
    ) -> None:
        """
        Test the label page route to ensure it renders
        the label.html template with the correct image data.
        """
        mock_get_image_data.return_value = ['image1', 'image2']
        mock_render_template.return_value = 'rendered_template'

        response = self.client.get('/label/test_label')

        mock_get_image_data.assert_called_once_with(
            self.mock_redis_instance, 'test_label',
        )
        mock_render_template.assert_called_once_with(
            'label.html', label='test_label', image_data=['image1', 'image2'],
        )
        self.assertEqual(response.data.decode(), 'rendered_template')

    def test_image_not_found(self) -> None:
        """
        Test the image route to ensure it returns a 404 error
        if the image is not found in Redis.
        """
        self.mock_redis_instance.get.return_value = None

        response = self.client.get('/image/test_label/test_image.png')

        self.assertEqual(response.status_code, 404)

    def test_image_found(self) -> None:
        """
        Test the image route to ensure it returns
        the correct image data when found in Redis.
        """
        self.mock_redis_instance.get.return_value = b'image_data'

        response = self.client.get('/image/test_label/test_image.png')

        self.mock_redis_instance.get.assert_called_once_with(
            'test_label_test_image',
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, b'image_data')
        self.assertEqual(response.headers['Content-Type'], 'image/png')

    @patch('examples.streaming_web.routes.render_template')
    def test_camera_page(self, mock_render_template: MagicMock) -> None:
        """
        Test the camera page route to ensure it renders
        the camera.html template with the correct camera ID and label.
        """
        mock_render_template.return_value = 'rendered_template'

        response = self.client.get('/camera/test_label/test_camera')

        mock_render_template.assert_called_once_with(
            'camera.html', label='test_label', camera_id='test_camera',
        )
        self.assertEqual(response.data.decode(), 'rendered_template')


if __name__ == '__main__':
    unittest.main()
