from __future__ import annotations

import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

from flask import Flask
from flask_limiter import Limiter

from examples.streaming_web.routes import register_routes


class TestRoutes(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.limiter = Limiter(self.app)
        self.r = MagicMock()
        register_routes(self.app, self.limiter, self.r)
        self.client = self.app.test_client()

    @patch('examples.streaming_web.routes.get_labels')
    @patch('examples.streaming_web.routes.render_template')
    def test_index(self, mock_render_template, mock_get_labels):
        """
        Test the index route to ensure it renders the index.html template
        with the correct labels.
        """
        mock_get_labels.return_value = ['label1', 'label2']
        mock_render_template.return_value = 'rendered_template'

        response = self.client.get('/')

        mock_get_labels.assert_called_once_with(self.r)
        mock_render_template.assert_called_once_with(
            'index.html', labels=['label1', 'label2'],
        )
        self.assertEqual(response.data.decode(), 'rendered_template')

    @patch('examples.streaming_web.routes.get_image_data')
    @patch('examples.streaming_web.routes.render_template')
    def test_label_page(self, mock_render_template, mock_get_image_data):
        """
        Test the label page route to ensure it renders the label.html template
        with the correct image data.
        """
        mock_get_image_data.return_value = ['image1', 'image2']
        mock_render_template.return_value = 'rendered_template'

        response = self.client.get('/label/test_label')

        mock_get_image_data.assert_called_once_with(self.r, 'test_label')
        mock_render_template.assert_called_once_with(
            'label.html', label='test_label', image_data=['image1', 'image2'],
        )
        self.assertEqual(response.data.decode(), 'rendered_template')

    def test_image_not_found(self):
        """
        Test the image route to ensure it returns a 404 error if the image
        is not found in Redis.
        """
        self.r.get.return_value = None

        response = self.client.get('/image/test_label/test_image.png')

        self.assertEqual(response.status_code, 404)

    def test_image_found(self):
        """
        Test the image route to ensure it returns the correct image data when
        found in Redis.
        """
        self.r.get.return_value = b'image_data'

        response = self.client.get('/image/test_label/test_image.png')

        self.r.get.assert_called_once_with('test_label_test_image')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, b'image_data')
        self.assertEqual(response.headers['Content-Type'], 'image/png')

    @patch('examples.streaming_web.routes.render_template')
    def test_camera_page(self, mock_render_template):
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
