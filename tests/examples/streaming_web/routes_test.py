from __future__ import annotations

import unittest
from unittest.mock import AsyncMock
from unittest.mock import patch

import redis
from fastapi import FastAPI
from fastapi import status
from fastapi.testclient import TestClient
from fastapi_limiter import FastAPILimiter

from examples.streaming_web.routes import register_routes


class TestRoutes(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        """
        Sets up the FastAPI app and test client.
        """
        self.app = FastAPI()
        self.app.add_event_handler(
            'startup',
            self.init_redis,
        )
        register_routes(self.app)
        self.client = TestClient(self.app)

    def init_redis(self) -> None:
        redis_instance = redis.StrictRedis(host='localhost', port=6379, db=0)
        FastAPILimiter.init(redis_instance)

    @patch('examples.streaming_web.routes.redis_manager.get_labels', new_callable=AsyncMock)
    def test_index(self, mock_get_labels) -> None:
        """
        Tests the '/' endpoint to check if the labels are fetched and displayed correctly.
        """
        mock_get_labels.return_value = ['label1', 'label2']

        response = self.client.get('/')

        # Check if the response status code is 200
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        # Verify that the labels are included in the response HTML
        self.assertIn('label1', response.text)
        self.assertIn('label2', response.text)

    @patch('examples.streaming_web.routes.redis_manager.get_labels', new_callable=AsyncMock)
    def test_label_page_found(self, mock_get_labels) -> None:
        """
        Tests the '/label/{label}' endpoint to check if a valid label page is returned.
        """
        mock_get_labels.return_value = ['label1', 'label2']

        response = self.client.get('/label/label1')

        # Check if the response status code is 200
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        # Verify that the correct label is included in the response HTML
        self.assertIn('label1', response.text)

    @patch('examples.streaming_web.routes.redis_manager.get_labels', new_callable=AsyncMock)
    def test_label_page_not_found(self, mock_get_labels) -> None:
        """
        Tests the '/label/{label}' endpoint for an invalid label.
        """
        mock_get_labels.return_value = ['label1', 'label2']

        response = self.client.get('/label/invalid_label')

        # Check if the response status code is 404
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    # @patch('examples.streaming_web.routes.redis_manager.get_keys_for_label', new_callable=AsyncMock)
    # def test_websocket_label_stream(self, mock_get_keys_for_label) -> None:
    #     """
    #     Tests the '/ws/label/{label}' websocket endpoint.
    #     """
    #     mock_get_keys_for_label.return_value = ['key1', 'key2']

    #     with self.client.websocket_connect('/ws/label/label1') as websocket:
    #         # Verify that the connection was accepted and the first message is correct
    #         data = websocket.receive_json()
    #         self.assertNotIn('error', data)

    @patch('examples.streaming_web.routes.redis_manager.get_keys_for_label', new_callable=AsyncMock)
    def test_websocket_label_stream_no_keys(self, mock_get_keys_for_label) -> None:
        """
        Tests the '/ws/label/{label}' websocket endpoint with no keys for the label.
        """
        mock_get_keys_for_label.return_value = []

        with self.client.websocket_connect('/ws/label/label1') as websocket:
            # Expecting an error response
            data = websocket.receive_json()
            self.assertIn('error', data)

    @patch('builtins.print', new_callable=AsyncMock)
    def test_webhook(self, mock_print) -> None:
        """
        Tests the '/webhook' endpoint to check if the incoming request is logged and acknowledged.
        """
        body = {'key': 'value'}
        response = self.client.post('/webhook', json=body)

        # Verify that the webhook request was printed
        mock_print.assert_called_once_with(body)
        # Check if the response status code is 200 and contains correct content
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.json(), {'status': 'ok'})

    @patch('builtins.print', new_callable=AsyncMock)
    def test_upload_file(self, mock_print) -> None:
        """
        Tests the '/upload' endpoint to check if the file is uploaded and saved correctly.
        """
        filename = 'test.txt'
        content = b'Some content for testing'
        files = {'file': (filename, content)}

        response = self.client.post('/upload', files=files)

        # Verify if the response status code is 200 and contains the URL of the uploaded file
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('url', response.json())
        self.assertTrue(response.json()['url'].endswith(filename))


if __name__ == '__main__':
    unittest.main()
