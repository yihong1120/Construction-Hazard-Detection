from __future__ import annotations

import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

import requests
from flask import Flask

from examples.YOLO_server_api.model_downloader import models_blueprint


class ModelDownloaderTestCase(unittest.TestCase):
    def setUp(self):
        """
        Set up the test environment before each test.
        """
        self.app = Flask(__name__)
        self.app.register_blueprint(models_blueprint)
        self.client = self.app.test_client()

    def tearDown(self):
        """
        Clean up after each test.
        """
        # Delete the Flask app and test client instances
        del self.client
        del self.app

    @patch('examples.YOLO_server_api.model_downloader.requests.head')
    @patch('examples.YOLO_server_api.model_downloader.send_from_directory')
    def test_download_model_up_to_date(
        self,
        mock_send_from_directory,
        mock_requests_head,
    ):
        """
        Test the download_model endpoint when the local model is up-to-date.
        """
        # Prepare mocks
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {
            'Last-Modified': 'Wed, 22 Sep 2023 10:00:00 GMT',
        }
        mock_requests_head.return_value = mock_response

        # Simulate FileNotFoundError to ensure 404 is correctly handled
        mock_send_from_directory.side_effect = FileNotFoundError

        # Mock the stat method to simulate an up-to-date file
        with patch(
            'examples.YOLO_server_api.model_downloader.Path.stat',
        ) as mock_stat:
            # Corresponding to 'Wed, 22 Sep 2023 10:00:00 GMT'
            mock_stat.return_value.st_mtime = 1695386400.0

            # Ensure the path exists and is correct
            with patch(
                'examples.YOLO_server_api.model_downloader.Path.exists',
                return_value=True,
            ):
                response = self.client.get('/models/best_yolov8l.pt')

                self.assertEqual(response.status_code, 304)

    @patch('examples.YOLO_server_api.model_downloader.requests.head')
    @patch('examples.YOLO_server_api.model_downloader.send_from_directory')
    def test_download_model_not_found(
        self,
        mock_send_from_directory,
        mock_requests_head,
    ):
        """
        Test the download_model endpoint when the model is not found.
        """
        mock_send_from_directory.side_effect = FileNotFoundError
        response = self.client.get('/models/non_existent_model.pt')
        self.assertEqual(response.status_code, 404)
        self.assertIn(b'Model not found.', response.data)

    @patch('examples.YOLO_server_api.model_downloader.requests.head')
    @patch('examples.YOLO_server_api.model_downloader.send_from_directory')
    def test_download_model_success(
        self,
        mock_send_from_directory,
        mock_requests_head,
    ):
        """
        Test the download_model endpoint
        when the model is successfully downloaded.
        """
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {
            'Last-Modified': 'Wed, 22 Sep 2023 10:05:00 GMT',
        }
        mock_requests_head.return_value = mock_response

        mock_send_from_directory.return_value = MagicMock()

        with patch(
            'examples.YOLO_server_api.model_downloader.Path.stat',
        ) as mock_stat:
            # Set the timestamp to match the date in the Last-Modified header
            # Corresponding to 'Sat, 01 Jan 2000 00:00:00 GMT'
            mock_stat.return_value.st_mtime = 946684800.0

            # Ensure the path exists and is correct
            with patch(
                'examples.YOLO_server_api.model_downloader.Path.exists',
                return_value=True,
            ):
                # Update the mock to simulate the file being downloaded
                response = self.client.get('/models/best_yolov8l.pt')

                self.assertEqual(response.status_code, 200)
                mock_send_from_directory.assert_called_once()

    @patch('examples.YOLO_server_api.model_downloader.requests.head')
    def test_download_model_request_exception(self, mock_requests_head):
        """
        Test the download_model endpoint when there is a request exception.
        """
        mock_requests_head.side_effect = requests.RequestException
        response = self.client.get('/models/best_yolov8l.pt')
        self.assertEqual(response.status_code, 500)
        self.assertIn(b'Failed to fetch model information.', response.data)

    def test_download_model_invalid_name(self):
        """
        Test the download_model endpoint
        when an invalid model name is provided.
        """
        response = self.client.get('/models/invalid_model_name.pt')
        self.assertEqual(response.status_code, 404)
        self.assertIn(b'Model not found.', response.data)


if __name__ == '__main__':
    unittest.main()
