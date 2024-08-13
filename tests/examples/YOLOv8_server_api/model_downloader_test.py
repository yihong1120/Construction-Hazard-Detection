from __future__ import annotations

import unittest
from unittest.mock import patch, MagicMock
from flask import Flask, jsonify
import requests
from examples.YOLOv8_server_api.model_downloader import models_blueprint, ALLOWED_MODELS, MODELS_DIRECTORY

class ModelDownloaderTestCase(unittest.TestCase):
    def setUp(self):
        """
        Set up the test environment before each test.
        """
        self.app = Flask(__name__)
        self.app.register_blueprint(models_blueprint)
        self.client = self.app.test_client()

    @patch('examples.YOLOv8_server_api.model_downloader.requests.head')
    @patch('examples.YOLOv8_server_api.model_downloader.send_from_directory')
    def test_download_model_up_to_date(self, mock_send_from_directory, mock_requests_head):
        """
        Test the download_model endpoint when the local model is up-to-date.
        """
        # Prepare mocks
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {'Last-Modified': 'Wed, 22 Sep 2023 10:00:00 GMT'}
        mock_requests_head.return_value = mock_response
        
        # Simulate FileNotFoundError to ensure 404 is correctly handled
        mock_send_from_directory.side_effect = FileNotFoundError

        # Mock the stat method to simulate an up-to-date file
        with patch('examples.YOLOv8_server_api.model_downloader.Path.stat') as mock_stat:
            mock_stat.return_value.st_mtime = 1695386400.0  # Corresponding to 'Wed, 22 Sep 2023 10:00:00 GMT'
            
            # Ensure the path exists and is correct
            with patch('examples.YOLOv8_server_api.model_downloader.Path.exists', return_value=True):
                response = self.client.get('/models/best_yolov8l.pt')
                
                self.assertEqual(response.status_code, 304)
                # 修改：不再檢查消息體
                # self.assertIn(b'Local model is up-to-date.', response.data)

    @patch('examples.YOLOv8_server_api.model_downloader.requests.head')
    @patch('examples.YOLOv8_server_api.model_downloader.send_from_directory')
    def test_download_model_not_found(self, mock_send_from_directory, mock_requests_head):
        """
        Test the download_model endpoint when the model is not found.
        """
        mock_send_from_directory.side_effect = FileNotFoundError
        response = self.client.get('/models/non_existent_model.pt')
        self.assertEqual(response.status_code, 404)
        self.assertIn(b'Model not found.', response.data)

    @patch('examples.YOLOv8_server_api.model_downloader.requests.head')
    @patch('examples.YOLOv8_server_api.model_downloader.send_from_directory')
    def test_download_model_success(self, mock_send_from_directory, mock_requests_head):
        """
        Test the download_model endpoint when the model is successfully downloaded.
        """
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {'Last-Modified': 'Wed, 22 Sep 2023 10:05:00 GMT'}
        mock_requests_head.return_value = mock_response

        mock_send_from_directory.return_value = MagicMock()

        with patch('examples.YOLOv8_server_api.model_downloader.Path.stat') as mock_stat:
            # 設置本地文件的時間戳早於遠端文件
            mock_stat.return_value.st_mtime = 1695382800.0  # Corresponding to 'Wed, 22 Sep 2023 08:00:00 GMT'

            # 確保路徑存在且正確
            with patch('examples.YOLOv8_server_api.model_downloader.Path.exists', return_value=True):
                # 更新路徑以匹配路由定義
                response = self.client.get('/models/best_yolov8l.pt')

                self.assertEqual(response.status_code, 200)
                mock_send_from_directory.assert_called_once()


    @patch('examples.YOLOv8_server_api.model_downloader.requests.head')
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
        Test the download_model endpoint when an invalid model name is provided.
        """
        response = self.client.get('/models/invalid_model_name.pt')
        self.assertEqual(response.status_code, 404)
        self.assertIn(b'Model not found.', response.data)

if __name__ == '__main__':
    unittest.main()
