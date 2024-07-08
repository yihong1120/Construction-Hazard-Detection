from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import call
from unittest.mock import MagicMock
from unittest.mock import mock_open
from unittest.mock import patch

from src.model_fetcher import download_model
from src.model_fetcher import main


class TestModelFetcher(unittest.TestCase):

    @patch('src.model_fetcher.requests.get')
    @patch('src.model_fetcher.Path.exists')
    @patch('src.model_fetcher.Path.mkdir')
    @patch('builtins.open', new_callable=mock_open)
    def test_download_model_exists(
        self, mock_open, mock_mkdir, mock_exists, mock_get,
    ):
        # Mock the Path.exists() to return True
        mock_exists.return_value = True

        # Call the download_model function
        download_model('test_model.pt', 'http://example.com/test_model.pt')

        # Ensure the directory was checked for existence
        mock_exists.assert_called_with()

        # Ensure the model was not downloaded
        mock_get.assert_not_called()

    @patch('src.model_fetcher.requests.get')
    @patch('src.model_fetcher.Path.exists')
    @patch('src.model_fetcher.Path.mkdir')
    @patch('builtins.open', new_callable=mock_open)
    def test_download_model_new(
        self, mock_open, mock_mkdir, mock_exists, mock_get,
    ):
        # Mock the Path.exists() to return False
        mock_exists.return_value = False

        # Mock the requests.get() response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content = lambda chunk_size: [b'data']
        mock_get.return_value = mock_response

        # Call the download_model function
        download_model('test_model.pt', 'http://example.com/test_model.pt')

        # Ensure the directory was created
        mock_mkdir.assert_called_with(parents=True, exist_ok=True)

        # Ensure the model was downloaded
        mock_get.assert_called_once_with(
            'http://example.com/test_model.pt', stream=True,
        )

        # Ensure the file was opened and data was written
        mock_open.assert_called_once_with(
            Path('models/pt/test_model.pt'), 'wb',
        )
        mock_open().write.assert_called_once_with(b'data')

    @patch('src.model_fetcher.requests.get')
    @patch('src.model_fetcher.Path.exists')
    @patch('src.model_fetcher.Path.mkdir')
    @patch('builtins.open', new_callable=mock_open)
    def test_download_model_error(
        self, mock_open, mock_mkdir, mock_exists, mock_get,
    ):
        # Mock the Path.exists() to return False
        mock_exists.return_value = False

        # Mock the requests.get() response
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        # Call the download_model function
        download_model('test_model.pt', 'http://example.com/test_model.pt')

        # Ensure the directory was created
        mock_mkdir.assert_called_with(parents=True, exist_ok=True)

        # Ensure the model was requested
        mock_get.assert_called_once_with(
            'http://example.com/test_model.pt', stream=True,
        )

        # Ensure the file was not opened
        mock_open.assert_not_called()

    @patch('src.model_fetcher.download_model')
    def test_main(self, mock_download_model):
        main()
        calls = [
            call(
                'best_yolov8l.pt',
                'http://changdar-server.mooo.com:28000/models/best_yolov8l.pt',
            ),
            call(
                'best_yolov8x.pt',
                'http://changdar-server.mooo.com:28000/models/best_yolov8x.pt',
            ),
        ]
        mock_download_model.assert_has_calls(calls, any_order=True)


if __name__ == '__main__':
    unittest.main()
