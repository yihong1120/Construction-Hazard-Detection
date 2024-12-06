from __future__ import annotations

import datetime
import unittest
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import mock_open
from unittest.mock import patch

from src.model_fetcher import ModelFetcher


class TestModelFetcher(unittest.TestCase):
    @patch('src.model_fetcher.requests.get')
    def setUp(self, mock_requests_get):
        """Set up the test environment."""
        self.api_url = 'http://test-server/get_new_model'
        self.local_dir = 'test_models/pt'
        self.models = ['test_model1', 'test_model2']
        self.fetcher = ModelFetcher(
            api_url=self.api_url,
            models=self.models,
            local_dir=self.local_dir,
        )
        # Mock Path existence and directory creation
        Path(self.local_dir).mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """Clean up after tests."""
        for file in Path(self.local_dir).glob('*'):
            file.unlink()
        Path(self.local_dir).rmdir()

    @patch('src.model_fetcher.Path.stat')
    def test_get_last_update_time_existing_file(self, mock_stat):
        """Test fetching the last update time of an existing file."""
        mock_stat.return_value.st_mtime = datetime.datetime(
            2022, 1, 1,
        ).timestamp()
        file_path = Path(self.local_dir) / 'best_test_model1.pt'
        file_path.touch()

        last_update_time = self.fetcher.get_last_update_time('test_model1')
        self.assertEqual(last_update_time, '2022-01-01T00:00:00')

    def test_get_last_update_time_nonexistent_file(self):
        """Test fetching the last update time of a nonexistent file."""
        last_update_time = self.fetcher.get_last_update_time('test_model1')
        self.assertEqual(last_update_time, '1970-01-01T00:00:00')

    @patch('builtins.open', new_callable=mock_open)
    def test_download_and_save_model(self, mock_open_file):
        """Test downloading and saving a model file."""
        model_content = b'test content'
        self.fetcher.download_and_save_model('test_model1', model_content)

        file_path = Path(self.local_dir) / 'best_test_model1.pt'
        mock_open_file.assert_called_once_with(file_path, 'wb')
        mock_open_file().write.assert_called_once_with(model_content)

    @patch('src.model_fetcher.requests.get')
    @patch('src.model_fetcher.ModelFetcher.download_and_save_model')
    def test_request_new_model_success(
        self,
        mock_download_and_save,
        mock_requests_get,
    ):
        """Test requesting and saving a new model from the server."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'model_file': b'testcontent'.hex(),
        }
        mock_requests_get.return_value = mock_response

        self.fetcher.request_new_model('test_model1', '2022-01-01T00:00:00')
        mock_requests_get.assert_called_once_with(
            self.api_url,
            params={
                'model': 'test_model1',
                'last_update_time': '2022-01-01T00:00:00',
            },
            timeout=10,
        )
        mock_download_and_save.assert_called_once_with(
            'test_model1', b'testcontent',
        )

    @patch('src.model_fetcher.requests.get')
    def test_request_new_model_no_update_needed(self, mock_requests_get):
        """Test requesting a model when no update is needed."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_requests_get.return_value = mock_response

        self.fetcher.request_new_model('test_model1', '2022-01-01T00:00:00')
        mock_requests_get.assert_called_once()
        self.assertFalse(
            Path(self.local_dir).joinpath('best_test_model1.pt').exists(),
        )

    @patch('src.model_fetcher.requests.get')
    def test_request_new_model_server_error(self, mock_requests_get):
        """Test handling a server error during model request."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_requests_get.return_value = mock_response

        # 使用捕獲標準輸出來驗證錯誤訊息
        with patch('sys.stdout', new_callable=MagicMock) as mock_stdout:
            self.fetcher.request_new_model(
                'test_model1', '2022-01-01T00:00:00',
            )

            # 獲取完整的輸出內容
            written_output = ''.join(
                call[0][0] for call in mock_stdout.write.call_args_list
            )

            # 驗證輸出的錯誤訊息是否正確
            expected_message = (
                'Failed to fetch model test_model1. '
                'Server returned status code: 500'
            )
            assert expected_message in written_output

    @patch('src.model_fetcher.ModelFetcher.request_new_model')
    def test_update_all_models(self, mock_request_new_model):
        """Test updating all models."""
        self.fetcher.update_all_models()
        self.assertEqual(mock_request_new_model.call_count, len(self.models))
        mock_request_new_model.assert_any_call(
            'test_model1', '1970-01-01T00:00:00',
        )
        mock_request_new_model.assert_any_call(
            'test_model2', '1970-01-01T00:00:00',
        )


if __name__ == '__main__':
    unittest.main()
