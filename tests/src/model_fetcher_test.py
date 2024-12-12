from __future__ import annotations

import datetime
import unittest
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import mock_open
from unittest.mock import patch

import requests

from src.model_fetcher import ModelFetcher
from src.model_fetcher import run_scheduler_loop
from src.model_fetcher import schedule_task


class TestModelFetcher(unittest.TestCase):
    """
    Test suite for the ModelFetcher class and related functions.
    """

    def setUp(self) -> None:
        """
        Set up the test environment.
        """
        self.api_url: str = 'http://test-server/get_new_model'
        self.local_dir: str = 'test_models/pt'
        self.models: list[str] = ['test_model1', 'test_model2']
        self.fetcher: ModelFetcher = ModelFetcher(
            api_url=self.api_url,
            models=self.models,
            local_dir=self.local_dir,
        )

        # Ensure the local directory exists before running tests.
        Path(self.local_dir).mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        """
        Clean up the test environment.
        """
        test_dir = Path(self.local_dir)
        if test_dir.exists():
            for file in test_dir.glob('*'):
                file.unlink()  # Delete all files in the directory
            test_dir.rmdir()  # Delete the directory

    @patch('src.model_fetcher.Path.stat')
    def test_get_last_update_time_existing_file(
        self, mock_stat: MagicMock,
    ) -> None:
        """
        Test fetching the last update time of an existing model file.

        Args:
            mock_stat: Mocked version of Path.stat to control return values.
        """
        mock_stat.return_value.st_mtime = datetime.datetime(
            2022, 1, 1,
        ).timestamp()
        file_path = Path(self.local_dir) / 'best_test_model1.pt'
        file_path.touch()

        last_update_time = self.fetcher.get_last_update_time('test_model1')
        self.assertEqual(last_update_time, '2022-01-01T00:00:00')

    def test_get_last_update_time_nonexistent_file(self) -> None:
        """
        Test fetching the last update time of a nonexistent file.
        """
        last_update_time = self.fetcher.get_last_update_time('test_model1')
        self.assertEqual(last_update_time, '1970-01-01T00:00:00')

    @patch('builtins.open', new_callable=mock_open)
    @patch('src.model_fetcher.Path.mkdir')
    def test_download_and_save_model(
        self, mock_mkdir: MagicMock, mock_open_file: MagicMock,
    ) -> None:
        """
        Test downloading and saving a model file.

        Args:
            mock_mkdir: Mocked Path.mkdir method to
                avoid actual directory creation.
            mock_open_file: Mocked open function to
                avoid writing a real file.
        """
        model_content = b'test content'
        self.fetcher.download_and_save_model('test_model1', model_content)

        file_path = Path(self.local_dir) / 'best_test_model1.pt'
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_open_file.assert_called_once_with(file_path, 'wb')
        mock_open_file().write.assert_called_once_with(model_content)

    @patch('src.model_fetcher.requests.get')
    @patch('src.model_fetcher.ModelFetcher.download_and_save_model')
    def test_request_new_model_success(
        self,
        mock_download_and_save: MagicMock,
        mock_requests_get: MagicMock,
    ) -> None:
        """
        Test requesting and saving a new model from the server.

        Args:
            mock_download_and_save: Mocked method to save the model locally.
            mock_requests_get: Mocked requests.get function to
                simulate server response.
        """
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
    def test_request_new_model_no_update_needed(
        self, mock_requests_get: MagicMock,
    ) -> None:
        """
        Test requesting a model when no update is needed.

        Args:
            mock_requests_get: Mocked requests.get function to
                simulate server response.
        """
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_requests_get.return_value = mock_response

        with self.assertLogs(level='INFO') as log:
            self.fetcher.request_new_model(
                'test_model1', '2022-01-01T00:00:00',
            )
            self.assertIn(
                'Model test_model1 is already up to date.',
                log.output[0],
            )

        mock_requests_get.assert_called_once()
        self.assertFalse(
            Path(self.local_dir).joinpath('best_test_model1.pt').exists(),
        )

    @patch('src.model_fetcher.requests.get')
    def test_request_new_model_server_error(
        self, mock_requests_get: MagicMock,
    ) -> None:
        """
        Test handling a server error during model request.

        Args:
            mock_requests_get: Mocked requests.get function to
                simulate server response.
        """
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_requests_get.return_value = mock_response

        with self.assertLogs(level='ERROR') as log:
            self.fetcher.request_new_model(
                'test_model1', '2022-01-01T00:00:00',
            )
            self.assertIn(
                'Failed to fetch model test_model1. '
                'Server returned status code: 500',
                log.output[0],
            )

    @patch('src.model_fetcher.requests.get')
    def test_request_new_model_timeout(
        self, mock_requests_get: MagicMock,
    ) -> None:
        """
        Test handling a request timeout.

        Args:
            mock_requests_get: Mocked requests.get function to
                simulate a timeout.
        """
        mock_requests_get.side_effect = requests.exceptions.Timeout

        with self.assertLogs(level='ERROR') as log:
            self.fetcher.request_new_model(
                'test_model1', '2022-01-01T00:00:00',
            )
            self.assertIn(
                'Error requesting model test_model1:',
                log.output[0],
            )

    @patch('src.model_fetcher.requests.get')
    def test_request_new_model_connection_error(
        self, mock_requests_get: MagicMock,
    ) -> None:
        """
        Test handling a connection error.

        Args:
            mock_requests_get: Mocked requests.get function to
                simulate a connection error.
        """
        mock_requests_get.side_effect = requests.exceptions.ConnectionError

        with self.assertLogs(level='ERROR') as log:
            self.fetcher.request_new_model(
                'test_model1', '2022-01-01T00:00:00',
            )
            self.assertIn(
                'Error requesting model test_model1:',
                log.output[0],
            )

    @patch('src.model_fetcher.ModelFetcher.request_new_model')
    def test_update_all_models(
        self, mock_request_new_model: MagicMock,
    ) -> None:
        """
        Test updating all models.

        Args:
            mock_request_new_model: Mocked method
                to simulate requesting new models.
        """
        self.fetcher.update_all_models()
        self.assertEqual(mock_request_new_model.call_count, len(self.models))
        mock_request_new_model.assert_any_call(
            'test_model1', '1970-01-01T00:00:00',
        )
        mock_request_new_model.assert_any_call(
            'test_model2', '1970-01-01T00:00:00',
        )

    @patch('src.model_fetcher.ModelFetcher.request_new_model')
    def test_update_all_models_with_exception(
        self, mock_request_new_model: MagicMock,
    ) -> None:
        """
        Test handling exceptions during update_all_models.

        Args:
            mock_request_new_model: Mocked method that raises an exception.
        """
        # Mock request_new_model to raise an exception
        mock_request_new_model.side_effect = Exception('Simulated exception')

        with self.assertLogs(level='ERROR') as log:
            self.fetcher.update_all_models()
            self.assertEqual(
                mock_request_new_model.call_count, len(self.models),
            )
            for model in self.models:
                self.assertTrue(
                    any(
                        f"Failed to update model {model}: Simulated exception"
                        in message
                        for message in log.output
                    ),
                    f"Expected log for model {model} not found.",
                )

    @patch('src.model_fetcher.ModelFetcher.update_all_models')
    def test_schedule_task(self, mock_update_all_models: MagicMock) -> None:
        """
        Test scheduling the update task.

        Args:
            mock_update_all_models: Mocked update_all_models method
                to verify it is called.
        """
        schedule_task()
        mock_update_all_models.assert_called_once()

    @patch('src.model_fetcher.schedule.run_pending')
    @patch(
        'src.model_fetcher.time.sleep',
        side_effect=[None, Exception('Stop loop')],
    )
    def test_schedule_task_loop(
        self, mock_sleep: MagicMock, mock_run_pending: MagicMock,
    ) -> None:
        """
        Test the scheduled task loop.

        Args:
            mock_sleep: Mocked time.sleep to control loop iterations.
            mock_run_pending: Mocked schedule.run_pending
                to simulate task execution.
        """
        # Mock run_pending executes normally on the first call
        # and triggers Exception on the second call to end the loop test.
        mock_run_pending.side_effect = [None, Exception('Stop loop')]

        with self.assertLogs(level='INFO') as log:
            try:
                run_scheduler_loop()  # Run the loop
            except Exception as e:
                self.assertEqual(str(e), 'Stop loop')

            # Verify that the expected log message is present
            self.assertTrue(
                any(
                    'Starting scheduled tasks. Press Ctrl+C to exit.'
                    in message
                    for message in log.output
                ),
                'Expected log message not found.',
            )

        mock_run_pending.assert_called()


if __name__ == '__main__':
    unittest.main()
