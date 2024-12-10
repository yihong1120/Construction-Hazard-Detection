from __future__ import annotations

import datetime
import unittest
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import mock_open
from unittest.mock import patch

from examples.YOLO_server_api.model_files import get_new_model_file
from examples.YOLO_server_api.model_files import update_model_file


class TestModelFilesWithMock(unittest.IsolatedAsyncioTestCase):
    """
    Unit tests for model file operations using virtual files.
    """

    def setUp(self) -> None:
        """
        Set up variables for testing.
        """
        self.valid_model = 'yolo11n'
        self.invalid_model = 'yolo_invalid'
        self.model_file = Path('test_model.pt')
        self.updated_time = datetime.datetime(2023, 1, 1, 0, 0, 0)
        self.destination_path = Path(f'models/pt/best_{self.valid_model}.pt')

    @patch('torch.load')
    @patch('pathlib.Path.rename')
    @patch('pathlib.Path.is_file', return_value=True)
    @patch('pathlib.Path.suffix', new_callable=MagicMock(return_value='.pt'))
    async def test_update_model_file_valid(
        self,
        mock_suffix: MagicMock,
        mock_is_file: MagicMock,
        mock_rename: MagicMock,
        mock_torch_load: MagicMock,
    ) -> None:
        """
        Test updating a valid model file with virtual files.
        """
        mock_torch_load.return_value = True

        await update_model_file(self.valid_model, self.model_file)
        mock_torch_load.assert_called_once_with(self.model_file)
        mock_rename.assert_called_once_with(self.destination_path)

    async def test_update_model_file_invalid_model(self) -> None:
        """
        Test updating with an invalid model key.
        """
        with self.assertRaises(ValueError) as context:
            await update_model_file(self.invalid_model, self.model_file)
        self.assertIn('Invalid model key', str(context.exception))

    @patch('pathlib.Path.is_file', return_value=False)
    async def test_update_model_file_invalid_file(
        self,
        mock_is_file: MagicMock,
    ) -> None:
        """
        Test updating with an invalid file path.
        """
        with self.assertRaises(ValueError) as context:
            await update_model_file(self.valid_model, self.model_file)
        self.assertIn('Invalid file', str(context.exception))

    @patch('torch.load', side_effect=Exception('Invalid model format'))
    @patch('pathlib.Path.is_file', return_value=True)
    @patch('pathlib.Path.suffix', new_callable=MagicMock(return_value='.pt'))
    async def test_update_model_file_invalid_torch_file(
        self,
        mock_suffix: MagicMock,
        mock_is_file: MagicMock,
        mock_torch_load: MagicMock,
    ) -> None:
        """
        Test updating with an invalid `.pt` file.
        """
        with self.assertRaises(ValueError) as context:
            await update_model_file(self.valid_model, self.model_file)
        self.assertIn('Invalid PyTorch model file', str(context.exception))

    @patch('pathlib.Path.is_file', return_value=False)
    async def test_get_new_model_file_no_file(
        self, mock_is_file: MagicMock,
    ) -> None:
        """
        Test retrieving a model file when it doesn't exist.
        """
        result = await get_new_model_file(self.valid_model, self.updated_time)
        self.assertIsNone(result)

    @patch('pathlib.Path.stat')
    @patch('pathlib.Path.is_file', return_value=True)
    @patch(
        'pathlib.Path.open',
        new_callable=mock_open,
        read_data=b'model_data',
    )
    async def test_get_new_model_file_updated(
        self,
        mock_open: MagicMock,
        mock_is_file: MagicMock,
        mock_stat: MagicMock,
    ) -> None:
        """
        Test retrieving an updated model file.
        """
        mock_stat.return_value.st_mtime = (
            self.updated_time + datetime.timedelta(days=1)
        ).timestamp()

        result = await get_new_model_file(self.valid_model, self.updated_time)
        self.assertIsInstance(result, bytes)
        self.assertEqual(result, b'model_data')
        mock_open.assert_called_once_with('rb')  # 修正呼叫模式

    @patch('pathlib.Path.stat')
    @patch('pathlib.Path.is_file', return_value=True)
    async def test_get_new_model_file_not_updated(
        self, mock_is_file: MagicMock, mock_stat: MagicMock,
    ) -> None:
        """
        Test retrieving a model file that has not been updated.
        """
        mock_stat.return_value.st_mtime = self.updated_time.timestamp()

        result = await get_new_model_file(self.valid_model, self.updated_time)
        self.assertIsNone(result)

    @patch(
        'pathlib.Path.open',
        side_effect=FileNotFoundError('[Errno 2] No such file or directory'),
    )
    @patch('pathlib.Path.is_file', return_value=True)
    async def test_get_new_model_file_read_error(
        self, mock_is_file: MagicMock, mock_open: MagicMock,
    ) -> None:
        """
        Test error while reading the model file.
        """
        with self.assertRaises(OSError) as context:
            await get_new_model_file(self.valid_model, self.updated_time)
        self.assertIn('No such file or directory', str(context.exception))


if __name__ == '__main__':
    unittest.main()
