from __future__ import annotations

import datetime
import unittest
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import mock_open
from unittest.mock import patch

from examples.YOLO_server_api.backend.model_files import get_new_model_file
from examples.YOLO_server_api.backend.model_files import update_model_file


class TestModelFilesWithMock(unittest.IsolatedAsyncioTestCase):
    """
    Unit tests for model file operations using virtual files.
    """

    def setUp(self) -> None:
        """
        Set up common variables for testing.
        """
        self.valid_model = 'yolo11n'
        self.invalid_model = 'yolo_invalid'
        self.model_file = Path('test_model.pt')
        self.updated_time = datetime.datetime(2023, 1, 1, 0, 0, 0)
        self.destination_path = Path(f'models/pt/best_{self.valid_model}.pt')

    @patch('torch.jit.load')
    @patch('pathlib.Path.rename')
    @patch('pathlib.Path.is_file', return_value=True)
    @patch('pathlib.Path.suffix', new_callable=MagicMock(return_value='.pt'))
    async def test_update_model_file_valid(
        self,
        mock_suffix: MagicMock,
        mock_is_file: MagicMock,
        mock_rename: MagicMock,
        mock_torch_jit_load: MagicMock,
    ) -> None:
        """
        Test updating a valid model file with virtual files.

        Args:
            mock_suffix (MagicMock):
                Mock for the file suffix check.
            mock_is_file (MagicMock):
                Mock for checking if the file exists.
            mock_rename (MagicMock):
                Mock for renaming the file.
            mock_torch_jit_load (MagicMock):
                Mock for loading the PyTorch model.
        """
        mock_torch_jit_load.return_value = True

        await update_model_file(self.valid_model, self.model_file)

        mock_torch_jit_load.assert_called_once_with(str(self.model_file))
        expected_destination_path = self.destination_path.resolve()
        mock_rename.assert_called_once_with(expected_destination_path)

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

        Args:
            mock_suffix (MagicMock):
                Mock for the file suffix check.
            mock_is_file (MagicMock):
                Mock for checking if the file exists.
            mock_torch_load (MagicMock):
                Mock for loading the PyTorch model.
        """
        with self.assertRaises(ValueError) as context:
            await update_model_file(self.valid_model, self.model_file)
        self.assertIn('Invalid PyTorch model file', str(context.exception))

    @patch('pathlib.Path.is_file', return_value=False)
    async def test_get_new_model_file_no_file(
        self,
        mock_is_file: MagicMock,
    ) -> None:
        """
        Test retrieving a model file when it does not exist.

        Args:
            mock_is_file (MagicMock):
                Mock for checking if the file exists.
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

        Args:
            mock_open (MagicMock):
                Mock for opening the file.
            mock_is_file (MagicMock):
                Mock for checking if the file exists.
            mock_stat (MagicMock):
                Mock for getting file status.
        """
        mock_stat.return_value.st_mtime = (
            self.updated_time + datetime.timedelta(days=1)
        ).timestamp()

        result = await get_new_model_file(self.valid_model, self.updated_time)

        self.assertIsInstance(result, bytes)
        self.assertEqual(result, b'model_data')
        mock_open.assert_called_once_with('rb')

    @patch('pathlib.Path.stat')
    @patch('pathlib.Path.is_file', return_value=True)
    async def test_get_new_model_file_not_updated(
        self,
        mock_is_file: MagicMock,
        mock_stat: MagicMock,
    ) -> None:
        """
        Test retrieving a model file that has not been updated.

        Args:
            mock_is_file (MagicMock):
                Mock for checking if the file exists.
            mock_stat (MagicMock):
                Mock for getting file status.
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
        self,
        mock_is_file: MagicMock,
        mock_open: MagicMock,
    ) -> None:
        """
        Test error while reading the model file.

        Args:
            mock_is_file (MagicMock):
                Mock for checking if the file exists.
            mock_open (MagicMock):
                Mock for opening the file.
        """
        with self.assertRaises(OSError) as context:
            await get_new_model_file(self.valid_model, self.updated_time)
        self.assertIn('No such file or directory', str(context.exception))

    @patch('pathlib.Path.is_file', return_value=True)
    @patch('pathlib.Path.suffix', new_callable=MagicMock(return_value='.pt'))
    @patch('torch.jit.load', return_value=True)
    @patch('pathlib.Path.rename', side_effect=OSError('Cannot rename file'))
    async def test_update_model_file_rename_oserror(
        self,
        mock_rename: MagicMock,
        mock_torch_jit_load: MagicMock,
        mock_suffix: MagicMock,
        mock_is_file: MagicMock,
    ) -> None:
        """
        Test rename operation raising OSError.

        Args:
            mock_rename (MagicMock):
                Mock for renaming the file.
            mock_torch_jit_load (MagicMock):
                Mock for loading the PyTorch model.
            mock_suffix (MagicMock):
                Mock for the file suffix check.
            mock_is_file (MagicMock):
                Mock for checking if the file exists.
        """
        with self.assertRaises(OSError) as context:
            await update_model_file(self.valid_model, self.model_file)
        self.assertIn('Failed to update model file', str(context.exception))

    async def test_get_new_model_file_invalid_model(self) -> None:
        """
        Test invalid model key when retrieving new model file.
        """
        with self.assertRaises(ValueError) as context:
            await get_new_model_file(self.invalid_model, self.updated_time)
        self.assertIn('Invalid model key', str(context.exception))


if __name__ == '__main__':
    unittest.main()

'''
pytest \
    --cov=examples.YOLO_server_api.backend.model_files \
    --cov-report=term-missing \
    tests/examples/YOLO_server_api/backend/model_files_test.py
'''
