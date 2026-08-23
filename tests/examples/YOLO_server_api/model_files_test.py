from __future__ import annotations

import datetime
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

from examples.YOLO_server_api import model_files
from examples.YOLO_server_api.model_files import get_new_model_path
from examples.YOLO_server_api.model_files import model_file_checksum
from examples.YOLO_server_api.model_files import model_file_etag
from examples.YOLO_server_api.model_files import update_model_file


class ModelFileTests(unittest.IsolatedAsyncioTestCase):
    """Model delivery exposes paths and streams files; it never buffers
    them."""

    async def test_returns_updated_path_without_opening_file(self) -> None:
        """Test returns updated path without opening file."""
        timestamp = datetime.datetime(2023, 1, 1)
        with (
            patch('pathlib.Path.is_file', return_value=True),
            patch('pathlib.Path.stat') as stat,
        ):
            stat.return_value.st_mtime = (
                timestamp + datetime.timedelta(days=1)
            ).timestamp()
            path = await get_new_model_path('yolo26n', timestamp)
        self.assertIsInstance(path, Path)

    async def test_returns_none_for_missing_or_current_file(self) -> None:
        """Test returns none for missing or current file."""
        timestamp = datetime.datetime(2023, 1, 1)
        with patch('pathlib.Path.is_file', return_value=False):
            self.assertIsNone(await get_new_model_path('yolo26n', timestamp))

    async def test_invalid_model_is_rejected(self) -> None:
        """Test invalid model is rejected."""
        with self.assertRaises(ValueError):
            await get_new_model_path('invalid', datetime.datetime.now())

    @patch('torch.jit.load', return_value=True)
    @patch('pathlib.Path.rename')
    @patch('pathlib.Path.is_file', return_value=True)
    @patch('pathlib.Path.suffix', new_callable=MagicMock(return_value='.pt'))
    async def test_update_model_file_validates_before_atomic_move(
        self,
        _suffix: MagicMock,
        _is_file: MagicMock,
        rename: MagicMock,
        _jit_load: MagicMock,
    ) -> None:
        """Test update model file validates before atomic move.

        Args:
            _suffix: Value used by this callable.
            _is_file: Value used by this callable.
            rename: Value used by this callable.
            _jit_load: Value used by this callable.
        """
        await update_model_file('yolo26n', Path('candidate.pt'))
        self.assertTrue(rename.called)

    async def test_model_file_rejects_invalid_paths_and_model_contents(
        self,
    ) -> None:
        """Model replacement rejects bad files, models, and destinations."""
        with self.assertRaisesRegex(ValueError, 'Invalid file'):
            await update_model_file('yolo26n', Path('candidate.txt'))

        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / 'candidate.pt'
            candidate.write_bytes(b'model')
            with patch.object(
                model_files.torch.jit,
                'load',
                side_effect=RuntimeError('bad'),
            ):
                with self.assertRaisesRegex(ValueError, 'Invalid PyTorch'):
                    await update_model_file('yolo26n', candidate)

        with patch.dict(
            model_files.VALID_MODEL_FILES,
            {'yolo26n': '../../escape.pt'},
        ):
            with self.assertRaisesRegex(ValueError, 'path traversal'):
                model_files._model_destination_path('yolo26n')

    async def test_model_file_rename_failure_and_current_path_return_none(
        self,
    ) -> None:
        """Filesystem write failures and unchanged model paths are explicit."""
        timestamp = datetime.datetime(2023, 1, 1)
        with (
            patch.object(Path, 'is_file', return_value=True),
            patch.object(Path, 'stat') as stat,
        ):
            stat.return_value.st_mtime = timestamp.timestamp()
            self.assertIsNone(await get_new_model_path('yolo26n', timestamp))

        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / 'candidate.pt'
            candidate.write_bytes(b'model')
            with (
                patch.object(
                    model_files.torch.jit,
                    'load',
                    return_value=object(),
                ),
                patch.object(Path, 'rename', side_effect=OSError('read-only')),
            ):
                with self.assertRaisesRegex(OSError, 'Failed to update'):
                    await update_model_file('yolo26n', candidate)


class ModelChecksumTests(unittest.TestCase):
    """Provide ModelChecksumTests."""

    def test_checksum_and_etag_are_content_based(self) -> None:
        """Test checksum and etag are content based."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            path.write_bytes(b'model-bytes')
            checksum = model_file_checksum(path)
            self.assertEqual(len(checksum), 64)
            self.assertEqual(model_file_etag(path), f'"{checksum}"')
