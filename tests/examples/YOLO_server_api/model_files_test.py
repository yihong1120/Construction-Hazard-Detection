from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from examples.YOLO_server_api.model_files import model_file_checksum


class ModelChecksumTests(unittest.TestCase):
    """Model delivery uses content-based checksums."""

    def test_checksum_is_content_based(self) -> None:
        """The checksum matches SHA-256 regardless of the artifact filename."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            path.write_bytes(b'model-bytes')
            self.assertEqual(
                model_file_checksum(path),
                '357e5d6fafa34d27360fec24b4326d35349'
                '05e33c6acdee60198fb078b7b79e5',
            )
