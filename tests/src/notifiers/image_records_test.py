from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from src.notifiers.image_records import ImageRecordStore


class TestImageRecordStore(unittest.TestCase):
    """Verify image-record persistence rejects non-mapping JSON safely."""

    def test_load_warns_and_returns_empty_mapping_for_json_list(self) -> None:
        """A malformed record file cannot break notification delivery."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'records.json'
            path.write_text(json.dumps(['not', 'a mapping']), encoding='utf-8')
            logger = MagicMock()
            store = ImageRecordStore(str(path), logger)

            self.assertEqual(store.load(), {})

        logger.warning.assert_called_once()
