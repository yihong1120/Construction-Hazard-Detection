from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from examples.mcp_server.tools.model import ModelTools


class ModelToolsTests(unittest.IsolatedAsyncioTestCase):

    """Provide ModelToolsTests.
    """

    async def test_sync_reports_updated_local_model(self) -> None:
        """Test sync reports updated local model.
        """
        with tempfile.TemporaryDirectory() as directory:
            fetcher = MagicMock()
            fetcher.local_dir = Path(directory)
            fetcher.get_last_update_time.return_value = '1970-01-01T00:00:00'
            fetcher.request_new_model.side_effect = lambda *_args, **_kwargs: (
                (Path(directory) / 'best_yolo26n.pt').write_bytes(b'model')
                or True
            )
            tool = ModelTools(fetcher=fetcher)

            result = await tool.sync_model('yolo26n', force_download=True)

            self.assertTrue(result['success'])
            self.assertTrue(result['updated'])
            fetcher.request_new_model.assert_called_once_with(
                'yolo26n', '1970-01-01T00:00:00', force_download=True,
            )

    async def test_list_operations_do_not_keep_stale_cache(self) -> None:
        """Test list operations do not keep stale cache.
        """
        with tempfile.TemporaryDirectory() as directory:
            fetcher = MagicMock()
            fetcher.local_dir = Path(directory)
            fetcher.models = ['yolo26n', 'yolo26s']
            (Path(directory) / 'best_yolo26n.pt').write_bytes(b'model')
            tool = ModelTools(fetcher=fetcher)
            self.assertEqual((await tool.list_available_models())['count'], 2)
            local = await tool.get_local_models()
            self.assertEqual(local['count'], 1)
