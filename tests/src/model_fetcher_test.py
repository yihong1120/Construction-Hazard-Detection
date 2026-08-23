from __future__ import annotations

import hashlib
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import requests

from src.model_fetcher import ModelFetcher


class ModelFetcherTests(unittest.TestCase):
    """Provide ModelFetcherTests."""

    def setUp(self) -> None:
        """Perform setUp."""
        self.directory = tempfile.TemporaryDirectory()
        self.fetcher = ModelFetcher(
            api_url='http://test-server/get_new_model',
            models=['yolo26n'],
            local_dir=self.directory.name,
            bearer_token='test-token',
        )

    def tearDown(self) -> None:
        """Perform tearDown."""
        self.directory.cleanup()

    @patch('src.model_fetcher.requests.post')
    def test_posts_authenticated_stream_and_installs_atomically(
        self,
        post: MagicMock,
    ) -> None:
        """Test posts authenticated stream and installs atomically.

        Args:
            post: Value used by this callable.
        """
        content = b'model-content'
        response = MagicMock(status_code=200)
        response.headers = {
            'content-length': str(len(content)),
            'x-model-sha256': hashlib.sha256(content).hexdigest(),
        }
        response.iter_content.return_value = [content]
        post.return_value = response

        updated = self.fetcher.request_new_model(
            'yolo26n',
            '1970-01-01T00:00:00',
        )

        self.assertTrue(updated)
        self.assertEqual(
            (Path(self.directory.name) / 'best_yolo26n.pt').read_bytes(),
            content,
        )
        self.assertFalse(
            (Path(self.directory.name) / 'best_yolo26n.pt.part').exists(),
        )
        post.assert_called_once_with(
            'http://test-server/get_new_model',
            json={
                'model': 'yolo26n',
                'last_update_time': '1970-01-01T00:00:00',
            },
            headers={'Authorization': 'Bearer test-token'},
            timeout=(5, 120),
            stream=True,
        )

    @patch('src.model_fetcher.requests.post')
    def test_no_update_returns_false(self, post: MagicMock) -> None:
        """Test no update returns false.

        Args:
            post: Value used by this callable.
        """
        post.return_value = MagicMock(status_code=204)
        self.assertFalse(
            self.fetcher.request_new_model('yolo26n', '1970-01-01T00:00:00'),
        )

    @patch('src.model_fetcher.requests.post')
    def test_checksum_failure_keeps_existing_model(
        self, post: MagicMock,
    ) -> None:
        """Test checksum failure keeps existing model.

        Args:
            post: Value used by this callable.
        """
        destination = Path(self.directory.name) / 'best_yolo26n.pt'
        destination.write_bytes(b'old')
        response = MagicMock(status_code=200)
        response.headers = {'x-model-sha256': 'not-a-real-checksum'}
        response.iter_content.return_value = [b'new']
        post.return_value = response
        with self.assertRaises(ValueError):
            self.fetcher.request_new_model('yolo26n', '1970-01-01T00:00:00')
        self.assertEqual(destination.read_bytes(), b'old')

    def test_token_is_required(self) -> None:
        """Test token is required."""
        fetcher = ModelFetcher(local_dir=self.directory.name, bearer_token='')
        with self.assertRaises(ValueError):
            fetcher.request_new_model('yolo26n', '1970-01-01T00:00:00')

    @patch('src.model_fetcher.requests.post')
    def test_fetcher_handles_up_to_date_errors_and_force_download(
        self,
        post: MagicMock,
    ) -> None:
        """Fetcher closes every response and retains normal request failures.

        Args:
            post: Mocked synchronous model-download request.
        """
        response = MagicMock(status_code=503)
        response.headers = {}
        post.return_value = response
        self.assertFalse(
            self.fetcher.request_new_model('yolo26n', '2026-08-23T10:00:00'),
        )
        response.close.assert_called_once()

        post.side_effect = requests.exceptions.ConnectionError('offline')
        self.assertFalse(
            self.fetcher.request_new_model('yolo26n', '2026-08-23T10:00:00'),
        )

        response = MagicMock(status_code=204)
        response.headers = {}
        post.side_effect = None
        post.return_value = response
        self.assertFalse(
            self.fetcher.request_new_model(
                'yolo26n',
                '2026-08-23T10:00:00',
                force_download=True,
            ),
        )
        self.assertEqual(
            post.call_args.kwargs['json']['last_update_time'],
            '1970-01-01T00:00:00',
        )

    @patch('src.model_fetcher.requests.post')
    def test_fetcher_rejects_oversize_and_empty_streams(
        self,
        post: MagicMock,
    ) -> None:
        """Fetcher rejects unsafe streams and removes temporary output.

        Args:
            post: Mocked synchronous model-download request.
        """
        self.fetcher.max_download_bytes = 3
        too_large = MagicMock(status_code=200)
        too_large.headers = {'content-length': '4'}
        post.return_value = too_large
        with self.assertRaises(ValueError):
            self.fetcher.request_new_model('yolo26n', '2026-08-23T10:00:00')
        too_large.close.assert_called_once()

        empty = MagicMock(status_code=200)
        empty.headers = {'content-length': '0'}
        empty.iter_content.return_value = [b'', b'']
        post.return_value = empty
        with self.assertRaises(ValueError):
            self.fetcher.request_new_model('yolo26n', '2026-08-23T10:00:00')
        self.assertFalse(
            (Path(self.directory.name) / 'best_yolo26n.pt.part').exists(),
        )

        streamed_too_large = MagicMock(status_code=200)
        streamed_too_large.headers = {'content-length': '0'}
        streamed_too_large.iter_content.return_value = [b'four']
        post.return_value = streamed_too_large
        with self.assertRaises(ValueError):
            self.fetcher.request_new_model('yolo26n', '2026-08-23T10:00:00')

    def test_last_update_time_uses_epoch_then_file_timestamp(self) -> None:
        """Model timestamps use the epoch before a model is installed."""
        destination = Path(self.directory.name) / 'best_yolo26n.pt'
        self.assertEqual(
            self.fetcher.get_last_update_time('yolo26n'),
            datetime(1970, 1, 1).isoformat(),
        )
        destination.write_bytes(b'model')
        self.assertNotEqual(
            self.fetcher.get_last_update_time('yolo26n'),
            datetime(1970, 1, 1).isoformat(),
        )
