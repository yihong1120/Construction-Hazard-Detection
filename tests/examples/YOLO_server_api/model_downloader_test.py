from __future__ import annotations

import unittest
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.responses import FileResponse
from httpx import RequestError

from examples.YOLO_server_api.model_downloader import download_model
from examples.YOLO_server_api.model_downloader import MODELS_DIRECTORY


class TestDownloadModel(unittest.TestCase):
    """
    Unit tests for the download_model function.
    """

    def setUp(self) -> None:
        """
        Set up common attributes for tests.
        """
        # Define model name and local file path in MODELS_DIRECTORY
        self.model_name: str = 'best_yolo11x.pt'
        self.local_file_path: Path = MODELS_DIRECTORY / self.model_name

    @patch('examples.YOLO_server_api.model_downloader.httpx.AsyncClient')
    @patch('examples.YOLO_server_api.model_downloader.Path.exists')
    @patch('examples.YOLO_server_api.model_downloader.Path.stat')
    async def test_download_model_up_to_date(
        self,
        mock_stat: unittest.mock.Mock,
        mock_exists: unittest.mock.Mock,
        mock_httpx: unittest.mock.Mock,
    ) -> None:
        """
        Test when the local model file is up-to-date and no download is
        required.

        Args:
            mock_stat: Mock for Path.stat() to simulate file modification time.
            mock_exists: Mock for Path.exists() to simulate file existence.
            mock_httpx: Mock for AsyncClient to simulate HTTP interactions.
        """
        mock_exists.return_value = True
        # Set local modification time to one day ago
        mock_stat.return_value.st_mtime = (
            datetime.now() - timedelta(days=1)
        ).timestamp()

        # Set server's Last-Modified header to two days ago
        server_last_modified = datetime.now() - timedelta(days=2)
        mock_response = AsyncMock(status_code=200)
        mock_response.headers = {
            'Last-Modified': server_last_modified.strftime(
                '%a, %d %b %Y %H:%M:%S GMT',
            ),
        }
        mock_httpx.return_value.head.return_value = mock_response

        # Verify that the function returns a 304 status indicating no need
        # for download
        response = await download_model(self.model_name)
        self.assertEqual(
            response, ({'message': 'Local model is up-to-date'}, 304),
        )

    @patch('examples.YOLO_server_api.model_downloader.httpx.AsyncClient')
    async def test_download_model_network_error(
        self, mock_httpx: unittest.mock.Mock,
    ) -> None:
        """
        Test behaviour when a network error occurs during model download.

        Args:
            mock_httpx: Mock for AsyncClient to simulate a network error.
        """
        # Simulate network error
        mock_httpx.return_value.head.side_effect = RequestError(
            'Network error',
        )

        # Expect a 500 HTTPException to be raised due to network failure
        with self.assertRaises(HTTPException) as context:
            await download_model(self.model_name)
        self.assertEqual(context.exception.status_code, 500)
        self.assertEqual(
            context.exception.detail,
            'Failed to fetch model information',
        )

    @patch('examples.YOLO_server_api.model_downloader.Path.resolve')
    async def test_invalid_model_name_path_traversal(
        self, mock_resolve: unittest.mock.Mock,
    ) -> None:
        """
        Test behaviour when an invalid model name indicating path traversal
        is used.

        Args:
            mock_resolve: Mock for Path.resolve() to simulate invalid path.
        """
        # Simulate ValueError raised for invalid path
        # (e.g., path traversal attempt)
        mock_resolve.side_effect = ValueError('Invalid path')

        # Expect a 400 HTTPException due to invalid model path
        with self.assertRaises(HTTPException) as context:
            await download_model(self.model_name)
        self.assertEqual(context.exception.status_code, 400)

    @patch('examples.YOLO_server_api.model_downloader.httpx.AsyncClient')
    async def test_model_not_found_on_server(
        self, mock_httpx: unittest.mock.Mock,
    ) -> None:
        """
        Test behaviour when the model file is not found on the server.

        Args:
            mock_httpx: Mock for AsyncClient to simulate HTTP interactions.
        """
        # Simulate a 404 response from server indicating model not found
        mock_response = AsyncMock(status_code=404)
        mock_httpx.return_value.head.return_value = mock_response

        # Expect a 404 HTTPException due to missing model on server
        with self.assertRaises(HTTPException) as context:
            await download_model(self.model_name)
        self.assertEqual(context.exception.status_code, 404)

    @patch('examples.YOLO_server_api.model_downloader.httpx.AsyncClient')
    @patch('examples.YOLO_server_api.model_downloader.Path.exists')
    @patch('examples.YOLO_server_api.model_downloader.Path.stat')
    async def test_download_model_outdated(
        self,
        mock_stat: unittest.mock.Mock,
        mock_exists: unittest.mock.Mock,
        mock_httpx: unittest.mock.Mock,
    ) -> None:
        """
        Test when the local model file is outdated and requires re-downloading.

        Args:
            mock_stat: Mock for Path.stat() to simulate file modification time.
            mock_exists: Mock for Path.exists() to simulate file existence.
            mock_httpx: Mock for AsyncClient to simulate HTTP interactions.
        """
        mock_exists.return_value = True
        # Set local modification time to two days ago
        mock_stat.return_value.st_mtime = (
            datetime.now() - timedelta(days=2)
        ).timestamp()

        # Set server's Last-Modified header to one day ago
        server_last_modified = datetime.now() - timedelta(days=1)
        mock_response = AsyncMock(status_code=200)
        mock_response.headers = {
            'Last-Modified': server_last_modified.strftime(
                '%a, %d %b %Y %H:%M:%S GMT',
            ),
        }
        mock_httpx.return_value.head.return_value = mock_response

        # Mock FileResponse to return a file response for the outdated model
        with patch(
            'examples.YOLO_server_api.model_downloader.FileResponse',
        ) as mock_file_response:
            mock_file_response.return_value = FileResponse(
                self.local_file_path,
            )
            response = await download_model(self.model_name)

            # Verify that the response is of type FileResponse,
            # indicating a re-download
            self.assertIsInstance(response, FileResponse)


if __name__ == '__main__':
    unittest.main()
