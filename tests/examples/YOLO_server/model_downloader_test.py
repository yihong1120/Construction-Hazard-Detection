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

from examples.YOLO_server.model_downloader import download_model
from examples.YOLO_server.model_downloader import MODELS_DIRECTORY


class TestDownloadModel(unittest.TestCase):
    def setUp(self):
        self.model_name = 'best_yolo11x.pt'
        self.local_file_path = MODELS_DIRECTORY / self.model_name

    @patch('examples.YOLO_server.model_downloader.httpx.AsyncClient')
    @patch('examples.YOLO_server.model_downloader.Path.exists')
    @patch('examples.YOLO_server.model_downloader.Path.stat')
    async def test_download_model_up_to_date(self, mock_stat, mock_exists, mock_httpx):
        mock_exists.return_value = True
        mock_stat.return_value.st_mtime = (
            datetime.now() - timedelta(days=1)
        ).timestamp()
        server_last_modified = datetime.now() - timedelta(days=2)
        mock_response = AsyncMock(status_code=200)
        mock_response.headers = {
            'Last-Modified': server_last_modified.strftime('%a, %d %b %Y %H:%M:%S GMT'),
        }
        mock_httpx.return_value.head.return_value = mock_response

        response = await download_model(self.model_name)
        self.assertEqual(
            response, ({'message': 'Local model is up-to-date'}, 304),
        )

    @patch('examples.YOLO_server.model_downloader.httpx.AsyncClient')
    async def test_download_model_network_error(self, mock_httpx):
        mock_httpx.return_value.head.side_effect = RequestError(
            'Network error',
        )
        with self.assertRaises(HTTPException) as context:
            await download_model(self.model_name)
        self.assertEqual(context.exception.status_code, 500)
        self.assertEqual(
            context.exception.detail,
            'Failed to fetch model information',
        )

    @patch('examples.YOLO_server.model_downloader.Path.resolve')
    async def test_invalid_model_name_path_traversal(self, mock_resolve):
        mock_resolve.side_effect = ValueError('Invalid path')
        with self.assertRaises(HTTPException) as context:
            await download_model(self.model_name)
        self.assertEqual(context.exception.status_code, 400)

    @patch('examples.YOLO_server.model_downloader.httpx.AsyncClient')
    async def test_model_not_found_on_server(self, mock_httpx):
        mock_response = AsyncMock(status_code=404)
        mock_httpx.return_value.head.return_value = mock_response
        with self.assertRaises(HTTPException) as context:
            await download_model(self.model_name)
        self.assertEqual(context.exception.status_code, 404)

    @patch('examples.YOLO_server.model_downloader.httpx.AsyncClient')
    @patch('examples.YOLO_server.model_downloader.Path.exists')
    @patch('examples.YOLO_server.model_downloader.Path.stat')
    async def test_download_model_outdated(self, mock_stat, mock_exists, mock_httpx):
        mock_exists.return_value = True
        mock_stat.return_value.st_mtime = (
            datetime.now() - timedelta(days=2)
        ).timestamp()
        server_last_modified = datetime.now() - timedelta(days=1)
        mock_response = AsyncMock(status_code=200)
        mock_response.headers = {
            'Last-Modified': server_last_modified.strftime('%a, %d %b %Y %H:%M:%S GMT'),
        }
        mock_httpx.return_value.head.return_value = mock_response

        with patch('examples.YOLO_server.model_downloader.FileResponse') as mock_file_response:
            mock_file_response.return_value = FileResponse(
                self.local_file_path,
            )
            response = await download_model(self.model_name)
            self.assertIsInstance(response, FileResponse)
