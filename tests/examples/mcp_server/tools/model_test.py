from __future__ import annotations

import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from examples.mcp_server.tools.model import ModelTools


class BaseModelTest(unittest.IsolatedAsyncioTestCase):
    """Base test class providing common setup."""

    async def asyncSetUp(self):
        self.tool = ModelTools()
        self.fetcher_mock = MagicMock()
        self.fetcher_mock.local_dir = 'fake_dir'
        self.fetcher_mock.models = ['yolo', 'bge']
        self.fetcher_mock.get_last_update_time.return_value = 1234
        self.fetcher_mock.request_new_model = MagicMock()


class FetchModelTests(BaseModelTest):
    """Tests for fetch_model."""

    @patch('examples.mcp_server.tools.model.Path')
    async def test_fetch_model_success(self, mock_path):
        """Should mark success when file exists and update cache."""
        mock_file = MagicMock()
        mock_file.exists.return_value = True
        mock_file.stat.return_value = MagicMock(st_size=100, st_mtime=200)
        mock_path.return_value = mock_file

        self.tool._model_fetcher = self.fetcher_mock
        res = await self.tool.fetch_model('yolo')

        self.assertTrue(res['success'])
        self.assertIn('yolo', self.tool._current_models)
        self.assertIn('size', res['model_info'])
        self.assertIn('downloaded', res['message'])

    @patch('examples.mcp_server.tools.model.Path')
    async def test_fetch_model_failure(self, mock_path):
        """Should return failure when file does not exist."""
        mock_file = MagicMock()
        mock_file.exists.return_value = False
        mock_path.return_value = mock_file
        self.tool._model_fetcher = self.fetcher_mock
        res = await self.tool.fetch_model('notfound')
        self.assertFalse(res['success'])
        self.assertIn('Failed', res['message'])

    @patch('examples.mcp_server.tools.model.logging.getLogger')
    async def test_fetch_model_exception_logs(self, mock_logger):
        """Should log error and raise if something fails."""
        logger = mock_logger.return_value
        with patch.object(
            ModelTools,
            '_ensure_model_fetcher',
            AsyncMock(side_effect=RuntimeError('boom')),
        ):
            tool = ModelTools()
            tool.logger = logger
            with self.assertRaises(RuntimeError):
                await tool.fetch_model('oops')
            logger.error.assert_called_once()


class ListAvailableModelsTests(BaseModelTest):
    """Tests for list_available_models."""

    async def test_list_models_success(self):
        """Should list available models successfully."""
        self.tool._model_fetcher = self.fetcher_mock
        result = await self.tool.list_available_models()
        self.assertTrue(result['success'])
        self.assertEqual(result['count'], 2)
        self.assertIn('yolo', result['available_models'])

    @patch('examples.mcp_server.tools.model.logging.getLogger')
    async def test_list_models_exception(self, mock_logger):
        """Should log error when exception occurs."""
        logger = mock_logger.return_value
        with patch.object(
            ModelTools,
            '_ensure_model_fetcher',
            AsyncMock(side_effect=RuntimeError('boom')),
        ):
            tool = ModelTools()
            tool.logger = logger
            with self.assertRaises(RuntimeError):
                await tool.list_available_models()
            logger.error.assert_called_once()


class GetModelInfoTests(BaseModelTest):
    """Tests for get_model_info."""

    @patch('examples.mcp_server.tools.model.Path')
    async def test_get_model_info_found(self, mock_path):
        """Should return model info if file exists."""
        mock_file = MagicMock()
        mock_file.exists.return_value = True
        mock_file.stat.return_value = MagicMock(st_size=123, st_mtime=456)
        mock_path.return_value = mock_file
        self.tool._model_fetcher = self.fetcher_mock
        res = await self.tool.get_model_info('yolo')
        self.assertTrue(res['success'])
        self.assertIn('Local model found', res['message'])

    @patch('examples.mcp_server.tools.model.Path')
    async def test_get_model_info_not_found(self, mock_path):
        """Should return failure when file missing."""
        mock_file = MagicMock()
        mock_file.exists.return_value = False
        mock_path.return_value = mock_file
        self.tool._model_fetcher = self.fetcher_mock
        res = await self.tool.get_model_info('missing')
        self.assertFalse(res['success'])
        self.assertIn('not found', res['message'])

    @patch('examples.mcp_server.tools.model.logging.getLogger')
    async def test_get_model_info_exception(self, mock_logger):
        """Should log and re-raise on exception."""
        logger = mock_logger.return_value
        with patch.object(
            ModelTools,
            '_ensure_model_fetcher',
            AsyncMock(side_effect=RuntimeError('boom')),
        ):
            tool = ModelTools()
            tool.logger = logger
            with self.assertRaises(RuntimeError):
                await tool.get_model_info('fail')
            logger.error.assert_called_once()


class UpdateModelTests(BaseModelTest):
    """Tests for update_model."""

    @patch('examples.mcp_server.tools.model.Path')
    async def test_update_model_success(self, mock_path):
        """Should update model and mark success."""
        mock_file = MagicMock()
        mock_file.exists.return_value = True
        mock_file.stat.return_value = MagicMock(st_size=111, st_mtime=222)
        mock_path.return_value = mock_file
        self.tool._model_fetcher = self.fetcher_mock
        res = await self.tool.update_model('yolo')
        self.assertTrue(res['success'])
        self.assertIn('updated', res['message'])

    @patch('examples.mcp_server.tools.model.Path')
    async def test_update_model_failure(self, mock_path):
        """Should mark failure when file missing."""
        mock_file = MagicMock()
        mock_file.exists.return_value = False
        mock_path.return_value = mock_file
        self.tool._model_fetcher = self.fetcher_mock
        res = await self.tool.update_model('bad')
        self.assertFalse(res['success'])
        self.assertIn('failed', res['message'].lower())

    @patch('examples.mcp_server.tools.model.logging.getLogger')
    async def test_update_model_exception(self, mock_logger):
        """Should log and raise on exception."""
        logger = mock_logger.return_value
        with patch.object(
            ModelTools,
            '_ensure_model_fetcher',
            AsyncMock(side_effect=RuntimeError('boom')),
        ):
            tool = ModelTools()
            tool.logger = logger
            with self.assertRaises(RuntimeError):
                await tool.update_model('fail')
            logger.error.assert_called_once()


class ValidateModelTests(BaseModelTest):
    """Tests for validate_model."""

    @patch('examples.mcp_server.tools.model.Path')
    async def test_validate_model_exists(self, mock_path):
        """Should mark valid if file exists."""
        mock_file = MagicMock()
        mock_file.exists.return_value = True
        mock_path.return_value = mock_file
        self.tool._model_fetcher = self.fetcher_mock
        res = await self.tool.validate_model('yolo')
        self.assertTrue(res['is_valid'])
        self.assertIn('exists', res['message'])

    @patch('examples.mcp_server.tools.model.Path')
    async def test_validate_model_not_exists(self, mock_path):
        """Should mark invalid if file missing."""
        mock_file = MagicMock()
        mock_file.exists.return_value = False
        mock_path.return_value = mock_file
        self.tool._model_fetcher = self.fetcher_mock
        res = await self.tool.validate_model('none')
        self.assertFalse(res['validation_results']['exists'])

    @patch('examples.mcp_server.tools.model.logging.getLogger')
    async def test_validate_model_exception(self, mock_logger):
        """Should log and raise on exception."""
        logger = mock_logger.return_value
        with patch.object(
            ModelTools,
            '_ensure_model_fetcher',
            AsyncMock(side_effect=RuntimeError('boom')),
        ):
            tool = ModelTools()
            tool.logger = logger
            with self.assertRaises(RuntimeError):
                await tool.validate_model('fail')
            logger.error.assert_called_once()


class GetLocalModelsTests(BaseModelTest):
    """Tests for get_local_models."""

    @patch('examples.mcp_server.tools.model.Path')
    async def test_get_local_models_scans_files(self, mock_path):
        """Should return matching files from directories."""
        fake_file = MagicMock()
        fake_file.is_file.return_value = True
        fake_file.suffix = '.pt'
        mock_dir = MagicMock()
        mock_dir.exists.return_value = True
        mock_dir.rglob.return_value = [fake_file]
        mock_path.side_effect = [mock_dir, mock_dir, mock_dir]

        self.tool._model_fetcher = self.fetcher_mock
        res = await self.tool.get_local_models()
        self.assertTrue(res['success'])
        self.assertEqual(res['count'], 3)

    @patch('examples.mcp_server.tools.model.logging.getLogger')
    async def test_get_local_models_exception(self, mock_logger):
        """Should log and raise on error."""
        logger = mock_logger.return_value
        with patch.object(
            ModelTools,
            '_ensure_model_fetcher',
            AsyncMock(side_effect=RuntimeError('boom')),
        ):
            tool = ModelTools()
            tool.logger = logger
            with self.assertRaises(RuntimeError):
                await tool.get_local_models()
            logger.error.assert_called_once()


class CleanupOldModelsTests(BaseModelTest):
    """Tests for cleanup_old_models."""

    async def test_cleanup_old_models_success(self):
        """Should return cleanup summary."""
        self.tool._model_fetcher = self.fetcher_mock
        res = await self.tool.cleanup_old_models()
        self.assertTrue(res['success'])
        self.assertIn('nothing to do', res['message'])

    @patch('examples.mcp_server.tools.model.logging.getLogger')
    async def test_cleanup_old_models_exception(self, mock_logger):
        """Should log and raise on error."""
        logger = mock_logger.return_value
        with patch.object(
            ModelTools,
            '_ensure_model_fetcher',
            AsyncMock(side_effect=RuntimeError('boom')),
        ):
            tool = ModelTools()
            tool.logger = logger
            with self.assertRaises(RuntimeError):
                await tool.cleanup_old_models()
            logger.error.assert_called_once()


class EnsureModelFetcherTests(unittest.IsolatedAsyncioTestCase):
    """Tests for _ensure_model_fetcher."""

    async def test_initialises_model_fetcher_once(self):
        """Should initialise fetcher only if None."""
        tool = ModelTools()
        with patch(
            'examples.mcp_server.tools.model.ModelFetcher',
        ) as mock_fetcher:
            await tool._ensure_model_fetcher()
            mock_fetcher.assert_called_once()
            # call again should not recreate
            await tool._ensure_model_fetcher()
            mock_fetcher.assert_called_once()


if __name__ == '__main__':
    unittest.main()

'''
pytest --cov=examples.mcp_server.tools.model\
    --cov-report=term-missing\
    tests/examples/mcp_server/tools/model_test.py
'''
