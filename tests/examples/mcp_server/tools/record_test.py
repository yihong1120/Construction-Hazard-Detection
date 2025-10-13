from __future__ import annotations

import base64
import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from examples.mcp_server.tools.record import RecordTools


class SendViolationTests(unittest.IsolatedAsyncioTestCase):
    """Tests for send_violation method."""

    async def test_send_violation_success(self):
        """Should send violation successfully and return proper dict."""
        fake_sender = AsyncMock()
        fake_sender.send_violation.return_value = 'abc123'
        with patch(
            'examples.mcp_server.tools.record.ViolationSender',
            return_value=fake_sender,
        ):
            tool = RecordTools()
            img_b64 = base64.b64encode(b'img').decode()
            res = await tool.send_violation(
                image_base64=img_b64,
                detections=[{'bbox': [1, 2, 3, 4]}],
                warning_message='warn',
                timestamp='2025-10-12T12:00:00',
                site_id='siteA',
            )
        fake_sender.send_violation.assert_awaited_once()
        self.assertTrue(res['success'])
        self.assertIn('abc123', res['message'])

    async def test_send_violation_with_data_url_prefix(self):
        """Should handle base64 with data URL prefix correctly."""
        fake_sender = AsyncMock()
        fake_sender.send_violation.return_value = 'id123'
        img = (
            'data:image/png;base64,'
            + base64.b64encode(b'data').decode()
        )
        with patch(
            'examples.mcp_server.tools.record.ViolationSender',
            return_value=fake_sender,
        ):
            tool = RecordTools()
            res = await tool.send_violation(img, [], 'warn')
        self.assertTrue(res['success'])

    async def test_send_violation_demo_mode_saves_file(self):
        """Should save locally and return mock success in demo mode."""
        fake_sender = AsyncMock()
        with (
            patch(
                'examples.mcp_server.tools.record.ViolationSender',
                return_value=fake_sender,
            ),
            patch(
                'examples.mcp_server.tools.record.os.getenv',
                return_value='true',
            ),
            patch('examples.mcp_server.tools.record.Path.mkdir'),
            patch(
                'examples.mcp_server.tools.record.Path.write_bytes',
                return_value=None,
            ),
        ):
            tool = RecordTools()
            img = base64.b64encode(b'fake').decode()
            res = await tool.send_violation(img, [], 'warn')
        self.assertTrue(res['success'])
        self.assertIn('Demo mode', res['message'])

    async def test_send_violation_invalid_timestamp(self):
        """Should handle invalid ISO timestamp gracefully."""
        fake_sender = AsyncMock()
        fake_sender.send_violation.return_value = None
        with patch(
            'examples.mcp_server.tools.record.ViolationSender',
            return_value=fake_sender,
        ):
            tool = RecordTools()
            img = base64.b64encode(b'fake').decode()
            res = await tool.send_violation(
                img,
                [],
                'warn',
                timestamp='bad-time',
            )
        self.assertFalse(res['success'])
        self.assertIn('Failed', res['message'])

    async def test_send_violation_exception_logged(self):
        """Should catch exceptions and return failure dict."""
        with (
            patch(
                'examples.mcp_server.tools.record.ViolationSender',
                side_effect=RuntimeError('boom'),
            ),
            patch(
                'examples.mcp_server.tools.record.logging.getLogger',
            ) as mock_logger,
        ):
            logger = mock_logger.return_value
            tool = RecordTools()
            tool.logger = logger
            img = base64.b64encode(b'fake').decode()
            res = await tool.send_violation(img, [], 'warn')
            self.assertFalse(res['success'])
            logger.error.assert_called_once()


class BatchSendViolationsTests(unittest.IsolatedAsyncioTestCase):
    """Tests for batch_send_violations."""

    async def test_batch_send_all_success(self):
        """Should process all successfully."""
        tool = RecordTools()
        with patch.object(
            tool,
            'send_violation',
            AsyncMock(return_value={'success': True}),
        ):
            payload = {
                'image_base64': 'a',
                'detections': [],
                'warning_message': 'x',
            }
            res = await tool.batch_send_violations([
                payload for _ in range(3)
            ])
        self.assertTrue(res['success'])
        self.assertEqual(res['successful'], 3)

    async def test_batch_send_partial_failures(self):
        """Should handle partial failures."""
        tool = RecordTools()
        results = [{'success': True}, {'success': False}, {'success': True}]

        async def fake_send(**_):
            return results.pop(0)

        with patch.object(tool, 'send_violation', new=fake_send):
            payload = {
                'image_base64': 'a',
                'detections': [],
                'warning_message': 'x',
            }
            res = await tool.batch_send_violations([
                payload for _ in range(3)
            ])
        self.assertFalse(res['success'])
        self.assertEqual(res['failed'], 1)

    async def test_batch_send_raises_and_logs(self):
        """Should log an error and re-raise when send_violation raises."""
        tool = RecordTools()
        with (
            patch.object(
                tool,
                'send_violation',
                AsyncMock(side_effect=RuntimeError('boom')),
            ),
            patch(
                'examples.mcp_server.tools.record.logging.getLogger',
            ) as mock_logger,
        ):
            logger = mock_logger.return_value
            tool.logger = logger
            with self.assertRaises(RuntimeError):
                await tool.batch_send_violations([
                    {
                        'image_base64': 'a',
                        'detections': [],
                        'warning_message': 'x',
                    },
                ])
            logger.error.assert_called_once()


class BackupLocalRecordsTests(unittest.IsolatedAsyncioTestCase):
    """Tests for backup_local_records."""

    async def test_backup_with_custom_path_success(self):
        """Should backup using provided path."""
        fake_sender = AsyncMock()
        fake_sender.backup_to_local.return_value = (True, 5)
        with patch(
            'examples.mcp_server.tools.record.ViolationSender',
            return_value=fake_sender,
        ):
            tool = RecordTools()
            res = await tool.backup_local_records('/tmp/backup.json')
        self.assertTrue(res['success'])
        self.assertEqual(res['records_count'], 5)

    async def test_backup_default_path_failure(self):
        """Should handle backup failure and construct default path."""
        fake_sender = AsyncMock()
        fake_sender.backup_to_local.return_value = (False, 0)
        with patch(
            'examples.mcp_server.tools.record.ViolationSender',
            return_value=fake_sender,
        ):
            tool = RecordTools()
            res = await tool.backup_local_records(None)
        self.assertFalse(res['success'])
        self.assertIn('Backup failed', res['message'])

    async def test_backup_default_path_construction(self):
        """Should construct default backup path using dirname/join
        and loop time.
        """
        fake_sender = AsyncMock()
        fake_sender.backup_to_local.return_value = (True, 1)
        fake_loop = MagicMock()
        fake_loop.time.return_value = 123.0
        with (
            patch(
                'examples.mcp_server.tools.record.ViolationSender',
                return_value=fake_sender,
            ),
            patch(
                'examples.mcp_server.tools.record.os.path.dirname',
                return_value='/base',
            ) as mock_dirname,
            patch(
                'examples.mcp_server.tools.record.os.path.join',
                return_value='/base/static/violations_backup_123.json',
            ) as mock_join,
            patch(
                'examples.mcp_server.tools.record.asyncio.get_event_loop',
                return_value=fake_loop,
            ),
        ):
            tool = RecordTools()
            res = await tool.backup_local_records(None)
        mock_dirname.assert_called()
        mock_join.assert_called()
        self.assertTrue(res['success'])
        self.assertEqual(
            res['backup_path'],
            '/base/static/violations_backup_123.json',
        )

    async def test_compute_default_backup_path_helper(self):
        """Directly exercise the helper to ensure its lines are covered."""
        fake_loop = MagicMock()
        fake_loop.time.return_value = 42.0
        with (
            patch(
                'examples.mcp_server.tools.record.os.path.dirname',
                return_value='/root',
            ),
            patch(
                'examples.mcp_server.tools.record.os.path.join',
                return_value='/root/static/violations_backup_42.json',
            ),
            patch(
                'examples.mcp_server.tools.record.asyncio.get_event_loop',
                return_value=fake_loop,
            ),
        ):
            tool = RecordTools()
            path = tool._compute_default_backup_path()
        self.assertEqual(path, '/root/static/violations_backup_42.json')

    async def test_backup_raises_and_logs(self):
        """Should log and raise on exception."""
        with (
            patch(
                'examples.mcp_server.tools.record.ViolationSender',
                side_effect=RuntimeError('boom'),
            ),
            patch(
                'examples.mcp_server.tools.record.logging.getLogger',
            ) as mock_logger,
        ):
            logger = mock_logger.return_value
            tool = RecordTools()
            tool.logger = logger
            with self.assertRaises(RuntimeError):
                await tool.backup_local_records('/tmp/test.json')
            logger.error.assert_called_once()

    async def test_backup_raises_after_sender_created(self):
        """Should log and raise when backup_to_local raises after init."""
        fake_sender = AsyncMock()
        fake_sender.backup_to_local.side_effect = RuntimeError('ioerr')
        with (
            patch(
                'examples.mcp_server.tools.record.ViolationSender',
                return_value=fake_sender,
            ),
            patch(
                'examples.mcp_server.tools.record.logging.getLogger',
            ) as mock_logger,
        ):
            logger = mock_logger.return_value
            tool = RecordTools()
            tool.logger = logger
            with self.assertRaises(RuntimeError):
                await tool.backup_local_records('/tmp/test.json')
            logger.error.assert_called()


class SyncAndStatsAndCacheTests(unittest.IsolatedAsyncioTestCase):
    """Tests for sync_pending_records,
    get_upload_statistics, clear_local_cache.
    """

    async def test_sync_pending_records_default(self):
        """Should return default not supported message."""
        fake_sender = AsyncMock()
        with patch(
            'examples.mcp_server.tools.record.ViolationSender',
            return_value=fake_sender,
        ):
            tool = RecordTools()
            res = await tool.sync_pending_records()
        self.assertFalse(res['success'])
        self.assertIn('not supported', res['message'])

    async def test_get_upload_statistics_default(self):
        """Should return default placeholder statistics."""
        fake_sender = AsyncMock()
        with patch(
            'examples.mcp_server.tools.record.ViolationSender',
            return_value=fake_sender,
        ):
            tool = RecordTools()
            res = await tool.get_upload_statistics()
        self.assertTrue(res['success'])
        self.assertIn('pending', res['statistics'])

    async def test_clear_local_cache_default(self):
        """Should return default not supported response."""
        fake_sender = AsyncMock()
        with patch(
            'examples.mcp_server.tools.record.ViolationSender',
            return_value=fake_sender,
        ):
            tool = RecordTools()
            res = await tool.clear_local_cache()
        self.assertFalse(res['success'])
        self.assertIn('not supported', res['message'])

    async def test_sync_raises_and_logs(self):
        """Should log and raise on sync exception."""
        with (
            patch(
                'examples.mcp_server.tools.record.ViolationSender',
                side_effect=RuntimeError('boom'),
            ),
            patch(
                'examples.mcp_server.tools.record.logging.getLogger',
            ) as mock_logger,
        ):
            logger = mock_logger.return_value
            tool = RecordTools()
            tool.logger = logger
            with self.assertRaises(RuntimeError):
                await tool.sync_pending_records()
            logger.error.assert_called_once()

    async def test_stats_raises_and_logs(self):
        """Should log and raise on stats exception."""
        with (
            patch(
                'examples.mcp_server.tools.record.ViolationSender',
                side_effect=RuntimeError('boom'),
            ),
            patch(
                'examples.mcp_server.tools.record.logging.getLogger',
            ) as mock_logger,
        ):
            logger = mock_logger.return_value
            tool = RecordTools()
            tool.logger = logger
            with self.assertRaises(RuntimeError):
                await tool.get_upload_statistics()
            logger.error.assert_called_once()

    async def test_clear_cache_raises_and_logs(self):
        """Should log and raise on cache exception."""
        with (
            patch(
                'examples.mcp_server.tools.record.ViolationSender',
                side_effect=RuntimeError('boom'),
            ),
            patch(
                'examples.mcp_server.tools.record.logging.getLogger',
            ) as mock_logger,
        ):
            logger = mock_logger.return_value
            tool = RecordTools()
            tool.logger = logger
            with self.assertRaises(RuntimeError):
                await tool.clear_local_cache()
            logger.error.assert_called_once()


class EnsureViolationSenderTests(unittest.IsolatedAsyncioTestCase):
    """Tests for _ensure_violation_sender method."""

    async def test_initialises_once(self):
        """Should create ViolationSender only once."""
        with patch(
            'examples.mcp_server.tools.record.ViolationSender',
        ) as mock_sender:
            tool = RecordTools()
            await tool._ensure_violation_sender()
            await tool._ensure_violation_sender()
            mock_sender.assert_called_once()


if __name__ == '__main__':
    unittest.main()

'''
pytest --cov=examples.mcp_server.tools.record\
    --cov-report=term-missing\
    tests/examples/mcp_server/tools/record_test.py
'''
