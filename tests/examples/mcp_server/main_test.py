from __future__ import annotations

import runpy
import sys
import unittest
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import patch

import examples.mcp_server.main as main


class TestMainTools(unittest.IsolatedAsyncioTestCase):
    """Test suite."""

    async def asyncSetUp(self) -> None:
        # reset tools mocks
        """Prepare test fixtures."""
        main.inference_tools = AsyncMock()
        main.hazard_tools = AsyncMock()
        main.violations_tools = AsyncMock()
        main.notify_tools = AsyncMock()
        main.record_tools = AsyncMock()
        main.streaming_tools = AsyncMock()
        main.model_tools = AsyncMock()
        main.utils_tools = AsyncMock()

    # === inference ===
    async def test_inference_detect_frame(self) -> None:
        """Exercise this test."""
        main.inference_tools.detect_frame.return_value = {'ok': True}
        res = await main.inference_detect_frame('img', 0.7, True)
        self.assertTrue(res['ok'])
        main.inference_tools.detect_frame.assert_awaited_once()

    # === hazard ===
    async def test_hazard_detect_violations(self) -> None:
        """Exercise this test."""
        main.hazard_tools.detect_violations.return_value = {'hazard': True}
        res = await main.hazard_detect_violations([], 100, 200)
        self.assertTrue(res['hazard'])

    # === violations ===
    async def test_violations_search(self) -> None:
        """Exercise this test."""
        main.violations_tools.search.return_value = {'total': 1}
        res = await main.violations_search()
        self.assertEqual(res['total'], 1)

    async def test_violations_get(self) -> None:
        """Exercise this test."""
        main.violations_tools.get.return_value = {'id': 1}
        res = await main.violations_get(1)
        self.assertEqual(res['id'], 1)

    async def test_violations_get_image(self) -> None:
        """Exercise this test."""
        main.violations_tools.get_image.return_value = {'url': 'a'}
        res = await main.violations_get_image('a', False)
        self.assertIn('url', res)

    async def test_violations_get_image_by_id(self) -> None:
        """Exercise this test."""
        main.violations_tools.get_image_by_violation_id.return_value = {
            'img': 'x',
        }
        res = await main.violations_get_image_by_id(1)
        self.assertEqual(res['img'], 'x')

    async def test_violations_my_sites(self) -> None:
        """Exercise this test."""
        main.violations_tools.my_sites.return_value = [{'id': 1}]
        res = await main.violations_my_sites()
        self.assertIn('sites', res)

    # === notify ===
    async def test_notify_line_push(self) -> None:
        """Exercise this test."""
        main.notify_tools.line_push.return_value = {'msg': 'ok'}
        res = await main.notify_line_push('r', 'm')
        self.assertEqual(res['msg'], 'ok')

    async def test_notify_broadcast_send(self) -> None:
        """Exercise this test."""
        main.notify_tools.broadcast_send.return_value = {'sent': True}
        res = await main.notify_broadcast_send('m')
        self.assertTrue(res['sent'])

    async def test_notify_telegram_send(self) -> None:
        """Exercise this test."""
        main.notify_tools.telegram_send.return_value = {'t': 1}
        res = await main.notify_telegram_send('id', 'm')
        self.assertEqual(res['t'], 1)

    # === record ===
    async def test_record_send_violation(self) -> None:
        """Exercise this test."""
        main.record_tools.send_violation.return_value = {'ok': True}
        res = await main.record_send_violation('img', [], 'warn')
        self.assertTrue(res['ok'])

    async def test_record_batch_send(self) -> None:
        """Exercise this test."""
        main.record_tools.batch_send_violations.return_value = {'count': 1}
        res = await main.record_batch_send_violations([])
        self.assertEqual(res['count'], 1)

    async def test_record_sync_pending(self) -> None:
        """Exercise this test."""
        main.record_tools.sync_pending_records.return_value = {'done': 1}
        res = await main.record_sync_pending()
        self.assertIn('done', res)

    async def test_record_get_statistics(self) -> None:
        """Exercise this test."""
        main.record_tools.get_upload_statistics.return_value = {'stats': True}
        res = await main.record_get_statistics()
        self.assertTrue(res['stats'])

    # === streaming ===
    async def test_streaming_start_stop_status_capture(self) -> None:
        """Exercise this test."""
        main.streaming_tools.start_detection_stream.return_value = {
            'started': True,
        }
        main.streaming_tools.stop_detection_stream.return_value = {
            'stopped': True,
        }
        main.streaming_tools.get_stream_status.return_value = {'status': 'ok'}
        main.streaming_tools.capture_frame.return_value = {'frame': 'ok'}
        self.assertTrue(
            (await main.streaming_start_detection('url'))['started'],
        )
        self.assertTrue(
            (await main.streaming_stop_detection('id'))['stopped'],
        )
        self.assertEqual(
            (await main.streaming_get_status())['status'],
            'ok',
        )
        self.assertIn(
            'frame',
            (await main.streaming_capture_frame('url')),
        )

    # === model ===
    async def test_model_all(self) -> None:
        """Exercise this test."""
        main.model_tools.fetch_model.return_value = {'fetched': 1}
        main.model_tools.list_available_models.return_value = {'list': []}
        main.model_tools.update_model.return_value = {'updated': 1}
        main.model_tools.get_local_models.return_value = {'local': []}
        self.assertIn('fetched', (await main.model_fetch('a')))
        self.assertIn('list', (await main.model_list_available()))
        self.assertIn('updated', (await main.model_update('a')))
        self.assertIn('local', (await main.model_get_local()))

    # === utils ===
    async def test_utils_all(self) -> None:
        """Exercise this test."""
        main.utils_tools.calculate_polygon_area.return_value = {'area': 1}
        main.utils_tools.point_in_polygon.return_value = {'inside': True}
        main.utils_tools.bbox_intersection.return_value = {'area': 2}
        main.utils_tools.validate_detection_data.return_value = {'ok': True}
        self.assertEqual(
            (await main.utils_calculate_polygon_area([]))['area'],
            1,
        )
        self.assertTrue(
            (await main.utils_point_in_polygon([], []))['inside'],
        )
        self.assertEqual(
            (await main.utils_bbox_intersection([], []))['area'],
            2,
        )
        self.assertTrue(
            (await main.utils_validate_detections([], 1, 1))['ok'],
        )


class TestRunServer(unittest.IsolatedAsyncioTestCase):
    """Test suite."""

    @patch('examples.mcp_server.main.get_transport_config')
    async def test_run_server_stdio(self, mock_cfg: Any) -> None:
        """Exercise this test."""
        mock_cfg.return_value = {'transport': 'stdio'}
        main.mcp.run_stdio_async = AsyncMock()
        await main.run_server()
        main.mcp.run_stdio_async.assert_awaited_once()

    @patch('examples.mcp_server.main.get_transport_config')
    async def test_run_server_sse(self, mock_cfg: Any) -> None:
        """Exercise this test."""
        mock_cfg.return_value = {'transport': 'sse', 'host': 'h', 'port': 1}
        main.mcp.run_sse_async = AsyncMock()
        await main.run_server()
        main.mcp.run_sse_async.assert_awaited_once()

    @patch('examples.mcp_server.main.get_transport_config')
    async def test_run_server_http_variants(self, mock_cfg: Any) -> None:
        """Exercise this test."""
        for t in ('streamable-http', 'http'):
            mock_cfg.return_value = {'transport': t, 'host': 'h', 'port': 1}
            main.mcp.run_http_async = AsyncMock()
            await main.run_server()
            main.mcp.run_http_async.assert_awaited()
            main.mcp.run_http_async.reset_mock()

    @patch('examples.mcp_server.main.get_transport_config')
    async def test_run_server_invalid_transport(self, mock_cfg: Any) -> None:
        """Exercise this test."""
        mock_cfg.return_value = {'transport': 'xyz'}
        with self.assertRaises(ValueError):
            await main.run_server()

    @patch(
        'examples.mcp_server.main.get_transport_config',
        side_effect=RuntimeError('bad'),
    )
    async def test_run_server_exception(self, _cfg: Any) -> None:
        """Exercise this test."""
        with self.assertRaises(RuntimeError):
            await main.run_server()


class TestMainEntrypoint(unittest.TestCase):
    """Test suite."""

    def test_module_main_guard_executes(self) -> Any:
        # Ensure __main__ guard runs without starting the server
        """Exercise this test."""
        def _consume(coro: Any) -> Any:
            # Close the coroutine to avoid 'never awaited' warnings
            """Support _consume.

            Args:
                coro: Test helper value.
            """
            try:
                coro.close()
            except Exception:
                pass
            return None

        with patch('asyncio.run', side_effect=_consume) as mock_run:
            # Remove already-imported module to avoid runpy warning
            sys.modules.pop(
                'examples.mcp_server.main',
                None,
            )
            runpy.run_module(
                'examples.mcp_server.main',
                run_name='__main__',
            )
            mock_run.assert_called_once()


if __name__ == '__main__':
    unittest.main()

'''
pytest --cov=examples.mcp_server.main \
       --cov-report=term-missing \
       tests/examples/mcp_server/main_test.py
'''
