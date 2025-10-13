from __future__ import annotations

import runpy
import sys
import unittest
from unittest.mock import AsyncMock
from unittest.mock import patch

import examples.mcp_server.main as main


class TestMainTools(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # reset tools mocks
        main.inference_tools = AsyncMock()
        main.hazard_tools = AsyncMock()
        main.violations_tools = AsyncMock()
        main.notify_tools = AsyncMock()
        main.record_tools = AsyncMock()
        main.streaming_tools = AsyncMock()
        main.model_tools = AsyncMock()
        main.utils_tools = AsyncMock()

    # === inference ===
    async def test_inference_detect_frame(self):
        main.inference_tools.detect_frame.return_value = {'ok': True}
        res = await main.inference_detect_frame.fn('img', 0.7, True)
        self.assertTrue(res['ok'])
        main.inference_tools.detect_frame.assert_awaited_once()

    # === hazard ===
    async def test_hazard_detect_violations(self):
        main.hazard_tools.detect_violations.return_value = {'hazard': True}
        res = await main.hazard_detect_violations.fn([], 100, 200)
        self.assertTrue(res['hazard'])

    # === violations ===
    async def test_violations_search(self):
        main.violations_tools.search.return_value = {'total': 1}
        res = await main.violations_search.fn()
        self.assertEqual(res['total'], 1)

    async def test_violations_get(self):
        main.violations_tools.get.return_value = {'id': 1}
        res = await main.violations_get.fn(1)
        self.assertEqual(res['id'], 1)

    async def test_violations_get_image(self):
        main.violations_tools.get_image.return_value = {'url': 'a'}
        res = await main.violations_get_image.fn('a', False)
        self.assertIn('url', res)

    async def test_violations_get_image_by_id(self):
        main.violations_tools.get_image_by_violation_id.return_value = {
            'img': 'x',
        }
        res = await main.violations_get_image_by_id.fn(1)
        self.assertEqual(res['img'], 'x')

    async def test_violations_my_sites(self):
        main.violations_tools.my_sites.return_value = [{'id': 1}]
        res = await main.violations_my_sites.fn()
        self.assertIn('sites', res)

    # === notify ===
    async def test_notify_line_push(self):
        main.notify_tools.line_push.return_value = {'msg': 'ok'}
        res = await main.notify_line_push.fn('r', 'm')
        self.assertEqual(res['msg'], 'ok')

    async def test_notify_broadcast_send(self):
        main.notify_tools.broadcast_send.return_value = {'sent': True}
        res = await main.notify_broadcast_send.fn('m')
        self.assertTrue(res['sent'])

    async def test_notify_telegram_send(self):
        main.notify_tools.telegram_send.return_value = {'t': 1}
        res = await main.notify_telegram_send.fn('id', 'm')
        self.assertEqual(res['t'], 1)

    # === record ===
    async def test_record_send_violation(self):
        main.record_tools.send_violation.return_value = {'ok': True}
        res = await main.record_send_violation.fn('img', [], 'warn')
        self.assertTrue(res['ok'])

    async def test_record_batch_send(self):
        main.record_tools.batch_send_violations.return_value = {'count': 1}
        res = await main.record_batch_send_violations.fn([])
        self.assertEqual(res['count'], 1)

    async def test_record_sync_pending(self):
        main.record_tools.sync_pending_records.return_value = {'done': 1}
        res = await main.record_sync_pending.fn()
        self.assertIn('done', res)

    async def test_record_get_statistics(self):
        main.record_tools.get_upload_statistics.return_value = {'stats': True}
        res = await main.record_get_statistics.fn()
        self.assertTrue(res['stats'])

    # === streaming ===
    async def test_streaming_start_stop_status_capture(self):
        main.streaming_tools.start_detection_stream.return_value = {
            'started': True,
        }
        main.streaming_tools.stop_detection_stream.return_value = {
            'stopped': True,
        }
        main.streaming_tools.get_stream_status.return_value = {'status': 'ok'}
        main.streaming_tools.capture_frame.return_value = {'frame': 'ok'}
        self.assertTrue(
            (await main.streaming_start_detection.fn('url'))['started'],
        )
        self.assertTrue(
            (await main.streaming_stop_detection.fn('id'))['stopped'],
        )
        self.assertEqual(
            (await main.streaming_get_status.fn())['status'],
            'ok',
        )
        self.assertIn(
            'frame',
            (await main.streaming_capture_frame.fn('url')),
        )

    # === model ===
    async def test_model_all(self):
        main.model_tools.fetch_model.return_value = {'fetched': 1}
        main.model_tools.list_available_models.return_value = {'list': []}
        main.model_tools.update_model.return_value = {'updated': 1}
        main.model_tools.get_local_models.return_value = {'local': []}
        self.assertIn('fetched', (await main.model_fetch.fn('a')))
        self.assertIn('list', (await main.model_list_available.fn()))
        self.assertIn('updated', (await main.model_update.fn('a')))
        self.assertIn('local', (await main.model_get_local.fn()))

    # === utils ===
    async def test_utils_all(self):
        main.utils_tools.calculate_polygon_area.return_value = {'area': 1}
        main.utils_tools.point_in_polygon.return_value = {'inside': True}
        main.utils_tools.bbox_intersection.return_value = {'area': 2}
        main.utils_tools.validate_detection_data.return_value = {'ok': True}
        self.assertEqual(
            (await main.utils_calculate_polygon_area.fn([]))['area'],
            1,
        )
        self.assertTrue(
            (await main.utils_point_in_polygon.fn([], []))['inside'],
        )
        self.assertEqual(
            (await main.utils_bbox_intersection.fn([], []))['area'],
            2,
        )
        self.assertTrue(
            (await main.utils_validate_detections.fn([], 1, 1))['ok'],
        )


class TestRunServer(unittest.IsolatedAsyncioTestCase):

    @patch('examples.mcp_server.main.get_transport_config')
    async def test_run_server_stdio(self, mock_cfg):
        mock_cfg.return_value = {'transport': 'stdio'}
        main.mcp.run_stdio_async = AsyncMock()
        await main.run_server()
        main.mcp.run_stdio_async.assert_awaited_once()

    @patch('examples.mcp_server.main.get_transport_config')
    async def test_run_server_sse(self, mock_cfg):
        mock_cfg.return_value = {'transport': 'sse', 'host': 'h', 'port': 1}
        main.mcp.run_sse_async = AsyncMock()
        await main.run_server()
        main.mcp.run_sse_async.assert_awaited_once()

    @patch('examples.mcp_server.main.get_transport_config')
    async def test_run_server_http_variants(self, mock_cfg):
        for t in ('streamable-http', 'http'):
            mock_cfg.return_value = {'transport': t, 'host': 'h', 'port': 1}
            main.mcp.run_http_async = AsyncMock()
            await main.run_server()
            main.mcp.run_http_async.assert_awaited()
            main.mcp.run_http_async.reset_mock()

    @patch('examples.mcp_server.main.get_transport_config')
    async def test_run_server_invalid_transport(self, mock_cfg):
        mock_cfg.return_value = {'transport': 'xyz'}
        with self.assertRaises(ValueError):
            await main.run_server()

    @patch('examples.mcp_server.main.logger')
    @patch(
        'examples.mcp_server.main.get_transport_config',
        side_effect=RuntimeError('bad'),
    )
    async def test_run_server_exception(self, _cfg, mock_logger):
        with self.assertRaises(RuntimeError):
            await main.run_server()
        mock_logger.error.assert_called_once()


class TestMainEntrypoint(unittest.TestCase):
    def test_module_main_guard_executes(self):
        # Ensure __main__ guard runs without starting the server
        def _consume(coro):
            # Close the coroutine to avoid 'never awaited' warnings
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
