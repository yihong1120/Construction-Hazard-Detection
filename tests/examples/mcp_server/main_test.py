from __future__ import annotations

import runpy
import sys
import unittest
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import patch

import examples.mcp_server.main as main


def _transport_config(transport: str) -> dict[str, object]:
    """Build the configuration shape returned by get_transport_config."""
    return {
        'transport': transport,
        'host': 'h',
        'port': 1,
        'path': '/mcp',
        'sse_path': '/sse',
        'debug': False,
    }


class TestMainTools(unittest.IsolatedAsyncioTestCase):
    """Test suite."""

    async def asyncSetUp(self) -> None:
        """Prepare test fixtures."""
        self.inference_tools = AsyncMock(spec=main.InferenceTools)
        self.hazard_tools = AsyncMock(spec=main.HazardTools)
        self.violations_tools = AsyncMock(spec=main.ViolationsTools)
        self.notify_tools = AsyncMock(spec=main.NotifyTools)
        self.record_tools = AsyncMock(spec=main.RecordTools)
        self.streaming_tools = AsyncMock(spec=main.StreamingTools)
        self.model_tools = AsyncMock(spec=main.ModelTools)
        for patcher in (
            patch.object(main, 'inference_tools', self.inference_tools),
            patch.object(main, 'hazard_tools', self.hazard_tools),
            patch.object(main, 'violations_tools', self.violations_tools),
            patch.object(main, 'notify_tools', self.notify_tools),
            patch.object(main, 'record_tools', self.record_tools),
            patch.object(main, 'streaming_tools', self.streaming_tools),
            patch.object(main, 'model_tools', self.model_tools),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    # === inference ===
    async def test_inference_detect_frame(self) -> None:
        """Exercise this test."""
        self.inference_tools.detect_frame.return_value = {'ok': True}
        res = await main.inference_detect_frame('img')
        self.assertTrue(res['ok'])
        self.inference_tools.detect_frame.assert_awaited_once()

    # === hazard ===
    async def test_hazard_detect_violations(self) -> None:
        """Exercise this test."""
        self.hazard_tools.detect_violations.return_value = {'hazard': True}
        res = await main.hazard_detect_violations([])
        self.assertTrue(res['hazard'])

    # === violations ===
    async def test_violations_search(self) -> None:
        """Exercise this test."""
        self.violations_tools.search.return_value = {'total': 1}
        res = await main.violations_search()
        self.assertEqual(res['total'], 1)

    async def test_violations_get(self) -> None:
        """Exercise this test."""
        self.violations_tools.get.return_value = {'id': 1}
        res = await main.violations_get(1)
        self.assertEqual(res['id'], 1)

    async def test_violations_get_image(self) -> None:
        """Exercise this test."""
        self.violations_tools.get_image.return_value = {'url': 'a'}
        res = await main.violations_get_image('a', False)
        self.assertIn('url', res)

    async def test_violations_my_sites(self) -> None:
        """Exercise this test."""
        self.violations_tools.my_sites.return_value = [{'id': 1}]
        res = await main.violations_my_sites()
        self.assertIn('sites', res)

    # === notify ===
    async def test_notify_line_push(self) -> None:
        """Exercise this test."""
        self.notify_tools.line_push.return_value = {'msg': 'ok'}
        res = await main.notify_line_push('r', 'm')
        self.assertEqual(res['msg'], 'ok')

    async def test_notify_broadcast_send(self) -> None:
        """Exercise this test."""
        self.notify_tools.broadcast_send.return_value = {'sent': True}
        res = await main.notify_broadcast_send('m')
        self.assertTrue(res['sent'])

    async def test_notify_messenger_and_wechat_send(self) -> None:
        """Expose optional Messenger and WeChat Work feature handlers."""
        self.notify_tools.messenger_send.return_value = {'messenger': True}
        self.notify_tools.wechat_send.return_value = {'wechat': True}
        self.assertTrue(
            (await main.notify_messenger_send('recipient', 'message'))['messenger'],
        )
        self.assertTrue(
            (await main.notify_wechat_send('user', 'message'))['wechat'],
        )

    async def test_notify_telegram_send(self) -> None:
        """Exercise this test."""
        self.notify_tools.telegram_send.return_value = {'t': 1}
        res = await main.notify_telegram_send('id', 'm')
        self.assertEqual(res['t'], 1)

    # === record ===
    async def test_record_send_violation(self) -> None:
        """Exercise this test."""
        self.record_tools.send_violation.return_value = {'ok': True}
        res = await main.record_send_violation('img', [], 'warn')
        self.assertTrue(res['ok'])

    async def test_record_batch_send(self) -> None:
        """Exercise this test."""
        self.record_tools.batch_send_violations.return_value = {'count': 1}
        res = await main.record_batch_send_violations([])
        self.assertEqual(res['count'], 1)

    async def test_record_sync_pending(self) -> None:
        """Exercise this test."""
        self.record_tools.sync_pending_records.return_value = {'done': 1}
        res = await main.record_sync_pending()
        self.assertIn('done', res)

    async def test_record_get_statistics(self) -> None:
        """Exercise this test."""
        self.record_tools.get_upload_statistics.return_value = {'stats': True}
        res = await main.record_get_statistics()
        self.assertTrue(res['stats'])

    # === streaming ===
    async def test_streaming_start_stop_status_capture(self) -> None:
        """Exercise this test."""
        self.streaming_tools.start_detection_stream.return_value = {
            'started': True,
        }
        self.streaming_tools.stop_detection_stream.return_value = {
            'stopped': True,
        }
        self.streaming_tools.get_stream_status.return_value = {'status': 'ok'}
        self.streaming_tools.capture_frame.return_value = {'frame': 'ok'}
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
        self.model_tools.sync_model.return_value = {'updated': 1}
        self.model_tools.list_available_models.return_value = {'list': []}
        self.model_tools.get_local_models.return_value = {'local': []}
        self.assertIn('updated', (await main.model_sync('a')))
        self.assertIn('list', (await main.model_list_available()))
        self.assertIn('local', (await main.model_get_local()))

    # === utils ===
    async def test_utils_all(self) -> None:
        """Exercise this test."""
        self.assertEqual(
            main.utils_calculate_polygon_area([])['area'], 0.0,
        )
        self.assertTrue(
            not main.utils_point_in_polygon([0, 0], [])['is_inside'],
        )
        self.assertEqual(
            main.utils_bbox_intersection([0, 0, 1, 1], [2, 2, 3, 3])[
                'intersection_area'
            ],
            0.0,
        )
        self.assertTrue(
            main.utils_validate_detections([], 1, 1)['is_valid'],
        )


class TestRunServer(unittest.IsolatedAsyncioTestCase):
    """Test suite."""

    @patch('examples.mcp_server.main.get_transport_config')
    async def test_run_server_stdio(self, mock_cfg: Any) -> None:
        """Exercise this test."""
        mock_cfg.return_value = _transport_config('stdio')
        with patch.object(
            main.mcp,
            'run_stdio_async',
            new=AsyncMock(),
        ) as run_stdio:
            await main.run_server()

        run_stdio.assert_awaited_once()

    @patch('examples.mcp_server.main.get_transport_config')
    async def test_run_server_sse(self, mock_cfg: Any) -> None:
        """Exercise this test."""
        mock_cfg.return_value = _transport_config('sse')
        with patch.object(
            main.mcp,
            'run_sse_async',
            new=AsyncMock(),
        ) as run_sse:
            await main.run_server()

        run_sse.assert_awaited_once()

    @patch('examples.mcp_server.main.get_transport_config')
    async def test_run_server_streamable_http(self, mock_cfg: Any) -> None:
        """Exercise this test."""
        with patch.object(
            main.mcp,
            'run_streamable_http_async',
            new=AsyncMock(),
        ) as run_streamable_http:
            mock_cfg.return_value = _transport_config('streamable-http')
            await main.run_server()

        run_streamable_http.assert_awaited_once()
        self.assertEqual(main.mcp.settings.host, 'h')
        self.assertEqual(main.mcp.settings.port, 1)
        self.assertTrue(main.mcp.settings.stateless_http)

    @patch('examples.mcp_server.main.get_transport_config')
    async def test_run_server_invalid_transport(self, mock_cfg: Any) -> None:
        """Exercise this test."""
        mock_cfg.return_value = _transport_config('xyz')
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
