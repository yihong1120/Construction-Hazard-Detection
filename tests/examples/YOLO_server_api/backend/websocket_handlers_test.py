from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import redis
from fastapi import WebSocket
from fastapi import WebSocketDisconnect

from examples.YOLO_server_api.backend import websocket_handlers as ws_mod


class TestGetModelKeyFromWs(unittest.IsolatedAsyncioTestCase):
    """Tests for extracting model key from WebSocket context.

    Covers header, query parameter, and first message fallbacks, and
    validates error responses when the key is missing or invalid.
    """

    async def test_header_model_key(self) -> None:
        """Header value is preferred when present."""
        ws = MagicMock(spec=WebSocket)
        ws.headers = {'x-model-key': 'modelA'}
        ws.query_params = {}
        result = await ws_mod._get_model_key_from_ws(ws, '1.2.3.4', 'alice')
        self.assertEqual(result, 'modelA')

    async def test_query_param_model_key(self) -> None:
        """Query param is used when header is absent."""
        ws = MagicMock(spec=WebSocket)
        ws.headers = {}
        ws.query_params = {'model': 'modelB'}
        result = await ws_mod._get_model_key_from_ws(ws, '1.2.3.4', 'bob')
        self.assertEqual(result, 'modelB')

    async def test_first_message_model_key(self) -> None:
        """First message JSON is used when header and query are absent."""
        ws = MagicMock(spec=WebSocket)
        ws.headers = {}
        ws.query_params = {}
        ws.receive_text = AsyncMock(return_value='{"model_key": "modelC"}')
        result = await ws_mod._get_model_key_from_ws(ws, '1.2.3.4', 'carol')
        self.assertEqual(result, 'modelC')

    async def test_first_message_missing_key_closes(self) -> None:
        """Missing key in first message causes policy close (1008)."""
        ws = MagicMock(spec=WebSocket)
        ws.headers = {}
        ws.query_params = {}
        ws.receive_text = AsyncMock(return_value='{"foo": "bar"}')
        ws.close = AsyncMock()
        result = await ws_mod._get_model_key_from_ws(ws, '1.2.3.4', 'dave')
        self.assertIsNone(result)
        ws.close.assert_awaited_once()
        _, kwargs = ws.close.await_args
        self.assertEqual(kwargs.get('code'), 1008)

    async def test_first_message_invalid_json(self) -> None:
        """Invalid JSON in first message results in policy close (1008)."""
        ws = MagicMock(spec=WebSocket)
        ws.headers = {}
        ws.query_params = {}
        ws.receive_text = AsyncMock(side_effect=ValueError('bad json'))
        ws.close = AsyncMock()
        result = await ws_mod._get_model_key_from_ws(ws, '1.2.3.4', 'erin')
        self.assertIsNone(result)
        ws.close.assert_awaited_once()
        _, kwargs = ws.close.await_args
        self.assertEqual(kwargs.get('code'), 1008)


class TestSendReadyConfig(unittest.IsolatedAsyncioTestCase):
    """Tests for sending initial ready/handshake configuration."""

    async def test_send_ready_config_success(self) -> None:
        """Sends 'ready' payload and reports True on success."""
        ws = MagicMock(spec=WebSocket)
        with patch.object(
            ws_mod,
            '_safe_websocket_send_json',
            new=AsyncMock(return_value=True),
        ) as send_mock:
            ok = await ws_mod._send_ready_config(
                ws, 'modelX', '1.2.3.4', 'alice',
            )
            self.assertTrue(ok)
            send_mock.assert_awaited_once()
            call = getattr(send_mock, 'await_args', None)
            self.assertIsNotNone(call)
            if call is not None:
                payload = call.args[1]
                self.assertEqual(payload.get('status'), 'ready')
                self.assertEqual(payload.get('model'), 'modelX')

    async def test_send_ready_config_failure(self) -> None:
        """Reports False when sending the ready payload fails."""
        ws = MagicMock(spec=WebSocket)
        with patch.object(
            ws_mod,
            '_safe_websocket_send_json',
            new=AsyncMock(return_value=False),
        ):
            ok = await ws_mod._send_ready_config(
                ws, 'modelY', '1.2.3.4', 'bob',
            )
            self.assertFalse(ok)


class TestProcessFrameAndRespond(unittest.IsolatedAsyncioTestCase):
    """Tests for processing a single frame and responding via WebSocket."""

    async def test_process_and_send_success(self) -> None:
        """Successful detection sends response and returns True."""
        ws = MagicMock(spec=WebSocket)
        img = b'bytes'
        datas = {'result': 1}
        with (
                patch.object(
                    ws_mod,
                    'run_detection_from_bytes',
                    new=AsyncMock(return_value=(datas, None)),
                ) as detect_mock,
                patch.object(
                    ws_mod,
                    '_safe_websocket_send_json',
                    new=AsyncMock(return_value=True),
                ) as send_mock,
        ):
            ok = await ws_mod._process_frame_and_respond(
                ws, img, object(), '1.2.3.4', 'eve',
            )
            self.assertTrue(ok)
            detect_mock.assert_awaited_once()
            call = getattr(detect_mock, 'await_args', None)
            self.assertIsNotNone(call)
            if call is not None:
                self.assertIs(
                    call.kwargs.get('semaphore'),
                    ws_mod.WS_INFERENCE_SEMAPHORE,
                )
            send_mock.assert_awaited_once()

    async def test_process_and_send_failure(self) -> None:
        """When send fails, method returns False to stop the loop."""
        ws = MagicMock(spec=WebSocket)
        img = b'bytes'
        with (
                patch.object(
                    ws_mod,
                    'run_detection_from_bytes',
                    new=AsyncMock(return_value=({}, None)),
                ),
                patch.object(
                    ws_mod,
                    '_safe_websocket_send_json',
                    new=AsyncMock(return_value=False),
                ),
        ):
            ok = await ws_mod._process_frame_and_respond(
                ws, img, object(), '1.2.3.4', 'eve',
            )
            self.assertFalse(ok)


class TestPrepareModelAndNotify(unittest.IsolatedAsyncioTestCase):
    """Tests for model preparation and readiness notification."""

    async def test_no_model_key(self) -> None:
        """If no model key, returns None immediately."""
        ws = MagicMock(spec=WebSocket)
        loader: MagicMock = MagicMock()
        with patch.object(
            ws_mod,
            '_get_model_key_from_ws',
            new=AsyncMock(return_value=None),
        ):
            res = await ws_mod._prepare_model_and_notify(
                ws, '1.2.3.4', 'alice', loader,
            )
            self.assertIsNone(res)

    async def test_model_not_found(self) -> None:
        """If model cannot be loaded, close with code 1003 and return None."""
        ws = MagicMock(spec=WebSocket)
        ws.close = AsyncMock()
        loader: MagicMock = MagicMock()
        loader.get_model.return_value = None
        with patch.object(
            ws_mod,
            '_get_model_key_from_ws',
            new=AsyncMock(return_value='modelZ'),
        ):
            res = await ws_mod._prepare_model_and_notify(
                ws, '1.2.3.4', 'bob', loader,
            )
            self.assertIsNone(res)
            ws.close.assert_awaited_once()
            _, kwargs = ws.close.await_args
            self.assertEqual(kwargs.get('code'), 1003)

    async def test_send_ready_config_failed(self) -> None:
        """If ready notification fails, return None without raising."""
        ws = MagicMock(spec=WebSocket)
        loader: MagicMock = MagicMock()
        model_obj: object = object()
        loader.get_model.return_value = model_obj
        with (
                patch.object(
                    ws_mod, '_get_model_key_from_ws',
                    new=AsyncMock(return_value='modelZ'),
                ),
                patch.object(
                    ws_mod, '_send_ready_config',
                    new=AsyncMock(return_value=False),
                ),
        ):
            res = await ws_mod._prepare_model_and_notify(
                ws, '1.2.3.4', 'carol', loader,
            )
            self.assertIsNone(res)

    async def test_success(self) -> None:
        """Happy path returns the loaded model object."""
        ws = MagicMock(spec=WebSocket)
        loader: MagicMock = MagicMock()
        model_obj: object = object()
        loader.get_model.return_value = model_obj
        with (
                patch.object(
                    ws_mod, '_get_model_key_from_ws',
                    new=AsyncMock(return_value='modelZ'),
                ),
                patch.object(
                    ws_mod, '_send_ready_config',
                    new=AsyncMock(return_value=True),
                ),
        ):
            res = await ws_mod._prepare_model_and_notify(
                ws, '1.2.3.4', 'dave', loader,
            )
            self.assertIs(res, model_obj)


class TestDetectLoop(unittest.IsolatedAsyncioTestCase):
    """Tests for the detection loop that streams and processes frames."""

    async def test_timeout_immediately(self) -> None:
        """If timed out immediately, returns count 0 and stops."""
        ws = MagicMock(spec=WebSocket)
        with patch.object(
            ws_mod,
            'check_and_maybe_close_on_timeout',
            new=AsyncMock(return_value=True),
        ):
            count = await ws_mod._detect_loop(
                ws, 0.0, object(), '1.2.3.4', 'alice',
            )
            self.assertEqual(count, 0)

    async def test_receive_none_breaks(self) -> None:
        """Loop stops when no more frames are received."""
        ws = MagicMock(spec=WebSocket)
        with (
                patch.object(
                    ws_mod, 'check_and_maybe_close_on_timeout',
                    new=AsyncMock(side_effect=[False, True]),
                ),
                patch.object(
                    ws_mod, '_safe_websocket_receive_bytes',
                    new=AsyncMock(return_value=None),
                ),
        ):
            count = await ws_mod._detect_loop(
                ws, 0.0, object(), '1.2.3.4', 'bob',
            )
            self.assertEqual(count, 0)

    async def test_process_two_frames_then_none(self) -> None:
        """Processes two frames successfully, then stops on None."""
        ws = MagicMock(spec=WebSocket)
        timeouts = [False, False, False, True]
        with (
                patch.object(
                    ws_mod, 'check_and_maybe_close_on_timeout',
                    new=AsyncMock(side_effect=timeouts),
                ),
                patch.object(
                    ws_mod, '_safe_websocket_receive_bytes',
                    new=AsyncMock(side_effect=[b'a', b'b', None]),
                ),
                patch.object(
                    ws_mod, '_process_frame_and_respond',
                    new=AsyncMock(return_value=True),
                ),
                patch.object(ws_mod, 'log_every_n', new=MagicMock()),
        ):
            count = await ws_mod._detect_loop(
                ws, 0.0, object(), '1.2.3.4', 'carol',
            )
            self.assertEqual(count, 2)

    async def test_process_frame_failure_breaks(self) -> None:
        """Stops the loop when processing a frame fails."""
        ws = MagicMock(spec=WebSocket)
        with (
                patch.object(
                    ws_mod, 'check_and_maybe_close_on_timeout',
                    new=AsyncMock(side_effect=[False, True]),
                ),
                patch.object(
                    ws_mod, '_safe_websocket_receive_bytes',
                    new=AsyncMock(return_value=b'a'),
                ),
                patch.object(
                    ws_mod, '_process_frame_and_respond',
                    new=AsyncMock(return_value=False),
                ),
        ):
            count = await ws_mod._detect_loop(
                ws, 0.0, object(), '1.2.3.4', 'dave',
            )
            self.assertEqual(count, 1)


class TestHandleWebsocketDetect(unittest.IsolatedAsyncioTestCase):
    """Tests for the top-level WebSocket handler control flow."""

    def _make_ws(self) -> WebSocket:
        """Create a mock WebSocket with accept/close and client host."""
        ws = MagicMock(spec=WebSocket)
        ws.accept = AsyncMock()
        ws.close = AsyncMock()
        ws.client = SimpleNamespace(host='9.9.9.9')
        return ws

    async def test_auth_failed(self) -> None:
        """When authentication fails, handler returns early."""
        ws = self._make_ws()
        rds: MagicMock = MagicMock(spec=redis.Redis)
        settings: MagicMock = MagicMock()
        loader: MagicMock = MagicMock()
        with patch.object(
            ws_mod,
            'authenticate_ws_or_none',
            new=AsyncMock(return_value=(None, None)),
        ):
            await ws_mod.handle_websocket_detect(ws, rds, settings, loader)

    async def test_successful_flow(self) -> None:
        """Happy path: prepare model and run detection loop."""
        ws = self._make_ws()
        rds: MagicMock = MagicMock(spec=redis.Redis)
        settings: MagicMock = MagicMock()
        loader: MagicMock = MagicMock()
        model_obj: object = object()
        with (
                patch.object(
                    ws_mod, 'authenticate_ws_or_none',
                    new=AsyncMock(return_value=('alice', 'jti')),
                ),
                patch.object(
                    ws_mod,
                    '_prepare_model_and_notify',
                    new=AsyncMock(return_value=model_obj),
                ) as prep_mock,
                patch.object(
                    ws_mod,
                    '_detect_loop',
                    new=AsyncMock(return_value=3),
                ) as loop_mock,
        ):
            await ws_mod.handle_websocket_detect(ws, rds, settings, loader)
            prep_mock.assert_awaited_once()
            loop_mock.assert_awaited_once()

    async def test_websocket_disconnect_handled(self) -> None:
        """WebSocketDisconnect is handled without raising."""
        ws = self._make_ws()
        rds: MagicMock = MagicMock(spec=redis.Redis)
        settings: MagicMock = MagicMock()
        loader: MagicMock = MagicMock()
        model_obj: object = object()
        with (
                patch.object(
                    ws_mod, 'authenticate_ws_or_none',
                    new=AsyncMock(return_value=('bob', 'jti')),
                ),
                patch.object(
                    ws_mod, '_prepare_model_and_notify',
                    new=AsyncMock(return_value=model_obj),
                ),
                patch.object(
                    ws_mod,
                    '_detect_loop',
                    new=AsyncMock(side_effect=WebSocketDisconnect),
                ),
        ):
            await ws_mod.handle_websocket_detect(ws, rds, settings, loader)

    async def test_generic_exception_closes(self) -> None:
        """Generic exceptions close the socket and do not propagate."""
        ws = self._make_ws()
        rds: MagicMock = MagicMock(spec=redis.Redis)
        settings: MagicMock = MagicMock()
        loader: MagicMock = MagicMock()
        model_obj: object = object()
        with (
                patch.object(
                    ws_mod, 'authenticate_ws_or_none',
                    new=AsyncMock(return_value=('carol', 'jti')),
                ),
                patch.object(
                    ws_mod, '_prepare_model_and_notify',
                    new=AsyncMock(return_value=model_obj),
                ),
                patch.object(
                    ws_mod,
                    '_detect_loop',
                    new=AsyncMock(side_effect=RuntimeError('boom')),
                ),
        ):
            await ws_mod.handle_websocket_detect(ws, rds, settings, loader)
            ws.close.assert_awaited()

    async def test_prepare_returns_none_early_return(self) -> None:
        """If preparation returns None, detection loop is not invoked."""
        ws = self._make_ws()
        rds: MagicMock = MagicMock(spec=redis.Redis)
        settings: MagicMock = MagicMock()
        loader: MagicMock = MagicMock()
        with (
                patch.object(
                    ws_mod, 'authenticate_ws_or_none',
                    new=AsyncMock(return_value=('dave', 'jti')),
                ),
                patch.object(
                    ws_mod, '_prepare_model_and_notify',
                    new=AsyncMock(return_value=None),
                ),
                patch.object(
                    ws_mod, '_detect_loop', new=AsyncMock(),
                ) as loop_mock,
        ):
            await ws_mod.handle_websocket_detect(ws, rds, settings, loader)
            loop_mock.assert_not_awaited()

    async def test_generic_exception_close_raises_is_swallowed(self) -> None:
        """Close raising inside generic-exception path is swallowed."""
        ws = self._make_ws()
        # Make close raise to hit inner except Exception: pass
        ws.close = AsyncMock(side_effect=RuntimeError('close failed'))
        rds: MagicMock = MagicMock(spec=redis.Redis)
        settings: MagicMock = MagicMock()
        loader: MagicMock = MagicMock()
        model_obj: object = object()
        with (
                patch.object(
                    ws_mod, 'authenticate_ws_or_none',
                    new=AsyncMock(return_value=('erin', 'jti')),
                ),
                patch.object(
                    ws_mod, '_prepare_model_and_notify',
                    new=AsyncMock(return_value=model_obj),
                ),
                patch.object(
                    ws_mod,
                    '_detect_loop',
                    new=AsyncMock(side_effect=RuntimeError('boom')),
                ),
        ):
            # Should not raise despite close failing
            await ws_mod.handle_websocket_detect(ws, rds, settings, loader)


if __name__ == '__main__':
    unittest.main()

"""
pytest \
    --cov=examples.YOLO_server_api.backend.websocket_handlers \
    --cov-report=term-missing \
    tests/examples/YOLO_server_api/backend/websocket_handlers_test.py
"""
