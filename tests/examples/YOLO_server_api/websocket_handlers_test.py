from __future__ import annotations

import os
import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import redis
from fastapi import WebSocket
from fastapi import WebSocketDisconnect

from examples.YOLO_server_api import websocket_handlers as ws_mod
from examples.YOLO_server_api.image_input import DetectionInputError


async def _setUp_load(websocket, key, client_ip, loader):
    return loader.get_model(key)


async def _idle_receive_is_closed_at_receive(stalled, /, *args):
    await stalled.wait()


async def _model_preparation_is_part_of_prepare(asyncio, /, *args):
    await asyncio.Event().wait()


async def _authentication_is_part_of_the_authenticate(
    asyncio,
    /,
    *args,
    **kwargs,
):
    await asyncio.Event().wait()


class TestGetModelKeyFromWs(unittest.IsolatedAsyncioTestCase):
    """Tests for extracting model key from WebSocket context.

    Covers header and first message model keys, and validates error responses
    when the key is missing or invalid.
    """

    async def test_header_model_key(self) -> None:
        """Header value is preferred when present."""
        ws = MagicMock(spec=WebSocket)
        ws.scope = {}
        ws.headers = {'x-model-key': 'modelA'}
        ws.query_params = {}
        result = await ws_mod._get_model_key_from_ws(ws, '1.2.3.4', 'alice')
        self.assertEqual(result, 'modelA')

    async def test_first_message_model_key(self) -> None:
        """First message JSON is used when header and query are absent."""
        ws = MagicMock(spec=WebSocket)
        ws.scope = {}
        ws.headers = {}
        ws.query_params = {}
        ws.receive_text = AsyncMock(return_value='{"model_key": "modelC"}')
        result = await ws_mod._get_model_key_from_ws(ws, '1.2.3.4', 'carol')
        self.assertEqual(result, 'modelC')

    async def test_first_message_missing_key_closes(self) -> None:
        """Missing key in first message causes policy close (1008)."""
        ws = MagicMock(spec=WebSocket)
        ws.scope = {}
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
        ws.scope = {}
        ws.headers = {}
        ws.query_params = {}
        ws.receive_text = AsyncMock(side_effect=ValueError('bad json'))
        ws.close = AsyncMock()
        result = await ws_mod._get_model_key_from_ws(ws, '1.2.3.4', 'erin')
        self.assertIsNone(result)
        ws.close.assert_awaited_once()
        _, kwargs = ws.close.await_args
        self.assertEqual(kwargs.get('code'), 1008)


class TestWebsocketEnvironmentHelpers(unittest.TestCase):
    """Provide TestWebsocketEnvironmentHelpers."""

    def test_local_auth_bypass_handles_unset_and_non_ip_clients(self) -> None:
        """Local-only bypass handles absent flags and host names safely."""
        with patch.dict('os.environ', {}, clear=False):
            self.assertTrue(ws_mod.WebSocketSettings().allow_localhost_bypass)
        with patch.dict(
            'os.environ',
            {'YOLO_WS_ALLOW_LOCALHOST_BYPASS': 'true'},
        ):
            self.assertTrue(ws_mod._should_bypass_local_auth('localhost'))


class TestSendReadyConfig(unittest.IsolatedAsyncioTestCase):
    """Tests for sending initial ready/handshake configuration."""

    async def test_send_ready_config_success(self) -> None:
        """Sends 'ready' payload and reports True on success."""
        ws = MagicMock(spec=WebSocket)
        ws.scope = {}
        with patch.object(
            ws_mod,
            '_safe_websocket_send_json',
            new=AsyncMock(return_value=True),
        ) as send_mock:
            ok = await ws_mod._send_ready_config(
                ws,
                'modelX',
                '1.2.3.4',
                'alice',
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
        ws.scope = {}
        with patch.object(
            ws_mod,
            '_safe_websocket_send_json',
            new=AsyncMock(return_value=False),
        ):
            ok = await ws_mod._send_ready_config(
                ws,
                'modelY',
                '1.2.3.4',
                'bob',
            )
            self.assertFalse(ok)


class TestProcessFrameAndRespond(unittest.IsolatedAsyncioTestCase):

    async def test_invalid_frame_returns_shared_input_code_and_closes_session(
        self,
    ):
        selection = SimpleNamespace(
            detect=AsyncMock(side_effect=DetectionInputError()),
        )
        websocket = AsyncMock()
        with patch.object(
            ws_mod,
            '_safe_websocket_send_json',
            new=AsyncMock(return_value=True),
        ) as send:
            self.assertFalse(
                await ws_mod._process_frame_and_respond(
                    websocket,
                    b'bad',
                    selection,
                    'test',
                    'operator',
                ),
            )
        self.assertEqual(send.await_args.args[1]['code'], 'INVALID_IMAGE')
        self.assertEqual(send.await_args.args[1]['http_status'], 400)
        self.assertEqual(websocket.close.await_args.kwargs['code'], 1003)

    async def test_shared_runtime_selection_sends_detection_results(self):
        for sent in (True, False):
            selection = SimpleNamespace(
                detect=AsyncMock(
                    return_value=SimpleNamespace(rows=[[1, 2, 3, 4, 0.9, 2]]),
                ),
            )
            with patch.object(
                ws_mod,
                '_safe_websocket_send_json',
                new=AsyncMock(return_value=sent),
            ):
                result = await ws_mod._process_frame_and_respond(
                    MagicMock(),
                    b'image',
                    selection,
                    'test',
                    'operator',
                )
            self.assertEqual(result, sent)
            selection.detect.assert_awaited_once_with(b'image')


class TestPrepareModelAndNotify(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        load = _setUp_load

        patcher = patch.object(ws_mod, '_load_session_model', side_effect=load)
        patcher.start()
        self.addCleanup(patcher.stop)

    """Tests for model preparation and readiness notification."""

    async def test_no_model_key(self) -> None:
        """If no model key, returns None immediately."""
        ws = MagicMock(spec=WebSocket)
        ws.scope = {}
        loader: MagicMock = MagicMock()
        with patch.object(
            ws_mod,
            '_get_model_key_from_ws',
            new=AsyncMock(return_value=None),
        ):
            res = await ws_mod._prepare_model_and_notify(
                ws,
                '1.2.3.4',
                'alice',
                loader,
            )
            self.assertIsNone(res)

    async def test_model_not_found(self) -> None:
        """If model cannot be loaded, close with code 1003 and return None."""
        ws = MagicMock(spec=WebSocket)
        ws.scope = {}
        ws.close = AsyncMock()
        loader: MagicMock = MagicMock()
        loader.get_model.return_value = None
        with patch.object(
            ws_mod,
            '_get_model_key_from_ws',
            new=AsyncMock(return_value='modelZ'),
        ):
            res = await ws_mod._prepare_model_and_notify(
                ws,
                '1.2.3.4',
                'bob',
                loader,
            )
            self.assertIsNone(res)
            ws.close.assert_awaited_once()
            _, kwargs = ws.close.await_args
            self.assertEqual(kwargs.get('code'), 1003)

    async def test_send_ready_config_failed(self) -> None:
        """If ready notification fails, return None without raising."""
        ws = MagicMock(spec=WebSocket)
        ws.scope = {}
        loader: MagicMock = MagicMock()
        model_obj: object = object()
        loader.get_model.return_value = model_obj
        with (
            patch.object(
                ws_mod,
                '_get_model_key_from_ws',
                new=AsyncMock(return_value='modelZ'),
            ),
            patch.object(
                ws_mod,
                '_send_ready_config',
                new=AsyncMock(return_value=False),
            ),
        ):
            res = await ws_mod._prepare_model_and_notify(
                ws,
                '1.2.3.4',
                'carol',
                loader,
            )
            self.assertIsNone(res)

    async def test_success(self) -> None:
        """Happy path returns the loaded model object."""
        ws = MagicMock(spec=WebSocket)
        ws.scope = {}
        loader: MagicMock = MagicMock()
        model_obj: object = object()
        loader.get_model.return_value = model_obj
        with (
            patch.object(
                ws_mod,
                '_get_model_key_from_ws',
                new=AsyncMock(return_value='modelZ'),
            ),
            patch.object(
                ws_mod,
                '_send_ready_config',
                new=AsyncMock(return_value=True),
            ),
        ):
            res = await ws_mod._prepare_model_and_notify(
                ws,
                '1.2.3.4',
                'dave',
                loader,
            )
            self.assertIs(res, model_obj)


class TestDetectLoop(unittest.IsolatedAsyncioTestCase):
    """Tests for the detection loop that streams and processes frames."""

    async def test_timeout_immediately(self) -> None:
        """If timed out immediately, returns count 0 and stops."""
        ws = MagicMock(spec=WebSocket)
        ws.scope = {}
        with patch.object(
            ws_mod,
            'check_and_maybe_close_on_timeout',
            new=AsyncMock(return_value=True),
        ):
            count = await ws_mod._detect_loop(
                ws,
                0.0,
                MagicMock(spec=ws_mod.ModelSelection),
                '1.2.3.4',
                'alice',
            )
            self.assertEqual(count, 0)

    async def test_receive_none_breaks(self) -> None:
        """Loop stops when no more frames are received."""
        ws = MagicMock(spec=WebSocket)
        ws.scope = {}
        with (
            patch.object(
                ws_mod,
                'check_and_maybe_close_on_timeout',
                new=AsyncMock(side_effect=[False, True]),
            ),
            patch.object(
                ws_mod,
                '_safe_websocket_receive_bytes',
                new=AsyncMock(return_value=None),
            ),
        ):
            count = await ws_mod._detect_loop(
                ws,
                0.0,
                MagicMock(spec=ws_mod.ModelSelection),
                '1.2.3.4',
                'bob',
            )
            self.assertEqual(count, 0)

    async def test_process_two_frames_then_none(self) -> None:
        """Processes two frames successfully, then stops on None."""
        ws = MagicMock(spec=WebSocket)
        ws.scope = {}
        timeouts = [False, False, False, True]
        with (
            patch.object(
                ws_mod,
                'check_and_maybe_close_on_timeout',
                new=AsyncMock(side_effect=timeouts),
            ),
            patch.object(
                ws_mod,
                '_safe_websocket_receive_bytes',
                new=AsyncMock(side_effect=[b'a', b'b', None]),
            ),
            patch.object(
                ws_mod,
                '_process_frame_and_respond',
                new=AsyncMock(return_value=True),
            ),
        ):
            count = await ws_mod._detect_loop(
                ws,
                0.0,
                MagicMock(spec=ws_mod.ModelSelection),
                '1.2.3.4',
                'carol',
            )
            self.assertEqual(count, 2)

    async def test_process_frame_failure_breaks(self) -> None:
        """Stops the loop when processing a frame fails."""
        ws = MagicMock(spec=WebSocket)
        ws.scope = {}
        with (
            patch.object(
                ws_mod,
                'check_and_maybe_close_on_timeout',
                new=AsyncMock(side_effect=[False, True]),
            ),
            patch.object(
                ws_mod,
                '_safe_websocket_receive_bytes',
                new=AsyncMock(return_value=b'a'),
            ),
            patch.object(
                ws_mod,
                '_process_frame_and_respond',
                new=AsyncMock(return_value=False),
            ),
        ):
            count = await ws_mod._detect_loop(
                ws,
                0.0,
                MagicMock(spec=ws_mod.ModelSelection),
                '1.2.3.4',
                'dave',
            )
            self.assertEqual(count, 1)


class TestHandleWebsocketDetect(unittest.IsolatedAsyncioTestCase):
    """Tests for the top-level WebSocket handler control flow."""

    def _make_ws(self) -> WebSocket:
        """Create a mock WebSocket with accept/close and client host."""
        ws = MagicMock(spec=WebSocket)
        ws.scope = {}
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

    def test_should_bypass_local_auth(self) -> None:
        """Only loopback clients bypass JWT when explicitly enabled."""
        with patch.dict(
            os.environ,
            {'YOLO_WS_ALLOW_LOCALHOST_BYPASS': 'true'},
        ):
            self.assertTrue(ws_mod._should_bypass_local_auth('127.0.0.1'))
            self.assertTrue(ws_mod._should_bypass_local_auth('::1'))
            self.assertFalse(ws_mod._should_bypass_local_auth('9.9.9.9'))
        with patch.dict(
            os.environ,
            {'YOLO_WS_ALLOW_LOCALHOST_BYPASS': 'false'},
        ):
            self.assertFalse(ws_mod._should_bypass_local_auth('127.0.0.1'))

    async def test_localhost_bypass_skips_auth(self) -> None:
        """Local main.py pipeline calls can skip JWT auth."""
        ws = self._make_ws()
        ws.client = SimpleNamespace(host='127.0.0.1')
        rds: MagicMock = MagicMock(spec=redis.Redis)
        settings: MagicMock = MagicMock()
        loader: MagicMock = MagicMock()
        model_obj: object = object()
        with (
            patch.dict(
                os.environ,
                {'YOLO_WS_ALLOW_LOCALHOST_BYPASS': 'true'},
            ),
            patch.object(
                ws_mod,
                'authenticate_ws_or_none',
                new=AsyncMock(
                    return_value=(
                        'alice',
                        {
                            'subject': {
                                'username': 'alice',
                                'role': 'user',
                                'tenant_id': 't1',
                            },
                        },
                    ),
                ),
            ) as auth_mock,
            patch.object(
                ws_mod,
                '_prepare_model_and_notify',
                new=AsyncMock(return_value=model_obj),
            ) as prep_mock,
            patch.object(
                ws_mod,
                '_detect_loop',
                new=AsyncMock(return_value=1),
            ) as loop_mock,
        ):
            await ws_mod.handle_websocket_detect(ws, rds, settings, loader)

        auth_mock.assert_not_awaited()
        prep_mock.assert_awaited_once()
        loop_mock.assert_awaited_once()

    async def test_successful_flow(self) -> None:
        """Happy path: prepare model and run detection loop."""
        ws = self._make_ws()
        rds: MagicMock = MagicMock(spec=redis.Redis)
        settings: MagicMock = MagicMock()
        loader: MagicMock = MagicMock()
        model_obj: object = object()
        with (
            patch.object(
                ws_mod,
                'authenticate_ws_or_none',
                new=AsyncMock(
                    return_value=(
                        'alice',
                        {
                            'subject': {
                                'username': 'alice',
                                'role': 'user',
                                'tenant_id': 't1',
                            },
                        },
                    ),
                ),
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
                ws_mod,
                'authenticate_ws_or_none',
                new=AsyncMock(
                    return_value=(
                        'bob',
                        {
                            'subject': {
                                'username': 'bob',
                                'role': 'user',
                                'tenant_id': 't1',
                            },
                        },
                    ),
                ),
            ),
            patch.object(
                ws_mod,
                '_prepare_model_and_notify',
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
                ws_mod,
                'authenticate_ws_or_none',
                new=AsyncMock(
                    return_value=(
                        'carol',
                        {
                            'subject': {
                                'username': 'carol',
                                'role': 'user',
                                'tenant_id': 't1',
                            },
                        },
                    ),
                ),
            ),
            patch.object(
                ws_mod,
                '_prepare_model_and_notify',
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
                ws_mod,
                'authenticate_ws_or_none',
                new=AsyncMock(
                    return_value=(
                        'dave',
                        {
                            'subject': {
                                'username': 'dave',
                                'role': 'user',
                                'tenant_id': 't1',
                            },
                        },
                    ),
                ),
            ),
            patch.object(
                ws_mod,
                '_prepare_model_and_notify',
                new=AsyncMock(return_value=None),
            ),
            patch.object(
                ws_mod,
                '_detect_loop',
                new=AsyncMock(),
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
                ws_mod,
                'authenticate_ws_or_none',
                new=AsyncMock(
                    return_value=(
                        'erin',
                        {
                            'subject': {
                                'username': 'erin',
                                'role': 'user',
                                'tenant_id': 't1',
                            },
                        },
                    ),
                ),
            ),
            patch.object(
                ws_mod,
                '_prepare_model_and_notify',
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


class TestCatalogSessionModel(unittest.IsolatedAsyncioTestCase):
    async def test_session_model_uses_verified_scope_and_pinned_version(self):
        from types import SimpleNamespace

        ws = SimpleNamespace(
            scope={'_catalog_subject': {'tenant_id': 't1', 'role': 'user'}},
            headers={'x-model-version': 'commit'},
        )
        loader = MagicMock()
        loader.prepare = AsyncMock()
        entry = object()
        with patch.object(
            ws_mod,
            'select_model_async',
            return_value=entry,
        ) as require:
            result = await ws_mod._load_session_model(
                ws,
                'custom',
                '1.2.3.4',
                loader,
            )
        require.assert_called_once_with(
            'custom',
            'commit',
            't1',
            'user',
            'image',
        )
        loader.prepare.assert_awaited_once_with(entry)
        self.assertIs(result, loader.prepare.return_value)

    async def test_remote_session_cannot_use_missing_scope(self):
        from fastapi import HTTPException
        from types import SimpleNamespace

        ws = SimpleNamespace(scope={}, headers={})
        with self.assertRaises(HTTPException) as raised:
            await ws_mod._load_session_model(
                ws,
                'custom',
                '1.2.3.4',
                MagicMock(),
            )
        self.assertEqual(raised.exception.status_code, 403)


class WebsocketDeadlineTest(unittest.IsolatedAsyncioTestCase):
    async def test_idle_receive_is_closed_at_absolute_deadline(self):
        import asyncio

        ws = MagicMock(spec=WebSocket)
        ws.scope = {}
        ws.accept = AsyncMock()
        ws.close = AsyncMock()
        ws.client = SimpleNamespace(host='external')
        stalled = asyncio.Event()

        receive = partial(_idle_receive_is_closed_at_receive, stalled)

        selection = SimpleNamespace(detect=AsyncMock())
        with (
            patch.object(ws_mod, 'WS_MAX_SESSION_SECONDS', 0.03),
            patch.object(
                ws_mod,
                'authenticate_ws_or_none',
                new=AsyncMock(return_value=('operator', {'subject': {}})),
            ),
            patch.object(
                ws_mod,
                '_prepare_model_and_notify',
                new=AsyncMock(return_value=selection),
            ),
            patch.object(ws_mod, '_safe_websocket_receive_bytes', receive),
        ):
            await asyncio.wait_for(
                ws_mod.handle_websocket_detect(
                    ws,
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                ),
                1,
            )
        ws.close.assert_awaited_once_with(code=1000, reason='Session timeout')
        selection.detect.assert_not_awaited()

    async def test_model_preparation_is_part_of_the_same_deadline(self):
        import asyncio

        ws = MagicMock(spec=WebSocket)
        ws.scope = {}
        ws.accept = AsyncMock()
        ws.close = AsyncMock()
        ws.client = SimpleNamespace(host='external')

        prepare = partial(_model_preparation_is_part_of_prepare, asyncio)

        with (
            patch.object(ws_mod, 'WS_MAX_SESSION_SECONDS', 0.03),
            patch.object(
                ws_mod,
                'authenticate_ws_or_none',
                new=AsyncMock(return_value=('operator', {'subject': {}})),
            ),
            patch.object(ws_mod, '_prepare_model_and_notify', prepare),
        ):
            await asyncio.wait_for(
                ws_mod.handle_websocket_detect(
                    ws,
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                ),
                1,
            )
        ws.close.assert_awaited_once_with(code=1000, reason='Session timeout')

    async def test_authentication_is_part_of_the_same_deadline(self):
        import asyncio

        ws = MagicMock(spec=WebSocket)
        ws.scope = {}
        ws.accept = AsyncMock()
        ws.close = AsyncMock()
        ws.client = SimpleNamespace(host='external')

        authenticate = partial(
            _authentication_is_part_of_the_authenticate,
            asyncio,
        )

        with (
            patch.object(ws_mod, 'WS_MAX_SESSION_SECONDS', 0.03),
            patch.object(ws_mod, 'authenticate_ws_or_none', authenticate),
        ):
            await asyncio.wait_for(
                ws_mod.handle_websocket_detect(
                    ws,
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                ),
                1,
            )
        ws.close.assert_awaited_once_with(code=1000, reason='Session timeout')


class LocalWebsocketVersionTest(unittest.IsolatedAsyncioTestCase):
    async def test_local_session_still_passes_requested_version(self):
        ws = SimpleNamespace(scope={}, headers={'x-model-version': 'v1'})
        runtime = SimpleNamespace(prepare=AsyncMock())
        with patch.object(
            ws_mod,
            'select_local_model_async',
            new=AsyncMock(),
        ) as select:
            await ws_mod._load_session_model(ws, 'model', '127.0.0.1', runtime)
        select.assert_awaited_once_with('model', 'v1')
        runtime.prepare.assert_awaited_once_with(select.return_value)
