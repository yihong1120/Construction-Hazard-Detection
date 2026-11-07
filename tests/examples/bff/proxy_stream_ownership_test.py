from __future__ import annotations

import asyncio
import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import httpx
from starlette.requests import ClientDisconnect

from examples.bff import proxy_streams as module


async def _cancelled_open_closes_its_owned_send(
    entered,
    release,
    /,
    *args,
    **kwargs,
):
    entered.set()
    await release.wait()


async def _header_send_failure_closes_before_send(message):
    raise OSError('downstream closed before headers')


async def _shared_client_survives_failure_and_close(
    release,
    started,
    /,
):
    started.set()
    await release.wait()


class ProxyStreamOwnershipTest(unittest.IsolatedAsyncioTestCase):
    def request(self):
        return SimpleNamespace(
            method='GET',
            headers={},
            query_params={},
            url=SimpleNamespace(path='/bff/events'),
        )

    async def test_cancelled_open_closes_its_owned_client(self):
        entered, release = asyncio.Event(), asyncio.Event()

        send = partial(_cancelled_open_closes_its_owned_send, entered, release)

        client = SimpleNamespace(
            build_request=Mock(),
            send=send,
            aclose=AsyncMock(),
        )
        with patch.object(
            module,
            '_proxy_sse_http_client',
            AsyncMock(return_value=(client, True)),
        ), patch.object(module, '_proxy_request_headers', return_value={}):
            task = asyncio.create_task(
                module._open_proxy_stream(
                    self.request(),
                    'https://example.com',
                    'fake',
                ),
            )
            await entered.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        client.aclose.assert_awaited_once()

    async def test_header_send_failure_closes_before_body_iteration(self):
        upstream = httpx.Response(200, content=b'event: update\n\n')
        client = SimpleNamespace(aclose=AsyncMock())
        stream = module.ProxyStream(client, True)
        stream.response = upstream
        response = module.ProxyStreamingResponse(stream, '/bff/events')

        send = _header_send_failure_closes_before_send

        with self.assertRaises(ClientDisconnect):
            await response(
                {'type': 'http', 'asgi': {'spec_version': '2.4'}},
                AsyncMock(),
                send,
            )
        self.assertTrue(upstream.is_closed)
        client.aclose.assert_awaited_once()
        await stream.close()
        client.aclose.assert_awaited_once()

    async def test_shared_client_survives_failure_and_cancelled_cleanup_drains(
        self,
    ):
        client = SimpleNamespace(
            build_request=Mock(side_effect=ValueError('bad headers')),
            aclose=AsyncMock(),
        )
        with patch.object(
            module,
            '_proxy_sse_http_client',
            AsyncMock(return_value=(client, False)),
        ):
            with self.assertRaises(ValueError):
                await module._open_proxy_stream(
                    self.request(),
                    'https://example.com',
                    'fake',
                )
        client.aclose.assert_not_awaited()
        started, release = asyncio.Event(), asyncio.Event()

        close = partial(
            _shared_client_survives_failure_and_close,
            release,
            started,
        )

        stream = module.ProxyStream(SimpleNamespace(aclose=close), True)
        task = asyncio.create_task(stream.close())
        await started.wait()
        task.cancel()
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
