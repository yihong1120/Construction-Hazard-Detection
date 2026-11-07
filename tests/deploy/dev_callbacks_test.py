from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from httpx import ASGITransport
from httpx import AsyncClient

from deploy.dev import apps


class _LimiterSerializesInferenceForOneOwnerCallbacks:
    active: Any
    maximum: Any

    async def _call_call_next(self, request):

        self.active += 1
        self.maximum = max(self.maximum, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return JSONResponse({'status': True})


class _IndependentLimitersDoNotShareAGateCallbacks:
    entered: Any
    ready: Any

    async def _call_call_next(self, request):

        self.entered += 1
        if self.entered == 2:
            self.ready.set()
        await self.ready.wait()
        return JSONResponse({'status': True})


class DevCallbacksTest(unittest.IsolatedAsyncioTestCase):
    async def test_limiter_serializes_inference_for_one_owner(self):
        _callbacks = _LimiterSerializesInferenceForOneOwnerCallbacks()
        limiter = apps._InferenceLimit()
        _callbacks.active = 0
        _callbacks.maximum = 0

        call_next = _callbacks._call_call_next

        request = SimpleNamespace(
            method='POST',
            url=SimpleNamespace(path='/v2/detect'),
        )
        await asyncio.gather(
            *(limiter.dispatch(request, call_next) for _ in range(3)),
        )
        self.assertEqual(_callbacks.maximum, 1)

    async def test_independent_limiters_do_not_share_a_gate(self):
        _callbacks = _IndependentLimitersDoNotShareAGateCallbacks()
        _callbacks.ready = asyncio.Event()
        _callbacks.entered = 0

        call_next = _callbacks._call_call_next

        request = SimpleNamespace(
            method='POST',
            url=SimpleNamespace(path='/detect'),
        )
        async with asyncio.timeout(1):
            await asyncio.gather(
                apps._InferenceLimit().dispatch(request, call_next),
                apps._InferenceLimit().dispatch(request, call_next),
            )
        self.assertEqual(_callbacks.entered, 2)

    async def test_identity_guard_uses_the_current_app_and_keeps_denials(self):
        applications = [FastAPI(), FastAPI()]
        for index, app in enumerate(applications):
            app.state.redis_client = SimpleNamespace(client=index)
            app.middleware('http')(apps._protect_shared_identity)
        logout = AsyncMock(return_value=JSONResponse({'status': True}))
        with patch.object(apps, 'local_logout_response', logout):
            for index, app in enumerate(applications):
                async with AsyncClient(
                    transport=ASGITransport(app=app),
                    base_url='http://test',
                ) as client:
                    self.assertEqual(
                        (await client.get('/bff/auth/account')).status_code,
                        403,
                    )
                    self.assertEqual(
                        (
                            await client.post('/bff/auth/oidc/logout')
                        ).status_code,
                        405,
                    )
                    self.assertEqual(
                        (
                            await client.get('/bff/auth/oidc/logout?state=one')
                        ).status_code,
                        200,
                    )
                self.assertEqual(logout.await_args.args, (index, 'one'))
