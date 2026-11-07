from __future__ import annotations

import asyncio
import os
import time
import unittest
from functools import partial
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import patch

import jwt
from fastapi import HTTPException
from redis.asyncio import Redis

from examples.auth import session_store as store
from examples.bff import proxy_tokens as module


async def _cancelled_browser_waits_until_rotation_refresh(
    entered,
    release,
    /,
    *args,
):
    entered.set()
    await release.wait()
    return {
        'access_token': token('new'),
        'refresh_token': token('new-refresh', 3600),
    }


async def _deleted_session_is_never_revived_refresh(
    testcase,
    /,
    *args,
):
    await store.delete_auth_session(
        testcase.redis,
        testcase.session_id,
    )
    return {
        'access_token': token('new'),
        'refresh_token': token('new-r', 3600),
    }


async def _heartbeat_keeps_a_slow_rotation_acquire(redis, session_id):
    return await store.acquire_refresh_lock(
        redis,
        session_id,
        ttl_seconds=1,
    )


async def _heartbeat_keeps_a_slow_rotation_refresh(
    testcase,
    /,
    *args,
):
    await asyncio.sleep(1.2)
    testcase.assertIsNotNone(
        await testcase.redis.get(
            testcase.key + ':refresh-lock',
        ),
    )
    return {
        'access_token': token('slow'),
        'refresh_token': token('slow-r', 3600),
    }


class _RaceCallbacks:
    calls: Any
    entered: Any
    release: Any
    terminal: Any

    async def _call_refresh(self, *args):

        self.calls += 1
        version = self.calls
        if version == 1:
            self.entered.set()
            await self.release.wait()
            if self.terminal:
                raise HTTPException(401, 'oidc_token_exchange_failed')
        return {
            'access_token': token(str(version)),
            'refresh_token': token('r' + str(version), 3600),
        }


def token(mark, offset=300):
    return jwt.encode(
        {'mark': mark, 'exp': int(time.time()) + offset},
        key='',
        algorithm='none',
    )


@unittest.skipUnless(
    os.getenv('NOTIFICATION_REDIS_TEST_URL'),
    'Requires isolated Redis',
)
class RefreshOwnershipTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.redis = Redis.from_url(os.environ['NOTIFICATION_REDIS_TEST_URL'])
        self.session_id, _ = await store.create_auth_session(
            self.redis,
            {
                'access_token': token('old', -1),
                'refresh_token': token('old-refresh', 3600),
                'feature_names': [],
            },
            {'id': 1},
        )
        self.key = store.auth_session_key(self.session_id)

    async def asyncTearDown(self):
        await store.delete_auth_session(self.redis, self.session_id)
        await self.redis.aclose()

    async def mark(self):
        value = await store.get_auth_session(self.redis, self.session_id)
        if value is None:
            return None
        access, _ = store.auth_tokens(value)
        return jwt.decode(access, options={'verify_signature': False})['mark']

    async def race(self, terminal=False):
        _callbacks = _RaceCallbacks()
        _callbacks.terminal = terminal
        _callbacks.entered, _callbacks.release = (
            asyncio.Event(),
            asyncio.Event(),
        )
        _callbacks.calls = 0

        refresh = _callbacks._call_refresh

        with patch.object(module, 'refresh_oidc_tokens', refresh):
            old = asyncio.create_task(
                module.get_proxy_access_token(self.redis, self.session_id),
            )
            await _callbacks.entered.wait()
            await self.redis.delete(self.key + ':refresh-lock')
            await module.get_proxy_access_token(self.redis, self.session_id)
            self.assertEqual(await self.mark(), '2')
            _callbacks.release.set()
            winner, _ = await old
            self.assertEqual(
                jwt.decode(winner, options={'verify_signature': False})[
                    'mark'
                ],
                '2',
            )
        self.assertEqual(await self.mark(), '2')

    async def test_expired_refresher_cannot_overwrite_new_tokens(self):
        await self.race()

    async def test_expired_refresher_cannot_log_out_the_new_owner(self):
        await self.race(terminal=True)

    async def test_cancelled_browser_waits_until_rotation_is_saved(self):
        entered, release = asyncio.Event(), asyncio.Event()

        refresh = partial(
            _cancelled_browser_waits_until_rotation_refresh,
            entered,
            release,
        )

        with patch.object(module, 'refresh_oidc_tokens', refresh):
            task = asyncio.create_task(
                module.get_proxy_access_token(self.redis, self.session_id),
            )
            await entered.wait()
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(await self.mark(), 'new')
        self.assertIsNone(await self.redis.get(self.key + ':refresh-lock'))

    async def test_deleted_session_is_never_revived_by_rotation(self):
        refresh = partial(_deleted_session_is_never_revived_refresh, self)

        with patch.object(module, 'refresh_oidc_tokens', refresh):
            with self.assertRaises(HTTPException) as caught:
                await module.get_proxy_access_token(
                    self.redis,
                    self.session_id,
                )
        self.assertEqual(caught.exception.status_code, 401)
        self.assertIsNone(await self.mark())

    async def test_heartbeat_keeps_a_slow_rotation_owned(self):
        acquire = _heartbeat_keeps_a_slow_rotation_acquire

        refresh = partial(_heartbeat_keeps_a_slow_rotation_refresh, self)

        with patch.object(
            module,
            'acquire_refresh_lock',
            acquire,
        ), patch.object(
            module.ProxySessionRefresh,
            'lease_seconds',
            1,
        ), patch.object(
            module,
            'refresh_oidc_tokens',
            refresh,
        ):
            await module.get_proxy_access_token(self.redis, self.session_id)
        self.assertEqual(await self.mark(), 'slow')

    async def test_terminal_and_transient_refresh_failures(
        self,
    ):
        with patch.object(
            module,
            'refresh_oidc_tokens',
            new=AsyncMock(side_effect=HTTPException(503, 'unavailable')),
        ):
            with self.assertRaises(HTTPException) as caught:
                await module.get_proxy_access_token(
                    self.redis,
                    self.session_id,
                )
        self.assertEqual(caught.exception.status_code, 503)
        self.assertEqual(await self.mark(), 'old')
        with patch.object(
            module,
            'refresh_oidc_tokens',
            new=AsyncMock(
                side_effect=HTTPException(401, 'oidc_token_exchange_failed'),
            ),
        ):
            with self.assertRaises(HTTPException) as caught:
                await module.get_proxy_access_token(
                    self.redis,
                    self.session_id,
                )
        self.assertEqual(caught.exception.status_code, 401)
        self.assertIsNone(await self.mark())
