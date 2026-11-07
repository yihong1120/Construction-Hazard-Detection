from __future__ import annotations

import asyncio
import time
import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock

from examples.local_notification_server.notification_claim import (
    NotificationClaim,
)
from examples.local_notification_server.notification_claim import (
    NotificationLeaseLost,
)


async def _hung_renewal_stops_dispatch_within_evaluate(script, *args):
    if 'ARGV[2]' in script and "'done'" in script:
        return 1
    await asyncio.Event().wait()


async def _hung_renewal_stops_dispatch_within_send(
    calls,
    cancelled,
    /,
):
    calls.append('first')
    try:
        await asyncio.Event().wait()
        calls.append('second')
    finally:
        cancelled.set()


async def _acquire_timeout_and_unowned_context_hang(*args, **kwargs):
    await asyncio.Event().wait()


async def _cancelled_request_is_not_blocked_evaluate(*args):
    await asyncio.Event().wait()


class NotificationDeadlineTest(unittest.IsolatedAsyncioTestCase):
    async def test_hung_renewal_stops_dispatch_within_the_confirmed_lease(
        self,
    ):
        cancelled = asyncio.Event()
        calls = []

        evaluate = _hung_renewal_stops_dispatch_within_evaluate

        redis = SimpleNamespace(
            set=AsyncMock(return_value=True),
            eval=evaluate,
        )
        claim = NotificationClaim(redis, 'test', 600)
        claim.lease_seconds = 0.06
        claim.io_timeout_seconds = 0.02
        await claim.acquire()

        send = partial(
            _hung_renewal_stops_dispatch_within_send,
            calls,
            cancelled,
        )

        started = time.monotonic()
        async with claim:
            with self.assertRaises(NotificationLeaseLost):
                await asyncio.wait_for(claim.run(send()), 0.5)
        self.assertLess(time.monotonic() - started, 0.3)
        self.assertTrue(cancelled.is_set())
        self.assertEqual(calls, ['first'])

    async def test_expired_local_deadline_refuses_new_batch_without_redis_io(
        self,
    ):
        redis = SimpleNamespace(get=AsyncMock())
        claim = NotificationClaim(redis, 'test', 600)
        claim._deadline = time.monotonic() - 1
        with self.assertRaises(NotificationLeaseLost):
            await claim.guard()
        redis.get.assert_not_awaited()

    async def test_acquire_timeout_and_unowned_context_fail_closed(self):
        hang = _acquire_timeout_and_unowned_context_hang

        claim = NotificationClaim(SimpleNamespace(set=hang), 'test', 600)
        claim.io_timeout_seconds = 0.01
        with self.assertRaises(NotificationLeaseLost):
            await claim.acquire()
        with self.assertRaises(RuntimeError):
            async with claim:
                pass
        with self.assertRaises(RuntimeError):
            await claim.run(hang())

    async def test_cancelled_request_is_not_blocked_by_a_hung_release(self):
        evaluate = _cancelled_request_is_not_blocked_evaluate

        claim = NotificationClaim(
            SimpleNamespace(
                set=AsyncMock(return_value=True),
                eval=evaluate,
            ),
            'test',
            600,
        )
        claim.io_timeout_seconds = 0.01
        await claim.acquire()
        with self.assertRaises(asyncio.CancelledError):
            async with claim:
                raise asyncio.CancelledError
