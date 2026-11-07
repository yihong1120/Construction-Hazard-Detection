from __future__ import annotations

import asyncio
import os
import unittest
from functools import partial
from uuid import uuid4

from redis.asyncio import Redis

from examples.local_notification_server.notification_claim import (
    NotificationClaim,
)

REDIS_URL = os.getenv('NOTIFICATION_REDIS_TEST_URL')


async def _cancellation_releases_processing_state_deliver(
    claim,
    entered,
    /,
):
    async with claim:
        entered.set()
        await asyncio.Event().wait()


@unittest.skipUnless(
    REDIS_URL,
    'Requires isolated Redis for atomic lease tests',
)
class NotificationClaimTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.redis = Redis.from_url(REDIS_URL)
        self.key = 'test:notification:' + uuid4().hex

    async def asyncTearDown(self) -> None:
        await self.redis.delete(self.key)
        await self.redis.aclose()

    async def test_failure_releases_claim_for_immediate_retry(self) -> None:
        first = NotificationClaim(self.redis, self.key, 600)
        self.assertTrue(await first.acquire())
        with self.assertRaisesRegex(RuntimeError, 'delivery failed'):
            async with first:
                raise RuntimeError('delivery failed')
        self.assertTrue(
            await NotificationClaim(self.redis, self.key, 600).acquire(),
        )

    async def test_success_blocks_duplicates_with_completed_state(
        self,
    ) -> None:
        first = NotificationClaim(self.redis, self.key, 600)
        self.assertTrue(await first.acquire())
        async with first:
            other = NotificationClaim(self.redis, self.key, 600)
            self.assertFalse(await other.acquire())
            self.assertFalse(await other.is_completed())
            await first.finish()
        self.assertFalse(await other.acquire())
        self.assertTrue(await other.is_completed())
        self.assertGreater(await self.redis.ttl(self.key), 590)

    async def test_stale_owner_cannot_complete_or_release_replacement_claim(
        self,
    ) -> None:
        first = NotificationClaim(self.redis, self.key, 600)
        await first.acquire()
        async with first:
            await self.redis.set(self.key, 'replacement', ex=120)
            with self.assertRaisesRegex(RuntimeError, 'lease expired'):
                await first.finish()
        self.assertEqual(await self.redis.get(self.key), b'replacement')

    async def test_cancellation_releases_processing_state(self) -> None:
        claim = NotificationClaim(self.redis, self.key, 600)
        await claim.acquire()
        entered = asyncio.Event()

        deliver = partial(
            _cancellation_releases_processing_state_deliver,
            claim,
            entered,
        )

        task = asyncio.create_task(deliver())
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertIsNone(await self.redis.get(self.key))

    async def test_release_failure_does_not_mask_cancellation(self):
        from unittest.mock import AsyncMock
        from unittest.mock import patch

        claim = NotificationClaim(self.redis, self.key, 600)
        self.assertTrue(await claim.acquire())
        with patch.object(
            self.redis,
            'eval',
            new=AsyncMock(side_effect=ConnectionError('offline')),
        ):
            with self.assertRaises(asyncio.CancelledError):
                async with claim:
                    raise asyncio.CancelledError
