from __future__ import annotations

import asyncio
import unittest
from functools import partial

from redis.exceptions import TimeoutError as RedisReadTimeout

from examples.streaming_web.metadata_fanout import MetadataFanout
from examples.streaming_web.metadata_fanout import MetadataFanoutSettings


async def _last_unsubscribe_drains_reader_and_stuck(
    entered,
    stopped,
    /,
    *args,
):
    entered.set()
    try:
        await asyncio.Event().wait()
    finally:
        stopped.set()


async def _read_deadline_is_reported_and_stuck(*args):
    await asyncio.Event().wait()


class MetadataLifetimeTest(unittest.IsolatedAsyncioTestCase):
    async def test_last_unsubscribe_drains_reader_and_next_starts_fresh(
        self,
    ):
        fanout = MetadataFanout()
        entered, stopped = asyncio.Event(), asyncio.Event()

        stuck = partial(
            _last_unsubscribe_drains_reader_and_stuck,
            entered,
            stopped,
        )

        first = await fanout.subscribe(None, 'camera', fetcher=stuck)
        await entered.wait()
        second = await fanout.subscribe(None, 'camera', fetcher=stuck)
        await first.close()
        self.assertFalse(stopped.is_set())
        await second.close()
        self.assertTrue(stopped.is_set())
        self.assertFalse(fanout._streams)
        entered.clear()
        third = await fanout.subscribe(None, 'camera', fetcher=stuck)
        await entered.wait()
        await third.close()
        await third.close()
        self.assertFalse(fanout._streams)

    async def test_read_deadline_is_reported_and_lifespan_drains_reader(self):
        fanout = MetadataFanout(
            MetadataFanoutSettings(read_timeout_seconds=0.1),
        )

        stuck = _read_deadline_is_reported_and_stuck

        async with fanout.lifespan():
            subscription = await fanout.subscribe(
                None,
                'camera',
                fetcher=stuck,
            )
            error = await asyncio.wait_for(subscription.get(), 1)
            self.assertIsInstance(error, RedisReadTimeout)
        self.assertFalse(fanout._streams)
        with self.assertRaises(RuntimeError):
            await fanout.subscribe(None, 'camera')
        await subscription.close()
