from __future__ import annotations

import os
import unittest
from unittest.mock import Mock
from unittest.mock import patch
from uuid import uuid4

from redis.asyncio import Redis

from examples.streaming_web.metadata_keys import build_metadata_key
from examples.streaming_web.redis_service import fetch_latest_metadata_for_key


@unittest.skipUnless(
    os.getenv('NOTIFICATION_REDIS_TEST_URL'),
    'Requires isolated Redis',
)
class MetadataLatestTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.redis = Redis.from_url(os.environ['NOTIFICATION_REDIS_TEST_URL'])
        self.key = build_metadata_key('audit-' + uuid4().hex, 'Camera')

    async def asyncTearDown(self):
        await self.redis.delete(self.key)
        await self.redis.aclose()

    async def test_backlogged_reader_jumps_to_newest_in_two_bounded_reads(
        self,
    ):
        async with self.redis.pipeline(transaction=False) as pipeline:
            for _ in range(1000):
                pipeline.xadd(self.key, {'has_warning': '0'})
            results = await pipeline.execute()
        with patch.object(
            self.redis,
            'xread',
            new=Mock(wraps=self.redis.xread),
        ) as read, patch.object(
            self.redis,
            'xrevrange',
            new=Mock(wraps=self.redis.xrevrange),
        ) as newest:
            snapshot = await fetch_latest_metadata_for_key(
                self.redis,
                self.key,
                '0-0',
                block_ms=1,
            )
        self.assertEqual(snapshot['id'], results[-1].decode())
        self.assertEqual(read.call_count, 1)
        self.assertEqual(newest.call_count, 1)
        self.assertIsNone(
            await fetch_latest_metadata_for_key(
                self.redis,
                self.key,
                snapshot['id'],
                block_ms=1,
            ),
        )

    async def test_normal_batches_merge_snapshots_without_an_extra_round_trip(
        self,
    ):
        await self.redis.xadd(self.key, {'has_warning': '1'})
        latest = await self.redis.xadd(self.key, {'has_warning': '0'})
        with patch.object(
            self.redis,
            'xrevrange',
            new=Mock(wraps=self.redis.xrevrange),
        ) as newest:
            snapshot = await fetch_latest_metadata_for_key(
                self.redis,
                self.key,
                '0-0',
                block_ms=1,
            )
        self.assertEqual(snapshot['id'], latest.decode())
        self.assertFalse(snapshot['has_warning'])
        newest.assert_not_called()
