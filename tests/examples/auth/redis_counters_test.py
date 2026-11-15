"""Native social begin and API quotas share the same atomic fixed window."""
from __future__ import annotations

import asyncio
import unittest

from fastapi import HTTPException
from fastapi import Request

from examples.auth.cache import RateLimiterService
from examples.auth.redis_counters import increment_window
from examples.db_management.services import native_social_state
from tests.private_redis import RedisFixture


class AtomicCounterTest(RedisFixture, unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.redis = await self.start_redis()
        self.request = Request(
            {'type': 'http', 'client': ('192.0.2.1', 1), 'headers': []},
        )

    async def test_native_begin_has_twenty_slots_under_concurrent_requests(
        self,
    ):
        async def begin():
            try:
                await native_social_state._enforce_begin_rate_limit(
                    self.request, self.redis,
                )
                return True
            except HTTPException as error:
                self.assertEqual(error.status_code, 429)
                self.assertGreater(int(error.headers['Retry-After']), 0)
                return False

        results = await asyncio.gather(*(begin() for _ in range(50)))
        self.assertEqual(sum(results), 20)
        key = native_social_state._rate_limit_key(self.request)
        self.assertGreater(await self.redis.ttl(key), 0)

    async def test_native_begin_repairs_a_missing_ttl_without_resetting_count(
        self,
    ):
        key = native_social_state._rate_limit_key(self.request)
        await self.redis.set(key, 20)
        with self.assertRaises(HTTPException):
            await native_social_state._enforce_begin_rate_limit(
                self.request, self.redis,
            )
        self.assertEqual(await self.redis.get(key), b'21')
        self.assertGreater(await self.redis.ttl(key), 0)

    async def test_api_and_native_counters_keep_existing_windows(self):
        api = RateLimiterService()
        first = await api._incr_and_get_ttl(self.redis, 'shared-test', 30)
        self.assertEqual(first[0], 1)
        await self.redis.expire('shared-test', 10)
        count, ttl = await increment_window(self.redis, 'shared-test', 30)
        self.assertEqual(count, 2)
        self.assertLessEqual(ttl, 10)
        # EVALSHA reload after SCRIPT FLUSH still uses the same contract.
        await self.redis.script_flush()
        self.assertEqual(
            (await api._incr_and_get_ttl(self.redis, 'shared-test', 30))[0], 3,
        )
