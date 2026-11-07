from __future__ import annotations

import json
import os
import time
import unittest
from unittest.mock import patch
from uuid import uuid4

from redis.asyncio import Redis

from examples.streaming_web import playback_config as config
from examples.streaming_web import playback_session_store as module


@unittest.skipUnless(
    os.getenv('NOTIFICATION_REDIS_TEST_URL'),
    'Requires isolated Redis',
)
class PlaybackRenewalBoundsTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.redis = Redis.from_url(os.environ['NOTIFICATION_REDIS_TEST_URL'])
        self.prefix = uuid4().hex
        self.sessions = [
            module.new_playback_session(
                session_id=self.prefix + str(i),
                username='audit',
                label=self.prefix,
                stream_name='Cam',
                profile='clean',
                rendition='detail',
                language=None,
            )
            for i in range(5)
        ]
        self.path = self.sessions[0]['base_media_path']
        self.index = module.playback_media_sessions_key(self.path)
        self.demand = module.playback_demand_session_key(self.path, 'clean')
        stamp = time.time()
        for index, session in enumerate(self.sessions):
            await self.redis.set(
                module.playback_session_key(session['session_id']),
                json.dumps(session),
                ex=60,
            )
            await self.redis.zadd(
                self.index,
                {session['session_id']: stamp + index + 10},
            )
        expired = {
            self.prefix + 'expired-' + str(i): stamp - 10 for i in range(5)
        }
        await self.redis.zadd(self.index, expired)
        await self.redis.zadd(self.demand, expired)

    async def asyncTearDown(self):
        await self.redis.delete(
            self.index,
            self.demand,
            module.playback_session_refresh_key(self.path),
            *[
                module.playback_session_key(s['session_id'])
                for s in self.sessions
            ],
        )
        await self.redis.aclose()

    async def test_batches_prune_expired_and_renew_live_sessions(
        self,
    ):
        with patch.object(config, 'STREAM_PLAYBACK_REFRESH_BATCH_SIZE', 2):
            for _ in range(3):
                await self.redis.delete(
                    module.playback_session_refresh_key(self.path),
                )
                await module.refresh_playback_sessions_for_media_path(
                    self.redis,
                    self.path,
                )
        self.assertEqual(
            await self.redis.zcount(self.index, '-inf', time.time()),
            0,
        )
        self.assertEqual(
            await self.redis.zcount(self.demand, '-inf', time.time()),
            0,
        )
        self.assertEqual(await self.redis.zcard(self.index), 5)
        for session in self.sessions:
            self.assertGreater(
                await self.redis.ttl(
                    module.playback_session_key(session['session_id']),
                ),
                60,
            )
