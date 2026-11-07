from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch
from uuid import uuid4

from fastapi import HTTPException
from redis.asyncio import Redis

from examples.streaming_web import playback_config as config
from examples.streaming_web import playback_responses
from examples.streaming_web import playback_session_store as store
from examples.streaming_web import streaming_api_service as service
from examples.streaming_web.schemas import StreamPlaybackBatchRequest
from examples.streaming_web.schemas import StreamPlaybackRequest

REDIS_URL = os.getenv('NOTIFICATION_REDIS_TEST_URL')


class _ConcurrentDeleteDoesNotReviveOrPartiallyCallbacks:
    deleted: Any
    factory: Any
    testcase: PlaybackBatchAtomicTest
    session: Any

    def _call_pipeline(self, *args, **kwargs):
        _callbacks = _ConcurrentDeleteDoesNotReviveOrPartiallyCallbacks2()
        _callbacks.parent = self
        pipe = self.factory(*args, **kwargs)
        _callbacks.original = pipe.mget

        mget = _callbacks._call_mget

        pipe.mget = mget
        return pipe


class _ConcurrentDeleteDoesNotReviveOrPartiallyCallbacks2:
    parent: _ConcurrentDeleteDoesNotReviveOrPartiallyCallbacks
    original: Any

    async def _call_mget(self, keys):

        result = await self.original(keys)
        if not self.parent.deleted:
            self.parent.deleted = True
            await self.parent.testcase.redis.delete(
                store.playback_session_key(
                    self.parent.session['session_id'],
                ),
            )
        return result


@unittest.skipUnless(REDIS_URL, 'Requires isolated Redis')
class PlaybackBatchAtomicTest(unittest.IsolatedAsyncioTestCase):
    async def test_http_boundary_mixed_batch_failure_creates_no_session(self):
        before = await self.snapshot()
        request = StreamPlaybackBatchRequest(
            streams=[
                StreamPlaybackRequest(
                    label=self.prefix,
                    key='new',
                    profile='clean',
                ),
                StreamPlaybackRequest(
                    label=self.prefix,
                    key='existing',
                    profile='clean',
                    session_id='missing',
                ),
            ],
        )
        with patch.object(
            service,
            'authorise_label_access',
            new=AsyncMock(),
        ), patch.object(
            service.stream_catalog_service,
            'resolve_configured_stream_names',
            new=AsyncMock(return_value=['new', 'existing']),
        ):
            with self.assertRaises(HTTPException) as caught:
                await service.request_stream_playback_batch(
                    request,
                    SimpleNamespace(subject={'username': 'alice'}),
                    MagicMock(),
                    self.redis,
                )
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(await self.snapshot(), before)

    async def asyncSetUp(self):
        self.redis = Redis.from_url(REDIS_URL)
        self.prefix = 'test:batch:' + uuid4().hex
        self.producer_keys = set()
        for name in (
            'STREAM_PLAYBACK_SESSION_PREFIX',
            'STREAM_PLAYBACK_MEDIA_SESSION_PREFIX',
            'STREAM_PLAYBACK_DEMAND_SESSION_PREFIX',
        ):
            operation = patch.object(config, name, self.prefix + ':' + name)
            operation.start()
            self.addCleanup(operation.stop)

    async def asyncTearDown(self):
        keys = {
            key async for key in self.redis.scan_iter(match=self.prefix + '*')
        }
        if keys or self.producer_keys:
            await self.redis.delete(*keys, *self.producer_keys)
        await self.redis.aclose()

    def plan(self, camera='one', session_id=None):
        plan = store.PlaybackSessionPlan(
            self.prefix,
            camera,
            'clean',
            'detail',
            None,
            session_id,
        )
        self.producer_keys.add(store._producer_key(plan.session('alice')))
        return plan

    async def snapshot(self):
        keys = {
            key async for key in self.redis.scan_iter(match=self.prefix + '*')
        }
        for key in self.producer_keys:
            if await self.redis.exists(key):
                keys.add(key.encode())
        return {key: await self.redis.dump(key) for key in keys}

    async def test_missing_or_foreign_session_leaves_whole_batch_unchanged(
        self,
    ):
        own = (
            await store.apply_playback_session_plans(
                self.redis,
                'alice',
                [self.plan()],
            )
        )[0]
        foreign = (
            await store.apply_playback_session_plans(
                self.redis,
                'bob',
                [self.plan('foreign')],
            )
        )[0]
        for session_id, status in [
            ('missing', 404),
            (foreign['session_id'], 403),
        ]:
            before = await self.snapshot()
            with self.assertRaises(HTTPException) as caught:
                await store.apply_playback_session_plans(
                    self.redis,
                    'alice',
                    [
                        self.plan('new'),
                        self.plan(
                            'updated',
                            own['session_id'],
                        ),
                        self.plan('invalid', session_id),
                    ],
                )
            self.assertEqual(caught.exception.status_code, status)
            self.assertEqual(await self.snapshot(), before)

    async def test_mixed_batch_commits_sessions_demand_and_indexes_together(
        self,
    ):
        old = (
            await store.apply_playback_session_plans(
                self.redis,
                'alice',
                [self.plan()],
            )
        )[0]
        sessions = await store.apply_playback_session_plans(
            self.redis,
            'alice',
            [
                self.plan('updated', old['session_id']),
                self.plan('new'),
            ],
        )
        for session in sessions:
            self.assertEqual(
                await store.load_playback_session(
                    self.redis,
                    session['session_id'],
                ),
                session,
            )
            self.assertTrue(
                await self.redis.exists(
                    store._producer_key(session),
                ),
            )
            self.assertIsNotNone(
                await self.redis.zscore(
                    store.playback_media_sessions_key(
                        session['base_media_path'],
                    ),
                    session['session_id'],
                ),
            )
        before = await self.snapshot()
        bodies = await playback_responses.read_playback_session_responses(
            self.redis,
            sessions,
        )
        self.assertEqual(len(bodies), 2)
        self.assertEqual(await self.snapshot(), before)

    async def test_concurrent_delete_does_not_revive_or_partially_commit(self):
        _callbacks = _ConcurrentDeleteDoesNotReviveOrPartiallyCallbacks()
        _callbacks.testcase = self
        _callbacks.session = (
            await store.apply_playback_session_plans(
                self.redis,
                'alice',
                [self.plan()],
            )
        )[0]
        _callbacks.factory = self.redis.pipeline
        _callbacks.deleted = False

        pipeline = _callbacks._call_pipeline

        with patch.object(self.redis, 'pipeline', pipeline):
            with self.assertRaises(HTTPException) as caught:
                await store.apply_playback_session_plans(
                    self.redis,
                    'alice',
                    [
                        self.plan('new'),
                        self.plan(
                            'updated',
                            _callbacks.session['session_id'],
                        ),
                    ],
                )
        self.assertEqual(caught.exception.status_code, 404)
        self.assertFalse(
            await self.redis.exists(
                store.playback_session_key(_callbacks.session['session_id']),
            ),
        )
        remaining = [
            key
            async for key in self.redis.scan_iter(
                match=config.STREAM_PLAYBACK_SESSION_PREFIX + ':*',
            )
        ]
        self.assertEqual(remaining, [])

    async def test_duplicate_session_is_rejected_before_mutation(self):
        session = (
            await store.apply_playback_session_plans(
                self.redis,
                'alice',
                [self.plan()],
            )
        )[0]
        before = await self.snapshot()
        with self.assertRaises(HTTPException) as caught:
            await store.apply_playback_session_plans(
                self.redis,
                'alice',
                [
                    self.plan('one', session['session_id']),
                    self.plan('two', session['session_id']),
                ],
            )
        self.assertEqual(caught.exception.status_code, 422)
        self.assertEqual(await self.snapshot(), before)

    async def test_language_quota_rejects_whole_batch_before_any_write(self):
        from dataclasses import replace

        plans = [
            replace(self.plan(), profile='overlay', language=language)
            for language in ('en', 'zh-TW')
        ]
        for plan in plans:
            self.producer_keys.add(store._producer_key(plan.session('alice')))
        before = await self.snapshot()
        with patch.object(config, 'OVERLAY_MAX_ACTIVE_LANGUAGES', 1):
            with self.assertRaises(HTTPException) as caught:
                await store.apply_playback_session_plans(
                    self.redis,
                    'alice',
                    plans,
                )
        self.assertEqual(caught.exception.status_code, 429)
        self.assertEqual(await self.snapshot(), before)

    async def test_renditions_share_camera_language_quota(self):
        import asyncio
        from dataclasses import replace

        plans = [
            replace(self.plan(), profile='overlay', language='en'),
            replace(
                self.plan(),
                profile='overlay',
                rendition='preview',
                language='zh-TW',
            ),
        ]
        for plan in plans:
            self.producer_keys.add(store._producer_key(plan.session('alice')))
        with patch.object(config, 'OVERLAY_MAX_ACTIVE_LANGUAGES', 1):
            results = await asyncio.gather(
                *(
                    store.apply_playback_session_plans(
                        self.redis,
                        'alice',
                        [plan],
                    )
                    for plan in plans
                ),
                return_exceptions=True,
            )
            self.assertEqual(
                sum(isinstance(value, HTTPException) for value in results),
                1,
            )
            self.assertEqual(
                sum(isinstance(value, list) for value in results),
                1,
            )
            winner = next(
                index
                for index, value in enumerate(
                    results,
                )
                if isinstance(value, list)
            )
            # A renewal of the existing language still fits the quota.
            await store.apply_playback_session_plans(
                self.redis,
                'alice',
                [plans[winner]],
            )
        leases = [await self.redis.exists(key) for key in self.producer_keys]
        self.assertEqual(sum(leases), 1)

    async def test_stale_selection_renews_neither_lease_nor_indexes(self):
        from examples.streaming_web.playback_selection import (
            select_session_playback,
        )

        session = (
            await store.apply_playback_session_plans(
                self.redis,
                'alice',
                [self.plan()],
            )
        )[0]
        await store.delete_playback_session(self.redis, session)
        before = await self.snapshot()
        with self.assertRaises(HTTPException) as caught:
            await select_session_playback(self.redis, session)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(await self.snapshot(), before)

    async def test_metadata_rejects_overlay_quota_before_response_start(self):
        from dataclasses import replace
        from examples.streaming_web import (
            streaming_metadata_service as metadata,
        )
        from examples.streaming_web.media_paths import encode_media_segment

        plan = replace(self.plan(), profile='overlay', language='en')
        self.producer_keys.add(store._producer_key(plan.session('alice')))
        with patch.object(config, 'OVERLAY_MAX_ACTIVE_LANGUAGES', 1):
            await store.apply_playback_session_plans(
                self.redis,
                'alice',
                [plan],
            )
            with patch.object(
                metadata,
                'authorise_label_access',
                new=AsyncMock(),
            ):
                with self.assertRaises(HTTPException) as caught:
                    await metadata.metadata_stream_response(
                        MagicMock(),
                        self.prefix,
                        encode_media_segment('one'),
                        'backend',
                        'zh-TW',
                        MagicMock(),
                        AsyncMock(),
                        self.redis,
                    )
        self.assertEqual(caught.exception.status_code, 429)
