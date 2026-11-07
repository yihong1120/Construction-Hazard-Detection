from __future__ import annotations

import asyncio
import os
import unittest
from dataclasses import replace
from typing import Any
from unittest.mock import patch
from uuid import uuid4

from fastapi import HTTPException
from redis.asyncio import Redis

from examples.auth import session_store as store
from examples.auth.media_session_request import MediaSessionOwner
from examples.auth.media_session_request import MediaSessionRequest
from examples.streaming_web import playback_config as config
from examples.streaming_web import playback_session_store as playback

REDIS_URL = os.getenv('NOTIFICATION_REDIS_TEST_URL')


class _RefreshLockReleaseCannotDeleteReplacementOwnerCallbacks:
    changed: Any
    factory: Any
    key: Any
    testcase: AtomicSessionTest

    def _call_pipeline(self, *args, **kwargs):
        _callbacks = (
            _RefreshLockReleaseCannotDeleteReplacementOwnerCallbacks2()
        )
        _callbacks.parent = self
        pipe = self.factory(*args, **kwargs)
        _callbacks.original_get = pipe.get

        get = _callbacks._call_get

        pipe.get = get
        return pipe


class _RefreshLockReleaseCannotDeleteReplacementOwnerCallbacks2:
    parent: _RefreshLockReleaseCannotDeleteReplacementOwnerCallbacks
    original_get: Any

    async def _call_get(self, name):

        result = await self.original_get(name)
        if name == self.parent.key and not self.parent.changed:
            self.parent.changed = True
            await self.parent.testcase.redis.set(
                self.parent.key,
                b'replacement',
                ex=15,
            )
        return result


@unittest.skipUnless(
    REDIS_URL,
    'Requires isolated Redis for session transactions',
)
class AtomicSessionTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.redis = Redis.from_url(REDIS_URL)
        self.prefix = 'test:session:' + uuid4().hex
        for name in (
            'AUTH_PREFIX',
            'MEDIA_PREFIX',
            'MEDIA_PUBLIC_PREFIX',
            'MEDIA_PARENT_PREFIX',
        ):
            operation = patch.object(store, name, self.prefix + ':' + name)
            operation.start()
            self.addCleanup(operation.stop)
        for name in (
            'STREAM_PLAYBACK_SESSION_PREFIX',
            'STREAM_PLAYBACK_MEDIA_SESSION_PREFIX',
            'STREAM_PLAYBACK_DEMAND_SESSION_PREFIX',
            'STREAM_PLAYBACK_SESSION_REFRESH_PREFIX',
        ):
            operation = patch.object(config, name, self.prefix + ':' + name)
            operation.start()
            self.addCleanup(operation.stop)
        self.session_id, self.session = await store.create_auth_session(
            self.redis,
            {
                'access_token': 'access',
                'refresh_token': 'refresh',
                'feature_names': [],
            },
            {'id': 1},
        )
        self.parent = store.auth_session_key(self.session_id)
        self.owner = MediaSessionOwner('alice', 1, self.parent, 'web')

    async def asyncTearDown(self) -> None:
        keys = [
            key async for key in self.redis.scan_iter(match=self.prefix + '*')
        ]
        if keys:
            await self.redis.delete(*keys)
        await self.redis.aclose()

    async def create_capability(self, owner=None):
        return await store.create_media_session(
            self.redis,
            MediaSessionRequest(
                owner=owner or self.owner,
                site='Site',
                cameras=['Camera'],
                profile='clean',
                demand_keys=[self.prefix + ':demand'],
            ),
        )

    async def test_renew_racing_with_delete_never_revives_deleted_session(
        self,
    ) -> None:
        for _ in range(10):
            token, data = await self.create_capability()
            await asyncio.gather(
                store.renew_media_session(
                    self.redis,
                    data['id'],
                    owner=self.parent,
                ),
                store.delete_media_session(
                    self.redis,
                    data['id'],
                    owner=self.parent,
                ),
            )
            self.assertIsNone(await store.get_media_session(self.redis, token))
            self.assertIsNone(
                await store.get_media_session_by_id(self.redis, data['id']),
            )
            self.assertFalse(
                await self.redis.sismember(
                    f'{store.MEDIA_PARENT_PREFIX}:'
                    f'{store._digest(self.parent)}',
                    store.media_session_key(token),
                ),
            )

    async def test_logout_prevents_stale_token_refresh_and_new_capability(
        self,
    ) -> None:
        await store.delete_auth_session(self.redis, self.session_id)
        with self.assertRaises(HTTPException) as raised:
            await store.save_auth_tokens(
                self.redis,
                store.AuthTokenUpdate(
                    self.session_id,
                    'old-owner',
                    self.session,
                    'new-access',
                    'new-refresh',
                ),
            )
        self.assertEqual(raised.exception.status_code, 401)
        self.assertIsNone(await self.redis.get(self.parent))
        with self.assertRaises(HTTPException):
            await self.create_capability()

    async def test_revocation_invalidates_orphans_with_lost_index(
        self,
    ) -> None:
        token, data = await self.create_capability()
        await self.redis.delete(
            f'{store.MEDIA_PARENT_PREFIX}:' f'{store._digest(self.parent)}',
        )
        await store.revoke_media_for_parent(self.redis, self.parent)
        self.assertIsNone(await store.get_media_session(self.redis, token))
        self.assertIsNone(
            await store.renew_media_session(
                self.redis,
                data['id'],
                owner=self.parent,
            ),
        )

    async def test_creation_checks_the_authorised_parent_generation(
        self,
    ) -> None:
        self.parent = 'native:user:' + self.prefix
        self.owner = replace(self.owner, parent=self.parent, platform='native')
        generation = await store.media_parent_generation(
            self.redis,
            self.parent,
        )
        old = replace(self.owner, generation=generation)
        await store.revoke_media_for_parent(self.redis, self.parent)
        with self.assertRaises(HTTPException):
            await self.create_capability(old)
        fresh = replace(
            self.owner,
            generation=await store.media_parent_generation(
                self.redis,
                self.parent,
            ),
        )
        token, _ = await self.create_capability(fresh)
        self.assertIsNotNone(await store.get_media_session(self.redis, token))

    async def test_refresh_lock_release_cannot_delete_replacement_owner(
        self,
    ) -> None:
        _callbacks = _RefreshLockReleaseCannotDeleteReplacementOwnerCallbacks()
        _callbacks.testcase = self
        owner = await store.acquire_refresh_lock(self.redis, self.session_id)
        assert owner is not None
        _callbacks.key = f'{self.parent}:refresh-lock'
        _callbacks.factory = self.redis.pipeline
        _callbacks.changed = False

        pipeline = _callbacks._call_pipeline

        with patch.object(self.redis, 'pipeline', pipeline):
            await store.release_refresh_lock(
                self.redis,
                self.session_id,
                owner,
            )
        self.assertEqual(await self.redis.get(_callbacks.key), b'replacement')

    async def create_playback(self, session_id=None, profile='clean'):
        return await playback.create_or_update_playback_session(
            self.redis,
            'alice',
            playback.PlaybackSessionPlan(
                'Site',
                'Camera',
                profile,
                'detail',
                'en' if profile == 'overlay' else None,
                session_id,
            ),
        )

    async def test_stale_revision_cannot_reinsert_indexes_or_delete_session(
        self,
    ) -> None:
        old = await self.create_playback()
        await playback.register_playback_session_demand(self.redis, old)
        current = await self.create_playback(old['session_id'], 'overlay')
        await playback.register_playback_session_demand(self.redis, old)
        await playback.register_playback_session_media_path(
            self.redis,
            old,
            old['base_media_path'],
        )
        self.assertIsNone(
            await self.redis.zscore(
                playback.playback_media_sessions_key(old['base_media_path']),
                old['session_id'],
            ),
        )
        self.assertFalse(
            await playback.delete_playback_session(self.redis, old),
        )
        self.assertIsNotNone(
            await playback.load_playback_session(
                self.redis,
                old['session_id'],
            ),
        )
        await playback.register_playback_session_demand(self.redis, current)
        self.assertTrue(
            await playback.delete_playback_session(self.redis, current),
        )
        self.assertIsNone(
            await playback.load_playback_session(
                self.redis,
                old['session_id'],
            ),
        )

    async def test_concurrent_playback_updates_keep_one_current_revision(
        self,
    ) -> None:
        first = await self.create_playback()
        a, b = await asyncio.gather(
            self.create_playback(first['session_id'], 'overlay'),
            self.create_playback(first['session_id']),
        )
        current = await playback.load_playback_session(
            self.redis,
            first['session_id'],
        )
        assert current is not None
        self.assertIn(current['revision'], (a['revision'], b['revision']))
        await playback.register_playback_session_demand(self.redis, current)
        await playback.register_playback_session_media_path(
            self.redis,
            current,
            current['overlay_media_path'] or current['base_media_path'],
        )
        self.assertTrue(
            await playback.delete_playback_session(self.redis, current),
        )
        await playback.refresh_playback_sessions_for_media_path(
            self.redis,
            current['overlay_media_path'] or current['base_media_path'],
        )
        self.assertIsNone(
            await playback.load_playback_session(
                self.redis,
                first['session_id'],
            ),
        )
