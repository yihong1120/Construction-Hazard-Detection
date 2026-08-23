from __future__ import annotations

import unittest
from collections.abc import Set as AbstractSet
from datetime import timedelta
from functools import partial
from unittest.mock import AsyncMock
from unittest.mock import patch

import jwt

from examples.auth import session_store as store
from examples.auth.jwt_config import jwt_access
from examples.auth.jwt_config import jwt_refresh
from examples.auth.session_store import auth_session_key
from examples.auth.session_store import AUTH_SESSION_TTL_SECONDS
from examples.auth.session_store import auth_tokens
from examples.auth.session_store import create_auth_session
from examples.auth.session_store import create_media_session
from examples.auth.session_store import delete_auth_session
from examples.auth.session_store import get_auth_session
from examples.auth.session_store import get_media_session
from examples.auth.session_store import get_media_session_by_id
from examples.auth.session_store import media_session_cameras
from examples.auth.session_store import media_session_key
from examples.auth.session_store import MEDIA_SESSION_TTL_SECONDS
from examples.auth.session_store import renew_media_session
from examples.auth.session_store import touch_auth_session
from examples.bff.proxy import resolve_upstream
from examples.db_management.services import auth_services
from examples.streaming_web.playback_hls import (
    opaque_media_session_allows_path,
)


def _access_subject() -> dict[str, object]:
    """Perform access subject.

    Returns:
        The callable result.
    """
    return {
        'username': 'alice',
        'user_id': 1,
        'role': 'user',
        'jti': 'access-jti',
        'features': [],
    }


def _refresh_subject() -> dict[str, str]:
    """Perform refresh subject.

    Returns:
        The callable result.
    """
    return {
        'username': 'alice',
        'family_id': 'refresh-family',
        'token_id': 'refresh-token-id',
    }


class FakeRedis:
    """Provide FakeRedis."""

    def __init__(self) -> None:
        """Perform init."""
        self.data: dict[str, bytes] = {}
        self.ttls: dict[str, int] = {}
        self.sets: dict[str, set[bytes]] = {}

    async def get(self, key: str) -> bytes | None:
        """Perform get.

        Args:
            key: Value used by this callable.

        Returns:
            The callable result.
        """
        return self.data.get(key)

    async def getdel(self, key: str) -> bytes | None:
        """Perform getdel.

        Args:
            key: Value used by this callable.

        Returns:
            The callable result.
        """
        value = self.data.get(key)
        await self.delete(key)
        return value

    async def exists(self, key: str) -> int:
        """Perform exists.

        Args:
            key: Value used by this callable.

        Returns:
            The callable result.
        """
        return int(key in self.data or key in self.sets)

    async def set(
        self,
        key: str,
        value: bytes,
        *,
        ex: int | None = None,
        nx: bool = False,
    ) -> bool:
        """Perform set.

        Args:
            key: Value used by this callable.
            value: Value used by this callable.
            ex: Value used by this callable.
            nx: Value used by this callable.

        Returns:
            The callable result.
        """
        if nx and key in self.data:
            return False
        self.data[key] = value
        if ex is not None:
            self.ttls[key] = ex
        return True

    async def delete(self, *keys: str) -> int:
        """Perform delete.

        Args:
            *keys: Value used by this callable.

        Returns:
            The callable result.
        """
        count = 0
        for key in keys:
            count += int(key in self.data or key in self.sets)
            self.data.pop(key, None)
            self.sets.pop(key, None)
            self.ttls.pop(key, None)
        return count

    async def ttl(self, key: str) -> int:
        """Perform ttl.

        Args:
            key: Value used by this callable.

        Returns:
            The callable result.
        """
        return self.ttls.get(key, -1)

    async def sadd(self, key: str, *values: bytes) -> int:
        """Perform sadd.

        Args:
            key: Value used by this callable.
            *values: Value used by this callable.

        Returns:
            The callable result.
        """
        target = self.sets.setdefault(key, set())
        before = len(target)
        target.update(values)
        return len(target) - before

    async def smembers(self, key: str) -> AbstractSet[bytes]:
        """Perform smembers.

        Args:
            key: Value used by this callable.

        Returns:
            The callable result.
        """
        return set(self.sets.get(key, set()))

    async def srem(self, key: str, *values: bytes) -> int:
        """Perform srem.

        Args:
            key: Value used by this callable.
            *values: Value used by this callable.

        Returns:
            The callable result.
        """
        target = self.sets.get(key, set())
        before = len(target)
        target.difference_update(values)
        return before - len(target)

    async def expire(self, key: str, seconds: int) -> bool:
        """Perform expire.

        Args:
            key: Value used by this callable.
            seconds: Value used by this callable.

        Returns:
            The callable result.
        """
        self.ttls[key] = seconds
        return True


class SessionStoreTest(unittest.IsolatedAsyncioTestCase):
    """Provide SessionStoreTest."""

    def setUp(self) -> None:
        """Perform setUp."""
        self.redis = FakeRedis()

    async def test_bff_session_is_opaque_and_tokens_are_encrypted(
        self,
    ) -> None:
        """Test bff session is opaque and tokens are encrypted."""
        access = jwt_access.create_access_token(
            _access_subject(),
            timedelta(minutes=15),
        )
        refresh = jwt_refresh.create_access_token(
            _refresh_subject(),
            timedelta(days=30),
        )
        session_id, stored = await create_auth_session(
            self.redis,  # type: ignore[arg-type]
            {
                'access_token': access,
                'refresh_token': refresh,
                'feature_names': ['stream'],
            },
            {
                'id': 1,
                'username': 'alice',
                'display_name': 'Alice',
                'role': 'user',
                'group_id': 1,
                'status': 'active',
            },
        )

        self.assertNotIn(session_id, self.redis.data)
        self.assertNotIn(access, str(stored))
        self.assertNotIn(refresh, str(stored))
        loaded = await get_auth_session(
            self.redis,  # type: ignore[arg-type]
            session_id,
        )
        assert loaded is not None
        self.assertEqual(auth_tokens(loaded), (access, refresh))

    async def test_touch_auth_session_renews_idle_ttl(self) -> None:
        """Active BFF sessions receive a full rolling idle lifetime."""
        access = jwt_access.create_access_token(_access_subject())
        refresh = jwt_refresh.create_access_token(_refresh_subject())
        session_id, _ = await create_auth_session(
            self.redis,  # type: ignore[arg-type]
            {
                'access_token': access,
                'refresh_token': refresh,
                'feature_names': [],
            },
            {'id': 1, 'username': 'alice'},
        )
        self.redis.ttls[auth_session_key(session_id)] = 1

        touched = await touch_auth_session(
            self.redis,  # type: ignore[arg-type]
            session_id,
        )

        self.assertTrue(touched)
        self.assertEqual(
            self.redis.ttls[auth_session_key(session_id)],
            AUTH_SESSION_TTL_SECONDS,
        )

    async def test_parent_logout_revokes_media_without_revoking_auth_early(
        self,
    ) -> None:
        """Test parent logout revokes media without revoking auth early."""
        access = jwt_access.create_access_token(_access_subject())
        refresh = jwt_refresh.create_access_token(_refresh_subject())
        session_id, _ = await create_auth_session(
            self.redis,  # type: ignore[arg-type]
            {
                'access_token': access,
                'refresh_token': refresh,
                'feature_names': [],
            },
            {'id': 1, 'username': 'alice'},
        )
        media_token, _ = await create_media_session(
            self.redis,  # type: ignore[arg-type]
            user_id=1,
            username='alice',
            site='Site A',
            camera='Cam 1',
            profile='clean',
            parent=auth_session_key(session_id),
            platform='web',
        )
        self.assertIsNotNone(
            await get_media_session(
                self.redis,  # type: ignore[arg-type]
                media_token,
            ),
        )

        await delete_auth_session(
            self.redis,  # type: ignore[arg-type]
            session_id,
        )
        self.assertIsNone(
            await get_media_session(
                self.redis,  # type: ignore[arg-type]
                media_token,
            ),
        )

    async def test_media_session_can_be_loaded_by_public_id(self) -> None:
        """Test media session can be loaded by public id."""
        token, data = await create_media_session(
            self.redis,  # type: ignore[arg-type]
            user_id=1,
            username='alice',
            site='Site A',
            camera='Cam 1',
            profile='clean',
            parent='parent',
            platform='web',
        )

        loaded = await get_media_session_by_id(
            self.redis,  # type: ignore[arg-type]
            str(data['id']),
        )

        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded['id'], data['id'])
        self.assertNotIn(token, str(loaded))

    async def test_media_session_renew_keeps_capability_stable(self) -> None:
        """Renewal extends the original opaque media capability in place."""
        token, data = await create_media_session(
            self.redis,  # type: ignore[arg-type]
            user_id=1,
            username='alice',
            site='Site A',
            camera='Cam 1',
            profile='clean',
            parent='parent',
            platform='web',
        )

        renewed = await renew_media_session(
            self.redis,  # type: ignore[arg-type]
            str(data['id']),
            owner='parent',
        )

        self.assertIsNotNone(renewed)
        assert renewed is not None
        self.assertEqual(renewed['id'], data['id'])
        self.assertEqual(
            self.redis.ttls[media_session_key(token)],
            MEDIA_SESSION_TTL_SECONDS,
        )
        self.assertIsNotNone(
            await get_media_session(
                self.redis,  # type: ignore[arg-type]
                token,
            ),
        )

    async def test_media_session_renew_refreshes_producer_demands(
        self,
    ) -> None:
        """A valid playback lease keeps its exact publisher demand alive."""
        demand_key = 'media_clean_demand:hazard_U2l0ZSBB_Q2FtIDE_preview'
        _, data = await create_media_session(
            self.redis,  # type: ignore[arg-type]
            user_id=1,
            username='alice',
            site='Site A',
            camera='Cam 1',
            profile='clean',
            parent='parent',
            platform='web',
            quality='preview',
            demand_keys=[demand_key],
        )
        self.assertEqual(
            self.redis.ttls[demand_key],
            MEDIA_SESSION_TTL_SECONDS,
        )

        self.redis.ttls[demand_key] = 1
        renewed = await renew_media_session(
            self.redis,  # type: ignore[arg-type]
            str(data['id']),
            owner='parent',
        )

        self.assertIsNotNone(renewed)
        self.assertEqual(
            self.redis.ttls[demand_key],
            MEDIA_SESSION_TTL_SECONDS,
        )

    async def test_batch_media_session_preserves_bounded_scope(self) -> None:
        """Test batch media session preserves bounded scope."""
        token, data = await create_media_session(
            self.redis,  # type: ignore[arg-type]
            user_id=1,
            username='alice',
            site='Site A',
            cameras=['Cam 1', 'Cam 2'],
            profile='overlay',
            parent='parent',
            platform='web',
        )

        loaded = await get_media_session(
            self.redis,  # type: ignore[arg-type]
            token,
        )

        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(data['scope'], 'batch')
        self.assertEqual(media_session_cameras(loaded), ('Cam 1', 'Cam 2'))

    async def test_media_session_keeps_playback_session_descriptors(
        self,
    ) -> None:
        """Playback metadata survives media-session renewal and lookup."""
        _, session = await create_media_session(
            self.redis,  # type: ignore[arg-type]
            user_id=1,
            username='alice',
            site='Site A',
            camera='Cam 1',
            profile='clean',
            parent='parent',
            platform='web',
            quality='detail',
            playback_sessions={
                'stream-session-1': {
                    'label': 'Site A',
                    'stream_name': 'Cam 1',
                    'profile': 'clean',
                    'rendition': 'detail',
                },
            },
        )

        self.assertEqual(
            session['playback_sessions']['stream-session-1']['stream_name'],
            'Cam 1',
        )

    async def test_refresh_state_detects_reuse_and_revokes_family(
        self,
    ) -> None:
        """Test refresh state detects reuse and revokes family."""
        family = 'family-1'
        await auth_services.register_refresh_token_state(
            self.redis,  # type: ignore[arg-type]
            'refresh-token',
            'alice',
            family,
            refresh_ttl=auth_services.REFRESH_TTL,
        )
        await auth_services.consume_refresh_token_state(
            self.redis,  # type: ignore[arg-type]
            'refresh-token',
            family,
            'alice',
            refresh_ttl=auth_services.REFRESH_TTL,
            revoke_refresh_family_fn=partial(
                auth_services.revoke_refresh_family,
                refresh_ttl=auth_services.REFRESH_TTL,
            ),
            revoke_user_access_tokens_fn=AsyncMock(),
        )
        with self.assertRaisesRegex(Exception, 'Refresh token reused'):
            await auth_services.consume_refresh_token_state(
                self.redis,  # type: ignore[arg-type]
                'refresh-token',
                family,
                'alice',
                refresh_ttl=auth_services.REFRESH_TTL,
                revoke_refresh_family_fn=partial(
                    auth_services.revoke_refresh_family,
                    refresh_ttl=auth_services.REFRESH_TTL,
                ),
                revoke_user_access_tokens_fn=AsyncMock(),
            )
        self.assertEqual(
            await self.redis.get(
                auth_services._refresh_family_revoked_key(family),
            ),
            b'1',
        )

    def test_media_scope_is_exact(self) -> None:
        """Test media scope is exact."""
        session = {
            'site': 'Site A',
            'camera': 'Cam 1',
            'cameras': ['Cam 1'],
            'profile': 'clean',
            'quality': 'detail',
        }
        self.assertTrue(
            opaque_media_session_allows_path(
                session,
                'hazard_U2l0ZSBB_Q2FtIDE',
            ),
        )
        self.assertFalse(
            opaque_media_session_allows_path(
                session,
                'hazard_U2l0ZSBB_Q2FtIDI',
            ),
        )

    def test_batch_media_scope_allows_only_listed_cameras(self) -> None:
        """Test batch media scope allows only listed cameras."""
        clean = {
            'site': 'Site A',
            'camera': None,
            'cameras': ['Cam 1', 'Cam 2'],
            'scope': 'batch',
            'profile': 'clean',
            'quality': 'detail',
        }
        self.assertTrue(
            opaque_media_session_allows_path(
                clean,
                'hazard_U2l0ZSBB_Q2FtIDE',
            ),
        )
        self.assertTrue(
            opaque_media_session_allows_path(
                clean,
                'hazard_U2l0ZSBB_Q2FtIDI',
            ),
        )
        self.assertFalse(
            opaque_media_session_allows_path(
                clean,
                'hazard_U2l0ZSBB_Q2FtIDM',
            ),
        )

        overlay = {**clean, 'profile': 'overlay'}
        self.assertTrue(
            opaque_media_session_allows_path(
                overlay,
                'hazard_U2l0ZSBB_Q2FtIDE_annotated_emgtVFc',
            ),
        )
        self.assertFalse(
            opaque_media_session_allows_path(
                overlay,
                'hazard_U2l0ZSBB_Q2FtIDM_annotated_emgtVFc',
            ),
        )

        preview_overlay = {
            **overlay,
            'quality': 'preview',
        }
        self.assertTrue(
            opaque_media_session_allows_path(
                preview_overlay,
                'hazard_U2l0ZSBB_Q2FtIDE_preview_annotated_emgtVFc',
            ),
        )
        self.assertFalse(
            opaque_media_session_allows_path(
                preview_overlay,
                'hazard_U2l0ZSBB_Q2FtIDE_annotated_emgtVFc',
            ),
        )

    def test_bff_proxy_has_no_arbitrary_url_escape(self) -> None:
        """Test bff proxy has no arbitrary url escape."""
        with self.assertRaisesRegex(Exception, 'bff_route_not_allowed'):
            resolve_upstream('https://attacker.example/private')


if __name__ == '__main__':
    unittest.main()


class TestSessionStoreCoverage(unittest.IsolatedAsyncioTestCase):
    """Provide TestSessionStoreCoverage."""

    def setUp(self) -> None:
        """Perform setUp."""
        self.redis = FakeRedis()

    def test_text_crypto_and_jwt_helpers(self) -> None:
        """Test text crypto and jwt helpers."""
        self.assertIsNone(store._text(None))
        self.assertEqual(store._text(b'value'), 'value')
        self.assertEqual(store._digest('value'), store._digest('value'))
        with patch.dict(
            'os.environ',
            {'BFF_TOKEN_ENCRYPTION_KEY': 'test-key'},
        ):
            encrypted = store._encrypt('secret')
            self.assertNotEqual(encrypted, 'secret')
            self.assertEqual(store._decrypt(encrypted), 'secret')

        token = jwt.encode({'exp': 1234}, 'test-key' * 8, algorithm='HS256')
        self.assertEqual(store._jwt_exp(token), 1234)
        self.assertEqual(store._jwt_exp('not-a-jwt'), 0)

    async def test_auth_session_token_save(self) -> None:
        """Test auth session token save."""
        self.assertIsNone(await store.get_auth_session(self.redis, None))
        session_id = 'session'
        key = store.auth_session_key(session_id)
        self.assertIsNone(await store.get_auth_session(self.redis, session_id))
        self.redis.data[key] = b'{"revoked":true}'
        self.assertIsNone(await store.get_auth_session(self.redis, session_id))

        _, session = await store.create_auth_session(
            self.redis,
            {
                'access_token': 'access',
                'refresh_token': 'refresh',
                'feature_names': ['stream'],
            },
            {'id': 1, 'username': 'alice'},
        )
        self.assertEqual(session['feature_names'], ['stream'])
        await store.save_auth_tokens(
            self.redis,
            session_id,
            session,
            'next-access',
            'next-refresh',
            feature_names=['stream'],
        )
        self.assertEqual(
            store.auth_tokens(session),
            ('next-access', 'next-refresh'),
        )
        self.assertEqual(session['feature_names'], ['stream'])

        await store.delete_auth_session(self.redis, None)
        self.assertTrue(await store.touch_auth_session(self.redis, session_id))

    async def test_refresh_lock_acquisition_and_release_ownership(
        self,
    ) -> None:
        """Test refresh lock acquisition and release ownership."""
        session_id = 'session'
        owner = await store.acquire_refresh_lock(self.redis, session_id)
        self.assertIsNotNone(owner)
        self.assertIsNone(
            await store.acquire_refresh_lock(self.redis, session_id),
        )
        await store.release_refresh_lock(
            self.redis,
            session_id,
            'different-owner',
        )
        key = f"{store.auth_session_key(session_id)}:refresh-lock"
        self.assertIn(key, self.redis.data)
        await store.release_refresh_lock(self.redis, session_id, str(owner))
        self.assertNotIn(key, self.redis.data)

    async def test_media_creation_scope_and_helpers(self) -> None:
        """Test media creation scope and helpers."""
        with self.assertRaises(ValueError):
            await store.create_media_session(
                self.redis,
                user_id=1,
                username='alice',
                site='Site',
                profile='clean',
                parent='parent',
                platform='web',
            )

        _, session = await store.create_media_session(
            self.redis,
            user_id=1,
            username='alice',
            site='Site',
            cameras=['Cam 1', 'Cam 1', 'Cam 2'],
            profile='clean',
            parent='parent',
            platform='web',
            language='zh-TW',
            quality='preview',
            purpose='wall',
            demand_keys=['demand:one', 'demand:one'],
        )
        self.assertEqual(session['scope'], 'batch')
        self.assertEqual(session['cameras'], ['Cam 1', 'Cam 2'])
        self.assertEqual(session['demand_keys'], ['demand:one'])
        self.assertEqual(
            self.redis.ttls['demand:one'],
            store.MEDIA_SESSION_TTL_SECONDS,
        )
        self.assertEqual(
            store.media_session_demand_keys(
                {'demand_keys': ['one', 'one']},
            ),
            ('one',),
        )

    async def test_media_lookup_rejects_missing_and_expired_data(
        self,
    ) -> None:
        """Test media lookup rejects missing and expired data."""
        self.assertIsNone(await store.get_media_session(self.redis, None))
        token = 'token'
        token_key = store.media_session_key(token)
        self.redis.data[token_key] = b'{"expires_at":0}'
        self.assertIsNone(await store.get_media_session(self.redis, token))

        self.assertIsNone(
            await store.get_media_session_by_id(self.redis, None),
        )
        self.assertIsNone(
            await store.get_media_session_by_id(self.redis, 'public'),
        )
        public_key = f"{store.MEDIA_PUBLIC_PREFIX}:public"
        self.redis.data[public_key] = token_key.encode('utf-8')
        self.redis.data.pop(token_key)
        self.assertIsNone(
            await store.get_media_session_by_id(self.redis, 'public'),
        )
        self.redis.data[token_key] = b'{"expires_at":0}'
        self.assertIsNone(
            await store.get_media_session_by_id(self.redis, 'public'),
        )

    async def test_renew_media_session_rejects_unowned_or_expired_sessions(
        self,
    ) -> None:
        """Test renew media session rejects unowned or expired sessions."""
        public_id = 'public'
        public_key = f"{store.MEDIA_PUBLIC_PREFIX}:{public_id}"
        self.assertIsNone(
            await store.renew_media_session(
                self.redis,
                public_id,
                owner='parent',
            ),
        )

        token_key = 'media:key'
        self.redis.data[public_key] = token_key.encode('utf-8')
        self.assertIsNone(
            await store.renew_media_session(
                self.redis,
                public_id,
                owner='parent',
            ),
        )
        self.assertNotIn(public_key, self.redis.data)

        self.redis.data[public_key] = token_key.encode('utf-8')
        self.redis.data[token_key] = (
            b'{"parent":"other","expires_at":9999999999}'
        )
        self.assertIsNone(
            await store.renew_media_session(
                self.redis,
                public_id,
                owner='parent',
            ),
        )
        self.redis.data[token_key] = b'{"parent":"parent","expires_at":0}'
        self.assertIsNone(
            await store.renew_media_session(
                self.redis,
                public_id,
                owner='parent',
            ),
        )

    async def test_delete_and_revoke_media_session_edge_cases(self) -> None:
        """Test delete and revoke media session edge cases."""
        public_id = 'public'
        public_key = f"{store.MEDIA_PUBLIC_PREFIX}:{public_id}"
        self.assertFalse(
            await store.delete_media_session(self.redis, public_id),
        )

        token_key = 'media:key'
        self.redis.data[public_key] = token_key.encode('utf-8')
        self.assertFalse(
            await store.delete_media_session(self.redis, public_id),
        )
        self.assertNotIn(public_key, self.redis.data)
        self.redis.data[public_key] = token_key.encode('utf-8')
        self.redis.data[token_key] = b'{"parent":"other"}'
        self.assertFalse(
            await store.delete_media_session(
                self.redis,
                public_id,
                owner='parent',
            ),
        )
        self.assertTrue(
            await store.delete_media_session(
                self.redis,
                public_id,
                owner='other',
            ),
        )

        parent = 'parent'
        parent_key = f"{store.MEDIA_PARENT_PREFIX}:{store._digest(parent)}"
        self.redis.sets[parent_key] = {b'valid'}
        self.redis.data['valid'] = b'{"id":"public-id"}'
        self.redis.data[f"{store.MEDIA_PUBLIC_PREFIX}:public-id"] = b'valid'
        await store.revoke_media_for_parent(self.redis, parent)
        self.assertNotIn(parent_key, self.redis.sets)
        self.assertNotIn(
            f"{store.MEDIA_PUBLIC_PREFIX}:public-id",
            self.redis.data,
        )
        self.assertNotIn('valid', self.redis.data)


if __name__ == '__main__':
    unittest.main()
