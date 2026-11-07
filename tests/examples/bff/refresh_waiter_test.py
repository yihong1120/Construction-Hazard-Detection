from __future__ import annotations

import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import patch

from fastapi import HTTPException

from examples.bff import proxy_tokens as module


def _lost_ownership_does_not_reuse_callback(row):
    return (row['access'], 'refresh')


class RefreshWaiterTest(unittest.IsolatedAsyncioTestCase):
    async def wait(self, access, revision, observed=None):
        latest = {'access_expires_at': time.time() + 300, 'revision': revision}
        with (
            patch.object(
                module,
                'get_auth_session',
                AsyncMock(return_value=latest),
            ),
            patch.object(
                module,
                'auth_tokens',
                return_value=(access, 'refresh'),
            ),
            patch.object(
                module,
                'asyncio',
                SimpleNamespace(sleep=AsyncMock()),
            ),
        ):
            return await module._wait_for_proxy_session_refresh(
                None,
                'session',
                observed or module.RefreshObservation(4, 'rejected', True),
            )

    async def test_force_rejects_old_access_even_when_it_has_not_expired(self):
        for access, revision in [
            ('rejected', 4),
            ('different', 4),
            ('rejected', 5),
        ]:
            with self.subTest(access=access, revision=revision):
                with self.assertRaises(HTTPException) as error:
                    await self.wait(access, revision)
                self.assertEqual(error.exception.status_code, 503)

    async def test_force_accepts_a_valid_token_from_a_new_session_revision(
        self,
    ):
        access, latest = await self.wait('rotated', 5)
        self.assertEqual(access, 'rotated')
        self.assertEqual(latest['revision'], 5)

    async def test_peer_rotation_avoids_a_second_exchange(
        self,
    ):
        latest = {'access_expires_at': time.time() + 300, 'revision': 5}
        redis = SimpleNamespace(eval=AsyncMock(return_value=1))
        with (
            patch.object(
                module,
                'get_auth_session',
                AsyncMock(return_value=latest),
            ),
            patch.object(
                module,
                'auth_tokens',
                return_value=('peer', 'refresh'),
            ),
            patch.object(
                module,
                'refresh_oidc_tokens',
                AsyncMock(),
            ) as exchange,
            patch.object(module, 'release_refresh_lock', AsyncMock()),
        ):
            access, _ = await module.ProxySessionRefresh(
                redis,
                'sid',
                'owner',
            ).run('rejected')
        self.assertEqual(access, 'peer')
        exchange.assert_not_awaited()

    async def test_lost_ownership_does_not_reuse_the_rejected_snapshot(self):
        refresh = module.ProxySessionRefresh(None, 'sid', 'owner')
        refresh.snapshot = {'revision': 4, 'access': 'rejected'}
        for access, revision in [
            ('rejected', 4),
            ('rotated', 4),
            ('rejected', 5),
        ]:
            latest = {
                'access': access,
                'revision': revision,
                'access_expires_at': time.time() + 300,
            }
            with (
                patch.object(
                    module,
                    'get_auth_session',
                    AsyncMock(return_value=latest),
                ),
                patch.object(
                    module,
                    'auth_tokens',
                    side_effect=_lost_ownership_does_not_reuse_callback,
                ),
            ):
                with self.assertRaises(module.AuthRefreshConflict):
                    await refresh._latest_valid()
