from __future__ import annotations

import asyncio
import os
from functools import partial
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import patch
from uuid import uuid4

from redis.asyncio import Redis
from sqlalchemy import select

from examples.auth.models import FcmDeviceToken
from examples.auth.models import Tenant
from examples.auth.models import User
from examples.local_notification_server import services
from examples.local_notification_server.schemas import (
    DeviceRegistrationRequest,
)
from examples.local_notification_server.token_delivery_state import (
    FcmDeliveryState,
)
from tests.database_fixtures_test import OrmDatabase


async def _concurrent_transfers_reconcile_to_durable_register(
    _capture_testcase,
    /,
    user,
):
    async with _capture_testcase.sessions() as db:
        await services.record_fcm_token_registration(
            user,
            _capture_testcase.request,
            db,
            _capture_testcase.redis,
        )


async def _first_registrations_share_one_row_register(
    request,
    _capture_testcase,
    /,
    owner,
):
    async with _capture_testcase.sessions() as db:
        return await services.record_fcm_token_registration(
            owner,
            request,
            db,
            _capture_testcase.redis,
        )


async def _loading_retries_concurrent_registration_register_before_disable(
    original,
    _capture_testcase,
    /,
    db,
    row,
    occurred_at,
):
    async with _capture_testcase.sessions() as current:
        await services.record_fcm_token_registration(
            _capture_testcase.users[0],
            _capture_testcase.request,
            current,
            _capture_testcase.redis,
        )
    return await original(db, row, occurred_at)


class _LateUnregisterCleanupPreservesANewRegistrationCallbacks:
    execute: Any
    renewed: Any
    testcase: TokenOwnershipTest

    async def _call_delayed(self, *args, **kwargs):

        if not self.renewed and args[0] == services._REMOVE_REGISTRATION_CACHE:
            self.renewed = True
            async with self.testcase.sessions() as other:
                await services.record_fcm_token_registration(
                    self.testcase.users[0],
                    self.testcase.request,
                    other,
                    self.testcase.redis,
                )
                await services.ensure_fcm_token_cache_for_users(
                    self.testcase.users,
                    other,
                    self.testcase.redis,
                )
        return await self.execute(*args, **kwargs)


class TokenOwnershipTest(OrmDatabase):
    async def test_diagnostic_cache_failure_is_repaired_before_next_send(self):
        async with self.sessions() as db:
            delivery = FcmDeliveryState(db, self.redis)
            await delivery.capture(self.users)
            with patch.object(
                self.redis,
                'delete',
                side_effect=ConnectionError('offline'),
            ):
                await delivery.invalidate([self.token])
            row = await db.scalar(select(FcmDeviceToken))
            self.assertIsNotNone(row.disabled_at)
            await services.ensure_fcm_token_cache_for_users(
                self.users,
                db,
                self.redis,
            )
            self.assertFalse(
                await self.redis.hexists(
                    f'fcm_tokens:{self.users[0]}',
                    self.token,
                ),
            )

    async def test_late_success_cannot_reactivate_unregistered_token(self):
        async with self.sessions() as db:
            delivery = FcmDeliveryState(db, self.redis)
            await delivery.capture(self.users)
            async with self.sessions() as other:
                await services.delete_fcm_token_metadata(
                    self.users[0],
                    self.token,
                    other,
                    self.redis,
                )
            await delivery.success([self.token])
            row = await db.scalar(select(FcmDeviceToken))
            self.assertIsNotNone(row.disabled_at)
            self.assertIsNone(row.last_success_at)
            await services.ensure_fcm_token_cache_for_users(
                self.users,
                db,
                self.redis,
            )
            self.assertFalse(
                await self.redis.hexists(
                    f'fcm_tokens:{self.users[0]}',
                    self.token,
                ),
            )

    async def test_stale_failure_and_invalidation_cannot_change_reregistration(
        self,
    ):
        async with self.sessions() as db:
            delivery = FcmDeliveryState(db, self.redis)
            await delivery.capture(self.users)
            async with self.sessions() as other:
                await services.record_fcm_token_registration(
                    self.users[0],
                    self.request,
                    other,
                    self.redis,
                )
            await delivery.failure([self.token])
            await delivery.invalidate([self.token])
            row = await db.scalar(select(FcmDeviceToken))
            self.assertIsNone(row.disabled_at)
            self.assertIsNone(row.last_failure_at)
            self.assertIsNone(row.failure_reason)

    async def test_delivery_results_cannot_cross_an_ownership_transfer(self):
        async with self.sessions() as db:
            delivery = FcmDeliveryState(db, self.redis)
            await delivery.capture(self.users)
            async with self.sessions() as other:
                await services.record_fcm_token_registration(
                    self.users[1],
                    self.request,
                    other,
                    self.redis,
                )
            await delivery.success([self.token])
            await delivery.invalidate([self.token])
            row = await db.scalar(select(FcmDeviceToken))
            self.assertEqual(row.user_id, self.users[1])
            self.assertIsNone(row.disabled_at)
            self.assertIsNone(row.last_success_at)

    async def test_current_registration_results_are_durable(
        self,
    ):
        async with self.sessions() as db:
            delivery = FcmDeliveryState(db, self.redis)
            await delivery.capture(self.users)
            await delivery.success([self.token])
            row = await db.scalar(select(FcmDeviceToken))
            self.assertIsNotNone(row.last_success_at)
            await delivery.invalidate([self.token])
            await db.refresh(row)
            self.assertIsNotNone(row.disabled_at)
            await services.ensure_fcm_token_cache_for_users(
                self.users,
                db,
                self.redis,
            )
            self.assertFalse(
                await self.redis.hexists(
                    f'fcm_tokens:{self.users[0]}',
                    self.token,
                ),
            )

    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.redis = Redis.from_url(os.environ['NOTIFICATION_REDIS_TEST_URL'])
        first = int(uuid4().hex[:7], 16) + 100
        self.users = [first, first + 1]
        self.token = 'device-' + uuid4().hex
        self.request = DeviceRegistrationRequest(
            device_token=self.token,
            device_lang='en-GB',
            platform='web',
        )
        encryption = patch.object(
            services.settings,
            'fcm_token_encryption_key',
            'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=',
        )
        encryption.start()
        self.addCleanup(encryption.stop)
        async with self.sessions() as db:
            tenant = Tenant(name='token-test')
            db.add(tenant)
            await db.flush()
            db.add_all(
                [
                    User(
                        id=user,
                        username=str(user),
                        tenant_id=tenant.id,
                        role='user',
                    )
                    for user in self.users
                ],
            )
            await db.commit()
        async with self.sessions() as db:
            await services.record_fcm_token_registration(
                self.users[0],
                self.request,
                db,
                self.redis,
            )
            await services.ensure_fcm_token_cache_for_users(
                self.users,
                db,
                self.redis,
            )

    async def asyncTearDown(self):
        keys = []
        for user in self.users:
            keys.extend(
                [
                    f'fcm_tokens:{user}',
                    services._token_index_key(user),
                    services._token_cache_ready_key(user),
                ],
            )
            keys.extend(
                [
                    key
                    async for key in self.redis.scan_iter(
                        match=f'fcm_token_meta:{user}:*',
                    )
                ],
            )
        await self.redis.delete(*keys)
        await self.redis.aclose()
        await super().asyncTearDown()

    async def test_transfer_removes_previous_owner_cache_and_metadata(self):
        async with self.sessions() as db:
            await services.record_fcm_token_registration(
                self.users[1],
                self.request,
                db,
                self.redis,
            )
            row = await db.scalar(select(FcmDeviceToken))
            self.assertEqual(row.user_id, self.users[1])
        self.assertFalse(
            await self.redis.hexists(
                f'fcm_tokens:{self.users[0]}',
                self.token,
            ),
        )
        self.assertFalse(
            await self.redis.exists(
                services._token_meta_key(
                    self.users[0],
                    services.fcm_token_hash(self.token),
                ),
            ),
        )
        self.assertFalse(
            await self.redis.sismember(
                services._token_index_key(self.users[0]),
                services.fcm_token_hash(self.token),
            ),
        )
        self.assertTrue(
            await self.redis.hexists(
                f'fcm_tokens:{self.users[1]}',
                self.token,
            ),
        )

    async def test_failed_cache_write_is_repaired_before_next_send(self):
        async with self.sessions() as db:
            with patch.object(
                self.redis,
                'pipeline',
                side_effect=ConnectionError('offline'),
            ):
                await services.record_fcm_token_registration(
                    self.users[1],
                    self.request,
                    db,
                    self.redis,
                )
        self.assertTrue(
            await self.redis.hexists(
                f'fcm_tokens:{self.users[0]}',
                self.token,
            ),
        )
        async with self.sessions() as db:
            await services.ensure_fcm_token_cache_for_users(
                self.users,
                db,
                self.redis,
            )
        self.assertFalse(
            await self.redis.hexists(
                f'fcm_tokens:{self.users[0]}',
                self.token,
            ),
        )
        self.assertTrue(
            await self.redis.hexists(
                f'fcm_tokens:{self.users[1]}',
                self.token,
            ),
        )
        async with self.sessions() as db:
            with patch.object(
                services,
                'refresh_fcm_token_cache_for_users',
                side_effect=AssertionError('unchanged cache rebuilt'),
            ):
                self.assertEqual(
                    await services.ensure_fcm_token_cache_for_users(
                        self.users,
                        db,
                        self.redis,
                    ),
                    0,
                )

    async def test_concurrent_transfers_reconcile_to_durable_owner(self):
        register = partial(
            _concurrent_transfers_reconcile_to_durable_register,
            self,
        )

        await asyncio.gather(
            *(register(user) for user in reversed(self.users)),
        )
        async with self.sessions() as db:
            owner = (await db.scalar(select(FcmDeviceToken))).user_id
            await services.ensure_fcm_token_cache_for_users(
                self.users,
                db,
                self.redis,
            )
        for user in self.users:
            self.assertEqual(
                bool(
                    await self.redis.hexists(f'fcm_tokens:{user}', self.token),
                ),
                user == owner,
            )

    async def test_first_registrations_share_one_row(self):
        from sqlalchemy import func

        for owners in [(self.users[0], self.users[0]), tuple(self.users)]:
            token = 'new-device-' + uuid4().hex
            request = self.request.model_copy(update={'device_token': token})

            register = partial(
                _first_registrations_share_one_row_register,
                request,
                self,
            )

            results = await asyncio.gather(
                *(register(owner) for owner in owners),
            )
            self.assertEqual(
                results[0]['registered_at'],
                results[1]['registered_at'],
            )
            async with self.sessions() as db:
                count = await db.scalar(
                    select(func.count())
                    .select_from(FcmDeviceToken)
                    .where(
                        FcmDeviceToken.device_token_hash
                        == (services.fcm_token_hash(token)),
                    ),
                )
                self.assertEqual(count, 1)
                row = await db.scalar(
                    select(FcmDeviceToken).where(
                        FcmDeviceToken.device_token_hash
                        == (services.fcm_token_hash(token)),
                    ),
                )
                self.assertGreater(row.last_seen_at, row.created_at)
                await services.ensure_fcm_token_cache_for_users(
                    self.users,
                    db,
                    self.redis,
                )
                for user in self.users:
                    cached = await self.redis.hexists(
                        f'fcm_tokens:{user}',
                        token,
                    )
                    self.assertEqual(bool(cached), user == row.user_id)

    async def test_decryption_failure_preserves_new_registration(self):
        from datetime import datetime
        from datetime import timezone
        from examples.local_notification_server.fcm_token_crypto import (
            disable_undecryptable_token,
        )

        async with self.sessions() as stale:
            row = await stale.scalar(select(FcmDeviceToken))
            row.device_token_encrypted = 'invalid-ciphertext'
            await stale.commit()
            async with self.sessions() as current:
                await services.record_fcm_token_registration(
                    self.users[1],
                    self.request,
                    current,
                    self.redis,
                )
            self.assertFalse(
                await disable_undecryptable_token(
                    stale,
                    row,
                    datetime.now(timezone.utc),
                ),
            )
            await stale.commit()
        async with self.sessions() as db:
            row = await db.scalar(select(FcmDeviceToken))
            self.assertEqual(row.user_id, self.users[1])
            self.assertIsNone(row.disabled_at)
            self.assertEqual(
                await services.load_active_fcm_device_tokens(
                    self.users[1],
                    db,
                ),
                [self.token],
            )

    async def test_loading_retries_concurrent_registration(self):
        original = services._disable_undecryptable_fcm_token
        async with self.sessions() as db:
            row = await db.scalar(select(FcmDeviceToken))
            row.device_token_encrypted = 'broken'
            await db.commit()

        register_before_disable = partial(
            _loading_retries_concurrent_registration_register_before_disable,
            original,
            self,
        )

        async with self.sessions() as db:
            with patch.object(
                services,
                '_disable_undecryptable_fcm_token',
                register_before_disable,
            ):
                self.assertEqual(
                    await services.load_active_fcm_device_tokens(
                        self.users[0],
                        db,
                    ),
                    [self.token],
                )
            row = await db.scalar(select(FcmDeviceToken))
            self.assertIsNone(row.disabled_at)

    async def test_late_unregister_cleanup_preserves_a_new_registration(self):
        _callbacks = _LateUnregisterCleanupPreservesANewRegistrationCallbacks()
        _callbacks.testcase = self
        from examples.local_notification_server import (
            notification_delivery_service as delivery,
        )
        from examples.local_notification_server.schemas import (
            DeviceUnregistrationRequest,
        )

        _callbacks.execute = self.redis.eval
        _callbacks.renewed = False

        delayed = _callbacks._call_delayed

        async with self.sessions() as db:
            with patch.object(self.redis, 'eval', side_effect=delayed):
                await delivery.unregister_fcm_device(
                    self.users[0],
                    DeviceUnregistrationRequest(device_token=self.token),
                    db,
                    self.redis,
                )
            self.assertTrue(
                await self.redis.hexists(
                    f'fcm_tokens:{self.users[0]}',
                    self.token,
                ),
            )
            with patch.object(
                services,
                'refresh_fcm_token_cache_for_users',
                new=AsyncMock(),
            ) as rebuild:
                await services.ensure_fcm_token_cache_for_users(
                    self.users,
                    db,
                    self.redis,
                )
            rebuild.assert_not_awaited()

    async def test_failed_unregister_cache_cleanup_is_repaired_from_database(
        self,
    ):
        from redis.exceptions import ConnectionError as RedisConnectionError

        async with self.sessions() as db:
            with patch.object(
                self.redis,
                'eval',
                side_effect=RedisConnectionError('offline'),
            ):
                self.assertTrue(
                    await services.delete_fcm_token_metadata(
                        self.users[0],
                        self.token,
                        db,
                        self.redis,
                    ),
                )
            await services.ensure_fcm_token_cache_for_users(
                self.users,
                db,
                self.redis,
            )
        self.assertFalse(
            await self.redis.hexists(
                f'fcm_tokens:{self.users[0]}',
                self.token,
            ),
        )
