from __future__ import annotations

import asyncio
import os
from datetime import timedelta
from functools import partial
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch
from uuid import uuid4

from redis.asyncio import Redis
from sqlalchemy import delete
from sqlalchemy import select

from examples.auth.models import FcmDeviceToken
from examples.auth.models import Group
from examples.auth.models import Site
from examples.auth.models import SiteNotificationPreference
from examples.auth.models import Tenant
from examples.auth.models import User
from examples.auth.models import user_sites_table
from examples.local_notification_server import event_delivery
from examples.local_notification_server import site_recipient_cache as cache
from examples.local_notification_server.fcm_service import FcmSendResult
from examples.local_notification_server.notification_claim import (
    NotificationClaim,
)
from examples.local_notification_server.notification_claim import (
    NotificationLeaseLost,
)
from examples.local_notification_server.services import fcm_token_hash
from examples.local_notification_server.token_delivery_state import (
    FcmDeliveryState,
)
from tests.database_fixtures_test import OrmDatabase


async def _late_cache_rebuild_cannot_undo_rebuild(testcase, /):
    async with testcase.sessions() as db:
        return await cache.refresh_site_notification_user_cache(
            testcase.site_name,
            db,
            testcase.redis,
        )


async def _partial_success_and_new_registration_send(
    batches,
    /,
    req,
    language,
    tokens,
):
    batches.append(tokens)
    if len(batches) == 1:
        return FcmSendResult(1, 1, successful_tokens=(tokens[0],))
    return FcmSendResult(
        len(tokens),
        0,
        successful_tokens=tuple(tokens),
    )


async def _partial_success_and_new_registration_attempt(
    claim,
    testcase,
    /,
):
    async with testcase.sessions() as db:
        state = FcmDeliveryState(db, testcase.redis)
        await state.capture([testcase.user_id])
        return await event_delivery.FcmEventDelivery(
            state,
            claim,
        ).send_batch(
            SimpleNamespace(violation_id=42),
            'en',
            testcase.tokens,
        )


async def _lease_loss_cancels_dispatch_and_work(
    cancelled,
    entered,
    /,
):
    entered.set()
    try:
        await asyncio.Event().wait()
    finally:
        cancelled.set()


async def _expired_receipt_is_renewed_without_send(
    claim,
    testcase,
    /,
):
    async with testcase.sessions() as db:
        state = FcmDeliveryState(db, testcase.redis)
        await state.capture([testcase.user_id])
        return await event_delivery.FcmEventDelivery(
            state,
            claim,
        ).send_batch(
            SimpleNamespace(violation_id=42),
            'en',
            testcase.tokens,
        )


class _LateCacheRebuildCannotUndoUnsubscribeCallbacks:
    first: Any
    original: Any
    read: Any
    release: Any

    async def _call_delayed(self, site, db):

        users = await self.original(site, db)
        if self.first:
            self.first = False
            self.read.set()
            await self.release.wait()
        return users


class DeliveryConsistencyTest(OrmDatabase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.redis = Redis.from_url(os.environ['NOTIFICATION_REDIS_TEST_URL'])
        self.site_name = 'delivery-' + uuid4().hex
        self.claim_key = 'test:claim:' + uuid4().hex
        async with self.sessions() as db:
            tenant = Tenant(name='delivery')
            group = Group(name='delivery', uniform_number='12345678')
            site = Site(name=self.site_name, groups=[group])
            user = User(
                username='operator',
                role='user',
                tenant=tenant,
                group=group,
                sites=[site],
            )
            db.add_all([tenant, group, site, user])
            await db.flush()
            db.add(
                SiteNotificationPreference(
                    user_id=user.id,
                    site_id=site.id,
                    is_enabled=True,
                ),
            )
            await db.commit()
            self.user_id, self.site_id = user.id, site.id
            self.tokens = ['token-a-' + uuid4().hex, 'token-b-' + uuid4().hex]
            from datetime import datetime
            from datetime import timezone

            now = datetime.now(timezone.utc)
            db.add_all(
                [
                    FcmDeviceToken(
                        user_id=user.id,
                        device_token_hash=fcm_token_hash(token),
                        device_token_encrypted='not-used',
                        platform='web',
                        device_lang='en',
                        created_at=now,
                        updated_at=now,
                        last_seen_at=now,
                    )
                    for token in self.tokens
                ],
            )
            await db.commit()

    async def asyncTearDown(self):
        await cache.invalidate_site_notification_user_cache(
            [self.site_name],
            self.redis,
        )
        await self.redis.delete(self.claim_key)
        await self.redis.aclose()
        await super().asyncTearDown()

    async def test_late_cache_rebuild_cannot_undo_unsubscribe(self):
        _callbacks = _LateCacheRebuildCannotUndoUnsubscribeCallbacks()
        _callbacks.read = asyncio.Event()
        _callbacks.release = asyncio.Event()
        _callbacks.original = cache._fetch_site_notification_user_ids_from_db
        _callbacks.first = True

        delayed = _callbacks._call_delayed

        rebuild = partial(_late_cache_rebuild_cannot_undo_rebuild, self)

        with patch.object(
            cache,
            '_fetch_site_notification_user_ids_from_db',
            delayed,
        ):
            stale = asyncio.create_task(rebuild())
            await _callbacks.read.wait()
            async with self.sessions() as db:
                pref = await db.get(
                    SiteNotificationPreference,
                    (self.user_id, self.site_id),
                )
                pref.is_enabled = False
                await db.commit()
            await rebuild()
            _callbacks.release.set()
            self.assertEqual(await stale, [])
        async with self.sessions() as db:
            self.assertEqual(
                await cache.get_site_notification_user_ids_cached(
                    self.site_name,
                    db,
                    self.redis,
                ),
                [],
            )
        self.assertGreater(
            await self.redis.ttl(
                cache._site_user_cache_ready_key(self.site_name),
            ),
            0,
        )

    async def test_account_and_scope_changes_invalidate_without_redis_write(
        self,
    ):
        async with self.sessions() as db:
            self.assertEqual(
                await cache.get_site_notification_user_ids_cached(
                    self.site_name,
                    db,
                    self.redis,
                ),
                [self.user_id],
            )
        async with self.sessions() as db:
            user = await db.get(User, self.user_id)
            user.status = 'inactive'
            await db.commit()
        async with self.sessions() as db:
            self.assertEqual(
                await cache.get_site_notification_user_ids_cached(
                    self.site_name,
                    db,
                    self.redis,
                ),
                [],
            )
            user = await db.get(User, self.user_id)
            user.status = 'active'
            await db.commit()
        async with self.sessions() as db:
            self.assertEqual(
                await cache.get_site_notification_user_ids_cached(
                    self.site_name,
                    db,
                    self.redis,
                ),
                [self.user_id],
            )
            await db.execute(
                delete(user_sites_table).where(
                    user_sites_table.c.user_id == self.user_id,
                ),
            )
            await db.commit()
        async with self.sessions() as db:
            self.assertEqual(
                await cache.get_site_notification_user_ids_cached(
                    self.site_name,
                    db,
                    self.redis,
                ),
                [],
            )

    async def test_partial_success_and_new_registration(
        self,
    ):
        claim = NotificationClaim(self.redis, self.claim_key, 600)
        self.assertTrue(await claim.acquire())
        batches = []

        send = partial(_partial_success_and_new_registration_send, batches)

        attempt = partial(
            _partial_success_and_new_registration_attempt,
            claim,
            self,
        )

        with patch.object(event_delivery, 'build_push_task', send):
            await attempt()
            await attempt()
            await attempt()
            self.assertEqual(batches, [self.tokens, [self.tokens[1]]])
            async with self.sessions() as db:
                token = await db.scalar(
                    select(FcmDeviceToken).where(
                        FcmDeviceToken.device_token_hash
                        == fcm_token_hash(self.tokens[0]),
                    ),
                )
                token.last_seen_at += timedelta(seconds=1)
                await db.commit()
            await attempt()
            self.assertEqual(batches[-1], [self.tokens[0]])

    async def test_lease_loss_cancels_dispatch_and_preserves_new_owner(self):
        claim = NotificationClaim(self.redis, self.claim_key, 600)
        # Redis EX requires whole seconds; shorten only the heartbeat interval.
        claim.lease_seconds = 1
        self.assertTrue(await claim.acquire())
        cancelled = asyncio.Event()
        entered = asyncio.Event()

        work = partial(
            _lease_loss_cancels_dispatch_and_work,
            cancelled,
            entered,
        )

        async with claim:
            running = asyncio.create_task(claim.run(work()))
            await entered.wait()
            await self.redis.set(self.claim_key, 'new-owner', ex=60)
            with self.assertRaises(NotificationLeaseLost):
                await asyncio.wait_for(running, 2)
            self.assertTrue(cancelled.is_set())
        self.assertEqual(await self.redis.get(self.claim_key), b'new-owner')

    async def test_semantic_events_without_durable_id_do_not_share_receipts(
        self,
    ):
        claim = NotificationClaim(self.redis, self.claim_key, 15)
        self.assertTrue(await claim.acquire())
        from unittest.mock import AsyncMock

        sender = AsyncMock(return_value=FcmSendResult(2, 0))
        with patch.object(event_delivery, 'build_push_task', sender):
            for _ in range(2):
                async with self.sessions() as db:
                    state = FcmDeliveryState(db, self.redis)
                    await state.capture([self.user_id])
                    await event_delivery.FcmEventDelivery(
                        state,
                        claim,
                    ).send_batch(
                        SimpleNamespace(violation_id=None),
                        'en',
                        self.tokens,
                    )
        self.assertEqual(sender.await_count, 2)

    async def test_expired_receipt_is_renewed_without_waiting_for_cleanup(
        self,
    ):
        from datetime import datetime
        from datetime import timezone
        from unittest.mock import AsyncMock
        from examples.auth.models import NotificationDeliveryReceipt

        claim = NotificationClaim(self.redis, self.claim_key, 600)
        self.assertTrue(await claim.acquire())
        sender = AsyncMock(return_value=FcmSendResult(2, 0))

        send = partial(_expired_receipt_is_renewed_without_send, claim, self)

        with patch.object(event_delivery, 'build_push_task', sender):
            await send()
            async with self.sessions() as db:
                for receipt in (
                    await db.scalars(select(NotificationDeliveryReceipt))
                ).all():
                    receipt.expires_at = datetime.now(
                        timezone.utc,
                    ) - timedelta(days=1)
                await db.commit()
            await send()
            result = await send()
        self.assertEqual(sender.await_count, 2)
        self.assertEqual(result.skipped_count, 2)
        async with self.sessions() as db:
            for receipt in (
                await db.scalars(select(NotificationDeliveryReceipt))
            ).all():
                self.assertGreater(
                    receipt.expires_at,
                    datetime.now(timezone.utc),
                )
