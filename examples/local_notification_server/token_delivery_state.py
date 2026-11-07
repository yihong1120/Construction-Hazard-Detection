from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import datetime
from datetime import timezone
from typing import Literal

from redis.asyncio import Redis
from sqlalchemy import and_
from sqlalchemy import or_
from sqlalchemy import select
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.models import FcmDeviceToken
from examples.local_notification_server.fcm_token_crypto import fcm_token_hash
from examples.local_notification_server.fcm_token_keys import (
    _token_cache_ready_key,
)

logger = logging.getLogger(__name__)


class FcmDeliveryState:
    """Apply delivery results only to registrations captured before sending.

    Registration advances ``last_seen_at`` under its row lock.
    Diagnostics never change this revision or reactivate a token. A newer
    registration cannot be overwritten. Redis is a rebuildable delivery cache.
    """

    def __init__(self, db: AsyncSession, redis: Redis) -> None:
        self.db = db
        self.redis = redis
        self.registrations: list[tuple[int, str, datetime]] = []

    async def capture(self, user_ids: list[int]) -> None:
        result = await self.db.execute(
            select(
                FcmDeviceToken.user_id,
                FcmDeviceToken.device_token_hash,
                FcmDeviceToken.last_seen_at,
            ).where(
                FcmDeviceToken.user_id.in_(user_ids),
                FcmDeviceToken.disabled_at.is_(None),
            ),
        )
        self.registrations = [tuple(row) for row in result.all()]

    async def success(self, tokens: Iterable[str]) -> None:
        await self._record('success', tokens)

    async def failure(self, tokens: Iterable[str]) -> None:
        await self._record('failure', tokens)

    async def invalidate(self, tokens: Iterable[str]) -> None:
        await self._record('invalid', tokens)

    async def _record(
        self, outcome: Literal['success', 'failure', 'invalid'],
        tokens: Iterable[str],
    ) -> None:
        hashes = {fcm_token_hash(token) for token in tokens}
        conditions = [
            and_(
                FcmDeviceToken.user_id == user,
                FcmDeviceToken.device_token_hash == token_hash,
                FcmDeviceToken.last_seen_at == revision,
            )
            for user, token_hash, revision in self.registrations
            if token_hash in hashes
        ]
        if not conditions:
            return
        now = datetime.now(timezone.utc)
        values: dict[str, object] = {'updated_at': now}
        if outcome == 'success':
            values.update(
                last_success_at=now, last_failure_at=None, failure_reason=None,
            )
        else:
            values.update(
                last_failure_at=now,
                failure_reason='invalid_token' if outcome == 'invalid'
                else 'fcm_error',
            )
            if outcome == 'invalid':
                values['disabled_at'] = now
        result = await self.db.execute(
            update(FcmDeviceToken).where(
                FcmDeviceToken.disabled_at.is_(None), or_(*conditions),
            ).values(**values).returning(FcmDeviceToken.user_id),
        )
        changed_users = set(result.scalars().all())
        await self.db.commit()
        if changed_users:
            try:
                await self.redis.delete(
                    *[
                        _token_cache_ready_key(user) for user in changed_users
                    ],
                )
            except Exception:
                # The next send also compares database revisions, so a failed
                # invalidation cannot make stale ownership authoritative.
                logger.warning('FCM diagnostic cache reconciliation deferred')
