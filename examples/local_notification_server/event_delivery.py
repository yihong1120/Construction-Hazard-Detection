from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime
from datetime import timedelta
from datetime import timezone

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from examples.auth.models import FcmDeviceToken
from examples.auth.models import NotificationDeliveryReceipt
from examples.local_notification_server.fcm_service import FcmSendResult
from examples.local_notification_server.fcm_token_crypto import fcm_token_hash
from examples.local_notification_server.lang_config import NotificationLanguage
from examples.local_notification_server.notification_claim import (
    NotificationClaim,
)
from examples.local_notification_server.push_dispatch import build_push_task
from examples.local_notification_server.schemas import SiteNotifyRequest
from examples.local_notification_server.token_delivery_state import (
    FcmDeliveryState,
)


class FcmEventDelivery:
    """Remember confirmed delivery for each event and device registration."""

    def __init__(
        self,
        state: FcmDeliveryState,
        claim: NotificationClaim,
    ) -> None:
        self.state = state
        self.claim = claim
        self.digest = hashlib.sha256(claim.key.encode()).hexdigest()
        self._db_lock = asyncio.Lock()

    async def send_batch(
        self,
        req: SiteNotifyRequest,
        language: NotificationLanguage,
        tokens: list[str],
    ) -> FcmSendResult:
        async with self._db_lock:
            pending = await self._pending(tokens, req.violation_id is not None)
        await self.claim.guard()
        if not pending:
            return FcmSendResult(0, 0, skipped_count=len(tokens))
        result = await build_push_task(req, language, list(pending))
        successful = set(result.successful_tokens)
        if result.success_count == len(pending) and result.failure_count == 0:
            successful.update(pending)
        if successful and req.violation_id is not None:
            async with self._db_lock:
                now = datetime.now(timezone.utc)
                rows = [
                    dict(
                        event_digest=self.digest,
                        user_id=pending[token][0],
                        token_hash=fcm_token_hash(token),
                        registration_at=pending[token][1],
                        delivered_at=now,
                        expires_at=now + timedelta(days=7),
                    )
                    for token in successful
                    if token in pending
                ]
                if rows:
                    statement = insert(NotificationDeliveryReceipt).values(
                        rows,
                    )
                    await self.state.db.execute(
                        statement.on_conflict_do_update(
                            index_elements=[
                                NotificationDeliveryReceipt.event_digest,
                                NotificationDeliveryReceipt.user_id,
                                NotificationDeliveryReceipt.token_hash,
                                NotificationDeliveryReceipt.registration_at,
                            ],
                            set_={
                                'delivered_at': (
                                    statement.excluded.delivered_at
                                ),
                                'expires_at': statement.excluded.expires_at,
                            },
                            where=(
                                NotificationDeliveryReceipt.expires_at
                                <= statement.excluded.delivered_at
                            ),
                        ),
                    )
                    await self.state.db.commit()
        return result

    async def _pending(self, tokens: list[str], durable: bool) -> dict:
        db = self.state.db
        captured = {
            digest: (user, revision)
            for user, digest, revision in self.state.registrations
        }
        hashes = [fcm_token_hash(token) for token in tokens]
        live = (
            await db.execute(
                select(
                    FcmDeviceToken.user_id,
                    FcmDeviceToken.device_token_hash,
                    FcmDeviceToken.last_seen_at,
                ).where(
                    FcmDeviceToken.device_token_hash.in_(hashes),
                    FcmDeviceToken.disabled_at.is_(None),
                ),
            )
        ).all()
        current = {
            digest: (user, revision)
            for user, digest, revision in live
            if captured.get(digest) == (user, revision)
        }
        if not durable:
            await db.commit()
            return {
                token: current[digest]
                for token in tokens
                if (digest := fcm_token_hash(token)) in current
            }
        now = datetime.now(timezone.utc)
        receipts = (
            await db.execute(
                select(
                    NotificationDeliveryReceipt.user_id,
                    NotificationDeliveryReceipt.token_hash,
                    NotificationDeliveryReceipt.registration_at,
                ).where(
                    NotificationDeliveryReceipt.event_digest == self.digest,
                    NotificationDeliveryReceipt.token_hash.in_(hashes),
                    NotificationDeliveryReceipt.expires_at > now,
                ),
            )
        ).all()
        completed = {
            (user, digest, revision) for user, digest, revision in receipts
        }
        await db.commit()
        return {
            token: current[digest]
            for token in tokens
            if (digest := fcm_token_hash(token)) in current
            and (
                current[digest][0],
                digest,
                current[digest][1],
            )
            not in completed
        }
