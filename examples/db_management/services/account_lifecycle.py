from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from uuid import UUID
from uuid import uuid4

from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.database import engine
from examples.auth.models import AccountOperation
from examples.auth.models import User
from examples.auth.operation_database import operation_database
from examples.db_management.schemas.user import UserCreate
from examples.db_management.services import (
    keycloak_user_management_services as provider,
)
from examples.db_management.services.account_maintenance import (
    AccountMaintenance,
)
from examples.db_management.services.account_retention import (
    AccountMaintenanceSettings,
)
from examples.db_management.services.account_retention import AccountRetention
from examples.db_management.services.account_sync_steps import (
    AccountSynchronizer,
)
from examples.db_management.services.account_sync_steps import AccountSyncPlan
from src.async_tasks import finish_on_cancel

logger = logging.getLogger(__name__)


def now() -> datetime:
    return datetime.now(timezone.utc)


def account_mutable(user: User) -> None:
    if getattr(user, 'deletion_pending', False) is True:
        raise HTTPException(409, 'account_deletion_pending')


class AccountLifecycle:
    """Persist remote intent and reconcile by immutable subject."""

    def __init__(self, database: AsyncEngine = engine):
        self.database = operation_database(database)
        self.engine = self.database.engine
        self.sessions = self.database.sessions
        self.stop = asyncio.Event()
        self.settings = AccountMaintenanceSettings()
        self.retention = AccountRetention(self.sessions, self.settings)
        self.maintenance = AccountMaintenance(self)

    @classmethod
    def for_session(cls, db: AsyncSession) -> AccountLifecycle:
        bind = db.bind
        if bind is None:
            raise RuntimeError('Account operations require a bound database')
        return cls(bind)

    @asynccontextmanager
    async def ownership(
        self,
        operation_id: str,
        wait: bool = False,
    ) -> AsyncIterator[bool]:
        async with self.database.locks.begin() as connection:
            operation = (
                'pg_advisory_xact_lock'
                if wait
                else 'pg_try_advisory_xact_lock'
            )
            result = await connection.scalar(
                text(
                    f'SELECT {operation}(hashtextextended(:id, 4102))',
                ),
                {'id': operation_id},
            )
            yield True if wait else bool(result)

    async def update(self, operation_id, **values):
        if values.get('phase') == 'completed':
            values['completed_at'] = now()
        async with self.sessions() as db:
            record = await db.get(AccountOperation, operation_id)
            if record is None:
                raise RuntimeError('Account operation is missing')
            for name, value in values.items():
                setattr(record, name, value)
            await db.commit()

    @asynccontextmanager
    async def provisioning(
        self,
        payload: UserCreate,
        tenant_id: UUID,
    ) -> AsyncIterator[ProvisionedAccount]:
        operation_id = uuid4().hex
        profile = payload.profile
        assert profile is not None
        account = provider.KeycloakAccount(
            username=payload.username,
            password=payload.password,
            email=str(profile.email),
            given_name=profile.given_name,
            family_name=profile.family_name,
            temporary=payload.force_password_change,
            operation_id=operation_id,
        )
        async with self.ownership(operation_id, wait=True):
            async with self.sessions() as db:
                db.add(
                    AccountOperation(
                        id=operation_id,
                        kind='provision',
                        phase='creating',
                        tenant_id=tenant_id,
                        username=payload.username,
                        role=payload.role,
                    ),
                )
                await db.commit()
            provision = ProvisionedAccount(self, operation_id)

            try:
                await finish_on_cancel(provision.create_remote(account))
                await provider.reset_keycloak_password(
                    provision.subject,
                    password=account.password,
                    temporary=account.temporary,
                )
                yield provision
            except BaseException:
                await finish_on_cancel(self.reconcile_owned(operation_id))
                raise
            else:
                if not await finish_on_cancel(
                    self.reconcile_owned(operation_id),
                ):
                    raise HTTPException(
                        503,
                        detail={'code': 'account_reconciliation_required'},
                    )

    @asynccontextmanager
    async def updating(self, user: User, plan: AccountSyncPlan):
        operation_id = uuid4().hex
        async with self.ownership(operation_id, wait=True):
            async with self.sessions() as journal:
                journal.add(
                    AccountOperation(
                        id=operation_id,
                        kind=plan.kind,
                        phase='pending',
                        user_id=user.id,
                        tenant_id=user.tenant_id,
                        group_id=user.group_id,
                        username=user.username,
                        role=user.role,
                        subject=plan.subject,
                        sync_parts={part: 'pending' for part in plan.parts},
                        target_revision=user.state_revision,
                    ),
                )
                await journal.commit()
            try:
                yield
            except BaseException:
                await finish_on_cancel(self.reconcile_owned(operation_id))
                raise
            else:
                if not await finish_on_cancel(
                    self.reconcile_owned(operation_id),
                ):
                    raise HTTPException(
                        503,
                        detail={'code': 'account_reconciliation_required'},
                    )

    async def reconcile_owned(self, operation_id: str) -> bool:
        try:
            for _ in range(16):
                async with self.sessions() as db:
                    record = await db.get(AccountOperation, operation_id)
                    if record is None or record.phase == 'completed':
                        return True
                    complete = await AccountSynchronizer(db, record).step()
                    if complete:
                        await self.retention.complete(db, record)
                    await finish_on_cancel(db.commit())
                    if complete:
                        return True
                    if record.phase == 'uncertain':
                        return False
            raise RuntimeError(
                'Account changed repeatedly during synchronization',
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await self.defer(operation_id, type(error).__name__)
            return False

    async def defer(self, operation_id: str, error: str):
        logger.warning(
            'Account reconciliation deferred operation=%s error=%s',
            operation_id,
            error,
        )
        try:
            async with self.sessions() as db:
                record = await db.get(AccountOperation, operation_id)
                if record and record.phase != 'completed':
                    record.attempts += 1
                    record.last_error = error
                    record.next_attempt = now() + timedelta(
                        seconds=min(300, 2 ** min(record.attempts, 8)),
                    )
                    await db.commit()
        except Exception:
            logger.error(
                'Account journal unavailable operation=%s',
                operation_id,
            )

    async def purge_completed(self):
        await self.retention.purge()

    async def process(self, operation_id: str) -> bool:
        async with self.ownership(operation_id) as acquired:
            return (
                await self.reconcile_owned(operation_id) if acquired else False
            )

    async def sweep(self) -> None:
        await self.maintenance.sweep()

    def lifespan(self):
        return self.maintenance.lifespan()


class ProvisionedAccount:
    def __init__(self, lifecycle: AccountLifecycle, operation_id: str):
        self.lifecycle = lifecycle
        self.id = operation_id
        self.subject = ''

    async def create_remote(self, account: provider.KeycloakAccount):
        try:
            subject = await provider.create_keycloak_user(account)
        except HTTPException as error:
            await self.lifecycle.update(
                self.id,
                phase=(
                    'completed'
                    if error.status_code in (400, 409)
                    else 'uncertain'
                ),
                last_error=(
                    'remote_create_rejected'
                    if error.status_code in (400, 409)
                    else 'remote_create_uncertain'
                ),
            )
            raise
        self.subject = subject
        try:
            await self.lifecycle.update(
                self.id,
                subject=subject,
                phase='remote_created',
            )
        except Exception:
            logger.error(
                'Account subject requires recovery operation=%s ' 'subject=%s',
                self.id,
                subject,
            )
            raise

    async def adopt(self, db: AsyncSession, user: User) -> None:
        record = await db.get(AccountOperation, self.id)
        if record is None:
            raise RuntimeError('Provisioning journal is missing')
        record.user_id = user.id
        record.target_revision = user.state_revision
        record.group_id = user.group_id
        record.phase = 'adopted'
