from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import TypeAdapter
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.config import Settings
from examples.auth.models import AccountOperation
from examples.auth.models import User
from examples.auth.models import UserIdentity
from examples.db_management.services import (
    keycloak_user_management_services as provider,
)
from examples.db_management.services.account_state import lock_account

SyncPart = Literal['state', 'profile', 'roles']
SyncState = Literal['pending', 'completed', 'superseded']
PROGRESS = TypeAdapter(dict[SyncPart, SyncState])


@dataclass(frozen=True)
class AccountSyncPlan:
    kind: Literal['status', 'profile', 'rename', 'openwebui']
    subject: str
    parts: tuple[SyncPart, ...] = ()

    def __post_init__(self):
        if self.kind not in ('status', 'profile', 'rename', 'openwebui'):
            raise ValueError('Unsupported account update')
        if self.kind == 'openwebui' and (
            not self.parts
            or not set(self.parts) <= {'state', 'profile', 'roles'}
        ):
            raise ValueError('Invalid OpenWebUI synchronization parts')


class AccountSynchronizer:
    def __init__(self, db: AsyncSession, record: AccountOperation):
        self.db, self.record = db, record

    async def step(self) -> bool:
        handler = {
            'delete': self._delete,
            'provision': self._provision,
            'status': self._update,
            'profile': self._update,
            'rename': self._update,
            'openwebui': self._openwebui,
        }.get(self.record.kind)
        if handler is None:
            raise ValueError('Unsupported account operation')
        return await handler()

    async def _delete(self):
        db, record = self.db, self.record
        if record.user_id is None:
            raise ValueError('Deletion requires a local user ID')
        if record.subject:
            await provider.delete_keycloak_user(record.subject)
        user = await db.get(User, record.user_id)
        if user is not None:
            user = await lock_account(db, user, wait=False)
            if not user.deletion_pending:
                raise RuntimeError('Deletion intent changed')
            await db.delete(user)
        return True

    async def _update(self):
        db, record = self.db, self.record
        user = await db.get(User, record.user_id)
        if user is None:
            return True
        user = await lock_account(db, user, wait=False)
        if user.tenant_id != record.tenant_id:
            raise RuntimeError('Account update scope changed')
        record.target_revision = user.state_revision
        if user.deletion_pending:
            return True
        if not record.subject:
            raise RuntimeError('Account update subject is missing')
        if record.kind == 'status':
            await provider.set_keycloak_user_enabled(
                record.subject,
                enabled=user.status == 'active',
            )
        elif record.kind == 'rename':
            await provider.update_keycloak_user(
                record.subject,
                username=user.username,
            )
        elif user.profile is not None:
            await provider.update_keycloak_user(
                record.subject,
                email=user.profile.email,
                given_name=user.profile.given_name,
                family_name=user.profile.family_name,
            )
        return True

    async def _provision(self):
        db, record = self.db, self.record
        if not record.subject:
            record.phase = 'uncertain'
            record.last_error = 'remote_subject_unknown'
            return False
        user = await db.scalar(
            select(User)
            .join(UserIdentity)
            .where(
                UserIdentity.provider == Settings().oidc_identity_provider,
                UserIdentity.provider_user_id == record.subject,
            ),
        )
        if user is not None:
            user = await lock_account(db, user, wait=False)
            if user.tenant_id != record.tenant_id:
                raise RuntimeError('Identity adoption scope changed')
            record.target_revision = user.state_revision
            await provider.set_keycloak_user_enabled(
                record.subject,
                enabled=user.status == 'active' and not user.deletion_pending,
            )
            record.user_id = user.id
        else:
            await provider.delete_keycloak_user(record.subject)
        return True

    async def _openwebui(self):
        db, record = self.db, self.record
        progress = PROGRESS.validate_python(record.sync_parts)
        if not progress or not set(progress) <= {'state', 'profile', 'roles'}:
            raise ValueError('Invalid OpenWebUI operation')
        user = await db.get(User, record.user_id)
        if user is None:
            record.sync_parts = {part: 'superseded' for part in progress}
            return True
        user = await lock_account(db, user, wait=False)
        if user.tenant_id != record.tenant_id or not record.subject:
            raise RuntimeError('OpenWebUI synchronization scope changed')
        if user.deletion_pending:
            record.sync_parts = {part: 'superseded' for part in progress}
            return True
        if record.target_revision != user.state_revision:
            progress = {part: 'pending' for part in progress}
        record.target_revision = user.state_revision
        order = (
            ('roles', 'profile', 'state')
            if user.status == 'active'
            else ('state', 'roles', 'profile')
        )
        part = next(
            (part for part in order if progress.get(part) == 'pending'),
            None,
        )
        if part == 'state':
            await provider.set_keycloak_user_enabled(
                record.subject,
                enabled=user.status == 'active',
            )
        elif part == 'profile':
            if user.profile is None:
                raise RuntimeError('OpenWebUI profile is missing')
            await provider.update_keycloak_user(
                record.subject,
                email=user.profile.email,
                given_name=user.profile.given_name,
                family_name=user.profile.family_name,
            )
        elif part == 'roles':
            await provider.synchronize_openwebui_roles(
                record.subject,
                user.role,
            )
        if part is not None:
            progress[part] = 'completed'
        record.sync_parts = progress
        return all(value != 'pending' for value in progress.values())
