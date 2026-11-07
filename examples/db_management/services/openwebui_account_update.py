from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.models import User
from examples.db_management.deps import ensure_not_super
from examples.db_management.schemas.openwebui import OpenWebUIUserUpdate
from examples.db_management.schemas.user import UserProfileUpdate
from examples.db_management.services.account_lifecycle import account_mutable
from examples.db_management.services.account_lifecycle import AccountLifecycle
from examples.db_management.services.account_lifecycle import AccountSyncPlan
from examples.db_management.services.account_state import lock_account
from examples.db_management.services.account_sync_steps import SyncPart
from examples.db_management.services.account_transaction import (
    AccountTransaction,
)
from examples.db_management.services.keycloak_user_management_services import (
    reset_keycloak_password,
)
from examples.db_management.services.password_policy import (
    validate_password_minimum,
)
from examples.db_management.services.role_policy import validate_role
from examples.db_management.services.user_management_services import (
    ensure_user_management_scope,
)
from examples.db_management.services.user_management_services import (
    get_group_or_404,
)
from examples.db_management.services.user_management_services import (
    resolve_target_group_id,
)


class OpenWebUIAccountUpdate:
    def __init__(
        self,
        target: User,
        payload: OpenWebUIUserUpdate,
        operator: User,
        db: AsyncSession,
    ) -> None:
        self.target = target
        self.payload = payload
        self.operator = operator
        self.db = db
        self.profile: dict = {}

    async def validate(self) -> None:
        account_mutable(self.target)
        ensure_not_super(self.target)
        ensure_user_management_scope(self.target, self.operator)
        if self.payload.role in {'user', 'admin'}:
            validate_role(self.operator, self.payload.role, self.target)
        if self.payload.group_id is not None:
            group_id = resolve_target_group_id(
                self.payload.group_id,
                self.operator,
            )
            if group_id is None:
                raise HTTPException(400, 'group_id is required')
            await get_group_or_404(group_id, self.db)
        if self.payload.password:
            validate_password_minimum(self.payload.password)
        if self.payload.name is not None or self.payload.email is not None:
            if self.target.profile is None:
                raise HTTPException(400, 'User profile is required')
            self.profile = UserProfileUpdate(
                user_id=self.target.id,
                family_name='' if self.payload.name is not None else None,
                given_name=self.payload.name,
                email=self.payload.email,
            ).model_dump(exclude={'user_id'}, exclude_none=True)

    def apply_local(self) -> None:
        if self.payload.role is not None:
            self.target.status = (
                'suspended' if self.payload.role == 'pending' else 'active'
            )
            if self.payload.role != 'pending':
                self.target.role = self.payload.role
        if self.payload.group_id is not None:
            self.target.group_id = self.payload.group_id
        for field, value in self.profile.items():
            setattr(self.target.profile, field, value)
        if self.profile:
            self.target.state_revision += 1

    async def execute(self, subject: str) -> None:
        self.target = await lock_account(self.db, self.target)
        await self.validate()
        parts: list[SyncPart] = []
        if self.payload.role is not None:
            parts.append('state')
        if self.profile:
            parts.append('profile')
        if self.payload.role in ('user', 'admin'):
            parts.append('roles')
        if parts:
            lifecycle = AccountLifecycle.for_session(self.db)
            async with lifecycle.updating(
                self.target,
                AccountSyncPlan('openwebui', subject, tuple(parts)),
            ):
                await self._commit_local(subject)
        else:
            await self._commit_local(subject)

    async def _commit_local(self, subject):
        async with AccountTransaction(self.db) as transaction:
            self.apply_local()
            await self.db.flush()
            if self.payload.password:
                transaction.irreversible_effect()
                try:
                    await reset_keycloak_password(
                        subject,
                        password=self.payload.password,
                        temporary=False,
                    )
                except HTTPException as error:
                    if error.status_code < 500:
                        transaction.reconciled_effect()
                    raise
