from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from uuid import uuid4

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.models import AccountDeletionReceipt
from examples.auth.models import AccountOperation
from examples.auth.models import User
from examples.db_management.deps import ensure_not_super
from examples.db_management.schemas.user import SetUserStatus
from examples.db_management.schemas.user import UpdatePasswordById
from examples.db_management.schemas.user import UpdateUserGroup
from examples.db_management.schemas.user import UpdateUserRole
from examples.db_management.schemas.user import UserCreate
from examples.db_management.schemas.user import UserDelete
from examples.db_management.schemas.user import UserProfileUpdate
from examples.db_management.schemas.user import UserRead
from examples.db_management.services.account_lifecycle import account_mutable
from examples.db_management.services.account_lifecycle import AccountLifecycle
from examples.db_management.services.account_lifecycle import AccountSyncPlan
from examples.db_management.services.account_state import lock_account
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
from examples.db_management.services.site_services import (
    list_site_ids_for_group,
)
from examples.db_management.services.site_services import (
    seed_site_notification_preferences,
)
from examples.db_management.services.user_management_services import (
    ensure_user_management_scope,
)
from examples.db_management.services.user_management_services import (
    get_group_or_404,
)
from examples.db_management.services.user_management_services import (
    load_user_read,
)
from examples.db_management.services.user_management_services import (
    resolve_target_group_id,
)
from examples.db_management.services.user_services import (
    create_oidc_managed_user,
)
from examples.db_management.services.user_services import (
    ensure_keycloak_subject_for_user,
)
from examples.db_management.services.user_services import get_user_by_id
from examples.db_management.services.user_services import (
    keycloak_subject_for_user,
)


@dataclass(frozen=True)
class CreatedAccount:
    user: User
    subject: str


@asynccontextmanager
async def creating_account(
    payload: UserCreate,
    db: AsyncSession,
    me: User,
) -> AsyncIterator[CreatedAccount]:
    target_group_id = resolve_target_group_id(
        payload.group_id,
        me,
        default_to_operator_group=True,
    )
    validate_role(me, payload.role)

    if payload.profile is None:
        raise HTTPException(
            400,
            'A profile with email and name is required for Keycloak.',
        )
    validate_password_minimum(payload.password)
    lifecycle = AccountLifecycle.for_session(db)
    try:
        async with lifecycle.provisioning(
            payload,
            me.tenant_id,
        ) as remote, AccountTransaction(db):
            subject = remote.subject
            new_user = await create_oidc_managed_user(
                payload=payload.model_copy(
                    update={'group_id': target_group_id},
                ),
                tenant_id=me.tenant_id,
                keycloak_subject=subject,
                db=db,
            )
            if target_group_id:
                site_ids = await list_site_ids_for_group(target_group_id, db)
                await seed_site_notification_preferences(
                    user_ids=[new_user.id],
                    site_ids=site_ids,
                    db=db,
                )
            yield CreatedAccount(new_user, subject)
            await remote.adopt(db, new_user)
    except IntegrityError as exc:
        raise HTTPException(409, 'username_or_email_already_exists') from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, 'user_identity_persistence_failed') from exc


async def add_user(
    payload: UserCreate,
    db: AsyncSession,
    me: User,
) -> UserRead:
    async with creating_account(payload, db, me) as account:
        return await load_user_read(account.user.id, db)


async def remove_user(
    payload: UserDelete,
    db: AsyncSession,
    me: User,
) -> dict[str, str]:
    """Delete a user by user ID.

    Args:
        payload: Dictionary containing 'user_id'.
        db: Async database session.

    Returns:
        Confirmation message.
    """
    user = await db.scalar(
        select(User)
        .where(User.id == payload.user_id)
        .with_for_update(of=User),
    )
    operation = await db.scalar(
        select(AccountOperation)
        .where(
            AccountOperation.kind == 'delete',
            AccountOperation.user_id == payload.user_id,
        )
        .order_by(AccountOperation.created_at.desc())
        .limit(1),
    )
    if user is None:
        if operation is None:
            receipt = await db.get(AccountDeletionReceipt, payload.user_id)
            if receipt is None or receipt.expires_at <= datetime.now(
                timezone.utc,
            ):
                raise HTTPException(404, 'User not found.')
            target = User(
                id=receipt.user_id,
                username=receipt.username,
                role=receipt.role,
                tenant_id=receipt.tenant_id,
                group_id=receipt.group_id,
            )
            ensure_not_super(target)
            ensure_user_management_scope(target, me)
            return {'message': 'User deleted successfully.'}
        target = User(
            id=operation.user_id,
            username=operation.username,
            role=operation.role,
            tenant_id=operation.tenant_id,
            group_id=operation.group_id,
        )
        ensure_not_super(target)
        ensure_user_management_scope(target, me)
        if operation.phase == 'completed':
            return {'message': 'User deleted successfully.'}
    else:
        ensure_not_super(user)
        ensure_user_management_scope(user, me)
        if not user.deletion_pending:
            operation = AccountOperation(
                id=uuid4().hex,
                kind='delete',
                phase='pending',
                user_id=user.id,
                username=user.username,
                tenant_id=user.tenant_id,
                group_id=user.group_id,
                role=user.role,
                subject=await keycloak_subject_for_user(user, db),
            )
            db.add(operation)
            user.status = 'suspended'
            user.deletion_pending = True
        if operation is None:
            raise HTTPException(
                503,
                detail={'code': 'account_reconciliation_required'},
            )
    await db.commit()
    lifecycle = AccountLifecycle.for_session(db)
    if not await lifecycle.process(operation.id):
        raise HTTPException(503, detail={'code': 'account_deletion_pending'})
    return {'message': 'User deleted successfully.'}


async def admin_update_pwd_by_id(
    payload: UpdatePasswordById,
    db: AsyncSession,
    me: User,
    temporary: bool = True,
) -> dict[str, str]:
    """Admin update user's password by user ID.

    Args:
        payload: Contains user_id and new password.
        db: Async database session.

    Returns:
        Confirmation message.
    """
    user = await lock_account(db, await get_user_by_id(payload.user_id, db))
    account_mutable(user)
    ensure_not_super(user)
    ensure_user_management_scope(user, me)
    validate_password_minimum(payload.new_password)
    subject = await ensure_keycloak_subject_for_user(user, db)
    await reset_keycloak_password(
        subject,
        password=payload.new_password,
        temporary=temporary,
    )
    await db.commit()
    return {'message': 'Password updated successfully by user ID.'}


async def update_user_status(
    payload: SetUserStatus,
    db: AsyncSession,
    me: User,
) -> dict[str, str]:
    """Set a user's account status.

    Args:
        payload: User ID and target status.
        db: Async database session.

    Returns:
        Confirmation message.
    """
    user = await lock_account(db, await get_user_by_id(payload.user_id, db))
    account_mutable(user)
    ensure_not_super(user)
    ensure_user_management_scope(user, me)
    subject = await ensure_keycloak_subject_for_user(user, db)
    lifecycle = AccountLifecycle.for_session(db)
    async with lifecycle.updating(
        user,
        AccountSyncPlan('status', subject),
    ), AccountTransaction(
        db,
    ):
        user.status = payload.status
        await db.flush()
    return {'message': 'User status updated successfully.'}


async def change_role(
    payload: UpdateUserRole,
    db: AsyncSession,
    me: User,
) -> dict[str, str]:
    """Update a user's role (admin or user).

    Args:
        payload: User ID and new role.
        db: Async database session.
        me: Currently authenticated admin.

    Returns:
        Confirmation message.
    """
    user = await lock_account(db, await get_user_by_id(payload.user_id, db))
    account_mutable(user)
    ensure_not_super(user)
    ensure_user_management_scope(user, me)
    validate_role(me, payload.new_role, user)

    user.role = payload.new_role
    await db.commit()
    return {'message': 'User role updated successfully.'}


async def change_group(
    payload: UpdateUserGroup,
    db: AsyncSession,
    me: User,
) -> dict[str, str]:
    """Update the user's group membership.

    Args:
        payload: User ID and new group ID.
        db: Async database session.

    Returns:
        Confirmation message.
    """
    user = await lock_account(db, await get_user_by_id(payload.user_id, db))
    account_mutable(user)
    ensure_not_super(user)
    ensure_user_management_scope(user, me)
    target_group_id = resolve_target_group_id(payload.new_group_id, me)
    if target_group_id is None:
        raise HTTPException(400, 'group_id is required.')
    await get_group_or_404(target_group_id, db)
    user.group_id = target_group_id
    await db.commit()
    return {'message': 'User group updated successfully.'}


async def update_profile(
    payload: UserProfileUpdate,
    db: AsyncSession,
    me: User,
) -> dict[str, str]:
    """Update contact-profile details for a user.

    Args:
        payload: Partial profile fields for the target user.
        db: Database session used to persist the profile.
        me: Authenticated administrator performing the update.

    Returns:
        Confirmation message after the profile is updated.

    Raises:
        HTTPException: If the user is unavailable, protected, or outside the
            administrator's management scope.
    """
    user = await lock_account(db, await get_user_by_id(payload.user_id, db))
    account_mutable(user)
    ensure_not_super(user)
    ensure_user_management_scope(user, me)
    profile_data = payload.model_dump(
        exclude={'user_id'},
        exclude_none=True,
    )
    try:
        if profile_data:
            if user.profile is None:
                raise HTTPException(404, 'Profile not found.')
            subject = await ensure_keycloak_subject_for_user(user, db)
            lifecycle = AccountLifecycle.for_session(db)
            async with lifecycle.updating(
                user,
                AccountSyncPlan('profile', subject),
            ), AccountTransaction(db):
                for name, value in profile_data.items():
                    setattr(user.profile, name, value)
                user.state_revision += 1
                await db.flush()
    except IntegrityError as error:
        raise HTTPException(
            400,
            'Duplicate email or mobile number.',
        ) from error
    return {'message': 'User profile updated successfully.'}


async def rename_account(
    user: User,
    username: str,
    db: AsyncSession,
    operator: User,
):
    user = await lock_account(db, user)
    account_mutable(user)
    ensure_not_super(user)
    ensure_user_management_scope(user, operator)
    subject = await ensure_keycloak_subject_for_user(user, db)
    try:
        lifecycle = AccountLifecycle.for_session(db)
        async with lifecycle.updating(
            user,
            AccountSyncPlan('rename', subject),
        ), AccountTransaction(
            db,
        ):
            user.username = username
            await db.flush()
    except IntegrityError as error:
        raise HTTPException(400, 'Username already exists.') from error
    return {'message': 'Username updated successfully.'}
