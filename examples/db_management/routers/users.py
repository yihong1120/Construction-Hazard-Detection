from __future__ import annotations

from datetime import datetime
from typing import Protocol

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Request
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from examples.auth.database import get_db
from examples.auth.models import Group
from examples.auth.models import User
from examples.auth.models import USER_STATUS_ACTIVE
from examples.auth.models import USER_STATUS_EMAIL_UNVERIFIED
from examples.auth.models import USER_STATUS_PENDING
from examples.auth.models import USER_STATUS_REJECTED
from examples.auth.redis_pool import get_redis_pool
from examples.auth.user_service import invalidate_effective_site_cache
from examples.db_management.deps import ensure_admin_with_group
from examples.db_management.deps import ensure_not_super
from examples.db_management.deps import get_current_user
from examples.db_management.deps import is_super_admin
from examples.db_management.deps import require_admin
from examples.db_management.schemas.user import AdminUserApproval
from examples.db_management.schemas.user import ApproveUserSignup
from examples.db_management.schemas.user import PendingUserReviewRead
from examples.db_management.schemas.user import SetUserStatus
from examples.db_management.schemas.user import UpdateMyPassword
from examples.db_management.schemas.user import UpdatePassword
from examples.db_management.schemas.user import UpdatePasswordById
from examples.db_management.schemas.user import UpdateUserGroup
from examples.db_management.schemas.user import UpdateUsername
from examples.db_management.schemas.user import UpdateUsernameById
from examples.db_management.schemas.user import UpdateUserRole
from examples.db_management.schemas.user import UserCreate
from examples.db_management.schemas.user import UserProfileUpdate
from examples.db_management.schemas.user import UserRead
from examples.db_management.schemas.user import UserSignup
from examples.db_management.services.email_verification_services import (
    send_signup_verification_email,
)
from examples.db_management.services.legal_services import (
    record_user_consent,
)
from examples.db_management.services.legal_services import (
    validate_signup_consents,
)
from examples.db_management.services.site_services import \
    list_site_ids_for_group
from examples.db_management.services.site_services import \
    seed_site_notification_preferences
from examples.db_management.services.user_services import (
    create_or_update_profile,
)
from examples.db_management.services.user_services import create_user
from examples.db_management.services.user_services import delete_user
from examples.db_management.services.user_services import get_user_by_id
from examples.db_management.services.user_services import (
    list_users as list_users_service,
)
from examples.db_management.services.user_services import set_user_status
from examples.db_management.services.user_services import update_password
from examples.db_management.services.user_services import update_username

router = APIRouter(tags=['user-mgmt'])


class _GroupOperator(Protocol):
    """User fields needed for group-scoped account administration."""

    username: str
    role: str
    group_id: int | None


class _ManagedUser(Protocol):
    """User fields needed while checking management scope."""

    group_id: int | None


class _SignupApprovalUser(_ManagedUser, Protocol):
    """User fields mutated by the signup-approval workflow."""

    id: int
    status: str


async def _load_user_read(user_id: int, db: AsyncSession) -> UserRead:
    """Load a user with related group/profile data for API responses."""
    result = await db.execute(
        select(User)
        .options(
            selectinload(User.group),
            selectinload(User.profile),
        )
        .where(User.id == user_id),
    )
    return UserRead.model_validate(result.scalar_one())


def _pending_user_review_read(user: User) -> PendingUserReviewRead:
    """Build the admin review row with legal and provider metadata."""
    base = UserRead.model_validate(user).model_dump()
    profile = user.profile
    consents = list(getattr(user, 'consents', []) or [])
    identities = list(getattr(user, 'identities', []) or [])
    latest_consent = max(
        consents,
        key=lambda item: (
            getattr(item, 'accepted_at', None) or datetime.min,
            getattr(item, 'id', 0),
        ),
        default=None,
    )
    providers = sorted({
        str(identity.provider)
        for identity in identities
        if getattr(identity, 'provider', None)
    })
    return PendingUserReviewRead(
        **base,
        email=str(profile.email) if profile and profile.email else None,
        terms_version=(
            latest_consent.terms_version if latest_consent else None
        ),
        privacy_version=(
            latest_consent.privacy_version if latest_consent else None
        ),
        ai_terms_version=(
            latest_consent.ai_terms_version if latest_consent else None
        ),
        notification_consent=(
            latest_consent.notification_consent if latest_consent else None
        ),
        provider=','.join(providers) if providers else 'password',
    )


async def _get_group_or_404(group_id: int, db: AsyncSession) -> Group:
    """Load a group and raise 404 if it does not exist."""
    group = (
        await db.execute(select(Group).where(Group.id == group_id))
    ).unique().scalar_one_or_none()
    if group is None:
        raise HTTPException(404, 'Group not found.')
    return group


async def _register_signup_user(
    payload: UserSignup,
    request: Request,
    db: AsyncSession,
    redis_pool: Redis,
) -> UserRead:
    """Create an email-unverified account and send verification email."""
    await validate_signup_consents(payload, db)
    new_user = await create_user(
        username=payload.username,
        password=payload.password,
        role='user',
        group_id=None,
        db=db,
        profile=payload.profile.model_dump(),
        status=USER_STATUS_EMAIL_UNVERIFIED,
    )
    await record_user_consent(new_user.id, payload, db, request)
    await send_signup_verification_email(new_user, redis_pool)
    return await _load_user_read(new_user.id, db)


def _resolve_target_group_id(
    requested_group_id: int | None,
    operator: _GroupOperator,
    default_to_operator_group: bool = False,
) -> int | None:
    """Resolve the effective group ID that the operator may manage."""
    if is_super_admin(operator):
        return requested_group_id

    ensure_admin_with_group(operator)

    if requested_group_id is None:
        if default_to_operator_group:
            return operator.group_id
        raise HTTPException(400, 'group_id is required.')

    if requested_group_id != operator.group_id:
        raise HTTPException(403, 'Cannot operate on other group')

    return requested_group_id


def _ensure_user_management_scope(
    target: _ManagedUser,
    operator: _GroupOperator,
) -> None:
    """Ensure an admin can only manage users in their own group."""
    if is_super_admin(operator):
        return

    ensure_admin_with_group(operator)
    if target.group_id != operator.group_id:
        raise HTTPException(
            status_code=403,
            detail='Cannot manage users outside your group.',
        )


@router.post(
    '/add_user',
    response_model=UserRead,
    dependencies=[Depends(require_admin)],
)
async def add_user(
    payload: UserCreate,
    db: AsyncSession = Depends(get_db),
    me: User = Depends(get_current_user),
) -> UserRead:
    """Create a new user.

    Args:
        payload: Data for the new user.
        db: Async database session.
        me: The currently authenticated admin user.

    Returns:
        Newly created user's details.
    """
    target_group_id = _resolve_target_group_id(
        payload.group_id,
        me,
        default_to_operator_group=True,
    )

    new_user = await create_user(
        username=payload.username,
        password=payload.password,
        role=payload.role,
        group_id=target_group_id,
        db=db,
        profile=payload.profile.model_dump() if payload.profile else None,
    )
    if target_group_id:
        site_ids = await list_site_ids_for_group(target_group_id, db)
        await seed_site_notification_preferences(
            user_ids=[new_user.id],
            site_ids=site_ids,
            db=db,
        )
        if site_ids:
            await db.commit()
    invalidate_effective_site_cache()
    return await _load_user_read(new_user.id, db)


@router.post('/signup', response_model=UserRead, status_code=201)
async def signup_user(
    payload: UserSignup,
    request: Request,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis_pool),
) -> UserRead:
    """Create an email-unverified account for the signup flow."""
    return await _register_signup_user(payload, request, db, redis)


@router.post('/auth/register', response_model=UserRead, status_code=201)
async def register_user(
    payload: UserSignup,
    request: Request,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis_pool),
) -> UserRead:
    """Alias for the public registration endpoint."""
    return await _register_signup_user(payload, request, db, redis)


@router.get(
    '/list_pending_users',
    response_model=list[PendingUserReviewRead],
    dependencies=[Depends(require_admin)],
)
async def list_pending_users(
    db: AsyncSession = Depends(get_db),
) -> list[PendingUserReviewRead]:
    """List email-verified, ungrouped signups waiting for admin approval."""
    result = await db.execute(
        select(User)
        .options(
            selectinload(User.group),
            selectinload(User.profile),
            selectinload(User.consents),
            selectinload(User.identities),
        )
        .where(
            User.role == 'user',
            User.status == USER_STATUS_PENDING,
            User.email_verified_at.is_not(None),
            User.group_id.is_(None),
        ),
    )
    users = result.scalars().all()
    return [_pending_user_review_read(user) for user in users]


@router.get(
    '/admin/pending-users',
    response_model=list[PendingUserReviewRead],
    dependencies=[Depends(require_admin)],
)
async def admin_list_pending_users(
    db: AsyncSession = Depends(get_db),
) -> list[PendingUserReviewRead]:
    """Alias matching the admin review API shape."""
    return await list_pending_users(db)


async def _approve_signup_user(
    user: _SignupApprovalUser,
    group_id: int | None,
    db: AsyncSession,
    me: _GroupOperator,
) -> UserRead:
    """Assign an approved signup to a group and activate it."""
    target_group_id = _resolve_target_group_id(
        group_id,
        me,
        default_to_operator_group=True,
    )
    if target_group_id is None:
        raise HTTPException(400, 'group_id is required.')

    await _get_group_or_404(target_group_id, db)

    user.group_id = target_group_id
    user.status = USER_STATUS_ACTIVE
    site_ids = await list_site_ids_for_group(target_group_id, db)
    await seed_site_notification_preferences(
        user_ids=[user.id],
        site_ids=site_ids,
        db=db,
    )
    await db.commit()
    invalidate_effective_site_cache()
    return await _load_user_read(user.id, db)


@router.put('/approve_user_signup', response_model=UserRead)
async def approve_user_signup(
    payload: ApproveUserSignup,
    db: AsyncSession = Depends(get_db),
    me: User = Depends(require_admin),
) -> UserRead:
    """Approve a pending signup by assigning a group and activating it."""
    user = await get_user_by_id(payload.user_id, db)
    ensure_not_super(user)

    if (
        user.role != 'user'
        or user.status != USER_STATUS_PENDING
        or user.email_verified_at is None
        or user.group_id is not None
    ):
        raise HTTPException(400, 'User is not awaiting signup approval.')

    return await _approve_signup_user(user, payload.group_id, db, me)


@router.patch('/admin/users/{user_id}/approval', response_model=UserRead)
async def review_user_signup(
    user_id: int,
    payload: AdminUserApproval,
    db: AsyncSession = Depends(get_db),
    me: User = Depends(require_admin),
) -> UserRead:
    """Approve or reject an email-verified signup request."""
    user = await get_user_by_id(user_id, db)
    ensure_not_super(user)

    if (
        user.role != 'user'
        or user.status != USER_STATUS_PENDING
        or user.email_verified_at is None
        or user.group_id is not None
    ):
        raise HTTPException(400, 'User is not awaiting signup approval.')

    if payload.decision == 'rejected':
        user.status = USER_STATUS_REJECTED
        await db.commit()
        invalidate_effective_site_cache()
        return await _load_user_read(user.id, db)

    return await _approve_signup_user(user, payload.group_id, db, me)


@router.get(
    '/list_users', response_model=list[UserRead],
)
async def list_users(
    db: AsyncSession = Depends(get_db),
    me: User = Depends(require_admin),
) -> list[UserRead]:
    """List users with group information, scoped by the operator role.

    Args:
        db: Async database session.

    Returns:
        List of user details.
    """
    if is_super_admin(me):
        users = await list_users_service(db)
    else:
        ensure_admin_with_group(me)
        users = await list_users_service(db, group_id=me.group_id)

    return [UserRead.model_validate(u) for u in users]


@router.delete('/delete_user', dependencies=[Depends(require_admin)])
async def remove_user(
    payload: dict[str, int],
    db: AsyncSession = Depends(get_db),
    me: User = Depends(require_admin),
) -> dict[str, str]:
    """Delete a user by user ID.

    Args:
        payload: Dictionary containing 'user_id'.
        db: Async database session.

    Returns:
        Confirmation message.
    """
    user = await get_user_by_id(payload['user_id'], db)
    ensure_not_super(user)
    _ensure_user_management_scope(user, me)
    await delete_user(user, db)
    invalidate_effective_site_cache()
    return {'message': 'User deleted successfully.'}


@router.put('/admin_update_password', dependencies=[Depends(require_admin)])
async def admin_update_pwd(
    payload: UpdatePassword,
    db: AsyncSession = Depends(get_db),
    me: User = Depends(require_admin),
) -> dict[str, str]:
    """Admin update user's password by username.

    Args:
        payload: Contains username and new password.
        db: Async database session.

    Returns:
        Confirmation message.
    """
    user = (
        await db.execute(select(User).where(User.username == payload.username))
    ).scalar_one_or_none()
    if not user:
        raise HTTPException(404, 'User not found.')
    ensure_not_super(user)
    _ensure_user_management_scope(user, me)
    await update_password(user, payload.new_password, db)
    return {'message': 'Password updated successfully.'}


@router.put(
    '/admin_update_password_userid',
    dependencies=[Depends(require_admin)],
)
async def admin_update_pwd_by_id(
    payload: UpdatePasswordById,
    db: AsyncSession = Depends(get_db),
    me: User = Depends(require_admin),
) -> dict[str, str]:
    """Admin update user's password by user ID.

    Args:
        payload: Contains user_id and new password.
        db: Async database session.

    Returns:
        Confirmation message.
    """
    user = await get_user_by_id(payload.user_id, db)
    ensure_not_super(user)
    _ensure_user_management_scope(user, me)
    await update_password(user, payload.new_password, db)
    return {'message': 'Password updated successfully by user ID.'}


@router.put('/update_my_password', dependencies=[Depends(get_current_user)])
async def update_my_pwd(
    payload: UpdateMyPassword,
    db: AsyncSession = Depends(get_db),
    redis_pool: Redis = Depends(get_redis_pool),
    me: User = Depends(get_current_user),
) -> dict[str, str]:
    """Allow users to update their own password.

    Args:
        payload: Contains old and new passwords.
        db: Async database session.
        redis_pool: Redis connection.
        me: Currently authenticated user.

    Returns:
        Message indicating password change success.
    """
    if not await me.check_password(payload.old_password):
        raise HTTPException(401, 'Old password incorrect.')

    await update_password(me, payload.new_password, db)

    # Clear existing tokens from Redis cache
    from examples.auth.cache import get_user_data, set_user_data
    cache = await get_user_data(redis_pool, me.username)
    if cache:
        cache['jti_list'] = []
        cache['refresh_tokens'] = []
        await set_user_data(redis_pool, me.username, cache)

    return {'message': 'Password changed successfully, please log in again.'}


@router.put('/update_username', dependencies=[Depends(require_admin)])
async def change_username(
    payload: UpdateUsername,
    db: AsyncSession = Depends(get_db),
    me: User = Depends(require_admin),
) -> dict[str, str]:
    """Change username by providing old and new usernames.

    Args:
        payload: Old and new usernames.
        db: Async database session.

    Returns:
        Confirmation message.
    """
    user = (
        await db.execute(
            select(User).where(User.username == payload.old_username),
        )
    ).scalar_one_or_none()
    if not user:
        raise HTTPException(404, 'User not found.')
    ensure_not_super(user)
    _ensure_user_management_scope(user, me)
    await update_username(user, payload.new_username, db)
    return {'message': 'Username updated successfully.'}


@router.put('/update_username_id', dependencies=[Depends(require_admin)])
async def change_username_by_id(
    payload: UpdateUsernameById,
    db: AsyncSession = Depends(get_db),
    me: User = Depends(require_admin),
) -> dict[str, str]:
    """Change username by user ID.

    Args:
        payload: User ID and new username.
        db: Async database session.

    Returns:
        Confirmation message.
    """
    user = await get_user_by_id(payload.user_id, db)
    ensure_not_super(user)
    _ensure_user_management_scope(user, me)
    await update_username(user, payload.new_username, db)
    return {'message': 'Username updated successfully.'}


@router.put('/set_user_status', dependencies=[Depends(require_admin)])
async def update_user_status(
    payload: SetUserStatus,
    db: AsyncSession = Depends(get_db),
    me: User = Depends(require_admin),
) -> dict[str, str]:
    """Set a user's account status.

    Args:
        payload: User ID and target status.
        db: Async database session.

    Returns:
        Confirmation message.
    """
    user = await get_user_by_id(payload.user_id, db)
    ensure_not_super(user)
    _ensure_user_management_scope(user, me)
    await set_user_status(user, payload.status, db)
    return {'message': 'User status updated successfully.'}


@router.put('/update_user_role', dependencies=[Depends(require_admin)])
async def change_role(
    payload: UpdateUserRole,
    db: AsyncSession = Depends(get_db),
    me: User = Depends(require_admin),
) -> dict[str, str]:
    """Update a user's role (admin or user).

    Args:
        payload: User ID and new role.
        db: Async database session.
        me: Currently authenticated admin.

    Returns:
        Confirmation message.
    """
    user = await get_user_by_id(payload.user_id, db)
    ensure_not_super(user)
    _ensure_user_management_scope(user, me)

    if payload.new_role == 'admin' and not is_super_admin(me):
        raise HTTPException(403, 'Only super admin can assign admin role.')

    user.role = payload.new_role
    await db.commit()
    return {'message': 'User role updated successfully.'}


@router.put('/update_user_group', dependencies=[Depends(require_admin)])
async def change_group(
    payload: UpdateUserGroup,
    db: AsyncSession = Depends(get_db),
    me: User = Depends(require_admin),
) -> dict[str, str]:
    """Update the user's group membership.

    Args:
        payload: User ID and new group ID.
        db: Async database session.

    Returns:
        Confirmation message.
    """
    user = await get_user_by_id(payload.user_id, db)
    ensure_not_super(user)
    _ensure_user_management_scope(user, me)
    target_group_id = _resolve_target_group_id(payload.new_group_id, me)
    if target_group_id is None:
        raise HTTPException(400, 'group_id is required.')
    await _get_group_or_404(target_group_id, db)
    user.group_id = target_group_id
    await db.commit()
    invalidate_effective_site_cache()
    return {'message': 'User group updated successfully.'}


@router.put('/update_user_profile', dependencies=[Depends(require_admin)])
async def update_profile(
    payload: UserProfileUpdate,
    db: AsyncSession = Depends(get_db),
    me: User = Depends(require_admin),
) -> dict[str, str]:
    """Update contact profile details for a user."""
    user = await get_user_by_id(payload.user_id, db)
    ensure_not_super(user)
    _ensure_user_management_scope(user, me)
    await create_or_update_profile(
        user,
        data=payload.model_dump(exclude={'user_id'}, exclude_none=True),
        db=db,
    )
    return {'message': 'User profile updated successfully.'}
