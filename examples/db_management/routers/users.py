from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Query
from fastapi import Request
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from examples.auth.database import get_db
from examples.auth.models import User
from examples.auth.models import USER_STATUS_PENDING_ADMIN_APPROVAL
from examples.auth.models import USER_STATUS_REJECTED
from examples.auth.redis_pool import get_redis_pool
from examples.auth.user_service import invalidate_effective_site_cache
from examples.db_management.deps import ensure_not_super
from examples.db_management.deps import get_current_user
from examples.db_management.deps import is_super_admin
from examples.db_management.deps import require_admin
from examples.db_management.schemas.user import AdminUserApproval
from examples.db_management.schemas.user import ApproveUserSignup
from examples.db_management.schemas.user import PendingUserReviewPage
from examples.db_management.schemas.user import SetUserStatus
from examples.db_management.schemas.user import UpdateMyPassword
from examples.db_management.schemas.user import UpdatePassword
from examples.db_management.schemas.user import UpdatePasswordById
from examples.db_management.schemas.user import UpdateUserGroup
from examples.db_management.schemas.user import UpdateUsername
from examples.db_management.schemas.user import UpdateUsernameById
from examples.db_management.schemas.user import UpdateUserRole
from examples.db_management.schemas.user import UserCreate
from examples.db_management.schemas.user import UserDelete
from examples.db_management.schemas.user import UserPage
from examples.db_management.schemas.user import UserProfileUpdate
from examples.db_management.schemas.user import UserRead
from examples.db_management.schemas.user import UserSignup
from examples.db_management.services.site_services import \
    list_site_ids_for_group
from examples.db_management.services.site_services import \
    seed_site_notification_preferences
from examples.db_management.services.user_management_services import \
    approve_signup_user
from examples.db_management.services.user_management_services import \
    ensure_user_management_scope
from examples.db_management.services.user_management_services import \
    get_group_or_404
from examples.db_management.services.user_management_services import \
    list_users_for_operator
from examples.db_management.services.user_management_services import \
    load_user_read
from examples.db_management.services.user_management_services import \
    pending_user_review_read
from examples.db_management.services.user_management_services import \
    register_signup_user
from examples.db_management.services.user_management_services import \
    resolve_target_group_id
from examples.db_management.services.user_services import (
    create_or_update_profile,
)
from examples.db_management.services.user_services import create_user
from examples.db_management.services.user_services import delete_user
from examples.db_management.services.user_services import get_user_by_id
from examples.db_management.services.user_services import set_user_status
from examples.db_management.services.user_services import update_password
from examples.db_management.services.user_services import update_username

router = APIRouter(tags=['user-mgmt'])


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
    target_group_id = resolve_target_group_id(
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
        tenant_id=getattr(me, 'tenant_id', None),
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
    return await load_user_read(new_user.id, db)


@router.post('/signup', response_model=UserRead, status_code=201)
async def signup_user(
    payload: UserSignup,
    request: Request,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis_pool),
) -> UserRead:
    """Create an email-unverified account for the public signup flow.

    Args:
        payload: Registration details, profile, and legal consents.
        request: HTTP request used to capture consent audit metadata.
        db: Database session used to create the account.
        redis: Redis connection used to create verification-token state.

    Returns:
        Newly created account in its email-unverified state.

    Raises:
        HTTPException: If the username or email is unavailable, legal consent is
            invalid, or verification delivery cannot be initiated.
    """
    return await register_signup_user(payload, request, db, redis)


@router.get(
    '/list_pending_users',
    response_model=PendingUserReviewPage,
    dependencies=[Depends(require_admin)],
)
async def list_pending_users(
    db: AsyncSession = Depends(get_db),
    cursor: Annotated[int | None, Query(ge=0)] = None,
    page_size: Annotated[int, Query(ge=1, le=100)] = 50,
) -> PendingUserReviewPage:
    """List email-verified signups awaiting administrator approval.

    Args:
        db: Database session used to load pending accounts and their relations.

    Returns:
        Pending account records suitable for an administrator review queue.
    """
    query = (
        select(User)
        .options(
            selectinload(User.group),
            selectinload(User.profile),
            selectinload(User.consents),
            selectinload(User.identities),
        )
        .where(
            User.role == 'user',
            User.status == USER_STATUS_PENDING_ADMIN_APPROVAL,
            User.email_verified_at.is_not(None),
            User.group_id.is_(None),
        )
    )
    if cursor is not None:
        query = query.where(User.id > cursor)
    result = await db.execute(query.order_by(User.id).limit(page_size + 1))
    users = list(result.scalars().all())
    has_more = len(users) > page_size
    page = users[:page_size]
    return PendingUserReviewPage(
        items=[pending_user_review_read(user) for user in page],
        next_cursor=page[-1].id if has_more and page else None,
    )


@router.put('/approve_user_signup', response_model=UserRead)
async def approve_user_signup(
    payload: ApproveUserSignup,
    db: AsyncSession = Depends(get_db),
    me: User = Depends(require_admin),
) -> UserRead:
    """Approve a pending signup by assigning a group and activating it.

    Args:
        payload: Pending user and optional group assignment to approve.
        db: Database session used to update the pending account.
        me: Authenticated administrator performing the approval.

    Returns:
        Activated user details.

    Raises:
        HTTPException: If the account is not awaiting approval or is outside the
            administrator's scope.
    """
    user = await get_user_by_id(payload.user_id, db)
    ensure_not_super(user)

    if (
        user.role != 'user'
        or user.status != USER_STATUS_PENDING_ADMIN_APPROVAL
        or user.email_verified_at is None
        or user.group_id is not None
    ):
        raise HTTPException(400, 'User is not awaiting signup approval.')

    return await approve_signup_user(user, payload.group_id, db, me)


@router.patch('/admin/users/{user_id}/approval', response_model=UserRead)
async def review_user_signup(
    user_id: int,
    payload: AdminUserApproval,
    db: AsyncSession = Depends(get_db),
    me: User = Depends(require_admin),
) -> UserRead:
    """Approve or reject an email-verified signup request.

    Args:
        user_id: Identifier of the pending account to review.
        payload: Approval decision, optional group, and audit note.
        db: Database session used to update the pending account.
        me: Authenticated administrator performing the review.

    Returns:
        Updated user details after the decision is recorded.

    Raises:
        HTTPException: If the account is not awaiting approval or is outside the
            administrator's scope.
    """
    user = await get_user_by_id(user_id, db)
    ensure_not_super(user)

    if (
        user.role != 'user'
        or user.status != USER_STATUS_PENDING_ADMIN_APPROVAL
        or user.email_verified_at is None
        or user.group_id is not None
    ):
        raise HTTPException(400, 'User is not awaiting signup approval.')

    if payload.decision == 'rejected':
        user.status = USER_STATUS_REJECTED
        await db.commit()
        invalidate_effective_site_cache()
        return await load_user_read(user.id, db)

    return await approve_signup_user(user, payload.group_id, db, me)


@router.get(
    '/list_users', response_model=UserPage,
)
async def list_users(
    db: AsyncSession = Depends(get_db),
    me: User = Depends(require_admin),
    cursor: Annotated[int | None, Query(ge=0)] = None,
    page_size: Annotated[int, Query(ge=1, le=100)] = 50,
) -> UserPage:
    """Delegate the paginated, scoped user listing to its application service."""
    return await list_users_for_operator(
        me,
        db,
        cursor=cursor,
        page_size=page_size,
    )


@router.delete('/delete_user', dependencies=[Depends(require_admin)])
async def remove_user(
    payload: UserDelete,
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
    user = await get_user_by_id(payload.user_id, db)
    ensure_not_super(user)
    ensure_user_management_scope(user, me)
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
    ensure_user_management_scope(user, me)
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
    ensure_user_management_scope(user, me)
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
    from examples.auth.cache import rate_limiter_service
    cache = await rate_limiter_service.get_user_data(redis_pool, me.username)
    if cache:
        cache['jti_list'] = []
        cache['refresh_tokens'] = []
        await rate_limiter_service.set_user_data(redis_pool, me.username, cache)

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
    ensure_user_management_scope(user, me)
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
    ensure_user_management_scope(user, me)
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
    ensure_user_management_scope(user, me)
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
    ensure_user_management_scope(user, me)

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
    ensure_user_management_scope(user, me)
    target_group_id = resolve_target_group_id(payload.new_group_id, me)
    if target_group_id is None:
        raise HTTPException(400, 'group_id is required.')
    await get_group_or_404(target_group_id, db)
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
    user = await get_user_by_id(payload.user_id, db)
    ensure_not_super(user)
    ensure_user_management_scope(user, me)
    await create_or_update_profile(
        user,
        data=payload.model_dump(exclude={'user_id'}, exclude_none=True),
        db=db,
    )
    return {'message': 'User profile updated successfully.'}
