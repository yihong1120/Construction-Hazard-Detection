from __future__ import annotations

from typing import Annotated
from typing import cast

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Query
from fastapi import Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from examples.auth.config import Settings
from examples.auth.database import get_db
from examples.auth.models import User
from examples.auth.models import USER_STATUS_PENDING_ADMIN_APPROVAL
from examples.auth.models import USER_STATUS_REJECTED
from examples.db_management.deps import _NamedRoleUser
from examples.db_management.deps import ensure_not_super
from examples.db_management.deps import get_current_user
from examples.db_management.deps import is_super_admin
from examples.db_management.deps import require_admin
from examples.db_management.schemas.resource_options import Option
from examples.db_management.schemas.resource_options import resource_options
from examples.db_management.schemas.resource_options import ResourceOptions
from examples.db_management.schemas.user import AdminUserApproval
from examples.db_management.schemas.user import ApproveUserSignup
from examples.db_management.schemas.user import PendingUserReviewPage
from examples.db_management.schemas.user import SetUserStatus
from examples.db_management.schemas.user import UpdatePassword
from examples.db_management.schemas.user import UpdatePasswordById
from examples.db_management.schemas.user import UpdateUserGroup
from examples.db_management.schemas.user import UpdateUsername
from examples.db_management.schemas.user import UpdateUsernameById
from examples.db_management.schemas.user import UpdateUserRole
from examples.db_management.schemas.user import UserCreate
from examples.db_management.schemas.user import UserDelete
from examples.db_management.schemas.user import UserManagementCapabilities
from examples.db_management.schemas.user import UserPage
from examples.db_management.schemas.user import UserProfileUpdate
from examples.db_management.schemas.user import UserRead
from examples.db_management.services import user_action_services as actions
from examples.db_management.services.keycloak_user_management_services import (
    reset_keycloak_password,
)
from examples.db_management.services.password_policy import (
    validate_password_minimum,
)
from examples.db_management.services.role_policy import assignable_roles
from examples.db_management.services.user_management_services import (
    approve_signup_user,
)
from examples.db_management.services.user_management_services import (
    ensure_user_management_scope,
)
from examples.db_management.services.user_management_services import (
    list_all_users_for_operator,
)
from examples.db_management.services.user_management_services import (
    list_users_for_operator,
)
from examples.db_management.services.user_management_services import (
    load_user_read,
)
from examples.db_management.services.user_management_services import (
    pending_user_review_read,
)
from examples.db_management.services.user_services import (
    ensure_keycloak_subject_for_user,
)
from examples.db_management.services.user_services import get_user_by_id

router = APIRouter(tags=['user-mgmt'])


@router.get('/users/assignable-roles', response_model=ResourceOptions)
async def get_assignable_roles(
    response: Response,
    target_user_id: int | None = Query(None, gt=0),
    locale: str = 'zh-TW',
    me: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ResourceOptions:
    target = (
        await db.get(User, target_user_id)
        if target_user_id is not None
        else None
    )
    if target_user_id is not None and target is None:
        raise HTTPException(404, detail='User not found')
    labels = (
        {'admin': '管理員', 'user': '使用者', 'guest': '訪客'}
        if locale.lower().startswith('zh')
        else {}
    )
    options = [
        Option(id=r, label=labels.get(r, r))
        for r in assignable_roles(me, target)
    ]
    response.headers['Cache-Control'] = 'private, no-store'
    return resource_options(me, target_user_id, options)


settings = Settings()


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
    return await actions.add_user(payload, db, me)


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
        HTTPException: If the account is not awaiting approval or is outside
            the
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
        HTTPException: If the account is not awaiting approval or is outside
            the
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
        return await load_user_read(user.id, db)

    return await approve_signup_user(user, payload.group_id, db, me)


@router.get(
    '/list_users',
    response_model=list[UserRead],
)
async def list_users(
    db: AsyncSession = Depends(get_db),
    me: User = Depends(require_admin),
) -> list[UserRead]:
    """Return the scoped user array required by deployed clients.

    Pagination lives at ``/admin/users`` so this established endpoint never
    changes its JSON top-level type.
    """
    return await list_all_users_for_operator(me, db)


@router.get('/admin/users', response_model=UserPage)
async def list_users_page(
    db: AsyncSession = Depends(get_db),
    me: User = Depends(require_admin),
    cursor: Annotated[int | None, Query(ge=0)] = None,
    page_size: Annotated[int, Query(ge=1, le=100)] = 50,
) -> UserPage:
    """Return one cursor-based, administrator-scoped user page.

    New management clients should use this endpoint and preserve
    ``next_cursor`` when loading later pages.
    """
    return await list_users_for_operator(
        me,
        db,
        cursor=cursor,
        page_size=page_size,
    )


@router.get(
    '/admin/capabilities',
    response_model=UserManagementCapabilities,
)
async def user_management_capabilities(
    me: User = Depends(require_admin),
) -> UserManagementCapabilities:
    """Return UI capabilities without trusting client-side role inference.

    The backend still authorises every mutation independently.  This endpoint
    merely prevents Flutter from showing ChangDar or a group administrator a
    misleading read-only management screen.
    """
    super_admin = is_super_admin(cast(_NamedRoleUser, me))
    return UserManagementCapabilities(
        scope='all_groups' if super_admin else 'own_group',
        managed_group_id=None if super_admin else me.group_id,
        can_create_users=True,
        can_reset_passwords=True,
        can_suspend_users=True,
        can_delete_users=True,
        can_manage_groups=super_admin,
        can_manage_group_features=super_admin,
        can_assign_group_admins=super_admin,
        can_manage_website_access=super_admin,
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
    return await actions.remove_user(payload, db, me)


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
    validate_password_minimum(payload.new_password)
    subject = await ensure_keycloak_subject_for_user(user, db)
    await reset_keycloak_password(
        subject,
        password=payload.new_password,
        temporary=True,
    )
    await db.commit()
    return {'message': 'Password updated successfully.'}


@router.put(
    '/admin_update_password_userid',
    dependencies=[Depends(require_admin)],
)
async def admin_update_pwd_by_id(
    payload: UpdatePasswordById,
    db: AsyncSession = Depends(get_db),
    me: User = Depends(require_admin),
    temporary: bool = True,
) -> dict[str, str]:
    """Admin update user's password by user ID.

    Args:
        payload: Contains user_id and new password.
        db: Async database session.

    Returns:
        Confirmation message.
    """
    return await actions.admin_update_pwd_by_id(
        payload, db, me, temporary=temporary,
    )


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
    return await actions.rename_account(user, payload.new_username, db, me)


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
    return await actions.rename_account(user, payload.new_username, db, me)


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
    return await actions.update_user_status(payload, db, me)


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
    return await actions.change_role(payload, db, me)


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
    return await actions.change_group(payload, db, me)


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
    return await actions.update_profile(payload, db, me)
