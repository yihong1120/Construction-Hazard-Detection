from __future__ import annotations

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from examples.auth.database import get_db
from examples.auth.models import User
from examples.auth.redis_pool import get_redis_pool
from examples.db_management.deps import ensure_not_super
from examples.db_management.deps import get_current_user
from examples.db_management.deps import is_super_admin
from examples.db_management.deps import require_admin
from examples.db_management.deps import require_super_admin
from examples.db_management.schemas.user import SetUserActiveStatus
from examples.db_management.schemas.user import UpdateMyPassword
from examples.db_management.schemas.user import UpdatePassword
from examples.db_management.schemas.user import UpdatePasswordById
from examples.db_management.schemas.user import UpdateUserGroup
from examples.db_management.schemas.user import UpdateUsername
from examples.db_management.schemas.user import UpdateUsernameById
from examples.db_management.schemas.user import UpdateUserRole
from examples.db_management.schemas.user import UserCreate
from examples.db_management.schemas.user import UserRead
from examples.db_management.services.user_service import create_user
from examples.db_management.services.user_service import delete_user
from examples.db_management.services.user_service import get_user_by_id
from examples.db_management.services.user_service import set_active_status
from examples.db_management.services.user_service import update_password
from examples.db_management.services.user_service import update_username

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
    user = await create_user(
        username=payload.username,
        password=payload.password,
        role=payload.role,
        group_id=payload.group_id or me.group_id,
        db=db,
    )
    return UserRead.from_orm(user)


@router.get(
    '/list_users',
    response_model=list[UserRead],
    dependencies=[Depends(require_admin)],
)
async def list_users(
    db: AsyncSession = Depends(get_db),
) -> list[UserRead]:
    """List all users with group information.

    Args:
        db: Async database session.

    Returns:
        List of user details.
    """
    result = await db.execute(select(User).options(selectinload(User.group)))
    users = result.scalars().all()
    return [UserRead.model_validate(u) for u in users]


@router.delete('/delete_user', dependencies=[Depends(require_admin)])
async def remove_user(
    payload: dict[str, int],
    db: AsyncSession = Depends(get_db),
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
    await delete_user(user, db)
    return {'message': 'User deleted successfully.'}


@router.put('/admin_update_password', dependencies=[Depends(require_admin)])
async def admin_update_pwd(
    payload: UpdatePassword,
    db: AsyncSession = Depends(get_db),
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
    await update_password(user, payload.new_password, db)
    return {'message': 'Password updated successfully.'}


@router.put(
    '/admin_update_password_userid',
    dependencies=[Depends(require_admin)],
)
async def admin_update_pwd_by_id(
    payload: UpdatePasswordById,
    db: AsyncSession = Depends(get_db),
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
    await update_username(user, payload.new_username, db)
    return {'message': 'Username updated successfully.'}


@router.put('/update_username_id', dependencies=[Depends(require_admin)])
async def change_username_by_id(
    payload: UpdateUsernameById,
    db: AsyncSession = Depends(get_db),
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
    await update_username(user, payload.new_username, db)
    return {'message': 'Username updated successfully.'}


@router.put('/set_user_active_status', dependencies=[Depends(require_admin)])
async def activate_user(
    payload: SetUserActiveStatus,
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    """Set a user's active status.

    Args:
        payload: User ID and active status.
        db: Async database session.

    Returns:
        Confirmation message.
    """
    user = await get_user_by_id(payload.user_id, db)
    ensure_not_super(user)
    await set_active_status(user, payload.is_active, db)
    return {'message': 'User active status updated successfully.'}


@router.put('/update_user_role', dependencies=[Depends(require_admin)])
async def change_role(
    payload: UpdateUserRole,
    db: AsyncSession = Depends(get_db),
    me: User = Depends(get_current_user),
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

    if payload.new_role == 'admin' and not is_super_admin(me):
        raise HTTPException(403, 'Only super admin can assign admin role.')

    user.role = payload.new_role
    await db.commit()
    return {'message': 'User role updated successfully.'}


@router.put('/update_user_group', dependencies=[Depends(require_super_admin)])
async def change_group(
    payload: UpdateUserGroup,
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    """Update the user's group membership.

    Args:
        payload: User ID and new group ID.
        db: Async database session.

    Returns:
        Confirmation message.
    """
    user = await get_user_by_id(payload.user_id, db)
    user.group_id = payload.new_group_id
    await db.commit()
    return {'message': 'User group updated successfully.'}
