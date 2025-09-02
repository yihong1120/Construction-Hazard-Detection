from __future__ import annotations

from fastapi import Depends
from fastapi import HTTPException
from fastapi import Security
from fastapi_jwt import JwtAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from examples.auth.database import get_db
from examples.auth.jwt_config import jwt_access
from examples.auth.models import Site
from examples.auth.models import User

SUPER_ADMIN_NAME = 'ChangDar'


async def get_current_user(
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Retrieve the current authenticated user from JWT credentials.

    Args:
        credentials (JwtAuthorizationCredentials):
            JWT credentials obtained from the request.
        db (AsyncSession): Database session dependency.

    Returns:
        User: Authenticated User instance.

    Raises:
        HTTPException: If the username is invalid or the user does not exist.
    """
    username: str | None = credentials.subject.get('username')
    if not username:
        raise HTTPException(status_code=401, detail='Invalid token subject')

    result = await db.execute(select(User).where(User.username == username))
    user: User | None = result.unique().scalar_one_or_none()

    if user is None:
        raise HTTPException(status_code=401, detail='User not found')

    return user


def is_super_admin(user: User) -> bool:
    """
    Check if the given user is the super administrator.

    Args:
        user (User): User instance to check.

    Returns:
        bool: True if the user is the super admin, False otherwise.
    """
    return user.username == SUPER_ADMIN_NAME and user.role == 'admin'


def require_admin(user: User = Depends(get_current_user)) -> User:
    """
    Dependency ensuring the user has admin-level permissions.

    Args:
        user (User): The currently authenticated user.

    Returns:
        User: The authenticated admin user.

    Raises:
        HTTPException: If the user lacks admin privileges.
    """
    if user.role != 'admin' and not is_super_admin(user):
        raise HTTPException(status_code=403, detail='Admin required')

    return user


def require_super_admin(user: User = Depends(get_current_user)) -> User:
    """
    Dependency ensuring the user has super admin privileges.

    Args:
        user (User): The currently authenticated user.

    Returns:
        User: The authenticated super admin user.

    Raises:
        HTTPException: If the user is not a super admin.
    """
    if not is_super_admin(user):
        raise HTTPException(status_code=403, detail='Super admin only')

    return user


def ensure_not_super(target: User) -> None:
    """
    Ensure the target user is not the super admin.

    Args:
        target (User): The user to verify.

    Raises:
        HTTPException: If attempting to operate on the super admin.
    """
    if target.username == SUPER_ADMIN_NAME:
        raise HTTPException(
            status_code=403,
            detail='Cannot operate on super admin',
        )


def ensure_admin_with_group(user: User) -> None:
    """
    Ensure the user is an admin associated with a group.

    Args:
        user (User): The user to verify.

    Raises:
        HTTPException: If the user is not an admin or has no group assigned.
    """
    if user.role != 'admin':
        raise HTTPException(status_code=403, detail='Admin required')

    if user.group_id is None:
        raise HTTPException(status_code=403, detail='Admin without group')


def _site_permission(
    op: User,
    *,
    site: Site | None = None,
    group_id: int | None = None,
) -> None:
    """
    Verify if the operating user has permission for
        the specified site or group.

    Permission Rules:
        1. Super admin has unrestricted access.
        2. Only users with the admin role can perform the action; additionally,
            admin users must belong to a group.
        3. Admin users can only operate on sites within their own group.

    Args:
        op (User): The user performing the operation.
        site (Optional[Site], optional): Site instance to
            verify permissions against. Defaults to None.
        group_id (Optional[int], optional): Group ID to
            verify permissions against. Defaults to None.

    Raises:
        HTTPException: If the user lacks the necessary permissions.
    """
    # Super admin bypasses all checks.
    if is_super_admin(op):
        return

    # Ensure user has admin role.
    if op.role != 'admin':
        raise HTTPException(status_code=403, detail='Admin required')

    # Admin users must belong to a group.
    ensure_admin_with_group(op)

    # Verify if the site belongs to the admin user's group.
    if site and site.group_id != op.group_id:
        raise HTTPException(
            status_code=403,
            detail="Cannot manage other group's site",
        )

    # Verify if provided group_id matches the admin user's group.
    if group_id is not None and group_id != op.group_id:
        raise HTTPException(
            status_code=403,
            detail='Cannot operate on other group',
        )
