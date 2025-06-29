from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from examples.auth.models import User
from examples.auth.models import UserProfile


async def create_user(
    username: str,
    password: str,
    role: str,
    group_id: int | None,
    db: AsyncSession,
    profile: dict[str, Any] | None = None,
) -> User:
    """Create a new user in the database.

    Args:
        username (str): Username of the new user.
        password (str): Password of the new user.
        role (str): Role of the user (e.g., 'admin', 'user').
        group_id (Optional[int]): Group identifier the user belongs to, if any.
        db (AsyncSession): Database session.

    Returns:
        User: The newly created user instance.

    Raises:
        HTTPException: If the username already exists
            or a database error occurs.
    """
    try:
        new_user = User(
            username=username,
            role=role,
            group_id=group_id,
            is_active=True,
        )
        new_user.set_password(password)
        db.add(new_user)

        # ＜重點①＞先 flush 取得 new_user.id（還沒 commit）
        await db.flush()

        # ＜重點②＞如有 profile → 帶 user_id
        if profile:
            prof = UserProfile(user_id=new_user.id, **profile)
            db.add(prof)

        # 一次 commit
        await db.commit()

        # 刷最新狀態，含 profile
        await db.refresh(new_user, attribute_names=['profile', 'group'])
        return new_user

    except IntegrityError as e:
        await db.rollback()
        # username / email duplicate … 都可能在這炸出
        raise HTTPException(400, 'Username or e-mail already exists.') from e
    except Exception as e:
        await db.rollback()
        raise HTTPException(500, f"Database error: {e}") from e


async def list_users(db: AsyncSession) -> list[User]:
    """Retrieve a list of all users from the database.

    Args:
        db (AsyncSession): Database session.

    Returns:
        List[User]: List of user instances.
    """
    # Fetch all users from the database
    result = await db.execute(select(User))
    return result.unique().scalars().all()


async def get_user_by_id(user_id: int, db: AsyncSession) -> User:
    """Retrieve a specific user by their unique identifier.

    Args:
        user_id (int): Identifier of the user.
        db (AsyncSession): Database session.

    Returns:
        User: The user instance matching the provided ID.

    Raises:
        HTTPException: If no user is found with the specified ID.
    """
    user = await db.get(User, user_id)

    if not user:
        raise HTTPException(
            status_code=404, detail='User not found.',
        )

    return user


async def delete_user(user: User, db: AsyncSession) -> None:
    """Delete a user from the database.

    Args:
        user (User): User instance to delete.
        db (AsyncSession): Database session.

    Raises:
        HTTPException: If a database error occurs during deletion.
    """
    await db.delete(user)  # Mark user instance for deletion

    try:
        await db.commit()  # Commit the deletion transaction
    except Exception as e:
        await db.rollback()  # Rollback on failure
        raise HTTPException(
            status_code=500, detail=f'Database error: {e}',
        )


async def update_username(
    user: User,
    new_username: str,
    db: AsyncSession,
) -> None:
    """Update the username of an existing user.

    Args:
        user (User): User instance to update.
        new_username (str): The new username.
        db (AsyncSession): Database session.

    Raises:
        HTTPException: If the new username already exists
            or a database error occurs.
    """
    user.username = new_username  # Set the new username

    try:
        await db.commit()  # Commit changes to the database
    except IntegrityError:
        await db.rollback()  # Rollback if username conflict occurs
        raise HTTPException(
            status_code=400, detail='Username already exists.',
        )
    except Exception as e:
        await db.rollback()  # Rollback on unexpected errors
        raise HTTPException(
            status_code=500, detail=f'Database error: {e}',
        )


async def update_password(
    user: User,
    new_password: str,
    db: AsyncSession,
) -> None:
    """Update the password of an existing user.

    Args:
        user (User): User instance to update.
        new_password (str): The new password.
        db (AsyncSession): Database session.

    Raises:
        HTTPException: If a database error occurs during password update.
    """
    user.set_password(new_password)  # Securely hash and set the new password

    try:
        await db.commit()  # Save changes to database
    except Exception as e:
        await db.rollback()  # Rollback on error
        raise HTTPException(
            status_code=500, detail=f'Database error: {e}',
        )


async def set_active_status(
    user: User,
    is_active: bool,
    db: AsyncSession,
) -> None:
    """Activate or deactivate a user account.

    Args:
        user (User): User instance to update.
        is_active (bool): Activation status; True for active,
            False for inactive.
        db (AsyncSession): Database session.

    Raises:
        HTTPException: If a database error occurs during status update.
    """
    user.is_active = is_active  # Update user's active status

    try:
        await db.commit()  # Persist status change to database
    except Exception as e:
        await db.rollback()  # Rollback if error occurs
        raise HTTPException(
            status_code=500, detail=f'Database error: {e}',
        )


async def create_or_update_profile(
    user: User,
    data: dict[str, Any],
    db:   AsyncSession,
    create_if_missing: bool = False,
) -> None:
    """
    若 user.profile 不存在且 create_if_missing==True → 建新檔，
    否則僅更新有傳入的欄位。
    """
    profile = user.profile
    if not profile:
        if not create_if_missing:
            raise HTTPException(404, 'Profile not found.')
        profile = UserProfile(user_id=user.id)
        db.add(profile)

    for key, val in data.items():
        if val is not None and hasattr(profile, key):
            setattr(profile, key, val)

    try:
        await db.commit()
        await db.refresh(user, attribute_names=['profile'])
    except IntegrityError:
        await db.rollback()
        # email / mobile 皆設 UNIQUE → 捕捉重覆
        msg = 'Duplicate email or mobile number.'
        raise HTTPException(400, msg)
    except Exception as e:
        await db.rollback()
        raise HTTPException(500, f'Database error: {e}')
