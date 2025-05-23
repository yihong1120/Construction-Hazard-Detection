from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from examples.auth.models import User


async def create_user(
    username: str,
    password: str,
    role: str,
    group_id: int | None,
    db: AsyncSession,
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
    # Initialise new user with provided details
    new_user = User(
        username=username,
        role=role,
        group_id=group_id,
        is_active=True,
    )
    new_user.set_password(password)  # Securely hash and store user's password
    db.add(new_user)  # Add user to the current database session

    try:
        await db.commit()  # Commit transaction to database
        await db.refresh(new_user)  # Refresh instance to load generated fields
        return new_user
    except IntegrityError:
        await db.rollback()  # Rollback transaction if username already exists
        raise HTTPException(
            status_code=400, detail='Username already exists.',
        )
    except Exception as e:
        await db.rollback()  # Rollback on unexpected errors
        raise HTTPException(
            status_code=500, detail=f'Database error: {e}',
        )


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
