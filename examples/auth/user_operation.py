from __future__ import annotations

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from examples.auth.models import User


async def add_user(
    username: str,
    password: str,
    role: str,
    db: AsyncSession,
) -> dict:
    """
    Create a new user in the database.

    Args:
        username (str): The username of the new account.
        password (str): Plain-text password for the user to be hashed.
        role (str): The user's role (e.g., 'admin', 'user', 'guest').
        db (AsyncSession): The asynchronous database session.

    Returns:
        dict: Contains 'success' (bool), 'message' (str),
              and optional 'error' (str) if creation fails.
    """
    new_user: User = User(username=username, role=role)
    new_user.set_password(password)  # Securely hash the password
    db.add(new_user)

    try:
        await db.commit()
        return {
            'success': True,
            'message': f"User '{username}' added successfully.",
        }
    except IntegrityError:
        await db.rollback()
        return {
            'success': False,
            'error': 'IntegrityError',
            'message': f"Username '{username}' already exists.",
        }
    except Exception as e:
        await db.rollback()
        return {
            'success': False,
            'error': 'UnknownError',
            'message': f"Failed to add user: {str(e)}",
        }


async def delete_user(username: str, db: AsyncSession) -> dict:
    """
    Delete an existing user by username.

    Args:
        username (str): The username of the account to remove.
        db (AsyncSession): The asynchronous database session.

    Returns:
        dict: Contains 'success' (bool), 'message' (str),
              and optional 'error' (str) if deletion fails.
    """
    stmt = select(User).where(User.username == username)
    result = await db.execute(stmt)
    user: User | None = result.scalars().first()

    if not user:
        return {
            'success': False,
            'error': 'NotFound',
            'message': f"User '{username}' not found.",
        }

    await db.delete(user)
    try:
        await db.commit()
        return {
            'success': True,
            'message': f"User '{username}' deleted successfully.",
        }
    except Exception as e:
        await db.rollback()
        return {
            'success': False,
            'error': 'UnknownError',
            'message': f"Failed to delete user: {str(e)}",
        }


async def update_username(
    old_username: str,
    new_username: str,
    db: AsyncSession,
) -> dict:
    """
    Update a user's username to a new value.

    Args:
        old_username (str): The current username.
        new_username (str): The new username to assign.
        db (AsyncSession): The asynchronous database session.

    Returns:
        dict: Contains 'success' (bool), 'message' (str),
              and optional 'error' (str) if the update fails.
    """
    stmt = select(User).where(User.username == old_username)
    result = await db.execute(stmt)
    user: User | None = result.scalars().first()

    if not user:
        return {
            'success': False,
            'error': 'NotFound',
            'message': f"User '{old_username}' not found.",
        }

    user.username = new_username
    try:
        await db.commit()
        return {
            'success': True,
            'message': (
                f"Username updated from '{old_username}' to '{new_username}'."
            ),
        }
    except IntegrityError:
        await db.rollback()
        return {
            'success': False,
            'error': 'IntegrityError',
            'message': f"Username '{new_username}' already exists.",
        }
    except Exception as e:
        await db.rollback()
        return {
            'success': False,
            'error': 'UnknownError',
            'message': f"Failed to update username: {str(e)}",
        }


async def update_password(
    username: str,
    new_password: str,
    db: AsyncSession,
) -> dict:
    """
    Update a user's password.

    Args:
        username (str): The username of the account to update.
        new_password (str): The new plain-text password.
        db (AsyncSession): The asynchronous database session.

    Returns:
        dict: Contains 'success' (bool), 'message' (str),
              and optional 'error' (str) if the update fails.
    """
    stmt = select(User).where(User.username == username)
    result = await db.execute(stmt)
    user: User | None = result.scalars().first()

    if not user:
        return {
            'success': False,
            'error': 'NotFound',
            'message': f"User '{username}' not found.",
        }

    user.set_password(new_password)
    try:
        await db.commit()
        return {
            'success': True,
            'message': f"Password updated successfully for user '{username}'.",
        }
    except Exception as e:
        await db.rollback()
        return {
            'success': False,
            'error': 'UnknownError',
            'message': f"Failed to update password: {str(e)}",
        }


async def set_user_active_status(
    username: str,
    is_active: bool,
    db: AsyncSession,
) -> dict:
    """
    Enable or disable a user's account.

    Args:
        username (str): The username of the account to modify.
        is_active (bool): True to activate, False to deactivate.
        db (AsyncSession): The asynchronous database session.

    Returns:
        dict: Contains 'success' (bool), 'message' (str),
              and optional 'error' (str) if the update fails.
    """
    stmt = select(User).where(User.username == username)
    result = await db.execute(stmt)
    user: User | None = result.scalars().first()

    if not user:
        return {
            'success': False,
            'error': 'NotFound',
            'message': f"User '{username}' not found.",
        }

    user.is_active = is_active
    try:
        await db.commit()
        return {
            'success': True,
            'message': (
                f"User '{username}' is now "
                f"{'active' if is_active else 'inactive'}."
            ),
        }
    except Exception as e:
        await db.rollback()
        return {
            'success': False,
            'error': 'UnknownError',
            'message': f"Failed to update active status: {str(e)}",
        }
