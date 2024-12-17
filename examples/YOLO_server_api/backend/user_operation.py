from __future__ import annotations

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from examples.YOLO_server_api.backend.models import User


async def add_user(
    username: str, password: str, role: str, db: AsyncSession,
) -> dict:
    """
    Add a new user to the database.

    Args:
        username (str): The username for the new user.
        password (str): The password for the new user.
        role (str): The role assigned to the user.
        db (AsyncSession): The database session.

    Returns:
        dict: A dictionary containing the result of the operation.
    """
    new_user = User(username=username, role=role)
    new_user.set_password(password)
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
    Delete a user from the database.

    Args:
        username (str): The username of the user to delete.
        db (AsyncSession): The database session.

    Returns:
        dict: A dictionary containing the result of the operation.
    """
    stmt = select(User).where(User.username == username)
    result = await db.execute(stmt)
    user = result.scalars().first()

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
    old_username: str, new_username: str, db: AsyncSession,
) -> dict:
    """
    Update a user's username.

    Args:
        old_username (str): The current username of the user.
        new_username (str): The new username to assign to the user.
        db (AsyncSession): The database session.

    Returns:
        dict: A dictionary containing the result of the operation.
    """
    stmt = select(User).where(User.username == old_username)
    result = await db.execute(stmt)
    user = result.scalars().first()

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
                f"Username updated from '{old_username}' "
                f"to '{new_username}'."
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
    username: str, new_password: str, db: AsyncSession,
) -> dict:
    """
    Update a user's password.

    Args:
        username (str): The username of the user.
        new_password (str): The new password.
        db (AsyncSession): The database session.

    Returns:
        dict: A dictionary containing the result of the operation.
    """
    stmt = select(User).where(User.username == username)
    result = await db.execute(stmt)
    user = result.scalars().first()

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
    username: str, is_active: bool, db: AsyncSession,
) -> dict:
    """
    Update a user's active status.

    Args:
        username (str): The username of the user.
        is_active (bool): The new active status.
        db (AsyncSession): The database session.

    Returns:
        dict: A dictionary containing the result of the operation.
    """
    stmt = select(User).where(User.username == username)
    result = await db.execute(stmt)
    user = result.scalars().first()

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
