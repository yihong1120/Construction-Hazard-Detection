from __future__ import annotations

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from examples.user_management.models import User


async def add_user(
    username: str, password: str, role: str, db: AsyncSession,
) -> str:
    """
    Add a new user to the database with the specified role.

    Args:
        username (str): The username for the new user. Must be unique.
        password (str): The password for the new user, stored as a hash.
        role (str): The role assigned to the user (e.g., admin, user, guest).
        db (AsyncSession): The database session for executing queries.

    Returns:
        str: A message indicating the result of the operation.

    Raises:
        IntegrityError: If the username already exists.
        Exception: For any other errors during the operation.
    """
    # Create a new user instance and hash the password
    new_user = User(username=username, role=role)
    new_user.set_password(password)
    db.add(new_user)

    try:
        # Commit the transaction to save the user
        await db.commit()
        return f"User {username} with role {role} added successfully."
    except IntegrityError:
        # Handle duplicate username errors
        await db.rollback()
        return f"Error: Username {username} already exists."
    except Exception as e:
        # Handle other unexpected errors
        await db.rollback()
        return f"Error adding user: {str(e)}"


async def delete_user(username: str, db: AsyncSession) -> str:
    """
    Delete a user from the database by their username.

    Args:
        username (str): The username of the user to delete.
        db (AsyncSession): The database session for executing queries.

    Returns:
        str: A message indicating the result of the operation.

    Raises:
        Exception: If there is an error during the deletion process.
    """
    # Fetch the user by their username
    stmt = select(User).where(User.username == username)
    result = await db.execute(stmt)
    user = result.scalars().first()

    if user:
        # User exists; proceed with deletion
        await db.delete(user)
        try:
            await db.commit()
            return f"User {username} deleted successfully."
        except Exception as e:
            await db.rollback()
            return f"Error deleting user: {str(e)}"
    return f"User {username} not found."


async def update_username(
    old_username: str, new_username: str, db: AsyncSession,
) -> str:
    """
    Update a user's username in the database.

    Args:
        old_username (str): The current username of the user.
        new_username (str): The new username to assign to the user.
        db (AsyncSession): The database session for executing queries.

    Returns:
        str: A message indicating the result of the operation.

    Raises:
        IntegrityError: If the new username already exists.
        Exception: For any other errors during the operation.
    """
    # Locate the user by their current username
    stmt = select(User).where(User.username == old_username)
    result = await db.execute(stmt)
    user = result.scalars().first()

    if user:
        # Update the user's username
        user.username = new_username
        try:
            await db.commit()
            return f"Username updated successfully to {new_username}."
        except IntegrityError:
            # Handle duplicate username errors
            await db.rollback()
            return f"Error: Username {new_username} already exists."
        except Exception as e:
            # Handle other unexpected errors
            await db.rollback()
            return f"Error updating username: {str(e)}"
    return f"User {old_username} not found."


async def update_password(
    username: str,
    new_password: str,
    db: AsyncSession,
) -> str:
    """
    Update a user's password.

    Args:
        username (str): The username of the user whose password
            will be updated.
        new_password (str): The new password to assign to the user.
        db (AsyncSession): The database session for executing queries.

    Returns:
        str: A message indicating the result of the operation.

    Raises:
        Exception: If there is an error during the password update process.
    """
    # Locate the user by their username
    stmt = select(User).where(User.username == username)
    result = await db.execute(stmt)
    user = result.scalars().first()

    if user:
        # Update and hash the new password
        user.set_password(new_password)
        try:
            await db.commit()
            return f"Password updated successfully for user {username}."
        except Exception as e:
            # Handle unexpected errors
            await db.rollback()
            return f"Error updating password: {str(e)}"
    return f"User {username} not found."


async def set_user_active_status(
    username: str, is_active: bool, db: AsyncSession,
) -> str:
    """
    Update a user's active status.

    Args:
        username (str): The username of the user whose status will be updated.
        is_active (bool): The active status to assign
            (True for active, False for inactive).
        db (AsyncSession): The database session for executing queries.

    Returns:
        str: A message indicating the result of the operation.

    Raises:
        Exception: If there is an error during the status update process.
    """
    # Locate the user by their username
    stmt = select(User).where(User.username == username)
    result = await db.execute(stmt)
    user = result.scalars().first()

    if user:
        # Update the user's active status
        user.is_active = is_active
        try:
            await db.commit()
            await db.commit()
            return (
                f"User {username} is now "
                f"{'active' if is_active else 'inactive'}."
            )
        except Exception as e:
            # Handle unexpected errors
            await db.rollback()
            return f"Error updating active status: {str(e)}"
    return f"User {username} not found."
