from __future__ import annotations

from .models import db
from .models import User


def add_user(username: str, password: str) -> str:
    """
    Create a new user instance and add it to the database.

    Args:
        username (str): The username of the new user.
        password (str): The password for the new user.

    Returns:
        str: Success or error message.

    """
    # Create a new user instance with the provided username
    new_user = User(username=username)
    new_user.set_password(password)  # Set the password after hashing
    db.session.add(new_user)
    try:
        db.session.commit()  # Commit the transaction to the database
        return f"User {username} added successfully."
    except Exception as e:
        db.session.rollback()  # Roll back the transaction on error
        return f"Error adding user: {str(e)}"


def delete_user(username: str) -> str:
    """
    Delete a user from the database by username.

    Args:
        username (str): The username of the user to delete.

    Returns:
        str: Success or error message.
    """
    # Fetch the user from the database by username
    user = User.query.filter_by(username=username).first()
    if user:
        db.session.delete(user)
        try:
            db.session.commit()  # Commit the deletion
            return f"User {username} deleted successfully."
        except Exception as e:
            db.session.rollback()  # Roll back if commit fails
            return f"Error deleting user: {str(e)}"
    else:
        return f"User {username} not found."


def update_username(old_username: str, new_username: str) -> str:
    """
    Update a user's username.

    Args:
        old_username (str): The current username.
        new_username (str): The new username to update to.

    Returns:
        str: Success or error message.
    """
    # Locate the user by the old username
    user = User.query.filter_by(username=old_username).first()
    if user:
        user.username = new_username  # Assign the new username
        try:
            db.session.commit()  # Commit changes to the database
            return f"Username updated successfully to {new_username}."
        except Exception as e:
            db.session.rollback()  # Roll back if an error occurs during commit
            return f"Error updating username: {str(e)}"
    else:
        return f"User {old_username} not found."


def update_password(username: str, new_password: str) -> str:
    """
    Update a user's password.

    Args:
        username (str): The username for which to update the password.
        new_password (str): The new password to be set.

    Returns:
        str: Success or error message.
    """
    # Find the user by username
    user = User.query.filter_by(username=username).first()
    if user:
        user.set_password(new_password)  # Set the new password after hashing
        try:
            db.session.commit()  # Commit the changes
            return f"Password updated successfully for user {username}."
        except Exception as e:
            db.session.rollback()  # Roll back on failure
            return f"Error updating password: {str(e)}"
    else:
        return f"User {username} not found."
