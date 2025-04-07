from __future__ import annotations

from pydantic import BaseModel


# --------------------------
#   Authentication Schemas
# --------------------------
class UserLogin(BaseModel):
    """
    Schema for user login requests.

    Attributes:
        username (str): The user's unique identifier.
        password (str): The user's plain-text password.
    """
    username: str
    password: str


class LogoutRequest(BaseModel):
    """
    Schema for logout requests.

    Attributes:
        refresh_token (str): The refresh token to be invalidated or revoked.
    """
    refresh_token: str


class RefreshRequest(BaseModel):
    """
    Schema for token refresh requests.

    Attributes:
        refresh_token (str):
            A valid refresh token to exchange for a new access token.
    """
    refresh_token: str


# --------------------------
#   User Management Schemas
# --------------------------
class UserCreate(BaseModel):
    """
    Schema for creating a new user account.

    Attributes:
        username (str): The desired username.
        password (str): The initial plain-text password.
        role (str): The role of the user, defaults to 'user'.
    """
    username: str
    password: str
    role: str = 'user'


class DeleteUser(BaseModel):
    """
    Schema for deleting a user.

    Attributes:
        username (str): The username of the account to be deleted.
    """
    username: str


class UpdateUsername(BaseModel):
    """
    Schema for updating an existing user's username.

    Attributes:
        old_username (str): The current username.
        new_username (str): The new username to replace the old one.
    """
    old_username: str
    new_username: str


class UpdatePassword(BaseModel):
    """
    Schema for updating a user's password.

    Attributes:
        username (str): The user's current username.
        new_password (str): The new plain-text password.
        role (str): The user's role, defaults to 'user'.
    """
    username: str
    new_password: str
    role: str = 'user'


class SetUserActiveStatus(BaseModel):
    """
    Schema for activating or deactivating a user account.

    Attributes:
        username (str): The username of the account to update.
        is_active (bool): True to activate, False to deactivate.
    """
    username: str
    is_active: bool
