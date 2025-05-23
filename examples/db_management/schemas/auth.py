from __future__ import annotations

from pydantic import BaseModel


class UserLogin(BaseModel):
    """Schema representing a user's login credentials."""

    username: str
    password: str


class LogoutRequest(BaseModel):
    """Schema representing a logout request."""

    refresh_token: str


class RefreshRequest(BaseModel):
    """Schema representing a token refresh request."""

    refresh_token: str


class TokenPair(BaseModel):
    """Schema representing a pair of JWT tokens and user-related details."""

    access_token: str
    refresh_token: str
    username: str | None = None
    role: str | None = None
    user_id: int | None = None
    group_id: int | None = None
    feature_names: list[str] = []
