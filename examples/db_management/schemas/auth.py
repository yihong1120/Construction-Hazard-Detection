from __future__ import annotations

from typing import NotRequired
from typing import TypedDict

from pydantic import BaseModel


class DbUserInfo(TypedDict):
    """Database user information structure."""

    id: int
    username: str
    role: str
    group_id: int | None
    is_active: bool


class UserCache(TypedDict, total=False):
    """Redis cache structure for user session data."""

    db_user: DbUserInfo
    jti_list: list[str]
    jti_meta: dict[str, int]
    refresh_tokens: list[str]
    feature_names: list[str]


class SubjectUsername(TypedDict):
    """JWT subject containing username."""

    username: str


class JWTPayloadBase(TypedDict, total=False):
    """Base structure for JWT payload claims."""

    exp: NotRequired[int]
    iat: NotRequired[int]
    nbf: NotRequired[int]
    iss: NotRequired[str]
    aud: NotRequired[str]


class RefreshTokenPayload(JWTPayloadBase, total=False):
    """Refresh token payload structure."""

    subject: SubjectUsername


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
