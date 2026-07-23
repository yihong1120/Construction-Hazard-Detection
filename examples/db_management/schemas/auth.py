from __future__ import annotations

from typing import NotRequired
from typing import TypedDict

from pydantic import BaseModel
from pydantic import EmailStr


class DbUserInfo(TypedDict):
    """Database user information structure."""

    id: int
    username: str
    role: str
    group_id: int | None
    status: str


class UserCache(TypedDict, total=False):
    """Redis cache structure for user session data."""

    db_user: DbUserInfo
    jti_list: list[str]
    jti_meta: dict[str, int]
    refresh_tokens: list[str]
    refresh_token_hashes: list[str]
    refresh_token_families: dict[str, str]
    feature_names: list[str]


class SubjectUsername(TypedDict):
    """JWT subject containing username."""

    username: str


class RefreshTokenSubject(SubjectUsername, total=False):
    """Refresh rotation and family-reuse claims."""

    family_id: str
    token_id: str


class JWTPayloadBase(TypedDict, total=False):
    """Base structure for JWT payload claims."""

    exp: NotRequired[int]
    iat: NotRequired[int]
    nbf: NotRequired[int]
    iss: NotRequired[str]
    aud: NotRequired[str]


class RefreshTokenPayload(JWTPayloadBase, total=False):
    """Refresh token payload structure."""

    subject: RefreshTokenSubject


class UserLogin(BaseModel):
    """Schema representing a user's login credentials."""

    identifier: str
    password: str
    hcaptcha_token: str | None = None


class VerifyEmailRequest(BaseModel):
    """Payload for verifying an email verification link."""

    token: str


class ResendVerificationRequest(BaseModel):
    """Payload for requesting another verification email."""

    email: EmailStr


class AuthMessageResponse(BaseModel):
    """Simple message response used by auth lifecycle endpoints."""

    message: str
    code: str | None = None
    status: str | None = None


class LegalConsentFields(BaseModel):
    """Legal consent fields required when a provider creates a new account."""

    accepted_terms: bool = False
    terms_version: str | None = None
    privacy_version: str | None = None
    notification_consent: bool = False
    ai_terms_accepted: bool = False
    ai_terms_version: str | None = None


class GoogleAuthRequest(LegalConsentFields):
    """Payload for Google Sign-In token authentication."""

    id_token: str
    email: str | None = None
    display_name: str | None = None
    device_lang: str | None = None


class AppleAuthRequest(LegalConsentFields):
    """Payload for Sign in with Apple token authentication."""

    identity_token: str | None = None
    authorization_code: str
    email: str | None = None
    given_name: str | None = None
    family_name: str | None = None
    nonce: str | None = None
    device_lang: str | None = None


class IdentityRead(BaseModel):
    """Linked external login identity returned to account settings."""

    id: int
    provider: str
    email: str | None = None
    display_name: str | None = None
    linked_at: str


class IdentityListResponse(BaseModel):
    """Linked login methods for the current user."""

    identities: list[IdentityRead]
    has_password: bool


class LogoutRequest(BaseModel):
    """Schema representing a logout request."""

    refresh_token: str | None = None


class RefreshRequest(BaseModel):
    """Schema representing a token refresh request."""

    refresh_token: str | None = None


class TokenPairData(TypedDict, total=False):
    """Typed dictionary for TokenPair-compatible response payloads."""

    access_token: str
    refresh_token: str | None
    username: str
    role: str
    user_id: int
    group_id: int | None
    feature_names: list[str]


class TokenPair(BaseModel):
    """Schema representing a pair of JWT tokens and user-related details."""

    access_token: str
    refresh_token: str | None = None
    username: str | None = None
    role: str | None = None
    user_id: int | None = None
    group_id: int | None = None
    feature_names: list[str] = []
