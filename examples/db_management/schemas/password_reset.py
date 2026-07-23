from __future__ import annotations

from pydantic import BaseModel
from pydantic import EmailStr


class ForgotPasswordRequest(BaseModel):
    """Request payload for issuing a password reset email."""

    email: EmailStr


class ResetPasswordRequest(BaseModel):
    """Request payload for resetting a password with a raw reset token."""

    token: str | None = None
    new_password: str | None = None


class PasswordMessageResponse(BaseModel):
    """Generic password reset response message."""

    message: str
