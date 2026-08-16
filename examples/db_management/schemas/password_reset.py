from __future__ import annotations

from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import EmailStr


class ForgotPasswordRequest(BaseModel):
    """Define a request to issue a password-reset email.

    Attributes:
        email: Address associated with the account to recover.
    """

    email: EmailStr


class ResetPasswordRequest(BaseModel):
    """Define a request to reset a password using a reset token.

    Attributes:
        token: Raw one-time token received by the user.
        new_password: Replacement password for the account.
    """

    token: str | None = None
    new_password: str | None = None


class PasswordMessageResponse(BaseModel):
    """Represent a successful password-operation response.

    Attributes:
        message: User-facing result message.
    """

    message: str


class PasswordErrorResponse(BaseModel):
    """Represent the standard error returned by password operations.

    Attributes:
        code: Stable machine-readable error identifier.
        message: User-facing explanation of the error.
        min_length: Required password length when applicable.
    """

    model_config = ConfigDict(extra='forbid', strict=True)

    code: Literal[
        'database_error',
        'password_too_short',
        'reset_token_invalid',
    ]
    message: str
    min_length: int | None = None
