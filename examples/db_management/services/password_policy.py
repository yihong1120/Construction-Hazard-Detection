from __future__ import annotations

from fastapi import HTTPException

from examples.db_management.schemas.password_reset import (
    PasswordErrorResponse,
)

# Keep the backend rule explicit so all password entry points share it.
MIN_PASSWORD_LENGTH = 8


def validate_password_minimum(password: str | None) -> None:
    """Validate the backend-enforced minimum password length.

    Args:
        password: Candidate password supplied by an account-management flow.

    Raises:
        HTTPException: If the password is absent or shorter than the minimum.
    """
    if not password or len(password) < MIN_PASSWORD_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=PasswordErrorResponse(
                code='password_too_short',
                message='Password is too short.',
                min_length=MIN_PASSWORD_LENGTH,
            ).model_dump(exclude_none=True),
        )
