from __future__ import annotations

from fastapi import HTTPException

MIN_PASSWORD_LENGTH = 8


def validate_password_minimum(password: str | None) -> None:
    """Validate the backend-enforced minimum password length."""
    if not password or len(password) < MIN_PASSWORD_LENGTH:
        raise HTTPException(
            status_code=400,
            detail={
                'code': 'password_too_short',
                'min_length': MIN_PASSWORD_LENGTH,
            },
        )
