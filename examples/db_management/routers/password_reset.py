from __future__ import annotations

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Request
from fastapi.responses import JSONResponse
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.database import get_db
from examples.auth.identity_provider import require_local_password_management
from examples.auth.redis_pool import get_redis_pool
from examples.db_management.schemas.password_reset import (
    ForgotPasswordRequest,
)
from examples.db_management.schemas.password_reset import (
    PasswordErrorResponse,
)
from examples.db_management.schemas.password_reset import (
    PasswordMessageResponse,
)
from examples.db_management.schemas.password_reset import ResetPasswordRequest
from examples.db_management.services.password_reset_services import (
    request_password_reset,
)
from examples.db_management.services.password_reset_services import (
    reset_password,
)

router = APIRouter(prefix='/password', tags=['password-reset'])


@router.post('/forgot', response_model=PasswordMessageResponse)
async def forgot_password(
    payload: ForgotPasswordRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis_pool),
) -> PasswordMessageResponse:
    """Issue a password-reset link when the email belongs to an account.

    Args:
        payload: Email address for the account to recover.
        request: HTTP request used to apply IP-based rate limiting.
        db: Database session used to locate the account.
        redis: Redis connection used for token and rate-limit state.

    Returns:
        Generic reset-request message that does not disclose account existence.

    Raises:
        HTTPException: If the request exceeds reset rate limits.
    """
    require_local_password_management()
    client_ip = request.client.host if request.client else None
    result = await request_password_reset(
        str(payload.email),
        db,
        redis,
        client_ip=client_ip,
    )
    return PasswordMessageResponse.model_validate(result)


@router.post('/reset', response_model=PasswordMessageResponse)
async def reset_password_endpoint(
    payload: ResetPasswordRequest,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis_pool),
) -> PasswordMessageResponse | JSONResponse:
    """Reset a password using a one-time raw reset token.

    Args:
        payload: Raw reset token and requested replacement password.
        db: Database session used to update the account password.
        redis: Redis connection used to consume token state.

    Returns:
        Success response, or a structured password-operation error response.
    """
    require_local_password_management()
    try:
        result = await reset_password(
            payload.token,
            payload.new_password,
            db,
            redis,
        )
    except HTTPException as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content=PasswordErrorResponse.model_validate(
                exc.detail,
            ).model_dump(exclude_none=True),
        )
    return PasswordMessageResponse.model_validate(result)
