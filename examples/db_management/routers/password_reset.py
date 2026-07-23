from __future__ import annotations

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Request
from fastapi.responses import JSONResponse
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.database import get_db
from examples.auth.redis_pool import get_redis_pool
from examples.db_management.schemas.password_reset import (
    ForgotPasswordRequest,
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
    """Send a password reset link when the e-mail belongs to a user."""
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
    """Reset a password using a one-time raw reset token."""
    try:
        result = await reset_password(
            payload.token,
            payload.new_password,
            db,
            redis,
        )
    except HTTPException as exc:
        content = (
            exc.detail
            if isinstance(exc.detail, dict)
            else {'message': str(exc.detail)}
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=content,
        )
    return PasswordMessageResponse.model_validate(result)
