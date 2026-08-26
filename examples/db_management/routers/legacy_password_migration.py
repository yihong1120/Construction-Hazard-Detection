"""Loopback-only endpoints used by Keycloak's legacy-password bridge."""
from __future__ import annotations

from fastapi import APIRouter
from fastapi import Depends
from fastapi import Request
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.database import get_db
from examples.auth.redis_pool import get_redis_pool
from examples.db_management.services import (
    legacy_password_migration_services as legacy_password_service,
)

router = APIRouter(prefix='/auth/legacy-password', tags=['internal'])


@router.post('/keycloak/verify', include_in_schema=False)
async def verify_for_keycloak(
    request: Request,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis_pool),
) -> dict[str, object]:
    """Verify a legacy password only for the local Keycloak authenticator."""
    return await legacy_password_service.verify_legacy_password(
        request,
        db,
        redis,
    )


@router.post('/keycloak/complete', include_in_schema=False)
async def complete_for_keycloak(
    request: Request,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis_pool),
) -> dict[str, str]:
    """Disable the legacy verifier after Keycloak saved its own credential."""
    return await legacy_password_service.complete_legacy_password_migration(
        request,
        db,
        redis,
    )
