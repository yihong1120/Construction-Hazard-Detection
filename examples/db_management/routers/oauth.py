from __future__ import annotations

from fastapi import APIRouter
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.database import get_db
from examples.auth.models import User
from examples.db_management.deps import get_current_user
from examples.db_management.schemas.oauth import MeResponse
from examples.db_management.services.oauth_protocol_services import (
    current_user_profile,
)

router = APIRouter(tags=['profile'])


@router.get('/me', response_model=MeResponse)
async def me(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> MeResponse:
    """Return the authenticated user's public profile.

    Args:
        db: Database session used to load profile details.
        user: Authenticated user supplied by the access dependency.

    Returns:
        The user's public OAuth profile.
    """
    return await current_user_profile(user, db)
