"""Load the public account profile shared by login and session renewal."""
from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import lazyload
from sqlalchemy.orm import selectinload

from examples.auth.models import User
from examples.bff.schemas import UserSummary


async def load_user_summary(
    db: AsyncSession,
    user_id: int,
) -> UserSummary:
    """Load the local profile data stored in the BFF session response."""
    user = await db.scalar(
        select(User)
        .options(lazyload('*'), selectinload(User.profile))
        .where(User.id == user_id),
    )
    if user is None:
        raise HTTPException(status_code=401, detail='user_not_found')
    profile = user.profile
    display_name = user.username
    if profile is not None:
        display_name = (
            ' '.join(
                part
                for part in (profile.given_name, profile.family_name)
                if part
            )
            or user.username
        )
    return UserSummary(
        id=user.id,
        username=user.username,
        display_name=display_name,
        role=user.role,
        group_id=user.group_id,
        status=user.status,
    )
