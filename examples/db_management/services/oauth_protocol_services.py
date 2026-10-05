from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from examples.auth.models import Feature
from examples.auth.models import group_features_table
from examples.auth.models import User
from examples.db_management.schemas.oauth import MeResponse


async def _load_feature_names(
    db: AsyncSession, group_id: int | None,
) -> list[str]:
    if group_id is None:
        return []
    rows = await db.execute(
        select(Feature.feature_name)
        .join(
            group_features_table,
            Feature.id == group_features_table.c.feature_id,
        )
        .where(group_features_table.c.group_id == group_id),
    )
    return [row.feature_name for row in rows]


async def current_user_profile(
    user: User,
    db: AsyncSession,
) -> MeResponse:
    """Load an active user and construct their native-app profile response.

    Args:
        user: Authenticated user from the access-token dependency.
        db: Database session used to load profile details.

    Returns:
        Public native OAuth profile.
    """
    loaded = await db.scalar(
        select(User)
        .options(selectinload(User.profile))
        .where(User.id == user.id),
    )
    if loaded is None or loaded.status != 'active':
        raise HTTPException(status_code=401, detail='invalid_user')
    display_name = (
        f"{loaded.profile.given_name} {loaded.profile.family_name}".strip()
        if loaded.profile is not None
        else loaded.username
    )
    return MeResponse(
        id=loaded.id,
        username=loaded.username,
        display_name=display_name,
        tenant_id=getattr(loaded, 'tenant_id', None),
        role=loaded.role,
        group_id=loaded.group_id,
        status=loaded.status,
        feature_names=await _load_feature_names(db, loaded.group_id),
    )
