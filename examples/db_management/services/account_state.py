from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.models import User


async def lock_account(
    db: AsyncSession,
    user: User,
    *,
    wait: bool = True,
) -> User:
    """Refresh the account while owning its mutation lock."""
    current = await db.scalar(
        select(User)
        .where(User.id == user.id)
        .with_for_update(of=User, nowait=not wait)
        .execution_options(populate_existing=True),
    )
    if current is None:
        raise HTTPException(404, 'User not found.')
    return current
