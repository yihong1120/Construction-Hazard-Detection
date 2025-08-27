from __future__ import annotations

import time

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from examples.auth.models import User

# Cache mechanism for storing user site information
_user_sites_cache: dict[str, tuple[list[str], float]] = {}
_cache_ttl: int = 300  # Cache time-to-live in seconds (5 minutes)


async def get_user_sites_cached(username: str, db: AsyncSession) -> list[str]:
    """Return site names the user can access, with simple in-memory caching.

    Raises HTTPException(404) if the user is not found.
    """
    current_time: float = time.time()

    if username in _user_sites_cache:
        cached_names, cached_time = _user_sites_cache[username]
        if current_time - cached_time < _cache_ttl:
            return cached_names

    stmt_user = (
        select(User)
        .where(User.username == username)
        .options(selectinload(User.sites))
    )
    user_obj: User | None = (await db.execute(stmt_user)).scalar()
    if not user_obj:
        raise HTTPException(status_code=404, detail='User not found')

    site_names: list[str] = [site.name for site in user_obj.sites]
    _user_sites_cache[username] = (site_names, current_time)
    return site_names
