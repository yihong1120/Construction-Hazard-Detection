from __future__ import annotations

import time
from typing import TypeAlias

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from examples.auth.models import User

# A cache entry stores: (list of site names, cached-at epoch seconds).
CacheEntry: TypeAlias = tuple[list[str], float]

# Process-local cache for storing user site information.
_user_sites_cache: dict[str, CacheEntry] = {}

# Cache time-to-live in seconds (5 minutes).
_cache_ttl: int = 300


async def get_user_sites_cached(username: str, db: AsyncSession) -> list[str]:
    """
    Return site names the user may access, with simple in-memory caching.

    Args:
        username: The unique username to resolve.
        db: An asynchronous SQLAlchemy session used for the lookup.

    Returns:
        A list of site names that the user may access. The list order follows
        the ORM relationship ordering as returned by the database.

    Raises:
        HTTPException: With status code 404 if the user is not found.
    """
    current_time: float = time.time()

    if username in _user_sites_cache:
        # Fast path: honour TTL and return cached site names when still fresh.
        cached_names, cached_time = _user_sites_cache[username]
        if current_time - cached_time < _cache_ttl:
            return cached_names

    # Query the user and their sites in one round-trip.
    stmt_user = (
        select(User)
        .where(User.username == username)
        .options(selectinload(User.sites))
    )
    user_obj: User | None = (await db.execute(stmt_user)).scalar()
    if not user_obj:
        raise HTTPException(status_code=404, detail='User not found')

    # Extract and cache the site names with the current timestamp.
    site_names: list[str] = [site.name for site in user_obj.sites]
    _user_sites_cache[username] = (site_names, current_time)
    return site_names


async def get_user_and_sites(
    db: AsyncSession, username: str,
) -> tuple[User, list[str], str]:
    """
    Fetch the user, their site names, and role from the database.

    Args:
        db: An asynchronous SQLAlchemy session.
        username: The username to query.

    Returns:
        A 3-tuple of ``(user, site_names, role)`` where:
        - ``user`` is the fully loaded ``User`` ORM instance,
        - ``site_names`` is a list of the user's site names, and
        - ``role`` is the user's role as a string.

    Raises:
        HTTPException: With status code 401 if the user cannot be found.
    """
    stmt_user = (
        select(User)
        .where(User.username == username)
        .options(selectinload(User.sites))
    )
    result = await db.execute(stmt_user)
    user: User | None = result.scalars().first()
    if not user:
        raise HTTPException(status_code=401, detail='Invalid user')
    user_role: str = user.role
    user_site_names: list[str] = [site.name for site in user.sites]
    return user, user_site_names, user_role
