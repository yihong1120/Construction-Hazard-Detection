from __future__ import annotations

import time
from typing import TypeAlias

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import load_only
from sqlalchemy.orm import selectinload

from examples.auth.models import Site
from examples.auth.models import site_groups_table
from examples.auth.models import User
from examples.auth.models import user_sites_table

# A cache entry stores: (list of site names, cached-at epoch seconds).
CacheEntry: TypeAlias = tuple[list[str], float]

# Process-local cache for storing user site information.
_user_sites_cache: dict[str, CacheEntry] = {}

# Cache time-to-live in seconds (5 minutes).
_cache_ttl: int = 300


def invalidate_effective_site_cache(
    usernames: list[str] | None = None,
) -> None:
    """Invalidate cached effective site-name lookups for one or more users."""
    if usernames is None:
        _user_sites_cache.clear()
        return

    for username in usernames:
        _user_sites_cache.pop(username, None)


async def _load_user_by_username(
    username: str,
    db: AsyncSession,
    status_code: int,
    detail: str,
) -> User:
    """Load a user by username or raise the requested HTTP error."""
    stmt_user = (
        select(User)
        .options(
            selectinload(User.group),
            selectinload(User.profile),
            selectinload(User.sites),
        )
        .where(User.username == username)
    )
    user_obj: User | None = (
        (await db.execute(stmt_user)).unique().scalars().one_or_none()
    )
    if not user_obj:
        raise HTTPException(status_code=status_code, detail=detail)
    return user_obj


async def list_effective_sites_for_user(
    user: User,
    db: AsyncSession,
) -> list[Site]:
    """Return the sites a user may effectively access right now."""
    if user.role == 'super_admin':
        return list(
            (
                await db.execute(
                    select(Site).order_by(Site.id),
                )
            ).scalars().unique().all(),
        )

    if user.group_id is None:
        return []

    stmt_sites = (
        select(Site)
        .join(user_sites_table, user_sites_table.c.site_id == Site.id)
        .join(site_groups_table, site_groups_table.c.site_id == Site.id)
        .where(
            user_sites_table.c.user_id == user.id,
            site_groups_table.c.group_id == user.group_id,
        )
        .order_by(Site.id)
        .distinct()
    )
    return list((await db.execute(stmt_sites)).scalars().unique().all())


async def list_effective_site_names_for_user(
    user: User,
    db: AsyncSession,
) -> list[str]:
    """Return only the site names needed by request-time authorisation.

    Streaming authorisation does not need site ORM instances (or their
    relationships).  Keeping this projection narrow reduces both database
    transfer and identity-map growth on long-lived API workers.
    """
    if user.role == 'super_admin':
        result = await db.execute(select(Site.name).order_by(Site.id))
        return [str(getattr(name, 'name', name)) for name in result.scalars().all()]
    if user.group_id is None:
        return []
    result = await db.execute(
        select(Site.name)
        .join(user_sites_table, user_sites_table.c.site_id == Site.id)
        .join(site_groups_table, site_groups_table.c.site_id == Site.id)
        .where(
            user_sites_table.c.user_id == user.id,
            site_groups_table.c.group_id == user.group_id,
        )
        .order_by(Site.name)
        .distinct(),
    )
    return [str(getattr(name, 'name', name)) for name in result.scalars().all()]


async def load_user_with_effective_sites(
    username: str,
    db: AsyncSession,
    status_code: int = 404,
    detail: str = 'User not found',
) -> tuple[User, list[Site]]:
    """Load a user and compute their current effective site access."""
    user = await _load_user_by_username(
        username,
        db,
        status_code=status_code,
        detail=detail,
    )
    sites = await list_effective_sites_for_user(user, db)
    return user, sites


async def get_cached_effective_site_names(
    username: str,
    db: AsyncSession,
) -> list[str]:
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

    _, sites = await load_user_with_effective_sites(username, db)
    site_names: list[str] = [site.name for site in sites]
    _user_sites_cache[username] = (site_names, current_time)
    return site_names


async def load_user_access_context(
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
    # Keep the hot authorisation path independent of ``User.group``, profile,
    # and site relationship loading.  Callers only inspect status, role, and
    # the final allowed site-name projection.
    stmt_user = (
        select(User)
        .options(
            load_only(
                User.id,
                User.status,
                User.role,
                User.group_id,
            ),
        )
        .where(User.username == username)
    )
    user: User | None = (
        (await db.execute(stmt_user)).unique().scalars().one_or_none()
    )
    if user is None:
        raise HTTPException(status_code=401, detail='Invalid user')
    user_role = user.role
    user_site_names = await list_effective_site_names_for_user(user, db)
    return user, user_site_names, user_role
