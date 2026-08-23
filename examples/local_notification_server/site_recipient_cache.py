from __future__ import annotations

from collections.abc import Awaitable
from typing import cast
from typing import Final

import redis.asyncio as redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.models import Site
from examples.auth.models import SiteNotificationPreference
from examples.auth.models import User
from examples.auth.models import USER_STATUS_ACTIVE


_recipient_index_ready_value: Final[str] = '1'


def _site_user_cache_key(site_name: str) -> str:
    """Build the Redis set key for site notification recipients.

    Args:
        site_name: Site name used by notification requests.

    Returns:
        Redis set key containing recipient user IDs.
    """
    return f'site_notification_users:{site_name}'


def _site_user_cache_ready_key(site_name: str) -> str:
    """Build the Redis readiness key for a site recipient index.

    Args:
        site_name: Site name used by notification requests.

    Returns:
        Redis key indicating that the recipient index is ready.
    """
    return f'site_notification_users_ready:{site_name}'


async def _fetch_site_notification_user_ids_from_db(
    site_name: str,
    db: AsyncSession,
) -> list[int] | None:
    """Load current recipient user IDs for a site from the database.

    Args:
        site_name: Site name to look up.
        db: Async database session dependency.

    Returns:
        Active recipient user IDs, or None when the site does not exist.
    """
    stmt = select(Site.id).where(Site.name == site_name)
    site_id_row = (await db.execute(stmt)).first()
    if site_id_row is None:
        return None
    site_id = site_id_row[0]

    users_stmt = (
        select(SiteNotificationPreference.user_id)
        .join(User, User.id == SiteNotificationPreference.user_id)
        .where(
            SiteNotificationPreference.site_id == site_id,
            SiteNotificationPreference.is_enabled.is_(True),
            User.status == USER_STATUS_ACTIVE,
        )
    )
    # Only explicit opt-ins from active accounts may enter the delivery index.
    return list((await db.execute(users_stmt)).scalars().all())


async def refresh_site_notification_user_cache(
    site_name: str,
    db: AsyncSession,
    rds: redis.Redis,
) -> list[int] | None:
    """Rebuild the Redis recipient index for a site from the database.

    Args:
        site_name: Site name to rebuild.
        db: Async database session dependency.
        rds: Redis connection used to write the recipient index.

    Returns:
        Active recipient user IDs, or None when the site does not exist.
    """
    user_ids = await _fetch_site_notification_user_ids_from_db(site_name, db)
    if user_ids is None:
        await invalidate_site_notification_user_cache([site_name], rds)
        return None

    pipe = rds.pipeline()
    cache_key = _site_user_cache_key(site_name)
    ready_key = _site_user_cache_ready_key(site_name)
    # A ready marker distinguishes an empty subscription from a cold cache.
    pipe.delete(cache_key)
    if user_ids:
        pipe.sadd(cache_key, *user_ids)
    pipe.set(ready_key, _recipient_index_ready_value)
    await pipe.execute()
    return user_ids


async def _get_site_user_index_members(
    site_name: str,
    rds: redis.Redis,
) -> list[int]:
    """Read recipient IDs from the Redis set for a site.

    Args:
        site_name: Site name to read.
        rds: Redis connection used to read the recipient index.

    Returns:
        Recipient user IDs from Redis.
    """
    members = cast(
        Awaitable[set[bytes]],
        rds.smembers(_site_user_cache_key(site_name)),
    )
    return [int(member) for member in await members]


async def invalidate_site_notification_user_cache(
    site_names: list[str],
    rds: redis.Redis,
) -> None:
    """Delete Redis recipient indexes for the given sites.

    Args:
        site_names: Site names whose indexes should be removed.
        rds: Redis connection used to delete cache keys.
    """
    keys: list[str] = []
    for site_name in site_names:
        keys.extend([
            _site_user_cache_key(site_name),
            _site_user_cache_ready_key(site_name),
        ])
    if keys:
        await rds.delete(*keys)


async def get_site_notification_user_ids_cached(
    site_name: str,
    db: AsyncSession,
    rds: redis.Redis,
) -> list[int] | None:
    """Get notification recipient IDs using the Redis site index.

    Args:
        site_name: Site name to look up.
        db: Database session used for cold-cache rebuilds.
        rds: Redis connection used as the live recipient index.

    Returns:
        Recipient user IDs if the site exists; otherwise ``None``.
    """
    ready_key = _site_user_cache_ready_key(site_name)
    if await rds.exists(ready_key):
        return await _get_site_user_index_members(site_name, rds)
    return await refresh_site_notification_user_cache(site_name, db, rds)
