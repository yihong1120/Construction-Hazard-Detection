from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass
from typing import cast
from typing import Final

import redis.asyncio as redis
from redis.asyncio.client import Pipeline
from redis.exceptions import WatchError
from sqlalchemy import and_
from sqlalchemy import or_
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.models import Site
from examples.auth.models import site_groups_table
from examples.auth.models import SiteNotificationPreference
from examples.auth.models import User
from examples.auth.models import user_sites_table
from examples.auth.models import USER_STATUS_ACTIVE
from examples.auth.redis_transactions import optimistic

_recipient_cache_ttl_seconds: Final[int] = 300


def _site_user_cache_key(site_name: str) -> str:
    """Build the Redis set key for site notification recipients.

    Args:
        site_name: Site name used by notification requests.

    Returns:
        Redis set key containing recipient user IDs.
    """
    return f"site_notification_users:{site_name}"


def _site_user_cache_ready_key(site_name: str) -> str:
    """Build the Redis readiness key for a site recipient index.

    Args:
        site_name: Site name used by notification requests.

    Returns:
        Redis key indicating that the recipient index is ready.
    """
    return f"site_notification_users_ready:{site_name}"


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
            or_(
                User.role == 'super_admin',
                and_(
                    select(user_sites_table.c.user_id)
                    .where(
                        user_sites_table.c.user_id == User.id,
                        user_sites_table.c.site_id == site_id,
                    )
                    .exists(),
                    select(site_groups_table.c.site_id)
                    .where(
                        site_groups_table.c.group_id == User.group_id,
                        site_groups_table.c.site_id == site_id,
                    )
                    .exists(),
                ),
            ),
        )
    )
    # Only explicit opt-ins from active accounts may enter the delivery index.
    return list((await db.execute(users_stmt)).scalars().all())


async def _recipient_revision(site_name: str, db: AsyncSession) -> str | None:
    row = (
        await db.execute(
            select(
                Site.id,
                Site.recipient_revision,
            ).where(Site.name == site_name),
        )
    ).first()
    return f'{row[0]}:{row[1]}' if row is not None else None


@dataclass
class _SiteRecipientCacheRebuild:
    site_name: str
    db: AsyncSession

    async def apply(self, pipe: Pipeline) -> list[int] | None:
        cache_key = _site_user_cache_key(self.site_name)
        ready_key = _site_user_cache_ready_key(self.site_name)
        await pipe.watch(cache_key, ready_key)
        revision = await _recipient_revision(self.site_name, self.db)
        users = (
            await _fetch_site_notification_user_ids_from_db(
                self.site_name,
                self.db,
            )
            if revision is not None
            else None
        )
        if await _recipient_revision(self.site_name, self.db) != revision:
            raise WatchError('Subscription changed during cache rebuild')
        pipe.multi()
        pipe.delete(cache_key, ready_key)
        if users is not None:
            assert revision is not None
            if users:
                pipe.sadd(cache_key, *users)
                pipe.expire(cache_key, _recipient_cache_ttl_seconds)
            pipe.set(
                ready_key,
                revision.encode(),
                ex=_recipient_cache_ttl_seconds,
            )
        await pipe.execute()
        return users


async def refresh_site_notification_user_cache(
    site_name: str,
    db: AsyncSession,
    rds: redis.Redis,
) -> list[int] | None:
    return await optimistic(
        rds,
        _SiteRecipientCacheRebuild(site_name, db).apply,
    )


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
        keys.extend(
            [
                _site_user_cache_key(site_name),
                _site_user_cache_ready_key(site_name),
            ],
        )
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
    revision = await _recipient_revision(site_name, db)
    if revision is None:
        await invalidate_site_notification_user_cache([site_name], rds)
        return None
    async with rds.pipeline(transaction=True) as pipe:
        pipe.get(_site_user_cache_ready_key(site_name))
        pipe.smembers(_site_user_cache_key(site_name))
        ready, members = await pipe.execute()
    if ready == revision.encode():
        return [int(member) for member in members]
    return await refresh_site_notification_user_cache(site_name, db, rds)
