from __future__ import annotations

from typing import cast
from typing import Protocol

import redis.asyncio as redis
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.models import Site
from examples.auth.models import SiteNotificationPreference
from examples.auth.models import User
from examples.auth.user_service import list_effective_sites_for_user
from examples.db_management.deps import is_super_admin
from examples.db_management.services.site_services import list_sites
from examples.local_notification_server.schemas import (
    SiteNotificationPreferenceOut,
)
from examples.local_notification_server.schemas import (
    SiteNotificationPreferenceUpdateRequest,
)
from examples.local_notification_server.site_recipient_cache import (
    refresh_site_notification_user_cache,
)


class _NotificationScopeUser(Protocol):
    """Identity fields required to determine notification preference scope.

    This narrow protocol keeps scope evaluation independent from unrelated ORM
    user fields while retaining type checking for authorisation decisions.
    """

    username: str
    role: str
    group_id: int | None


async def _list_notification_scope_sites(
    db: AsyncSession,
    me: _NotificationScopeUser,
) -> list[Site]:
    """Return sites the user may manage notification preferences for.

    Args:
        db: Database session used to load site records.
        me: Authenticated user with role and group information.

    Returns:
        Sites in the user's notification-preference scope.

    Raises:
        HTTPException: If a non-super-admin user has no group.
    """
    if is_super_admin(me):
        return await list_sites(db)
    # Group membership is the authoritative boundary for non-admin settings.
    if me.group_id is None:
        raise HTTPException(status_code=403, detail='User without group.')
    return await list_sites(db, group_id=me.group_id)


async def list_site_notification_preferences(
    db: AsyncSession,
    me: User,
) -> list[SiteNotificationPreferenceOut]:
    """List notification preferences for all sites in the user's scope.

    Args:
        db: Database session used to load preferences and site access.
        me: Authenticated user whose preferences are requested.

    Returns:
        Per-site notification preferences.
    """
    sites = await _list_notification_scope_sites(
        db,
        cast(_NotificationScopeUser, me),
    )
    if not sites:
        return []
    site_ids = [site.id for site in sites]
    preference_result = await db.execute(
        select(
            SiteNotificationPreference.site_id,
            SiteNotificationPreference.is_enabled,
        ).where(
            SiteNotificationPreference.user_id == me.id,
            SiteNotificationPreference.site_id.in_(site_ids),
        ),
    )
    explicit_preferences = {row[0]: row[1] for row in preference_result.all()}
    # Effective site access provides the default until an explicit preference
    # exists.
    access_site_ids = {
        site.id
        for site in await list_effective_sites_for_user(me, db)
        if site.id in site_ids
    }
    return [
        SiteNotificationPreferenceOut(
            site_id=site.id,
            site_name=site.name,
            group_name=site.groups[0].name if site.groups else None,
            is_enabled=explicit_preferences.get(
                site.id,
                site.id in access_site_ids,
            ),
        )
        for site in sites
    ]


async def update_site_notification_preferences(
    payload: SiteNotificationPreferenceUpdateRequest,
    db: AsyncSession,
    me: User,
    rds: redis.Redis,
) -> list[SiteNotificationPreferenceOut]:
    """Update explicit site preferences and refresh affected recipient caches.

    Args:
        payload: Requested preference values.
        db: Database session used to store preference changes.
        me: Authenticated user whose preferences are updated.
        rds: Redis connection used to refresh recipient caches.

    Returns:
        Current preference state after the update.

    Raises:
        HTTPException: If the request includes a site outside the user's scope.
    """
    sites = await _list_notification_scope_sites(
        db,
        cast(_NotificationScopeUser, me),
    )
    allowed_site_ids = {site.id for site in sites}
    requested_site_ids = {item.site_id for item in payload.preferences}
    # Reject cross-scope updates before creating or modifying any preference
    # rows.
    invalid_site_ids = requested_site_ids - allowed_site_ids
    if invalid_site_ids:
        raise HTTPException(
            status_code=403,
            detail='Cannot subscribe to sites outside your scope.',
        )
    preference_result = await db.execute(
        select(SiteNotificationPreference).where(
            SiteNotificationPreference.user_id == me.id,
            SiteNotificationPreference.site_id.in_(allowed_site_ids),
        ),
    )
    existing_preferences = {
        preference.site_id: preference
        for preference in preference_result.scalars().all()
    }
    access_site_ids = {
        site.id
        for site in await list_effective_sites_for_user(me, db)
        if site.id in allowed_site_ids
    }
    requested_preferences = {
        item.site_id: item.is_enabled for item in payload.preferences
    }
    changed_site_names: list[str] = []
    for site in sites:
        if site.id not in requested_preferences:
            continue
        desired_enabled = requested_preferences[site.id]
        preference = existing_preferences.get(site.id)
        current_enabled = (
            preference.is_enabled
            if preference is not None
            else site.id in access_site_ids
        )
        if preference is None:
            db.add(
                SiteNotificationPreference(
                    user_id=me.id,
                    site_id=site.id,
                    is_enabled=desired_enabled,
                ),
            )
        else:
            preference.is_enabled = desired_enabled
        if current_enabled != desired_enabled:
            changed_site_names.append(site.name)

    await db.commit()
    for site_name in changed_site_names:
        await refresh_site_notification_user_cache(site_name, db, rds)
    return await list_site_notification_preferences(db, me)
