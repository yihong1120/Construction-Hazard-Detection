from __future__ import annotations

from typing import cast

import redis.asyncio as redis
from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.database import get_db
from examples.auth.models import Site
from examples.auth.models import User
from examples.auth.redis_pool import get_redis_pool
from examples.auth.user_service import invalidate_effective_site_cache
from examples.db_management.deps import _site_permission
from examples.db_management.deps import ensure_admin_with_group
from examples.db_management.deps import get_current_user
from examples.db_management.deps import is_super_admin
from examples.db_management.deps import require_admin
from examples.db_management.deps import SUPER_ADMIN_NAME
from examples.db_management.schemas.site import SiteCreate
from examples.db_management.schemas.site import SiteDelete
from examples.db_management.schemas.site import SiteGroupOp
from examples.db_management.schemas.site import SiteRead
from examples.db_management.schemas.site import SiteUpdate
from examples.db_management.schemas.site import SiteUserOp
from examples.db_management.services.site_services import add_group_to_site
from examples.db_management.services.site_services import add_user_to_site
from examples.db_management.services.site_services import create_site
from examples.db_management.services.site_services import delete_matching_redis_keys
from examples.db_management.services.site_services import delete_site
from examples.db_management.services.site_services import encode_site_name
from examples.db_management.services.site_services import list_sites
from examples.db_management.services.site_services import \
    remove_group_from_site
from examples.db_management.services.site_services import remove_user_from_site
from examples.db_management.services.site_services import site_to_read
from examples.db_management.services.site_services import update_site
from examples.local_notification_server.services import (
    invalidate_site_notification_user_cache,
)
from examples.local_notification_server.services import \
    refresh_site_notification_user_cache

router: APIRouter = APIRouter(tags=['site-mgmt'])


@router.get('/list_sites', response_model=list[SiteRead])
async def endpoint_list_sites(
    db: AsyncSession = Depends(get_db),
    me: User = Depends(get_current_user),
) -> list[SiteRead]:
    """Retrieve a list of sites accessible to the user.

    Args:
        db (AsyncSession): Database session.
        me (User): Current authenticated user.

    Returns:
        list[SiteRead]: List of accessible sites.

    Raises:
        HTTPException: If the user lacks admin privileges.
    """
    # Super admin retrieves all sites; admin retrieves group-specific sites
    if is_super_admin(me):
        sites = await list_sites(db)
        visible_group_id = None
    elif me.role == 'admin':
        ensure_admin_with_group(me)
        sites = await list_sites(db, group_id=me.group_id)
        visible_group_id = me.group_id
    else:
        raise HTTPException(status_code=403, detail='Admin role required.')

    # Convert Site objects to SiteRead schemas for response
    return [
        site_to_read(site, visible_group_id=visible_group_id)
        for site in sites
    ]


@router.post(
    '/create_site',
    response_model=SiteRead,
    dependencies=[Depends(require_admin)],
)
async def endpoint_create_site(
    payload: SiteCreate,
    db: AsyncSession = Depends(get_db),
    me: User = Depends(get_current_user),
) -> SiteRead:
    """Create a new site.

    Args:
        payload (SiteCreate): Data required to create a site.
        db (AsyncSession): Database session.
        me (User): Current authenticated user.

    Returns:
        SiteRead: Details of the created site.

    Raises:
        HTTPException: If permission check fails.
    """
    if is_super_admin(me):
        group_ids: list[int] = payload.group_ids
        visible_group_id = None
    else:
        ensure_admin_with_group(me)
        # Validate any explicitly supplied group IDs before overriding
        for gid in payload.group_ids:
            _site_permission(me, group_id=gid)
        group_ids = [cast(int, me.group_id)]
        visible_group_id = me.group_id

    site: Site = await create_site(payload.name, group_ids, db)
    invalidate_effective_site_cache()

    return site_to_read(site, visible_group_id=visible_group_id)


@router.put(
    '/update_site',
    dependencies=[Depends(require_admin)],
)
async def endpoint_update_site(
    payload: SiteUpdate,
    db: AsyncSession = Depends(get_db),
    me: User = Depends(get_current_user),
    rds: redis.Redis = Depends(get_redis_pool),
) -> dict[str, str]:
    """Update an existing site's name.

    Args:
        payload (SiteUpdate): Contains site ID and the new site name.
        db (AsyncSession): Database session.
        me (User): Current authenticated user.

    Returns:
        dict[str, str]: Confirmation message.

    Raises:
        HTTPException: If the site is not found or permission fails.
    """
    # Retrieve the site to update
    site: Site | None = (
        await db.execute(
            select(Site).where(Site.id == payload.site_id),
        )
    ).scalar_one_or_none()

    if not site:
        raise HTTPException(status_code=404, detail='Site not found.')

    # Permission check before updating the site
    _site_permission(me, site=site)

    old_name = site.name
    await update_site(site, payload.new_name, db)
    invalidate_effective_site_cache()
    await invalidate_site_notification_user_cache([old_name], rds)
    await refresh_site_notification_user_cache(payload.new_name, db, rds)
    return {'message': 'Site updated successfully.'}


@router.delete(
    '/delete_site',
    dependencies=[Depends(require_admin)],
)
async def endpoint_delete_site(
    payload: SiteDelete,
    db: AsyncSession = Depends(get_db),
    me: User = Depends(get_current_user),
    rds: redis.Redis = Depends(get_redis_pool),
) -> dict[str, str]:
    """Delete an existing site and related data, and remove related Redis keys.

    Args:
        payload (SiteDelete): Contains site ID to delete.
        db (AsyncSession): Database session.
        me (User): Current authenticated user.
        rds (redis.Redis): Redis connection pool.

    Returns:
        dict[str, str]: Confirmation message.

    Raises:
        HTTPException: If the site is not found or permission fails.
    """
    # Retrieve the site to delete
    site: Site | None = (
        await db.execute(
            select(Site).where(Site.id == payload.site_id),
        )
    ).unique().scalar_one_or_none()

    if not site:
        raise HTTPException(status_code=404, detail='Site not found.')

    # Check permissions for deletion
    _site_permission(me, site=site)

    encoded_name: str = encode_site_name(site.name)
    key_pattern: str = f'stream_metadata:{encoded_name}*'
    await delete_matching_redis_keys(rds, key_pattern)

    await invalidate_site_notification_user_cache([site.name], rds)

    await delete_site(site, db)
    invalidate_effective_site_cache()
    return {'message': 'Site and related data deleted successfully.'}


@router.post(
    '/add_user_to_site',
    dependencies=[Depends(require_admin)],
)
async def endpoint_add_user_to_site(
    payload: SiteUserOp,
    db: AsyncSession = Depends(get_db),
    me: User = Depends(get_current_user),
    rds: redis.Redis = Depends(get_redis_pool),
) -> dict[str, str]:
    """Associate a user with a specific site.

    Args:
        payload (SiteUserOp): Contains user ID and site ID.
        db (AsyncSession): Database session.
        me (User): Current authenticated user.

    Returns:
        dict[str, str]: Confirmation message.

    Raises:
        HTTPException: If site or user is not found, or permission fails.
    """
    # Retrieve the site to which the user will be added
    site: Site | None = (
        await db.execute(
            select(Site).where(Site.id == payload.site_id),
        )
    ).unique().scalar_one_or_none()

    if not site:
        raise HTTPException(status_code=404, detail='Site not found.')

    # Permission check
    _site_permission(me, site=site)

    # Retrieve the user to add
    user: User | None = (
        await db.execute(
            select(User).where(User.id == payload.user_id),
        )
    ).unique().scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail='User not found.')

    if user.username == SUPER_ADMIN_NAME:
        raise HTTPException(
            status_code=403,
            detail="Cannot modify super admin's site membership.",
        )

    if not is_super_admin(me) and user.group_id is None:
        raise HTTPException(
            status_code=403,
            detail='User and site must belong to the same group.',
        )
    _site_permission(me, group_id=user.group_id)

    if user.group_id not in {g.id for g in site.groups}:
        raise HTTPException(
            status_code=403,
            detail='User and site must belong to the same group.',
        )

    await add_user_to_site(user.id, site.id, db)
    invalidate_effective_site_cache()
    await refresh_site_notification_user_cache(site.name, db, rds)
    return {'message': 'User linked to site successfully.'}


@router.post(
    '/remove_user_from_site',
    dependencies=[Depends(require_admin)],
)
async def endpoint_remove_user_from_site(
    payload: SiteUserOp,
    db: AsyncSession = Depends(get_db),
    me: User = Depends(get_current_user),
    rds: redis.Redis = Depends(get_redis_pool),
) -> dict[str, str]:
    """Dissociate a user from a specific site.

    Args:
        payload (SiteUserOp): Contains user ID and site ID.
        db (AsyncSession): Database session.
        me (User): Current authenticated user.

    Returns:
        dict[str, str]: Confirmation message.

    Raises:
        HTTPException: If site or user is not found, or permission fails.
    """
    # Retrieve the site from which the user will be removed
    site: Site | None = (
        await db.execute(
            select(Site).where(Site.id == payload.site_id),
        )
    ).unique().scalar_one_or_none()

    if not site:
        raise HTTPException(status_code=404, detail='Site not found.')

    _site_permission(me, site=site)

    # Retrieve the user to remove
    user: User | None = (
        await db.execute(
            select(User).where(User.id == payload.user_id),
        )
    ).unique().scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail='User not found.')

    if user.username == SUPER_ADMIN_NAME:
        raise HTTPException(
            status_code=403, detail='Cannot remove super admin from site.',
        )

    if not is_super_admin(me) and user.group_id is None:
        raise HTTPException(
            status_code=403,
            detail='Cannot manage users outside your group.',
        )
    _site_permission(me, group_id=user.group_id)

    await remove_user_from_site(user.id, site.id, db)
    invalidate_effective_site_cache()
    await refresh_site_notification_user_cache(site.name, db, rds)
    return {'message': 'User unlinked from site successfully.'}


@router.post(
    '/add_group_to_site',
    dependencies=[Depends(require_admin)],
)
async def endpoint_add_group_to_site(
    payload: SiteGroupOp,
    db: AsyncSession = Depends(get_db),
    me: User = Depends(get_current_user),
) -> dict[str, str]:
    """Associate a group with a specific site.

    Args:
        payload (SiteGroupOp): Contains site ID and group ID.
        db (AsyncSession): Database session.
        me (User): Current authenticated user.

    Returns:
        dict[str, str]: Confirmation message.

    Raises:
        HTTPException: If site is not found or permission fails.
    """
    site: Site | None = (
        await db.execute(
            select(Site).where(Site.id == payload.site_id),
        )
    ).unique().scalar_one_or_none()

    if not site:
        raise HTTPException(status_code=404, detail='Site not found.')

    _site_permission(me, site=site)
    _site_permission(me, group_id=payload.group_id)

    await add_group_to_site(site.id, payload.group_id, db)
    invalidate_effective_site_cache()
    return {'message': 'Group linked to site successfully.'}


@router.post(
    '/remove_group_from_site',
    dependencies=[Depends(require_admin)],
)
async def endpoint_remove_group_from_site(
    payload: SiteGroupOp,
    db: AsyncSession = Depends(get_db),
    me: User = Depends(get_current_user),
) -> dict[str, str]:
    """Dissociate a group from a specific site.

    Args:
        payload (SiteGroupOp): Contains site ID and group ID.
        db (AsyncSession): Database session.
        me (User): Current authenticated user.

    Returns:
        dict[str, str]: Confirmation message.

    Raises:
        HTTPException: If site is not found or permission fails.
    """
    site: Site | None = (
        await db.execute(
            select(Site).where(Site.id == payload.site_id),
        )
    ).unique().scalar_one_or_none()

    if not site:
        raise HTTPException(status_code=404, detail='Site not found.')

    _site_permission(me, site=site)
    _site_permission(me, group_id=payload.group_id)

    await remove_group_from_site(site.id, payload.group_id, db)
    invalidate_effective_site_cache()
    return {'message': 'Group unlinked from site successfully.'}
