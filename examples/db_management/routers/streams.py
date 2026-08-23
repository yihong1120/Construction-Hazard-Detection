from __future__ import annotations

from typing import Any
from typing import cast

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.database import get_db
from examples.auth.models import Group
from examples.auth.models import StreamConfig
from examples.auth.models import User
from examples.db_management.deps import _site_permission
from examples.db_management.deps import get_current_user
from examples.db_management.deps import is_super_admin
from examples.db_management.deps import require_admin
from examples.db_management.schemas.stream_config import SiteStreamConfigUpsert
from examples.db_management.schemas.stream_config import StreamConfigCreate
from examples.db_management.schemas.stream_config import StreamConfigRead
from examples.db_management.schemas.stream_config import StreamConfigUpdate
from examples.db_management.services.stream_config_services import \
    _get_site_or_404
from examples.db_management.services.stream_config_services import \
    _list_site_stream_config_reads
from examples.db_management.services.stream_config_services import \
    _resolve_stream_group_id
from examples.db_management.services.stream_config_services import create_stream_config
from examples.db_management.services.stream_config_services import (
    delete_stream_config,
)
from examples.db_management.services.stream_config_services import (
    get_group_stream_limit,
)
from examples.db_management.services.stream_config_services import (
    list_stream_configs,
)
from examples.db_management.services.stream_config_services import update_stream_config

router = APIRouter(tags=['stream-config'])


@router.get('/list_stream_configs', response_model=list[StreamConfigRead])
async def endpoint_list_stream_configs(
    site_id: int,
    db: AsyncSession = Depends(get_db),
    me: User = Depends(get_current_user),
) -> list[StreamConfigRead]:
    """List all stream configurations for a given site.

    Args:
        site_id (int): The identifier of the site.
        db (AsyncSession): The database session.
        me (User): The currently authenticated user.

    Returns:
        List[StreamConfigRead]: A list of stream configuration details.

    Raises:
        HTTPException: If the site does not exist or the user lacks permission.
    """
    return await _list_site_stream_config_reads(site_id, db, me)


@router.get(
    '/sites/{site_id}/stream-config',
    response_model=list[StreamConfigRead],
)
async def endpoint_get_site_stream_config(
    site_id: int,
    db: AsyncSession = Depends(get_db),
    me: User = Depends(get_current_user),
) -> list[StreamConfigRead]:
    """List stream settings for a site without accepting a group selection.

    Args:
        site_id: Identifier of the site whose streams are requested.
        db: Database session used to load stream configurations.
        me: Authenticated user whose site scope is enforced.

    Returns:
        Stream configurations visible to the authenticated user.

    Raises:
        HTTPException: If the site does not exist or is outside the user's
            management scope.
    """
    return await _list_site_stream_config_reads(site_id, db, me)


@router.post(
    '/create_stream_config',
    dependencies=[Depends(require_admin)],
)
async def endpoint_create_stream_config(
    payload: StreamConfigCreate,
    db: AsyncSession = Depends(get_db),
    me: User = Depends(get_current_user),
) -> dict[str, str | int]:
    """Create a new stream configuration for a site.

    Args:
        payload (StreamConfigCreate): The stream configuration data.
        db (AsyncSession): The database session.
        me (User): The currently authenticated user.

    Returns:
        Dict[str, str | int]: Confirmation message and the created config ID.

    Raises:
        HTTPException: If the site does not exist
            or group stream limit is exceeded.
    """
    site = await _get_site_or_404(payload.site_id, db)

    _site_permission(me, site=site)
    group_id = _resolve_stream_group_id(
        site,
        me,
        requested_group_id=payload.group_id,
    )

    current, limit = await get_group_stream_limit(group_id, db)
    if current >= limit:
        raise HTTPException(
            status_code=403, detail='Stream limit reached for group.',
        )

    payload_with_group = payload.model_copy(update={'group_id': group_id})
    cfg = await create_stream_config(payload_with_group, db)

    return {
        'id': cfg.id,
        'message': 'Stream configuration created successfully.',
    }


@router.put(
    '/sites/{site_id}/stream-config',
    response_model=list[StreamConfigRead],
    dependencies=[Depends(require_admin)],
)
async def endpoint_put_site_stream_config(
    site_id: int,
    payload: SiteStreamConfigUpsert,
    db: AsyncSession = Depends(get_db),
    me: User = Depends(get_current_user),
) -> list[StreamConfigRead]:
    """Replace or create stream settings for a site-owned group.

    Args:
        site_id: Identifier of the site whose streams are being updated.
        payload: Desired site-scoped stream configuration items.
        db: Database session used to persist configurations.
        me: Authenticated administrator whose scope determines the group.

    Returns:
        Current stream configurations visible to the administrator.

    Raises:
        HTTPException: If the site or an existing configuration is unavailable,
            a name is duplicated, or the group's stream limit is reached.
    """
    site = await _get_site_or_404(site_id, db)
    _site_permission(me, site=site)
    group_id = _resolve_stream_group_id(site, me)
    visible_group_id = (
        None if is_super_admin(cast(Any, me)) else group_id
    )

    # Validate the complete replacement before creating or changing any row.
    stream_names = [item.stream_name for item in payload.streams]
    if len(stream_names) != len(set(stream_names)):
        raise HTTPException(
            status_code=400,
            detail='Duplicate stream names are not allowed.',
        )

    # Read all site configurations once.  Besides making name validation a
    # local lookup, this preserves the site-wide uniqueness rule even when an
    # administrator can only update one of the site's groups.
    site_configs = await list_stream_configs(site_id, db)
    existing_configs = [
        cfg for cfg in site_configs
        if visible_group_id is None or cfg.group_id == visible_group_id
    ]
    existing_by_id = {cfg.id: cfg for cfg in existing_configs}
    config_id_by_name = {cfg.stream_name: cfg.id for cfg in site_configs}

    new_items = [item for item in payload.streams if item.id is None]
    for item in payload.streams:
        cfg = existing_by_id.get(item.id) if item.id is not None else None
        if item.id is not None and cfg is None:
            raise HTTPException(
                status_code=404,
                detail='Stream configuration not found.',
            )
        owner_id = config_id_by_name.get(item.stream_name)
        if owner_id is not None and owner_id != item.id:
            raise HTTPException(
                status_code=400,
                detail='Stream name already exists in site.',
            )

    if new_items:
        # Serialise quota checks for this group.  Without the row lock, two
        # concurrent replacement requests can both pass a stale COUNT(*).
        await db.execute(
            select(Group.id).where(Group.id == group_id).with_for_update(),
        )
        current, limit = await get_group_stream_limit(group_id, db)
        if current + len(new_items) > limit:
            raise HTTPException(
                status_code=403,
                detail='Stream limit reached for group.',
            )

    for item in payload.streams:
        item_data = item.model_dump(exclude={'id'})
        if item.id is None:
            db.add(
                StreamConfig(
                    site_id=site_id,
                    group_id=group_id,
                    **item_data,
                ),
            )
            continue

        cfg = existing_by_id[item.id]
        for field, value in item_data.items():
            setattr(cfg, field, value)

    try:
        # Flush assigns identifiers and checks all constraints before the one
        # commit.  It avoids N transactions and N round-trips for a wall's
        # configuration replacement.
        await db.flush()
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=409,
            detail='Stream configuration update conflicted.',
        ) from exc
    except Exception as exc:
        await db.rollback()
        raise HTTPException(
            status_code=500,
            detail='Unable to update stream configurations.',
        ) from exc

    return await _list_site_stream_config_reads(site_id, db, me)


@router.put(
    '/stream_config/update/{cfg_id}',
    dependencies=[Depends(require_admin)],
)
async def endpoint_update_stream_config(
    cfg_id: int,
    payload: StreamConfigUpdate,
    db: AsyncSession = Depends(get_db),
    me: User = Depends(get_current_user),
) -> dict[str, str]:
    """Update an existing stream configuration.

    Args:
        cfg_id (int): The stream configuration identifier.
        payload (StreamConfigUpdate): Updated stream configuration details.
        db (AsyncSession): The database session.
        me (User): The currently authenticated user.

    Returns:
        Dict[str, str]: Confirmation message of successful update.

    Raises:
        HTTPException: If config does not exist, permission denied,
            or name conflict.
    """
    cfg = await db.get(StreamConfig, cfg_id)
    if not cfg:
        raise HTTPException(
            status_code=404, detail='Stream configuration not found.',
        )

    _site_permission(me, site=cfg.site)
    _site_permission(me, group_id=cfg.group_id)

    # Check for name duplication within the same site
    if payload.stream_name and payload.stream_name != cfg.stream_name:
        exists = await db.scalar(
            select(StreamConfig).where(
                StreamConfig.site_id == cfg.site_id,
                StreamConfig.stream_name == payload.stream_name,
            ),
        )
        if exists:
            raise HTTPException(
                status_code=400, detail='Stream name already exists in site.',
            )

    await update_stream_config(cfg, payload, db)

    return {'message': 'Stream configuration updated successfully.'}


@router.delete(
    '/delete_stream_config/{cfg_id}',
    dependencies=[Depends(require_admin)],
)
async def endpoint_delete_stream_config(
    cfg_id: int,
    db: AsyncSession = Depends(get_db),
    me: User = Depends(get_current_user),
) -> dict[str, str]:
    """Delete an existing stream configuration.

    Args:
        cfg_id (int): The identifier of the stream configuration.
        db (AsyncSession): The database session.
        me (User): The currently authenticated user.

    Returns:
        Dict[str, str]: Confirmation message.

    Raises:
        HTTPException: If configuration not found or permission denied.
    """
    cfg = await db.get(StreamConfig, cfg_id)
    if not cfg:
        raise HTTPException(
            status_code=404, detail='Stream configuration not found.',
        )

    _site_permission(me, site=cfg.site)
    _site_permission(me, group_id=cfg.group_id)
    await delete_stream_config(cfg, db)

    return {'message': 'Stream configuration deleted successfully.'}


@router.get(
    '/group_stream_limit',
    dependencies=[Depends(get_current_user)],
)
async def endpoint_group_stream_limit(
    group_id: int,
    db: AsyncSession = Depends(get_db),
    me: User = Depends(get_current_user),
) -> dict[str, int]:
    """Retrieve the stream limit and current usage for a group.

    Args:
        group_id (int): The identifier of the group.
        db (AsyncSession): The database session.
        me (User): The currently authenticated user.

    Returns:
        Dict[str, int]: Details of stream usage and limits.

    Raises:
        HTTPException: If the user lacks permission to view the group's limits.
    """
    # Super admin has unlimited access, admins restricted to their group
    if not (
        is_super_admin(cast(Any, me))
        or (me.role == 'admin' and me.group_id == group_id)
    ):
        raise HTTPException(status_code=403, detail='Permission denied.')

    current, limit = await get_group_stream_limit(group_id, db)

    return {
        'group_id': group_id,
        'max_allowed_streams': limit,
        'current_streams_count': current,
    }
