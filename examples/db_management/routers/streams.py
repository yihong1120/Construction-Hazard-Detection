from __future__ import annotations

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.database import get_db
from examples.auth.models import Site
from examples.auth.models import StreamConfig
from examples.auth.models import User
from examples.db_management.deps import _site_permission
from examples.db_management.deps import ensure_admin_with_group
from examples.db_management.deps import get_current_user
from examples.db_management.deps import is_super_admin
from examples.db_management.deps import require_admin
from examples.db_management.schemas.stream_config import SiteStreamConfigUpsert
from examples.db_management.schemas.stream_config import StreamConfigCreate
from examples.db_management.schemas.stream_config import StreamConfigRead
from examples.db_management.schemas.stream_config import StreamConfigUpdate
from examples.db_management.services.stream_config_services import (
    create_stream_config,
)
from examples.db_management.services.stream_config_services import (
    delete_stream_config,
)
from examples.db_management.services.stream_config_services import (
    get_group_stream_limit,
)
from examples.db_management.services.stream_config_services import (
    list_stream_configs,
)
from examples.db_management.services.stream_config_services import (
    update_stream_config,
)

router = APIRouter(tags=['stream-config'])


async def _get_site_or_404(site_id: int, db: AsyncSession) -> Site:
    """Load a site or raise a stable 404."""
    site = await db.get(Site, site_id)
    if not site:
        raise HTTPException(status_code=404, detail='Site not found.')
    return site


def _site_group_ids(site: Site) -> set[int]:
    """Return group IDs associated with a site."""
    return {group.id for group in site.groups}


def _primary_site_group_id(site: Site) -> int:
    """Return the deterministic default owner group for a site."""
    group_ids = sorted(_site_group_ids(site))
    if not group_ids:
        raise HTTPException(
            status_code=400,
            detail='Site must have a group before configuring streams.',
        )
    return group_ids[0]


def _resolve_stream_group_id(
    site: Site,
    me: User,
    requested_group_id: int | None = None,
) -> int:
    """Resolve stream ownership from the authenticated user and site."""
    if is_super_admin(me):
        group_id = requested_group_id or _primary_site_group_id(site)
    else:
        ensure_admin_with_group(me)
        group_id = me.group_id

    if group_id is None:
        raise HTTPException(
            status_code=400,
            detail='group_id is required to create a stream.',
        )

    if group_id not in _site_group_ids(site):
        raise HTTPException(
            status_code=403,
            detail='Group is not associated with this site.',
        )

    return group_id


async def _stream_config_to_read(
    cfg: StreamConfig,
    db: AsyncSession,
    group_limit_cache: dict[int, tuple[int, int]],
) -> StreamConfigRead:
    """Serialize a StreamConfig and include current group usage limits."""
    if cfg.group_id not in group_limit_cache:
        group_limit_cache[cfg.group_id] = await get_group_stream_limit(
            cfg.group_id, db,
        )
    current, max_streams = group_limit_cache[cfg.group_id]
    return StreamConfigRead(
        id=cfg.id,
        stream_name=cfg.stream_name,
        video_url=cfg.video_url,
        model_key=cfg.model_key,
        recognition_enabled=cfg.recognition_enabled,
        work_start_hour=cfg.work_start_hour,
        work_end_hour=cfg.work_end_hour,
        detect_no_safety_vest_or_helmet=(
            cfg.detect_no_safety_vest_or_helmet
        ),
        detect_near_machinery_or_vehicle=(
            cfg.detect_near_machinery_or_vehicle
        ),
        detect_in_restricted_area=cfg.detect_in_restricted_area,
        detect_in_utility_pole_restricted_area=(
            cfg.detect_in_utility_pole_restricted_area
        ),
        detect_machinery_close_to_pole=(
            cfg.detect_machinery_close_to_pole
        ),
        expire_date=cfg.expire_date,
        total_stream_in_group=current,
        max_allowed_streams=max_streams,
        updated_at=cfg.updated_at,
    )


async def _ensure_stream_name_available(
    site_id: int,
    stream_name: str,
    db: AsyncSession,
    exclude_config_id: int | None = None,
) -> None:
    """Ensure a stream name is unique within a site."""
    query = select(StreamConfig).where(
        StreamConfig.site_id == site_id,
        StreamConfig.stream_name == stream_name,
    )
    if exclude_config_id is not None:
        query = query.where(StreamConfig.id != exclude_config_id)

    exists = await db.scalar(query)
    if exists:
        raise HTTPException(
            status_code=400,
            detail='Stream name already exists in site.',
        )


async def _list_site_stream_config_reads(
    site_id: int,
    db: AsyncSession,
    me: User,
) -> list[StreamConfigRead]:
    """List stream configs visible to the current admin for a site."""
    site = await _get_site_or_404(site_id, db)
    _site_permission(me, site=site)

    visible_group_id: int | None = None
    if not is_super_admin(me):
        ensure_admin_with_group(me)
        visible_group_id = me.group_id

    stream_configs = await list_stream_configs(
        site_id, db, group_id=visible_group_id,
    )

    group_limit_cache: dict[int, tuple[int, int]] = {}
    return [
        await _stream_config_to_read(c, db, group_limit_cache)
        for c in stream_configs
    ]


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
    """List stream settings by site without frontend group selection."""
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
    """Upsert stream settings by site without accepting frontend group_id."""
    site = await _get_site_or_404(site_id, db)
    _site_permission(me, site=site)
    group_id = _resolve_stream_group_id(site, me)
    visible_group_id = None if is_super_admin(me) else group_id

    stream_names = [item.stream_name for item in payload.streams]
    if len(stream_names) != len(set(stream_names)):
        raise HTTPException(
            status_code=400,
            detail='Duplicate stream names are not allowed.',
        )

    existing_configs = await list_stream_configs(
        site_id,
        db,
        group_id=visible_group_id,
    )
    existing_by_id = {cfg.id: cfg for cfg in existing_configs}

    for item in payload.streams:
        item_data = item.model_dump(exclude={'id'})
        if item.id is None:
            await _ensure_stream_name_available(
                site_id,
                item.stream_name,
                db,
            )
            current, limit = await get_group_stream_limit(group_id, db)
            if current >= limit:
                raise HTTPException(
                    status_code=403,
                    detail='Stream limit reached for group.',
                )
            await create_stream_config(
                StreamConfigCreate(
                    site_id=site_id,
                    group_id=group_id,
                    **item_data,
                ),
                db,
            )
            continue

        cfg = existing_by_id.get(item.id)
        if cfg is None:
            raise HTTPException(
                status_code=404,
                detail='Stream configuration not found.',
            )

        if item.stream_name != cfg.stream_name:
            await _ensure_stream_name_available(
                site_id,
                item.stream_name,
                db,
                exclude_config_id=cfg.id,
            )
        await update_stream_config(
            cfg,
            StreamConfigUpdate(**item_data),
            db,
        )

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
        is_super_admin(me)
        or (me.role == 'admin' and me.group_id == group_id)
    ):
        raise HTTPException(status_code=403, detail='Permission denied.')

    current, limit = await get_group_stream_limit(group_id, db)

    return {
        'group_id': group_id,
        'max_allowed_streams': limit,
        'current_streams_count': current,
    }
