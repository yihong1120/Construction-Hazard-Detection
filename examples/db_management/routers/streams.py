from __future__ import annotations

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.database import get_db
from examples.auth.models import Site
from examples.auth.models import StreamConfig
from examples.db_management.deps import _site_permission
from examples.db_management.deps import get_current_user
from examples.db_management.deps import is_super_admin
from examples.db_management.deps import require_admin
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


@router.get('/list_stream_configs', response_model=list[StreamConfigRead])
async def endpoint_list_stream_configs(
    site_id: int,
    db: AsyncSession = Depends(get_db),
    me=Depends(get_current_user),
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
    site = await db.get(Site, site_id)
    if not site:
        raise HTTPException(status_code=404, detail='Site not found.')

    _site_permission(me, site=site)

    stream_configs = await list_stream_configs(site_id, db)

    # Retrieve current stream count and group limit
    current, _ = await get_group_stream_limit(site.group_id, db)

    return [
        StreamConfigRead(
            id=c.id,
            stream_name=c.stream_name,
            video_url=c.video_url,
            model_key=c.model_key,
            detect_with_server=c.detect_with_server,
            store_in_redis=c.store_in_redis,
            work_start_hour=c.work_start_hour,
            work_end_hour=c.work_end_hour,
            detect_no_safety_vest_or_helmet=c.detect_no_safety_vest_or_helmet,
            detect_near_machinery_or_vehicle=(
                c.detect_near_machinery_or_vehicle
            ),
            detect_in_restricted_area=c.detect_in_restricted_area,
            detect_in_utility_pole_restricted_area=(
                c.detect_in_utility_pole_restricted_area
            ),
            detect_machinery_close_to_pole=c.detect_machinery_close_to_pole,
            expire_date=c.expire_date,
            total_stream_in_group=current,
            max_allowed_streams=site.group.max_allowed_streams,
            updated_at=c.updated_at,
        )
        for c in stream_configs
    ]


@router.post(
    '/create_stream_config',
    dependencies=[Depends(require_admin)],
)
async def endpoint_create_stream_config(
    payload: StreamConfigCreate,
    db: AsyncSession = Depends(get_db),
    me=Depends(get_current_user),
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
    site = await db.get(Site, payload.site_id)
    if not site:
        raise HTTPException(status_code=404, detail='Site not found.')

    _site_permission(me, site=site)

    current, limit = await get_group_stream_limit(site.group_id, db)
    if current >= limit:
        raise HTTPException(
            status_code=403, detail='Stream limit reached for group.',
        )

    data = payload.model_dump()
    data['group_id'] = site.group_id
    cfg = await create_stream_config(data, db)

    return {
        'id': cfg.id,
        'message': 'Stream configuration created successfully.',
    }


@router.put(
    '/stream_config/update/{cfg_id}',
    dependencies=[Depends(require_admin)],
)
async def endpoint_update_stream_config(
    cfg_id: int,
    payload: StreamConfigUpdate,
    db: AsyncSession = Depends(get_db),
    me=Depends(get_current_user),
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

    updates = payload.model_dump(exclude_none=True)
    await update_stream_config(cfg, updates, db)

    return {'message': 'Stream configuration updated successfully.'}


@router.delete(
    '/delete_stream_config/{cfg_id}',
    dependencies=[Depends(require_admin)],
)
async def endpoint_delete_stream_config(
    cfg_id: int,
    db: AsyncSession = Depends(get_db),
    me=Depends(get_current_user),
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
    await delete_stream_config(cfg, db)

    return {'message': 'Stream configuration deleted successfully.'}


@router.get(
    '/group_stream_limit',
    dependencies=[Depends(get_current_user)],
)
async def endpoint_group_stream_limit(
    group_id: int,
    db: AsyncSession = Depends(get_db),
    me=Depends(get_current_user),
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
