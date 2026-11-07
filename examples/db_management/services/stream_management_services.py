from __future__ import annotations

from typing import Any
from typing import cast

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.models import StreamConfig
from examples.auth.models import User
from examples.db_management.deps import _site_permission
from examples.db_management.deps import is_super_admin
from examples.db_management.deps import require_admin
from examples.db_management.schemas.resource_options import Option
from examples.db_management.schemas.resource_options import resource_options
from examples.db_management.schemas.resource_options import ResourceOptions
from examples.db_management.schemas.stream_config import SiteStreamConfigUpsert
from examples.db_management.schemas.stream_config import StreamConfigCreate
from examples.db_management.schemas.stream_config import StreamConfigRead
from examples.db_management.schemas.stream_config import StreamConfigUpdate
from examples.db_management.services.site_access import require_site
from examples.db_management.services.stream_config_services import (
    _get_site_or_404,
)
from examples.db_management.services.stream_config_services import (
    _list_site_stream_config_reads,
)
from examples.db_management.services.stream_config_services import (
    _resolve_stream_group_id,
)
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
    require_stream_capacity,
)
from examples.db_management.services.stream_config_services import (
    stream_config_transaction,
)
from examples.db_management.services.stream_config_services import (
    update_stream_config,
)
from examples.YOLO_server_api.model_registry import load_registry_async
from examples.YOLO_server_api.model_registry import require_model_async
from examples.YOLO_server_api.model_registry import revision_for


async def get_stream_model_options(
    site_id: int,
    me: User,
    db: AsyncSession,
) -> ResourceOptions:
    options = []
    require_admin(me)
    await require_site(db, me, site_id)
    registry = await load_registry_async()
    source_revision = revision_for(registry.model_dump())
    for model in registry.models:
        if model.allowed(str(me.tenant_id), me.role, 'stream', site_id):
            options.append(Option(id=model.id, label=model.display_name))
    default = (
        registry.default_id
        if registry.default_id in {o.id for o in options}
        else None
    )
    return resource_options(
        me,
        site_id,
        options,
        default,
        source=source_revision,
    )


async def list_site_stream_configs(
    site_id: int,
    db: AsyncSession,
    me: User,
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


async def create_stream_config_for_operator(
    payload: StreamConfigCreate,
    db: AsyncSession,
    me: User,
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
    await require_site(db, me, site.id)
    group_id = _resolve_stream_group_id(
        site,
        me,
        requested_group_id=payload.group_id,
    )

    await require_model_async(
        payload.model_key,
        None,
        str(me.tenant_id),
        me.role,
        'stream',
        site.id,
    )
    payload_with_group = payload.model_copy(update={'group_id': group_id})
    async with stream_config_transaction(db):
        await require_stream_capacity(group_id, 1, db)
        cfg = await create_stream_config(payload_with_group, db)

    return {
        'id': cfg.id,
        'message': 'Stream configuration created successfully.',
    }


async def upsert_site_stream_configs(
    site_id: int,
    payload: SiteStreamConfigUpsert,
    db: AsyncSession,
    me: User,
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
    await require_site(db, me, site.id)
    group_id = _resolve_stream_group_id(site, me)
    visible_group_id = None if is_super_admin(cast(Any, me)) else group_id

    for item in payload.streams:
        await require_model_async(
            item.model_key,
            None,
            str(me.tenant_id),
            me.role,
            'stream',
            site.id,
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
        cfg
        for cfg in site_configs
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

    async with stream_config_transaction(db):
        if new_items:
            await require_stream_capacity(group_id, len(new_items), db)
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

    return await _list_site_stream_config_reads(site_id, db, me)


async def update_stream_config_for_operator(
    cfg_id: int,
    payload: StreamConfigUpdate,
    db: AsyncSession,
    me: User,
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
            status_code=404,
            detail='Stream configuration not found.',
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
                status_code=400,
                detail='Stream name already exists in site.',
            )

    await require_site(db, me, cfg.site_id)
    await require_model_async(
        payload.model_key or cfg.model_key,
        None,
        str(me.tenant_id),
        me.role,
        'stream',
        cfg.site_id,
    )
    async with stream_config_transaction(db):
        await update_stream_config(cfg, payload, db)

    return {'message': 'Stream configuration updated successfully.'}


async def delete_stream_config_for_operator(
    cfg_id: int,
    db: AsyncSession,
    me: User,
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
            status_code=404,
            detail='Stream configuration not found.',
        )

    _site_permission(me, site=cfg.site)
    _site_permission(me, group_id=cfg.group_id)
    async with stream_config_transaction(db):
        await delete_stream_config(cfg, db)

    return {'message': 'Stream configuration deleted successfully.'}


async def get_group_stream_usage(
    group_id: int,
    db: AsyncSession,
    me: User,
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
