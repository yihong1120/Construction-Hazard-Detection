from __future__ import annotations

from datetime import datetime
from datetime import timezone

from fastapi import HTTPException
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from examples.auth.models import Group
from examples.auth.models import Site
from examples.auth.models import StreamConfig
from examples.auth.models import User
from examples.db_management.deps import _site_permission
from examples.db_management.deps import ensure_admin_with_group
from examples.db_management.deps import is_super_admin
from examples.db_management.schemas.stream_config import StreamConfigCreate
from examples.db_management.schemas.stream_config import StreamConfigRead
from examples.db_management.schemas.stream_config import StreamConfigUpdate


async def _get_site_or_404(site_id: int, db: AsyncSession) -> Site:
    """Load a site or raise a not-found response.

    Args:
        site_id: Identifier of the site to load.
        db: Database session used to query the site.

    Returns:
        Loaded site.

    Raises:
        HTTPException: If no site has the supplied identifier.
    """
    site = await db.get(Site, site_id)
    if site is None:
        raise HTTPException(status_code=404, detail='Site not found.')
    return site


def _site_group_ids(site: Site) -> set[int]:
    """Return identifiers of groups assigned to a site.

    Args:
        site: Site whose group membership is inspected.

    Returns:
        Set of assigned group identifiers.
    """
    return {group.id for group in site.groups}


def _primary_site_group_id(site: Site) -> int:
    """Return the deterministic primary group identifier for a site.

    Args:
        site: Site whose configured groups are inspected.

    Returns:
        Lowest assigned group identifier.

    Raises:
        HTTPException: If the site is not assigned to any group.
    """
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
    """Resolve the group authorised to own a stream-configuration change.

    Args:
        site: Site whose group membership constrains the change.
        me: Authenticated administrator performing the change.
        requested_group_id: Optional group requested by a super administrator.

    Returns:
        Authorised group identifier for the configuration.

    Raises:
        HTTPException: If the selected group is not assigned to the site.
    """
    if is_super_admin(me):
        group_id = requested_group_id or _primary_site_group_id(site)
    else:
        ensure_admin_with_group(me)
        group_id = me.group_id
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
    """Convert a stream configuration into its response schema.

    Args:
        cfg: Stream configuration to serialise.
        db: Database session used to obtain group limits when not cached.
        group_limit_cache: Per-group current-count and limit cache for this
            response batch.

    Returns:
        Validated stream-configuration response.
    """
    if cfg.group_id not in group_limit_cache:
        # Configurations commonly share a group, so fetch its limit once.
        group_limit_cache[cfg.group_id] = await get_group_stream_limit(
            cfg.group_id, db,
        )
    current, max_streams = group_limit_cache[cfg.group_id]
    return StreamConfigRead(
        id=cfg.id, stream_name=cfg.stream_name, video_url=cfg.video_url,
        model_key=cfg.model_key, recognition_enabled=cfg.recognition_enabled,
        work_start_hour=cfg.work_start_hour, work_end_hour=cfg.work_end_hour,
        detect_no_safety_vest_or_helmet=cfg.detect_no_safety_vest_or_helmet,
        detect_near_machinery_or_vehicle=cfg.detect_near_machinery_or_vehicle,
        detect_in_restricted_area=cfg.detect_in_restricted_area,
        detect_in_utility_pole_restricted_area=(
            cfg.detect_in_utility_pole_restricted_area
        ),
        detect_machinery_close_to_pole=cfg.detect_machinery_close_to_pole,
        expire_date=cfg.expire_date, total_stream_in_group=current,
        max_allowed_streams=max_streams, updated_at=cfg.updated_at,
    )


async def _ensure_stream_name_available(
    site_id: int,
    stream_name: str,
    db: AsyncSession,
    exclude_config_id: int | None = None,
) -> None:
    """Reject a stream name already used by another site configuration.

    Args:
        site_id: Site in which the stream name must be unique.
        stream_name: Candidate stream name.
        db: Database session used to query existing configurations.
        exclude_config_id: Existing configuration retained during an update.

    Raises:
        HTTPException: If another configuration in the site uses the name.
    """
    query = select(StreamConfig).where(
        StreamConfig.site_id == site_id,
        StreamConfig.stream_name == stream_name,
    )
    if exclude_config_id is not None:
        query = query.where(StreamConfig.id != exclude_config_id)
    if await db.scalar(query):
        raise HTTPException(
            status_code=400,
            detail='Stream name already exists in site.',
        )


async def _list_site_stream_config_reads(
    site_id: int,
    db: AsyncSession,
    me: User,
) -> list[StreamConfigRead]:
    """Return stream configurations visible to the requesting administrator.

    Args:
        site_id: Identifier of the site whose configurations are requested.
        db: Database session used to load site and stream data.
        me: Authenticated user whose site and group scope is enforced.

    Returns:
        Response models for configurations visible within the user's scope.

    Raises:
        HTTPException: If the site is missing or outside the user's scope.
    """
    site = await _get_site_or_404(site_id, db)
    _site_permission(me, site=site)
    visible_group_id = None
    if not is_super_admin(me):
        ensure_admin_with_group(me)
        visible_group_id = me.group_id
    configs = await list_stream_configs(site_id, db, group_id=visible_group_id)
    cache: dict[int, tuple[int, int]] = {}
    return [await _stream_config_to_read(cfg, db, cache) for cfg in configs]


async def list_stream_configs(
    site_id: int,
    db: AsyncSession,
    group_id: int | None = None,
) -> list[StreamConfig]:
    """Retrieve a list of StreamConfig objects associated with a specific site.

    Args:
        site_id (int): The ID of the site.
        db (AsyncSession): The asynchronous database session.

    Returns:
        List[StreamConfig]: A list of StreamConfig instances.
    """
    query = select(StreamConfig).where(StreamConfig.site_id == site_id)
    if group_id is not None:
        query = query.where(StreamConfig.group_id == group_id)

    result = await db.execute(query)
    return list(result.scalars().all())


async def create_stream_config(
    payload: StreamConfigCreate,
    db: AsyncSession,
) -> StreamConfig:
    """Create a new StreamConfig instance with provided data.

    Args:
        payload (StreamConfigCreate): A dictionary containing the data
            for the new stream configuration.
        db (AsyncSession): The asynchronous database session.

    Returns:
        StreamConfig: The newly created StreamConfig object.

    Raises:
        HTTPException: If a database error occurs during creation.
    """
    cfg = StreamConfig(**payload.model_dump())
    db.add(cfg)

    try:
        await db.commit()
        await db.refresh(cfg)
        return cfg
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f'Database error: {e}')


async def update_stream_config(
    cfg: StreamConfig,
    updates: StreamConfigUpdate,
    db: AsyncSession,
) -> None:
    """Update an existing StreamConfig object with new data.

    Args:
        cfg (StreamConfig): The existing StreamConfig instance to update.
        updates (StreamConfigUpdate): A dictionary containing updated values.
        db (AsyncSession): The asynchronous database session.

    Raises:
        HTTPException: If a database error occurs during updating.
    """
    # Apply updates to the configuration object
    for key, value in updates.model_dump(exclude_unset=True).items():
        setattr(cfg, key, value)

    # Update the timestamp to the current time
    cfg.updated_at = datetime.now(timezone.utc)

    try:
        await db.commit()
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f'Database error: {e}')


async def delete_stream_config(
    cfg: StreamConfig,
    db: AsyncSession,
) -> None:
    """Delete an existing StreamConfig from the database.

    Args:
        cfg (StreamConfig): The StreamConfig instance to delete.
        db (AsyncSession): The asynchronous database session.

    Raises:
        HTTPException: If a database error occurs during deletion.
    """
    await db.delete(cfg)

    try:
        await db.commit()
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f'Database error: {e}')


async def get_group_stream_limit(
    group_id: int,
    db: AsyncSession,
) -> tuple[int, int]:
    """Retrieve the current number of streams and the maximum allowed streams
    for a specific group.

    Args:
        group_id (int): The ID of the group.
        db (AsyncSession): The asynchronous database session.

    Returns:
        Tuple[int, int]: A tuple containing the current number of streams
            and the maximum allowed streams.

    Raises:
        HTTPException: If the specified group is not found.
    """
    grp = await db.get(Group, group_id)

    if not grp:
        raise HTTPException(status_code=404, detail='Group not found')

    # Count current number of StreamConfig entries for the group
    current = (
        await db.scalar(
            select(func.count())
            .select_from(StreamConfig)
            .where(StreamConfig.group_id == group_id),
        )
    ) or 0

    return current, grp.max_allowed_streams
