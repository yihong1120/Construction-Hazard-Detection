from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from examples.auth.models import Group
from examples.auth.models import StreamConfig
from examples.db_management.schemas.stream_config import StreamConfigCreate
from examples.db_management.schemas.stream_config import StreamConfigUpdate


async def list_stream_configs(
    site_id: int,
    db: AsyncSession,
) -> list[StreamConfig]:
    """Retrieve a list of StreamConfig objects associated with a specific site.

    Args:
        site_id (int): The ID of the site.
        db (AsyncSession): The asynchronous database session.

    Returns:
        List[StreamConfig]: A list of StreamConfig instances.
    """
    result = await db.execute(
        select(StreamConfig).where(StreamConfig.site_id == site_id),
    )
    return result.scalars().all()


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
    cfg = StreamConfig(**payload)
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
    for key, value in updates.items():
        setattr(cfg, key, value)

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
