from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from examples.auth.models import Feature
from examples.auth.models import Group
from examples.auth.models import group_features_table


async def list_features(db: AsyncSession) -> list[Feature]:
    """Retrieve all available features.

    Args:
        db (AsyncSession): Database session.

    Returns:
        List[Feature]: A list of all features.
    """
    result = await db.execute(select(Feature))
    return result.unique().scalars().all()


async def create_feature(
    name: str,
    description: str | None,
    db: AsyncSession,
) -> Feature:
    """Create a new feature.

    Args:
        name (str): Name of the feature.
        description (Optional[str]): Description of the feature.
        db (AsyncSession): Database session.

    Returns:
        Feature: The newly created feature object.

    Raises:
        HTTPException: If there is a database error during creation.
    """
    feat = Feature(feature_name=name, description=description)
    db.add(feat)

    try:
        await db.commit()
        await db.refresh(feat)
        return feat
    except Exception as e:
        await db.rollback()
        raise HTTPException(500, f'Database error: {e}')


async def update_feature(
    feat: Feature,
    new_name: str | None,
    new_description: str | None,
    db: AsyncSession,
) -> None:
    """Update an existing feature's details.

    Args:
        feat (Feature): Feature object to update.
        new_name (Optional[str]): New name for the feature.
        new_description (Optional[str]): New description for the feature.
        db (AsyncSession): Database session.

    Raises:
        HTTPException: If no fields provided for update,
            or if database error occurs.
    """
    if new_name:
        feat.feature_name = new_name.strip()

    if new_description is not None:
        feat.description = new_description.strip()

    if not (new_name or new_description is not None):
        raise HTTPException(400, 'Nothing to update')

    try:
        await db.commit()
    except Exception as e:
        await db.rollback()
        raise HTTPException(500, f'Database error: {e}')


async def delete_feature(feat: Feature, db: AsyncSession) -> None:
    """Delete a specified feature.

    Args:
        feat (Feature): Feature object to delete.
        db (AsyncSession): Database session.

    Raises:
        HTTPException: If there is a database error during deletion.
    """
    await db.delete(feat)

    try:
        await db.commit()
    except Exception as e:
        await db.rollback()
        raise HTTPException(500, f'Database error: {e}')


async def update_group_features(
    group: Group,
    feature_ids: list[int],
    db: AsyncSession,
) -> None:
    """Update the features associated with a group.

    This method first clears existing feature associations
    and then adds the new associations.

    Args:
        group (Group): The group object to update.
        feature_ids (List[int]): List of feature IDs to
            associate with the group.
        db (AsyncSession): Database session.

    Raises:
        HTTPException: If a database error occurs during update.
    """
    # Clear existing feature associations
    await db.execute(
        group_features_table.delete().where(
            group_features_table.c.group_id == group.id,
        ),
    )
    await db.commit()

    # Insert new feature associations
    await db.execute(
        group_features_table.insert(),
        [{'group_id': group.id, 'feature_id': fid} for fid in feature_ids],
    )
    await db.commit()


async def list_group_features(
    db: AsyncSession,
) -> list[tuple[Group, list[int]]]:
    """List groups with their associated feature IDs.

    Args:
        db (AsyncSession): Database session.

    Returns:
        List[Tuple[Group, List[int]]]: A list of tuples,
            each containing a group
            and a list of its associated feature IDs.
    """
    groups = (
        await db.execute(
            select(Group).options(selectinload(Group.features)),
        )
    ).unique().scalars().all()

    return [
        (group, [feature.id for feature in group.features])
        for group in groups
    ]
