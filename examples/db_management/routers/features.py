from __future__ import annotations

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from examples.auth.database import get_db
from examples.auth.models import Feature
from examples.auth.models import Group
from examples.auth.models import User
from examples.db_management.deps import get_current_user
from examples.db_management.deps import require_super_admin
from examples.db_management.schemas.feature import FeatureCreate
from examples.db_management.schemas.feature import FeatureDelete
from examples.db_management.schemas.feature import FeatureRead
from examples.db_management.schemas.feature import FeatureUpdate
from examples.db_management.schemas.feature import GroupFeatureUpdate
from examples.db_management.schemas.group import GroupFeatureRead
from examples.db_management.services.feature_services import create_feature
from examples.db_management.services.feature_services import delete_feature
from examples.db_management.services.feature_services import list_features
from examples.db_management.services.feature_services import (
    list_group_features,
)
from examples.db_management.services.feature_services import update_feature
from examples.db_management.services.feature_services import (
    update_group_features,
)

router = APIRouter(tags=['feature-mgmt'])


@router.get('/list_features', response_model=list[FeatureRead])
async def endpoint_list_features(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[FeatureRead]:
    """Retrieve a list of all available features.

    Args:
        db (AsyncSession): Database session dependency.
        current_user (User): Current logged-in user, must be super admin.

    Returns:
        List[FeatureRead]: A list of features.

    Raises:
        HTTPException: If user is not a super admin.
    """
    require_super_admin(current_user)
    features = await list_features(db)
    return [
        FeatureRead(
            id=f.id,
            feature_name=f.feature_name,
            description=f.description,
        )
        for f in features
    ]


@router.post(
    '/create_feature',
    response_model=FeatureRead,
    dependencies=[Depends(require_super_admin)],
)
async def endpoint_create_feature(
    payload: FeatureCreate,
    db: AsyncSession = Depends(get_db),
) -> FeatureRead:
    """Create a new feature.

    Args:
        payload (FeatureCreate): Data required to create a feature.
        db (AsyncSession): Database session dependency.

    Returns:
        FeatureRead: The created feature.
    """
    feature = await create_feature(
        payload.feature_name,
        payload.description,
        db,
    )
    return FeatureRead.model_validate(feature)


@router.put(
    '/update_feature',
    dependencies=[Depends(require_super_admin)],
)
async def endpoint_update_feature(
    payload: FeatureUpdate,
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    """Update an existing feature's details.

    Args:
        payload (FeatureUpdate): Data for updating the feature.
        db (AsyncSession): Database session dependency.

    Returns:
        dict[str, str]: Confirmation message.

    Raises:
        HTTPException: If the feature does not exist.
    """
    feature = (
        await db.execute(
            select(Feature).where(Feature.id == payload.feature_id),
        )
    ).unique().scalar_one_or_none()

    if not feature:
        raise HTTPException(404, 'Feature not found')

    await update_feature(
        feature,
        payload.new_name,
        payload.new_description,
        db,
    )
    return {'message': 'Feature updated successfully.'}


@router.delete(
    '/delete_feature',
    dependencies=[Depends(require_super_admin)],
)
async def endpoint_delete_feature(
    payload: FeatureDelete,
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    """Delete an existing feature.

    Args:
        payload (FeatureDelete): Contains the ID of the feature to delete.
        db (AsyncSession): Database session dependency.

    Returns:
        dict[str, str]: Confirmation message.

    Raises:
        HTTPException: If the feature does not exist.
    """
    feature = (
        await db.execute(
            select(Feature).where(Feature.id == payload.feature_id),
        )
    ).unique().scalar_one_or_none()

    if not feature:
        raise HTTPException(404, 'Feature not found')

    await delete_feature(feature, db)
    return {'message': 'Feature deleted successfully.'}


@router.post(
    '/update_group_feature',
    dependencies=[Depends(require_super_admin)],
)
async def endpoint_update_group_feature(
    payload: GroupFeatureUpdate,
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    """Update the features associated with a specific group.

    Args:
        payload (GroupFeatureUpdate): Data specifying group and new features.
        db (AsyncSession): Database session dependency.

    Returns:
        dict[str, str]: Confirmation message.

    Raises:
        HTTPException: If the group does not exist.
    """
    group = (
        await db.execute(
            select(Group).where(Group.id == payload.group_id),
        )
    ).unique().scalar_one_or_none()

    if not group:
        raise HTTPException(404, 'Group not found')

    await update_group_features(group, payload.feature_ids, db)
    return {'message': 'Group features updated successfully.'}


@router.get(
    '/list_group_features',
    response_model=list[GroupFeatureRead],
    dependencies=[Depends(require_super_admin)],
)
async def endpoint_list_group_features(
    db: AsyncSession = Depends(get_db),
) -> list[GroupFeatureRead]:
    """Retrieve a list of groups along with their associated feature IDs.

    Args:
        db (AsyncSession): Database session dependency.

    Returns:
        List[GroupFeatureRead]: List containing groups and their features.
    """
    data = await list_group_features(db)
    return [
        GroupFeatureRead(
            group_id=group.id,
            group_name=group.name,
            feature_ids=feature_ids,
        )
        for group, feature_ids in data
    ]
