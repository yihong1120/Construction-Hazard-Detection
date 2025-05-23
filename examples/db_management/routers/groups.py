from __future__ import annotations

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from examples.auth.database import get_db
from examples.auth.models import Group
from examples.db_management.deps import require_super_admin
from examples.db_management.schemas.group import GroupCreate
from examples.db_management.schemas.group import GroupDelete
from examples.db_management.schemas.group import GroupRead
from examples.db_management.schemas.group import GroupUpdate
from examples.db_management.services.group_service import create_group
from examples.db_management.services.group_service import delete_group
from examples.db_management.services.group_service import list_groups
from examples.db_management.services.group_service import update_group

router = APIRouter(tags=['group-mgmt'])


@router.get(
    '/list_groups',
    response_model=list[GroupRead],
    dependencies=[Depends(require_super_admin)],
)
async def endpoint_list_groups(
    db: AsyncSession = Depends(get_db),
) -> list[GroupRead]:
    """Retrieve a list of all groups.

    Args:
        db (AsyncSession): Database session dependency.

    Returns:
        List[GroupRead]: A list containing the details of each group.
    """
    # Fetch all groups from the database
    groups = await list_groups(db)

    # Format and return the group details as a list of GroupRead
    return [
        GroupRead(
            id=g.id,
            name=g.name,
            uniform_number=g.uniform_number,
        )
        for g in groups
    ]


@router.post(
    '/create_group',
    response_model=GroupRead,
    dependencies=[Depends(require_super_admin)],
)
async def endpoint_create_group(
    payload: GroupCreate,
    db: AsyncSession = Depends(get_db),
) -> GroupRead:
    """Create a new group.

    Args:
        payload (GroupCreate): Data containing the group's name
            and uniform number.
        db (AsyncSession): Database session dependency.

    Returns:
        GroupRead: Details of the newly created group.
    """
    # Create a new group entry in the database
    group = await create_group(
        name=payload.name,
        uniform_number=payload.uniform_number,
        db=db,
    )

    # Return the newly created group's details
    return GroupRead.from_orm(group)


@router.put(
    '/update_group',
    dependencies=[Depends(require_super_admin)],
)
async def endpoint_update_group(
    payload: GroupUpdate,
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    """Update details of an existing group.

    Args:
        payload (GroupUpdate): Contains group ID and new details to update.
        db (AsyncSession): Database session dependency.

    Returns:
        Dict[str, str]: Confirmation message of successful update.

    Raises:
        HTTPException: Raised if the group is not found
            or if no update data is provided.
    """
    # Ensure that at least one field is provided for updating
    if payload.new_name is None and payload.new_uniform_number is None:
        raise HTTPException(status_code=400, detail='Nothing to update.')

    # Retrieve the group by ID from the database
    group = (
        await db.execute(
            select(Group).where(Group.id == payload.group_id),
        )
    ).unique().scalar_one_or_none()

    # Check if the group exists
    if not group:
        raise HTTPException(status_code=404, detail='Group not found.')

    # Update the group's details in the database
    await update_group(
        grp=group,
        new_name=payload.new_name,
        new_uniform_number=payload.new_uniform_number,
        db=db,
    )

    # Return a confirmation message
    return {'message': 'Group updated successfully.'}


@router.delete(
    '/delete_group',
    dependencies=[Depends(require_super_admin)],
)
async def endpoint_delete_group(
    payload: GroupDelete,
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    """Delete an existing group.

    Args:
        payload (GroupDelete): Data containing the ID of the group to delete.
        db (AsyncSession): Database session dependency.

    Returns:
        Dict[str, str]: Confirmation message of successful deletion.

    Raises:
        HTTPException: Raised if the specified group does not exist.
    """
    # Retrieve the group by ID from the database
    group = (
        await db.execute(
            select(Group).where(Group.id == payload.group_id),
        )
    ).unique().scalar_one_or_none()

    # Ensure the group exists before deletion
    if not group:
        raise HTTPException(status_code=404, detail='Group not found.')

    # Delete the specified group from the database
    await delete_group(grp=group, db=db)

    # Return a confirmation message
    return {'message': 'Group deleted successfully.'}
