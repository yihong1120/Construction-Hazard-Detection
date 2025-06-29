from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from examples.auth.models import Group


async def list_groups(db: AsyncSession) -> list[Group]:
    """Retrieve a list of all groups from the database.

    Args:
        db (AsyncSession): The asynchronous database session.

    Returns:
        List[Group]: A list containing all Group objects retrieved.
    """
    # Execute query to fetch all groups
    result = await db.execute(select(Group))
    # Return a unique set of groups
    return result.unique().scalars().all()


async def create_group(
    name: str,
    uniform_number: str,
    db: AsyncSession,
) -> Group:
    """Create a new group with the specified name and uniform number.

    Args:
        name (str): Name of the new group.
        uniform_number (str): The group's unique uniform number
            (must be exactly 8 digits).
        db (AsyncSession): The asynchronous database session.

    Returns:
        Group: The newly created Group object.

    Raises:
        HTTPException: If the uniform number is not unique or invalid,
            or if a database error occurs.
    """
    # Instantiate a new Group object with provided details
    grp = Group(name=name, uniform_number=uniform_number)
    db.add(grp)

    try:
        # Commit changes to the database
        await db.commit()
        # Refresh the object to get updated details
        await db.refresh(grp)
        return grp
    except IntegrityError:
        # Roll back changes if integrity constraints fail
        await db.rollback()
        raise HTTPException(
            status_code=400,
            detail='Uniform number must be unique and exactly 8 digits.',
        )
    except Exception as e:
        # General exception handling, rollback changes
        await db.rollback()
        raise HTTPException(status_code=500, detail=f'Database error: {e}')


async def update_group(
    grp: Group,
    new_name: str | None,
    new_uniform_number: str | None,
    db: AsyncSession,
) -> None:
    """
    Update details of an existing group,
    including its name and/or uniform number.

    Args:
        grp (Group): The existing Group object to be updated.
        new_name (Optional[str]): New name for the group, if provided.
        new_uniform_number (Optional[str]): New uniform number,
            must be exactly 8 digits, if provided.
        db (AsyncSession): The asynchronous database session.

    Raises:
        HTTPException: If neither field is provided for update,
            or if the uniform number is invalid,
            or if a database error occurs.
    """
    # Update the group name if a new name is provided
    if new_name:
        grp.name = new_name.strip()

    # Update the uniform number after validating its format
    if new_uniform_number:
        if not new_uniform_number.isdigit() or len(new_uniform_number) != 8:
            raise HTTPException(
                status_code=400,
                detail='Uniform number must be exactly 8 digits.',
            )
        grp.uniform_number = new_uniform_number

    # Check if at least one field has been updated
    if not (new_name or new_uniform_number):
        raise HTTPException(status_code=400, detail='Nothing to update.')

    try:
        # Commit changes to the database
        await db.commit()
    except Exception as e:
        # Handle general exceptions, rolling back any changes
        await db.rollback()
        raise HTTPException(status_code=500, detail=f'Database error: {e}')


async def delete_group(grp: Group, db: AsyncSession) -> None:
    """Delete an existing group from the database.

    Args:
        grp (Group): The Group object to delete.
        db (AsyncSession): The asynchronous database session.

    Raises:
        HTTPException: If a database error occurs during deletion.
    """
    # Mark the group for deletion
    await db.delete(grp)

    try:
        # Commit the deletion to the database
        await db.commit()
    except Exception as e:
        # Roll back if an error occurs during the deletion process
        await db.rollback()
        raise HTTPException(status_code=500, detail=f'Database error: {e}')
