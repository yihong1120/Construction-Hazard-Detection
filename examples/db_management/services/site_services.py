from __future__ import annotations

from pathlib import Path
from typing import Final

from fastapi import HTTPException
from sqlalchemy import delete
from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from examples.auth.models import Site
from examples.auth.models import site_groups_table
from examples.auth.models import SiteNotificationPreference
from examples.auth.models import StreamConfig
from examples.auth.models import User
from examples.auth.models import user_sites_table
from examples.auth.models import Violation
from examples.db_management.deps import SUPER_ADMIN_NAME

_bulk_insert_chunk_size: Final[int] = 1000


async def _list_user_ids_for_groups(
    group_ids: list[int],
    db: AsyncSession,
) -> list[int]:
    """Load all user IDs for the provided groups."""
    if not group_ids:
        return []

    result = await db.execute(
        select(User.id).where(User.group_id.in_(group_ids)),
    )
    return list(result.scalars().all())


async def list_site_ids_for_group(
    group_id: int,
    db: AsyncSession,
) -> list[int]:
    """Load all site IDs linked to a group."""
    result = await db.execute(
        select(site_groups_table.c.site_id).where(
            site_groups_table.c.group_id == group_id,
        ),
    )
    return list(result.scalars().all())


async def seed_site_notification_preferences(
    user_ids: list[int],
    site_ids: list[int],
    db: AsyncSession,
) -> None:
    """Insert default-enabled notification preferences in bulk."""
    if not user_ids or not site_ids:
        return

    stmt = insert(SiteNotificationPreference).prefix_with('IGNORE')
    rows: list[dict[str, int | bool]] = []
    for user_id in user_ids:
        for site_id in site_ids:
            rows.append({
                'user_id': user_id,
                'site_id': site_id,
                'is_enabled': True,
            })
            if len(rows) >= _bulk_insert_chunk_size:
                await db.execute(stmt, rows)
                rows = []

    if rows:
        await db.execute(stmt, rows)


async def list_sites(
    db: AsyncSession,
    group_id: int | None = None,
) -> list[Site]:
    """Retrieve a list of sites based on the provided group identifier.

    This function fetches all Site objects from the database, optionally
    filtering them by a specific group identifier. If no group identifier
    is provided, all sites are retrieved.

    Args:
        db (AsyncSession): The asynchronous database session.
        group_id (Optional[int]): The identifier of the group to
            filter sites by.

    Returns:
        List[Site]: A list of retrieved Site objects.
    """
    # Construct the query to select all sites and the relationships used by
    # SiteRead serialization.
    query = select(Site).options(
        selectinload(Site.groups),
        selectinload(Site.users),
    )

    # If a group_id is provided, filter the sites via the association table
    if group_id is not None:
        query = query.join(
            site_groups_table,
            Site.id == site_groups_table.c.site_id,
        ).where(site_groups_table.c.group_id == group_id)

    result = await db.execute(query)
    return list(result.unique().scalars().all())


async def create_site(
    name: str,
    group_ids: list[int],
    db: AsyncSession,
) -> Site:
    """Create a new site with the specified name and group associations.

    The function also grants access to the super admin user ('ChangDar')
    for the newly created site.

    Args:
        name (str): The name of the new site.
        group_ids (list[int]): The group identifiers the site belongs to.
        db (AsyncSession): The asynchronous database session.

    Returns:
        Site: The newly created Site object.

    Raises:
        HTTPException: If a database error occurs during creation.
    """
    group_ids = list(dict.fromkeys(group_ids))
    if not group_ids:
        raise HTTPException(400, 'group_id is required for new site')

    site: Site = Site(name=name)
    db.add(site)

    try:
        await db.flush()  # Get site.id without committing

        # Link the site to all specified groups
        await db.execute(
            site_groups_table.insert().prefix_with('IGNORE'),
            [
                {'site_id': site.id, 'group_id': gid}
                for gid in group_ids
            ],
        )

        # Seed notification preferences for all users in the linked groups
        group_user_ids = await _list_user_ids_for_groups(group_ids, db)
        await seed_site_notification_preferences(
            user_ids=group_user_ids,
            site_ids=[site.id],
            db=db,
        )

        # Automatically grant the super admin (ChangDar) access to the new site
        super_admin: User | None = (
            await db.execute(
                select(User).where(User.username == SUPER_ADMIN_NAME),
            )
        ).unique().scalar_one_or_none()
        if super_admin:
            await db.execute(
                user_sites_table.insert().prefix_with('IGNORE').values(
                    user_id=super_admin.id, site_id=site.id,
                ),
            )
            await seed_site_notification_preferences(
                user_ids=[super_admin.id],
                site_ids=[site.id],
                db=db,
            )

        await db.commit()

        # Refresh the site object and load its users
        refreshed_site: Site = (
            await db.execute(
                select(Site)
                .options(
                    selectinload(Site.groups),
                    selectinload(Site.users),
                )
                .where(Site.id == site.id),
            )
        ).unique().scalar_one()

        return refreshed_site

    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f'Database error: {e}')


async def update_site(
    site: Site,
    new_name: str,
    db: AsyncSession,
) -> None:
    """Update the name of an existing site.

    Args:
        site (Site): The Site object to update.
        new_name (str): The new name for the site.
        db (AsyncSession): The asynchronous database session.

    Raises:
        HTTPException: If a database error occurs during the update.
    """
    # Update the site's name
    site.name = new_name

    try:
        await db.commit()
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f'Database error: {e}')


async def delete_site(
    site: Site,
    db: AsyncSession,
) -> None:
    """
    Delete an existing site, along with associated violations,
    stream configurations, and images.

    The deletion process includes:
        - Deleting related violation image files from the file system.
        - Removing related StreamConfig and Violation records
            from the database.
        - Deleting the Site itself.

    Args:
        site (Site): The Site object to delete.
        db (AsyncSession): The asynchronous database session.

    Raises:
        HTTPException: If a database error occurs during deletion.
    """
    # Step 1: Delete violation image files associated with the site
    image_paths = list((
        await db.execute(
            select(Violation.image_path).where(Violation.site == site.name),
        )
    ).scalars().all())

    for path_str in image_paths:
        if path_str:
            image_path: Path = Path(path_str)
            # Remove the file if it exists
            if image_path.is_file():
                image_path.unlink(missing_ok=True)

    # Step 2: Delete related database records
    # (StreamConfig, Violation, and Site itself)
    await db.execute(
        delete(StreamConfig).where(StreamConfig.site_id == site.id),
    )
    await db.execute(delete(Violation).where(Violation.site == site.name))
    await db.delete(site)

    try:
        await db.commit()
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f'Database error: {e}')


async def add_user_to_site(
    user_id: int,
    site_id: int,
    db: AsyncSession,
) -> None:
    """Grant a user access to a specified site.

    Args:
        user_id (int): The identifier of the user to add.
        site_id (int): The identifier of the site to grant access to.
        db (AsyncSession): The asynchronous database session.
    """
    # Insert a new record to grant the user access to the site
    await db.execute(
        user_sites_table.insert()
        .prefix_with('IGNORE')  # Prevent duplicate entries
        .values(user_id=user_id, site_id=site_id),
    )
    # Pre-seed a default enabled notification preference
    await db.execute(
        insert(SiteNotificationPreference)
        .prefix_with('IGNORE')
        .values(user_id=user_id, site_id=site_id, is_enabled=True),
    )
    await db.commit()


async def remove_user_from_site(
    user_id: int,
    site_id: int,
    db: AsyncSession,
) -> None:
    """Revoke a user's access to a specified site.

    Args:
        user_id (int): The identifier of the user to remove.
        site_id (int): The identifier of the site from which to revoke access.
        db (AsyncSession): The asynchronous database session.
    """
    # Delete the record that grants the user access to the site
    await db.execute(
        user_sites_table.delete().where(
            user_sites_table.c.user_id == user_id,
            user_sites_table.c.site_id == site_id,
        ),
    )
    has_group_access = (
        select(1)
        .select_from(User)
        .join(
            site_groups_table,
            User.group_id == site_groups_table.c.group_id,
        )
        .where(
            User.id == user_id,
            site_groups_table.c.site_id == site_id,
        )
    ).exists()
    await db.execute(
        delete(SiteNotificationPreference).where(
            SiteNotificationPreference.user_id == user_id,
            SiteNotificationPreference.site_id == site_id,
            ~has_group_access,
        ),
    )
    await db.commit()


async def add_group_to_site(
    site_id: int,
    group_id: int,
    db: AsyncSession,
) -> None:
    """Associate a group with a specified site.

    Args:
        site_id (int): The identifier of the site.
        group_id (int): The identifier of the group to add.
        db (AsyncSession): The asynchronous database session.
    """
    await db.execute(
        site_groups_table.insert()
        .prefix_with('IGNORE')
        .values(site_id=site_id, group_id=group_id),
    )
    # Seed notification preferences for all users in this group
    user_ids = await _list_user_ids_for_groups([group_id], db)
    await seed_site_notification_preferences(
        user_ids=user_ids,
        site_ids=[site_id],
        db=db,
    )
    await db.commit()


async def remove_group_from_site(
    site_id: int,
    group_id: int,
    db: AsyncSession,
) -> None:
    """Dissociate a group from a specified site.

    Args:
        site_id (int): The identifier of the site.
        group_id (int): The identifier of the group to remove.
        db (AsyncSession): The asynchronous database session.
    """
    await db.execute(
        site_groups_table.delete().where(
            site_groups_table.c.site_id == site_id,
            site_groups_table.c.group_id == group_id,
        ),
    )
    group_user_ids = select(User.id).where(User.group_id == group_id)
    has_direct_access = (
        select(1)
        .select_from(user_sites_table)
        .where(
            user_sites_table.c.site_id == site_id,
            user_sites_table.c.user_id == SiteNotificationPreference.user_id,
        )
    ).exists()
    await db.execute(
        delete(SiteNotificationPreference).where(
            SiteNotificationPreference.site_id == site_id,
            SiteNotificationPreference.user_id.in_(group_user_ids),
            ~has_direct_access,
        ),
    )
    await db.commit()
