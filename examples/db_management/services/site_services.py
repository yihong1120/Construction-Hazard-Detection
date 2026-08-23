from __future__ import annotations

from typing import Final

from fastapi import HTTPException
from sqlalchemy import column
from sqlalchemy import delete
from sqlalchemy import Integer
from sqlalchemy import literal
from sqlalchemy import values
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from examples.auth.models import Site
from examples.auth.models import site_groups_table
from examples.auth.models import SiteNotificationPreference
from examples.auth.models import User
from examples.auth.models import user_sites_table
from examples.db_management.deps import SUPER_ADMIN_NAME
from examples.db_management.schemas.site import SiteRead
from examples.db_management.services.site_media_cleanup import (
    enqueue_site_media_cleanup_for_site,
)

_bulk_insert_chunk_size: Final[int] = 250


def _chunks(values_: list[int], size: int) -> list[list[int]]:
    """Split identifiers into bounded SQL VALUES inputs."""
    return [
        values_[start:start + size]
        for start in range(0, len(values_), size)
    ]


def site_to_read(
    site: Site,
    visible_group_id: int | None = None,
) -> SiteRead:
    """Serialise a site within the requesting administrator's group scope.

    Args:
        site: Site and relationships to serialise.
        visible_group_id: Group visible to a scoped administrator, if any.

    Returns:
        Validated site response with only visible group information.
    """
    groups = list(site.groups)
    users = list(site.users)
    if visible_group_id is not None:
        groups = [group for group in groups if group.id == visible_group_id]
        users = [
            user for user in users
            if user.group_id == visible_group_id
        ]
    return SiteRead(
        id=site.id,
        name=site.name,
        group_ids=[group.id for group in groups],
        group_names=[group.name for group in groups],
        user_ids=[user.id for user in users],
    )


async def _list_user_ids_for_groups(
    group_ids: list[int],
    db: AsyncSession,
) -> list[int]:
    """Load all user identifiers for the provided groups.

    Args:
        group_ids: Group identifiers whose members are requested.
        db: Database session used to query users.

    Returns:
        User identifiers belonging to any supplied group.
    """
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
    """Load all site identifiers linked to a group.

    Args:
        group_id: Identifier of the group.
        db: Database session used to query site membership.

    Returns:
        Site identifiers accessible to the group.
    """
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
    """Insert default-enabled notification preferences in bulk.

    Args:
        user_ids: Users that should receive default preferences.
        site_ids: Sites for which preferences should be created.
        db: Database session used for bulk insertion.
    """
    if not user_ids or not site_ids:
        return

    # Let PostgreSQL construct the Cartesian product.  Materialising it in
    # Python used O(users * sites) application memory before any SQL ran.
    unique_user_ids = list(dict.fromkeys(user_ids))
    unique_site_ids = list(dict.fromkeys(site_ids))
    for user_chunk in _chunks(unique_user_ids, _bulk_insert_chunk_size):
        user_values = values(
            column('user_id', Integer),
            name='notification_preference_user_ids',
        ).data([(user_id,) for user_id in user_chunk])
        for site_chunk in _chunks(
            unique_site_ids,
            _bulk_insert_chunk_size,
        ):
            site_values = values(
                column('site_id', Integer),
                name='notification_preference_site_ids',
            ).data([(site_id,) for site_id in site_chunk])
            rows = select(
                user_values.c.user_id,
                site_values.c.site_id,
                literal(True),
            )
            await db.execute(
                pg_insert(SiteNotificationPreference)
                .from_select(['user_id', 'site_id', 'is_enabled'], rows)
                .on_conflict_do_nothing(
                    index_elements=['user_id', 'site_id'],
                ),
            )


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
            pg_insert(site_groups_table)
            .values([
                {'site_id': site.id, 'group_id': group_id}
                for group_id in group_ids
            ])
            .on_conflict_do_nothing(
                index_elements=[
                    site_groups_table.c.site_id,
                    site_groups_table.c.group_id,
                ],
            ),
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
                pg_insert(user_sites_table)
                .values(user_id=super_admin.id, site_id=site.id)
                .on_conflict_do_nothing(
                    index_elements=[
                        user_sites_table.c.user_id,
                        user_sites_table.c.site_id,
                    ],
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
    Delete an existing site and queue its evidence files for post-commit cleanup.

    Args:
        site (Site): The Site object to delete.
        db (AsyncSession): The asynchronous database session.

    Raises:
        HTTPException: If a database error occurs during deletion.
    """
    # Queue image deletion inside the same database transaction.  Physical
    # removal only runs after that transaction commits, so a failed site
    # deletion cannot orphan a surviving violation record from its image.
    await enqueue_site_media_cleanup_for_site(site.name, db)

    # PostgreSQL foreign keys cascade stream, violation, group, and access
    # rows.  Deleting only the site eliminates two redundant DELETE round trips.
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
        pg_insert(user_sites_table)
        .values(user_id=user_id, site_id=site_id)
        .on_conflict_do_nothing(
            index_elements=[
                user_sites_table.c.user_id,
                user_sites_table.c.site_id,
            ],
        ),
    )
    # Pre-seed a default enabled notification preference
    await db.execute(
        pg_insert(SiteNotificationPreference)
        .values(user_id=user_id, site_id=site_id, is_enabled=True)
        .on_conflict_do_nothing(index_elements=['user_id', 'site_id']),
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
        pg_insert(site_groups_table)
        .values(site_id=site_id, group_id=group_id)
        .on_conflict_do_nothing(
            index_elements=[
                site_groups_table.c.site_id,
                site_groups_table.c.group_id,
            ],
        ),
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
