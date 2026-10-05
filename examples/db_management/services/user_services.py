from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from examples.auth.config import Settings
from examples.auth.models import User
from examples.auth.models import USER_STATUS_ACTIVE
from examples.auth.models import USER_STATUS_VALUES
from examples.auth.models import UserIdentity
from examples.auth.models import UserProfile

settings = Settings()


async def create_oidc_managed_user(
    *,
    username: str,
    role: str,
    group_id: int | None,
    tenant_id: UUID,
    keycloak_subject: str,
    profile: dict[str, Any],
    db: AsyncSession,
) -> User:
    """Persist one Keycloak-managed identity and Visionnaire permissions.

    The caller must provision Keycloak first and compensates by deleting that
    identity when this transaction fails.  No clear-text or reusable local
    password exists for these accounts.
    """
    try:
        new_user = User(
            username=username,
            role=role,
            group_id=group_id,
            tenant_id=tenant_id,
            status=USER_STATUS_ACTIVE,
        )
        db.add(new_user)
        await db.flush()
        db.add(UserProfile(user_id=new_user.id, **profile))
        db.add(
            UserIdentity(
                user_id=new_user.id,
                provider=settings.oidc_identity_provider,
                provider_user_id=keycloak_subject,
                email=profile['email'],
                email_verified=True,
                display_name=(
                    f"{profile['family_name']} {profile['given_name']}"
                ).strip(),
            ),
        )
        await db.commit()
        await db.refresh(new_user, attribute_names=['profile', 'group'])
        return new_user
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=409,
            detail='username_or_email_already_exists',
        ) from exc
    except Exception as exc:
        await db.rollback()
        raise HTTPException(
            status_code=500,
            detail='user_identity_persistence_failed',
        ) from exc


async def keycloak_subject_for_user(
    user: User,
    db: AsyncSession,
) -> str | None:
    """Return the immutable Keycloak subject linked to a local user."""
    return await db.scalar(
        select(UserIdentity.provider_user_id).where(
            UserIdentity.user_id == user.id,
            UserIdentity.provider == settings.oidc_identity_provider,
        ),
    )


async def ensure_keycloak_subject_for_user(
    user: User, db: AsyncSession,
) -> str:
    """Require an immutable, already-linked identity-provider subject."""
    subject = await keycloak_subject_for_user(user, db)
    if not subject:
        raise HTTPException(409, detail='keycloak_identity_not_linked')
    return subject


async def list_users(
    db: AsyncSession,
    group_id: int | None = None,
    tenant_id: UUID | None = None,
    *,
    after_id: int | None = None,
    page_size: int = 50,
) -> tuple[list[User], int | None]:
    """Retrieve users, optionally scoped to a group.

    Args:
        db: Async SQLAlchemy session.
        group_id: Optional group identifier used to scope admin results.

    Returns:
        A page of ``User`` instances and the next keyset cursor, if any.
    """
    query = select(User).options(
        selectinload(User.group),
        selectinload(User.profile),
        selectinload(User.sites),
    )
    if group_id is not None:
        query = query.where(User.group_id == group_id)
    if tenant_id is not None:
        query = query.where(User.tenant_id == tenant_id)
    if after_id is not None:
        query = query.where(User.id > after_id)

    result = await db.execute(
        query.order_by(User.id).limit(page_size + 1),
    )
    users = list(result.unique().scalars().all())
    has_more = len(users) > page_size
    page = users[:page_size]
    return page, page[-1].id if has_more and page else None


async def get_user_by_id(user_id: int, db: AsyncSession) -> User:
    """Retrieve a user by its unique identifier.

    Args:
        user_id: Numeric identifier of the user.
        db: Async SQLAlchemy session.

    Returns:
        The matching ``User`` instance.

    Raises:
        HTTPException: If no user is found (404).
    """
    user = (
        (
            await db.execute(
                select(User)
                .options(
                    selectinload(User.group),
                    selectinload(User.profile),
                    selectinload(User.sites),
                )
                .where(User.id == user_id),
            )
        )
        .unique()
        .scalar_one_or_none()
    )

    if not user:
        raise HTTPException(status_code=404, detail='User not found.')

    return user


async def delete_user(user: User, db: AsyncSession) -> None:
    """Delete a user.

    Args:
        user: ``User`` instance to delete.
        db: Async SQLAlchemy session.

    Raises:
        HTTPException: If a database error occurs during deletion (500).
    """
    # Mark the user instance for deletion.
    await db.delete(user)

    try:
        # Commit the deletion transaction.
        await db.commit()
    except Exception as e:
        # Roll back on failure.
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {e}")


async def update_username(
    user: User,
    new_username: str,
    db: AsyncSession,
) -> None:
    """Update the username of an existing user.

    Args:
        user: ``User`` instance to update.
        new_username: The new username.
        db: Async SQLAlchemy session.

    Raises:
        HTTPException: If the new username already exists (400) or a generic
            database error occurs (500).
    """
    # Set the new username.
    user.username = new_username

    try:
        # Commit changes to the database.
        await db.commit()
    except IntegrityError:
        # Roll back if a username conflict occurs.
        await db.rollback()
        raise HTTPException(status_code=400, detail='Username already exists.')
    except Exception as e:
        # Roll back on unexpected errors.
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {e}")


async def set_user_status(
    user: User,
    status: str,
    db: AsyncSession,
) -> None:
    """Update a user's status.

    Args:
        user: ``User`` instance to update.
        status: New account status.
        db: Async SQLAlchemy session.

    Raises:
        HTTPException: If a database error occurs during status update (500).
    """
    if status not in USER_STATUS_VALUES:
        raise HTTPException(status_code=400, detail='Invalid user status.')

    user.status = status

    try:
        # Persist status change to the database.
        await db.commit()
    except Exception as e:
        # Roll back if an error occurs.
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {e}")


async def create_or_update_profile(
    user: User,
    data: dict[str, Any],
    db: AsyncSession,
    create_if_missing: bool = False,
) -> None:
    """Create a new profile if absent, or update allowed fields.

    Args:
        user: ``User`` whose profile is to be created or updated.
        data: Mapping of fields to update; keys outside the allowed set are
            ignored. ``None`` values are ignored as well.
        db: Async SQLAlchemy session.
        create_if_missing: Whether to create a profile if none exists.

    Raises:
        HTTPException: If the profile is missing (404) and not allowed to be
            created, a duplicate constraint is violated (400), or a generic
            database error occurs (500).
    """
    profile = user.profile
    if not profile:
        if not create_if_missing:
            raise HTTPException(404, 'Profile not found.')
        profile = UserProfile(user_id=user.id)
        db.add(profile)

    # Allow only known profile fields to be updated (safer than ``hasattr``).
    allowed_fields = {
        'family_name',
        'middle_name',
        'given_name',
        'email',
        'mobile_number',
    }
    for key, val in data.items():
        if val is not None and key in allowed_fields:
            setattr(profile, key, val)

    try:
        await db.commit()
        await db.refresh(user, attribute_names=['profile'])
    except IntegrityError:
        await db.rollback()
        # ``email``/``mobile`` are UNIQUE → catch duplicates.
        raise HTTPException(400, 'Duplicate email or mobile number.')
    except Exception as e:
        await db.rollback()
        raise HTTPException(500, f"Database error: {e}")
