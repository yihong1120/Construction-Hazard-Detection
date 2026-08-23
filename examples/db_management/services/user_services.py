from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from examples.auth.models import User
from examples.auth.models import USER_STATUS_ACTIVE
from examples.auth.models import USER_STATUS_VALUES
from examples.auth.models import UserProfile
from examples.db_management.services.password_policy import (
    validate_password_minimum,
)


async def create_user(
    username: str,
    password: str,
    role: str,
    group_id: int | None,
    db: AsyncSession,
    tenant_id: UUID | None = None,
    profile: dict[str, Any] | None = None,
    status: str = USER_STATUS_ACTIVE,
) -> User:
    """Create a new user and optionally its profile.

    Args:
        username: Username of the new user.
        password: Plain-text password to be hashed and stored securely.
        role: Role of the user (for example, ``"admin"`` or ``"user"``).
        group_id: Group identifier the user belongs to, if any.
        db: Async SQLAlchemy session.
        profile: Optional dictionary of profile fields used to create a
            ``UserProfile`` (for example, ``display_name``, ``email``).
        status: Account status to assign when creating the user.

    Returns:
        The newly created ``User`` instance, refreshed to include relationships
        ``profile`` and ``group``.

    Raises:
        HTTPException: If the username/email already exists (400) or a generic
            database error occurs (500).
    """
    validate_password_minimum(password)

    try:
        new_user = User(
            username=username,
            role=role,
            group_id=group_id,
            status=status,
            **({'tenant_id': tenant_id} if tenant_id is not None else {}),
        )
        new_user.set_password(password)
        db.add(new_user)

        # Important ①: flush to obtain ``new_user.id`` (not yet committed).
        await db.flush()

        # Important ②: if a profile payload exists, create it with ``user_id``.
        if profile:
            prof = UserProfile(user_id=new_user.id, **profile)
            db.add(prof)

        # Single commit for both user and profile operations.
        await db.commit()

        # Refresh to include the latest state, including profile and group.
        await db.refresh(new_user, attribute_names=['profile', 'group'])
        return new_user

    except IntegrityError as e:
        await db.rollback()
        # Duplicate username/email likely triggers an integrity error.
        raise HTTPException(400, 'Username or e-mail already exists.') from e
    except Exception as e:
        await db.rollback()
        raise HTTPException(500, f"Database error: {e}") from e


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


async def update_password(
    user: User,
    new_password: str,
    db: AsyncSession,
) -> None:
    """Update the password of an existing user.

    Args:
        user: ``User`` instance to update.
        new_password: The new password in plain text.
        db: Async SQLAlchemy session.

    Raises:
        HTTPException: If the password is too short (400) or a database error
            occurs during password update (500).
    """
    validate_password_minimum(new_password)

    # Securely hash and set the new password.
    user.set_password(new_password)

    try:
        # Save changes to the database.
        await db.commit()
    except Exception as e:
        # Roll back on error.
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
