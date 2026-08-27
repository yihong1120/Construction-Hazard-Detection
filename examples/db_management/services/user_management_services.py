from __future__ import annotations

from uuid import UUID

from fastapi import HTTPException
from fastapi import Request
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from examples.auth.deployment_context import DeploymentBinding
from examples.auth.deployment_context import resolve_request_deployment
from examples.auth.models import Group
from examples.auth.models import User
from examples.auth.models import USER_STATUS_ACTIVE
from examples.auth.models import USER_STATUS_EMAIL_UNVERIFIED
from examples.auth.user_service import invalidate_effective_site_cache
from examples.db_management.deps import ensure_admin_with_group
from examples.db_management.deps import is_super_admin
from examples.db_management.schemas.user import PendingUserReviewRead
from examples.db_management.schemas.user import UserPage
from examples.db_management.schemas.user import UserRead
from examples.db_management.schemas.user import UserSignup
from examples.db_management.services.email_verification_services import (
    send_signup_verification_email,
)
from examples.db_management.services.legal_services import record_user_consent
from examples.db_management.services.legal_services import (
    validate_signup_consents,
)
from examples.db_management.services.site_services import (
    list_site_ids_for_group,
)
from examples.db_management.services.site_services import (
    seed_site_notification_preferences,
)
from examples.db_management.services.user_services import create_user
from examples.db_management.services.user_services import list_users


async def list_users_for_operator(
    operator: User,
    db: AsyncSession,
    *,
    cursor: int | None,
    page_size: int,
) -> UserPage:
    """Return one keyset page within an administrator's permitted scope."""
    group_id: int | None = None
    tenant_id: UUID | None = None
    if not is_super_admin(operator):
        ensure_admin_with_group(operator)
        group_id = operator.group_id
        if isinstance(operator.tenant_id, UUID):
            tenant_id = operator.tenant_id

    users, next_cursor = await list_users(
        db,
        group_id=group_id,
        tenant_id=tenant_id,
        after_id=cursor,
        page_size=page_size,
    )
    return UserPage(
        items=[UserRead.model_validate(user) for user in users],
        next_cursor=next_cursor,
    )


async def list_all_users_for_operator(
    operator: User,
    db: AsyncSession,
) -> list[UserRead]:
    """Return every user visible to an operator using bounded keyset reads.

    The long-standing ``/list_users`` contract is an array.  Keep that
    contract for deployed clients while the explicit administrator endpoint
    exposes the cursor-based API for screens that need pagination.

    Args:
        operator: Authenticated administrator whose scope is enforced.
        db: Database session used to retrieve the user pages.

    Returns:
        All user records visible to the administrator, ordered by identifier.
    """
    cursor: int | None = None
    users: list[UserRead] = []

    while True:
        page = await list_users_for_operator(
            operator,
            db,
            cursor=cursor,
            page_size=100,
        )
        users.extend(page.items)
        if page.next_cursor is None:
            return users
        cursor = page.next_cursor


async def load_user_read(user_id: int, db: AsyncSession) -> UserRead:
    """Load a user and relations required by the public response schema.

    Args:
        user_id: Identifier of the user to load.
        db: Database session used to load the user graph.

    Returns:
        Validated user response including group and profile details.

    Raises:
        NoResultFound: If no user exists with the supplied identifier.
    """
    result = await db.execute(
        select(User)
        .options(
            selectinload(User.group),
            selectinload(User.profile),
        )
        .where(User.id == user_id),
    )
    return UserRead.model_validate(result.scalar_one())


def pending_user_review_read(user: User) -> PendingUserReviewRead:
    """Build an administrator-review row from a fully loaded user graph.

    Args:
        user: Pending user with consent, identity, group, and profile
            relations.

    Returns:
        Read model containing the latest consent and linked providers.
    """
    # Consent records are immutable; the most recent event reflects the
    # versions and choices currently shown in the review queue.
    latest_consent = max(
        user.consents,
        key=lambda consent: (consent.accepted_at, consent.id),
        default=None,
    )
    providers = sorted(
        {str(identity.provider) for identity in user.identities},
    )
    return PendingUserReviewRead(
        **UserRead.model_validate(user).model_dump(),
        email=user.profile.email,
        terms_version=(
            latest_consent.terms_version if latest_consent else None
        ),
        privacy_version=(
            latest_consent.privacy_version if latest_consent else None
        ),
        ai_terms_version=(
            latest_consent.ai_terms_version if latest_consent else None
        ),
        notification_consent=(
            latest_consent.notification_consent if latest_consent else None
        ),
        provider=','.join(providers) if providers else 'password',
    )


async def get_group_or_404(group_id: int, db: AsyncSession) -> Group:
    """Load a group or raise a not-found response.

    Args:
        group_id: Identifier of the group to load.
        db: Database session used to query the group.

    Returns:
        Loaded group.

    Raises:
        HTTPException: If no group has the supplied identifier.
    """
    group = (
        (await db.execute(select(Group).where(Group.id == group_id)))
        .unique()
        .scalar_one_or_none()
    )
    if group is None:
        raise HTTPException(404, 'Group not found.')
    return group


async def register_signup_user(
    payload: UserSignup,
    request: Request,
    db: AsyncSession,
    redis_pool: Redis,
    deployment: DeploymentBinding | None = None,
) -> UserRead:
    """Create an email-unverified account and start verification.

    Args:
        payload: Registration details, profile, and legal consents.
        request: HTTP request used to record consent audit metadata.
        db: Database session used to create the account.
        redis_pool: Redis connection used to store verification-token state.

    Returns:
        Newly created account in the email-unverified state.

    Raises:
        HTTPException: If consent validation, account creation, or verification
            delivery fails.
    """
    if deployment is None and isinstance(request, Request):
        deployment = await resolve_request_deployment(request, db)
    if deployment is None and isinstance(request, Request):
        raise HTTPException(
            status_code=409,
            detail='deployment_required',
        )
    await validate_signup_consents(payload, db)
    create_user_kwargs = {
        'username': payload.username,
        'password': payload.password,
        'role': 'user',
        'group_id': None,
        'db': db,
        'profile': payload.profile.model_dump(),
        'status': USER_STATUS_EMAIL_UNVERIFIED,
    }
    if deployment is not None:
        create_user_kwargs['tenant_id'] = deployment.tenant_id
    new_user = await create_user(
        **create_user_kwargs,
    )
    await record_user_consent(new_user.id, payload, db, request)
    await send_signup_verification_email(new_user, redis_pool)
    return await load_user_read(new_user.id, db)


def resolve_target_group_id(
    requested_group_id: int | None,
    operator: User,
    default_to_operator_group: bool = False,
) -> int | None:
    """Resolve the group an operator is permitted to manage.

    Args:
        requested_group_id: Group requested by the caller, if any.
        operator: Authenticated administrator performing the operation.
        default_to_operator_group: Whether missing group input uses the
            operator's group.

    Returns:
        Authorised group identifier, or ``None`` for a super administrator
        without a requested group.

    Raises:
        HTTPException: If a non-super administrator lacks a group or requests a
            different group.
    """
    if is_super_admin(operator):
        return requested_group_id

    ensure_admin_with_group(operator)
    if requested_group_id is None:
        if default_to_operator_group:
            return operator.group_id
        raise HTTPException(400, 'group_id is required.')
    if requested_group_id != operator.group_id:
        raise HTTPException(403, 'Cannot operate on other group')
    return requested_group_id


def ensure_user_management_scope(
    target: User,
    operator: User,
) -> None:
    """Ensure a group administrator manages only users in their own group.

    Args:
        target: User targeted by the management operation.
        operator: Authenticated administrator performing the operation.

    Raises:
        HTTPException: If the target is outside the operator's group scope.
    """
    if is_super_admin(operator):
        return
    ensure_admin_with_group(operator)
    # A group administrator manages members, not peer administrators.  Only
    # ChangDar may appoint, demote, suspend, reset, or otherwise change an
    # administrator account.
    if getattr(target, 'role', None) == 'admin':
        raise HTTPException(
            status_code=403,
            detail='Only super admin can manage administrator accounts.',
        )
    target_tenant_id = getattr(target, 'tenant_id', None)
    operator_tenant_id = getattr(operator, 'tenant_id', None)
    if (
        isinstance(target_tenant_id, UUID)
        and isinstance(operator_tenant_id, UUID)
        and target_tenant_id != operator_tenant_id
    ):
        raise HTTPException(
            status_code=403,
            detail='Cannot manage users in another tenant.',
        )
    if target.group_id != operator.group_id:
        raise HTTPException(
            status_code=403,
            detail='Cannot manage users outside your group.',
        )


async def approve_signup_user(
    user: User,
    group_id: int | None,
    db: AsyncSession,
    operator: User,
) -> UserRead:
    """Assign an approved signup to a group and activate it.

    Args:
        user: Pending user approved by an administrator.
        group_id: Requested group assignment, if explicitly supplied.
        db: Database session used to update user and notification data.
        operator: Authenticated administrator performing the approval.

    Returns:
        Activated user response including its assigned group.

    Raises:
        HTTPException: If the target group is missing or outside the operator's
            scope.
    """
    target_group_id = resolve_target_group_id(
        group_id,
        operator,
        default_to_operator_group=True,
    )
    if target_group_id is None:
        raise HTTPException(400, 'group_id is required.')
    await get_group_or_404(target_group_id, db)

    user.group_id = target_group_id
    user.status = USER_STATUS_ACTIVE
    # New group members receive preferences for the group's existing sites.
    site_ids = await list_site_ids_for_group(target_group_id, db)
    await seed_site_notification_preferences(
        user_ids=[user.id],
        site_ids=site_ids,
        db=db,
    )
    await db.commit()
    invalidate_effective_site_cache()
    return await load_user_read(user.id, db)
