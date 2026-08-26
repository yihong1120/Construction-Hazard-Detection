"""Map verified external identities to local Visionnaire authorisation."""
from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from jwt.exceptions import InvalidTokenError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.deployment_context import DeploymentBinding
from examples.auth.models import Feature
from examples.auth.models import group_features_table
from examples.auth.models import User
from examples.auth.models import USER_STATUS_ACTIVE
from examples.auth.models import UserIdentity
from examples.db_management.schemas.auth import AccessTokenSubject
from examples.db_management.schemas.auth import AccessTokenSubjectModel


async def _feature_names(
    db: AsyncSession,
    group_id: int | None,
) -> list[str]:
    """Return the current Visionnaire feature grants for one local user."""
    if group_id is None:
        return []
    rows = await db.execute(
        select(Feature.feature_name)
        .join(
            group_features_table,
            Feature.id == group_features_table.c.feature_id,
        )
        .where(group_features_table.c.group_id == group_id),
    )
    return list(rows.scalars())


async def subject_from_oidc_identity(
    db: AsyncSession,
    claims: Mapping[str, object],
    *,
    provider: str,
    binding: DeploymentBinding | None,
) -> AccessTokenSubject:
    """Resolve a verified OIDC subject to existing local permissions.

    Keycloak owns authentication. Visionnaire deliberately keeps its local
    user, tenant, group, site, and feature authorisation records authoritative.
    An unlinked provider subject never falls back to matching a username or an
    email address because either would permit accidental account takeover.
    """
    external_subject = claims.get('sub')
    issuer = claims.get('iss')
    if (
        not isinstance(external_subject, str)
        or not external_subject
        or not isinstance(issuer, str)
        or not issuer
    ):
        raise InvalidTokenError('OIDC claims do not contain a stable identity')

    identity = await db.scalar(
        select(UserIdentity).where(
            UserIdentity.provider == provider,
            UserIdentity.provider_user_id == external_subject,
        ),
    )
    if identity is None:
        raise InvalidTokenError('OIDC identity is not linked to a local user')
    user = await db.get(User, identity.user_id)
    if user is None or user.status != USER_STATUS_ACTIVE:
        raise InvalidTokenError('OIDC identity is not an active local user')
    if binding is not None and user.tenant_id != binding.tenant_id:
        raise InvalidTokenError(
            'OIDC identity is not allowed for this deployment',
        )

    token_id = claims.get('jti')
    issued_at = claims.get('iat')
    stable_token_id = (
        token_id
        if isinstance(token_id, str) and token_id
        else f'{external_subject}:{issued_at}'
    )
    subject: dict[str, object] = {
        'username': user.username,
        'user_id': user.id,
        'role': user.role,
        # Namespace external JTIs so a provider value can never collide with
        # a legacy application-issued token in the local revocation store.
        'jti': f'oidc:{issuer}:{stable_token_id}',
        'features': await _feature_names(db, user.group_id),
        'tenant_id': str(user.tenant_id),
    }
    if binding is not None:
        subject.update(
            {
                'deployment_id': str(binding.deployment_id),
                'config_revision': binding.config_revision,
            },
        )
    return cast(
        AccessTokenSubject,
        AccessTokenSubjectModel.model_validate(subject).model_dump(
            exclude_none=True,
        ),
    )
