from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal
from urllib.parse import quote

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.keycloak_admin import keycloak_admin
from examples.auth.models import User
from examples.auth.models import USER_STATUS_ACTIVE
from examples.auth.models import UserIdentity
from examples.auth.models import UserProfile
from examples.db_management.services.account_transaction import (
    AccountTransaction,
)
from examples.db_management.services.native_social_proof import (
    VerifiedSocialIdentity,
)
from examples.db_management.services.native_social_state import (
    _KEYCLOAK_SUBJECT_RE,
)
from examples.db_management.services.native_social_state import (
    _required_string,
)
from examples.db_management.services.native_social_state import (
    NativeSocialProvider,
)
from examples.db_management.services.native_social_state import settings


@dataclass(frozen=True, slots=True)
class EmailLinkCandidate:
    """One local account eligible for a verified-email link confirmation."""

    local_user_id: int
    keycloak_subject: str


_keycloak_admin_request = keycloak_admin.request


async def _find_email_link_candidate(
    db: AsyncSession,
    identity: VerifiedSocialIdentity,
) -> EmailLinkCandidate | None:
    """Find one active, already-Keycloak-linked local account by email.

    A provider assertion must already prove possession of a verified email.
    Even then this function does not authenticate the account; its result can
    only be used after a separate recent Keycloak reauthentication that proves
    control of the target Keycloak account.
    """
    email = identity.email
    if email is None:
        return None
    existing_provider_identity = await db.scalar(
        select(UserIdentity.id).where(
            UserIdentity.provider == identity.provider,
            UserIdentity.provider_user_id == identity.subject,
        ),
    )
    if existing_provider_identity is not None:
        # A local mirror indicates this provider subject has already gone
        # through an explicit Keycloak link. Let the regular Keycloak proof
        # path resolve it without requesting another password confirmation.
        return None
    rows = await db.execute(
        select(User.id, UserIdentity.provider_user_id)
        .join(UserProfile, UserProfile.user_id == User.id)
        .join(UserIdentity, UserIdentity.user_id == User.id)
        .where(
            User.status == USER_STATUS_ACTIVE,
            User.email_verified_at.is_not(None),
            UserProfile.email == email,
            UserIdentity.provider == settings.oidc_identity_provider,
        ),
    )
    candidates = [
        EmailLinkCandidate(local_user_id=user_id, keycloak_subject=subject)
        for user_id, subject in rows.all()
        if isinstance(user_id, int)
        and isinstance(subject, str)
        and _KEYCLOAK_SUBJECT_RE.fullmatch(subject)
    ]
    return candidates[0] if len(candidates) == 1 else None


def _provider_subject_from_record(record: Mapping[str, object]) -> str:
    """Decode and validate the provider subject retained in a proof record."""
    encoded_subject = _required_string(record, 'provider_subject_b64')
    try:
        padded = encoded_subject + '=' * (-len(encoded_subject) % 4)
        subject = base64.urlsafe_b64decode(
            padded.encode('ascii'),
        ).decode('utf-8')
    except (UnicodeDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=401,
            detail='native_social_exchange_invalid',
        ) from exc
    if not subject or len(subject.encode('utf-8')) > 512:
        raise HTTPException(
            status_code=401,
            detail='native_social_exchange_invalid',
        )
    return subject


async def _ensure_local_identity_available(
    db: AsyncSession,
    *,
    user_id: int,
    provider: NativeSocialProvider,
    provider_subject: str,
) -> UserIdentity | None:
    """Reject a provider subject that belongs to another local user."""
    identity = await db.scalar(
        select(UserIdentity).where(
            UserIdentity.provider == provider,
            UserIdentity.provider_user_id == provider_subject,
        ),
    )
    if identity is not None and identity.user_id != user_id:
        raise HTTPException(
            status_code=409,
            detail='provider_identity_already_linked',
        )
    identity_for_user = await db.scalar(
        select(UserIdentity).where(
            UserIdentity.user_id == user_id,
            UserIdentity.provider == provider,
        ),
    )
    if (
        identity_for_user is not None
        and identity_for_user.provider_user_id != provider_subject
    ):
        raise HTTPException(status_code=409, detail='provider_already_linked')

    return identity_for_user


@dataclass(frozen=True)
class NativeIdentityLink:
    user_id: int
    keycloak_subject: str
    identity: VerifiedSocialIdentity

    async def execute(
        self,
        db: AsyncSession,
    ) -> Literal['linked', 'already_linked']:
        """Stage the mirror first; retries reconcile an existing remote link.

        A remote identity can be observed by another concurrent request. Never
        delete it as rollback compensation. After an uncertain remote change,
        report reconciliation and let a fresh verified proof repair the mirror.
        """
        async with AccountTransaction(db) as transaction:
            existing = await _ensure_local_identity_available(
                db,
                user_id=self.user_id,
                provider=self.identity.provider,
                provider_subject=self.identity.subject,
            )
            if existing is None:
                try:
                    async with db.begin_nested():
                        db.add(
                            UserIdentity(
                                user_id=self.user_id,
                                provider=self.identity.provider,
                                provider_user_id=self.identity.subject,
                                email=None,
                                email_verified=False,
                                display_name=None,
                                raw_profile=None,
                                raw_email_is_private=False,
                            ),
                        )
                        await db.flush()
                except IntegrityError as error:
                    existing = await _ensure_local_identity_available(
                        db,
                        user_id=self.user_id,
                        provider=self.identity.provider,
                        provider_subject=self.identity.subject,
                    )
                    if existing is None:
                        raise HTTPException(
                            409,
                            'provider_identity_already_linked',
                        ) from error
            return await self.link_remote(transaction)

    async def link_remote(
        self,
        transaction: AccountTransaction,
    ) -> Literal['linked', 'already_linked']:
        """Attach an immutable provider subject using Keycloak's Admin API.

        The backend selects the target using the verified current Keycloak
        token.  It never accepts a user ID or an email in the client payload.
        """
        user_path = (
            f"/users/{quote(self.keycloak_subject, safe='')}"
            '/federated-identity'
        )
        existing = await _keycloak_admin_request('GET', user_path)
        if existing.status_code != 200:
            raise HTTPException(
                status_code=503,
                detail='keycloak_identity_service_unavailable',
            )
        try:
            identities = existing.json()
        except ValueError as exc:
            raise HTTPException(
                status_code=503,
                detail='keycloak_identity_service_unavailable',
            ) from exc
        if not isinstance(identities, list):
            raise HTTPException(
                status_code=503,
                detail='keycloak_identity_service_unavailable',
            )
        for identity in identities:
            if (
                not isinstance(identity, Mapping)
                or identity.get('identityProvider') != self.identity.provider
            ):
                continue
            if identity.get('userId') == self.identity.subject:
                return 'already_linked'
            raise HTTPException(
                status_code=409,
                detail='provider_already_linked',
            )
        transaction.uncompensated_effect('provider identity link')
        response = await _keycloak_admin_request(
            'POST',
            f'{user_path}/{self.identity.provider}',
            json_body={
                'identityProvider': self.identity.provider,
                'userId': self.identity.subject,
                'userName': self.identity.subject,
            },
        )
        if response.status_code == 204:
            return 'linked'
        if response.status_code == 409:
            transaction.reconciled_effect()
            latest = await _keycloak_admin_request('GET', user_path)
            if latest.status_code == 200:
                try:
                    identities = latest.json()
                except ValueError:
                    identities = []
                if isinstance(
                    identities, list,
                ) and any(
                    isinstance(identity, Mapping) and identity.get(
                        'identityProvider',
                    ) == self.identity.provider
                    and (identity.get('userId') == self.identity.subject)
                    for identity in identities
                ):
                    return 'already_linked'
            raise HTTPException(
                status_code=409,
                detail='provider_identity_already_linked',
            )
        raise HTTPException(
            status_code=503,
            detail='keycloak_identity_service_unavailable',
        )
