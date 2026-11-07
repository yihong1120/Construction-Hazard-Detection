from __future__ import annotations

from dataclasses import dataclass

from fastapi import HTTPException

from examples.db_management.schemas.auth import NativeSocialCredential
from examples.db_management.services.native_social_state import (
    NativeSocialProvider,
)
from examples.db_management.services.oauth_services import ProviderClaims
from examples.db_management.services.oauth_services import (
    verify_apple_identity_token,
)
from examples.db_management.services.oauth_services import (
    verify_google_id_token,
)


@dataclass(frozen=True)
class VerifiedSocialIdentity:
    provider: NativeSocialProvider
    subject: str
    email: str | None = None

    @classmethod
    async def verify(
        cls, provider: NativeSocialProvider,
        credential: NativeSocialCredential, nonce: str,
    ) -> VerifiedSocialIdentity:
        claims = await _verify_credential(provider, credential, nonce)
        return cls(
            provider, _provider_subject(claims),
            _normalised_verified_email(claims),
        )


async def _verify_credential(
    provider: NativeSocialProvider,
    credential: NativeSocialCredential,
    nonce: str,
) -> ProviderClaims:
    """Verify a provider assertion and require the transaction nonce.

    Email is intentionally not used here.  Only the issuer-verified provider
    ``sub`` can be linked or used to authenticate a Keycloak account.
    """
    if provider == 'google':
        if not credential.id_token or credential.authorization_code:
            raise HTTPException(
                status_code=401,
                detail='invalid_provider_credential',
            )
        return await verify_google_id_token(
            credential.id_token,
            expected_nonce=nonce,
            require_verified_email=False,
        )
    if not credential.authorization_code:
        raise HTTPException(
            status_code=401,
            detail='invalid_provider_credential',
        )
    return await verify_apple_identity_token(
        credential.id_token,
        credential.authorization_code,
        expected_nonce=nonce,
    )


def _provider_subject(claims: ProviderClaims) -> str:
    """Return a bounded, non-secret stable provider subject."""
    subject = claims.sub
    if not subject or len(subject.encode('utf-8')) > 512:
        raise HTTPException(
            status_code=401,
            detail='invalid_provider_credential',
        )
    return subject


def _normalised_verified_email(claims: ProviderClaims) -> str | None:
    """Return only a provider-verified, canonical email address.

    Email is deliberately used solely to offer a one-time account-link
    confirmation. It is never accepted as a login identifier or copied into
    the resulting Keycloak federated identity.
    """
    if not claims.email_verified or not isinstance(claims.email, str):
        return None
    email = claims.email.strip().lower()
    return email or None
