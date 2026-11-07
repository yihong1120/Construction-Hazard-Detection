from __future__ import annotations

import base64
import hmac
import time
from collections.abc import Mapping
from urllib.parse import urlencode

from fastapi import HTTPException
from fastapi import Request
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.jwt_config import JwtAuthorizationCredentials
from examples.auth.models import User
from examples.db_management.schemas.auth import (
    NativeSocialEmailLinkConfirmRequest,
)
from examples.db_management.schemas.auth import (
    NativeSocialExchangeBeginRequest,
)
from examples.db_management.schemas.auth import (
    NativeSocialExchangeBeginResponse,
)
from examples.db_management.schemas.auth import (
    NativeSocialExchangeCompleteRequest,
)
from examples.db_management.schemas.auth import (
    NativeSocialExchangeCompleteResponse,
)
from examples.db_management.schemas.auth import NativeSocialLinkBeginResponse
from examples.db_management.schemas.auth import NativeSocialLinkCompleteRequest
from examples.db_management.schemas.auth import NativeSocialLinkResponse
from examples.db_management.services.native_social_links import (
    _find_email_link_candidate,
)
from examples.db_management.services.native_social_links import (
    _provider_subject_from_record,
)
from examples.db_management.services.native_social_links import (
    EmailLinkCandidate,
)
from examples.db_management.services.native_social_links import (
    NativeIdentityLink,
)
from examples.db_management.services.native_social_proof import (
    VerifiedSocialIdentity,
)
from examples.db_management.services.native_social_state import (
    _consume_transaction,
)
from examples.db_management.services.native_social_state import (
    _enforce_begin_rate_limit,
)
from examples.db_management.services.native_social_state import (
    _KEYCLOAK_SUBJECT_RE,
)
from examples.db_management.services.native_social_state import (
    _opaque_identifier,
)
from examples.db_management.services.native_social_state import (
    _record_provider,
)
from examples.db_management.services.native_social_state import (
    _require_enabled,
)
from examples.db_management.services.native_social_state import (
    _required_string,
)
from examples.db_management.services.native_social_state import (
    _store_transaction,
)
from examples.db_management.services.native_social_state import (
    _validate_client_binding,
)
from examples.db_management.services.native_social_state import (
    NativeSocialProvider,
)
from examples.db_management.services.native_social_state import settings


async def _store_email_link_confirmation(
    redis: Redis,
    *,
    provider: NativeSocialProvider,
    provider_subject: str,
    candidate: EmailLinkCandidate,
) -> str:
    """Store the non-secret proof required after Keycloak reauthentication."""
    transaction_id = _opaque_identifier()
    await _store_transaction(
        redis,
        transaction_id,
        {
            'kind': 'email-link-confirmation',
            'provider': provider,
            'provider_subject_b64': base64.urlsafe_b64encode(
                provider_subject.encode('utf-8'),
            )
            .rstrip(b'=')
            .decode('ascii'),
            'local_user_id': candidate.local_user_id,
            'keycloak_subject': candidate.keycloak_subject,
        },
        settings.native_social_link_ttl_seconds,
    )
    return transaction_id


def _authorisation_url(
    transaction_id: str,
    record: Mapping[str, object],
) -> str:
    """Build an exact standard Keycloak authorisation-code request."""
    params = {
        'response_type': 'code',
        'client_id': _required_string(record, 'client_id'),
        'redirect_uri': _required_string(record, 'redirect_uri'),
        'code_challenge': _required_string(record, 'code_challenge'),
        'code_challenge_method': 'S256',
        'state': _required_string(record, 'state'),
        # The custom Keycloak authenticator consumes this private, one-use
        # proof.  It never receives the Google ID token or Apple code.
        'native_social_exchange': transaction_id,
    }
    return (
        f'{settings.oidc_issuer_url}/protocol/openid-connect/auth?'
        f'{urlencode(params)}'
    )


async def begin_native_social_exchange(
    payload: NativeSocialExchangeBeginRequest,
    request: Request,
    redis: Redis,
) -> NativeSocialExchangeBeginResponse:
    """Create a nonce- and PKCE-bound native social sign-in transaction."""
    _require_enabled()
    _validate_client_binding(payload.client_id, payload.redirect_uri)
    await _enforce_begin_rate_limit(request, redis)
    transaction_id = _opaque_identifier()
    nonce = _opaque_identifier()
    await _store_transaction(
        redis,
        transaction_id,
        {
            'kind': 'login-begin',
            'provider': payload.provider,
            'nonce': nonce,
            'client_id': payload.client_id,
            'redirect_uri': payload.redirect_uri,
            'code_challenge': payload.code_challenge,
            'state': payload.state,
        },
        settings.native_social_exchange_ttl_seconds,
    )
    return NativeSocialExchangeBeginResponse(
        transaction_id=transaction_id,
        nonce=nonce,
        expires_in=settings.native_social_exchange_ttl_seconds,
    )


async def complete_native_social_exchange(
    payload: NativeSocialExchangeCompleteRequest,
    redis: Redis,
    db: AsyncSession,
) -> NativeSocialExchangeCompleteResponse:
    """Validate native credentials and hand a one-use proof to Keycloak."""
    _require_enabled()
    record = await _consume_transaction(redis, payload.transaction_id)
    if record.get('kind') != 'login-begin':
        raise HTTPException(
            status_code=401,
            detail='native_social_exchange_expired',
        )
    provider = _record_provider(record)
    nonce = _required_string(record, 'nonce')
    identity = await VerifiedSocialIdentity.verify(provider, payload, nonce)
    subject = identity.subject
    candidate = await _find_email_link_candidate(
        db,
        identity,
    )
    if candidate is not None:
        confirmation_id = await _store_email_link_confirmation(
            redis,
            provider=provider,
            provider_subject=subject,
            candidate=candidate,
        )
        # Do not reveal any user, tenant, or email information. The app can
        # now start Keycloak with prompt=login/max_age=0. Confirmation accepts
        # only the exact Keycloak subject selected above.
        raise HTTPException(
            status_code=409,
            detail={
                'code': 'account_link_required',
                'link_transaction_id': confirmation_id,
                'expires_in': settings.native_social_link_ttl_seconds,
            },
        )

    # Keep only the Keycloak lookup key and the exact PKCE/client binding.
    # The record is consumed by Keycloak's loopback-only authenticator call.
    await _store_transaction(
        redis,
        payload.transaction_id,
        {
            'kind': 'login-redeem',
            'provider': provider,
            'provider_subject_b64': base64.urlsafe_b64encode(
                subject.encode('utf-8'),
            )
            .rstrip(b'=')
            .decode('ascii'),
            'client_id': _required_string(record, 'client_id'),
            'redirect_uri': _required_string(record, 'redirect_uri'),
            'code_challenge': _required_string(record, 'code_challenge'),
        },
        settings.native_social_exchange_ttl_seconds,
    )
    return NativeSocialExchangeCompleteResponse(
        authorization_url=_authorisation_url(payload.transaction_id, record),
        expires_in=settings.native_social_exchange_ttl_seconds,
    )


def _keycloak_identity_from_credentials(
    credentials: JwtAuthorizationCredentials,
) -> tuple[str, str | None]:
    """Require a recently reauthenticated Keycloak access token for linking."""
    payload = credentials.payload
    if payload.get('iss') != settings.oidc_issuer_url:
        raise HTTPException(
            status_code=401,
            detail='keycloak_reauthentication_required',
        )
    subject = payload.get('sub')
    auth_time = payload.get('auth_time')
    now = time.time()
    if (
        not isinstance(subject, str)
        or not _KEYCLOAK_SUBJECT_RE.fullmatch(subject)
        or isinstance(auth_time, bool)
        or not isinstance(auth_time, (int, float))
        or auth_time > now + 60
        or now - auth_time > settings.native_social_link_max_auth_age_seconds
    ):
        raise HTTPException(
            status_code=401,
            detail='keycloak_reauthentication_required',
        )
    sid = payload.get('sid') or payload.get('session_state')
    if sid is not None and not isinstance(sid, str):
        raise HTTPException(
            status_code=401,
            detail='keycloak_reauthentication_required',
        )
    return subject, sid


async def begin_native_social_link(
    provider: NativeSocialProvider,
    credentials: JwtAuthorizationCredentials,
    redis: Redis,
) -> NativeSocialLinkBeginResponse:
    """Start a nonce-bound link transaction after fresh Keycloak auth."""
    _require_enabled()
    keycloak_subject, session_id = _keycloak_identity_from_credentials(
        credentials,
    )
    transaction_id = _opaque_identifier()
    nonce = _opaque_identifier()
    await _store_transaction(
        redis,
        transaction_id,
        {
            'kind': 'link-begin',
            'provider': provider,
            'nonce': nonce,
            'keycloak_subject': keycloak_subject,
            'keycloak_session_id': session_id,
        },
        settings.native_social_link_ttl_seconds,
    )
    return NativeSocialLinkBeginResponse(
        transaction_id=transaction_id,
        nonce=nonce,
        expires_in=settings.native_social_link_ttl_seconds,
    )


async def complete_native_social_link(
    payload: NativeSocialLinkCompleteRequest,
    credentials: JwtAuthorizationCredentials,
    redis: Redis,
    db: AsyncSession,
    user: User,
) -> NativeSocialLinkResponse:
    """Validate provider proof and attach it to the current Keycloak user."""
    _require_enabled()
    current_subject, current_session_id = _keycloak_identity_from_credentials(
        credentials,
    )
    record = await _consume_transaction(redis, payload.transaction_id)
    if record.get('kind') != 'link-begin':
        raise HTTPException(
            status_code=401,
            detail='native_social_link_expired',
        )
    if not hmac.compare_digest(
        _required_string(record, 'keycloak_subject'),
        current_subject,
    ):
        raise HTTPException(
            status_code=401,
            detail='native_social_link_expired',
        )
    stored_session_id = record.get('keycloak_session_id')
    if isinstance(stored_session_id, str):
        if not isinstance(current_session_id, str) or not hmac.compare_digest(
            stored_session_id,
            current_session_id,
        ):
            raise HTTPException(
                status_code=401,
                detail='native_social_link_expired',
            )
    provider = _record_provider(record)
    identity = await VerifiedSocialIdentity.verify(
        provider, payload, _required_string(record, 'nonce'),
    )
    status = await NativeIdentityLink(
        user.id,
        current_subject,
        identity,
    ).execute(db)
    return NativeSocialLinkResponse(provider=provider, status=status)


async def confirm_native_social_email_link(
    payload: NativeSocialEmailLinkConfirmRequest,
    credentials: JwtAuthorizationCredentials,
    redis: Redis,
    db: AsyncSession,
    user: User,
) -> NativeSocialLinkResponse:
    """Finish a verified-email link only after fresh Keycloak proof.

    The original Google/Apple assertion has already been consumed. This route
    simply binds its immutable subject to the *preselected* local Keycloak
    account after the user proves possession of that account's own credential.
    """
    _require_enabled()
    current_subject, _ = _keycloak_identity_from_credentials(credentials)
    record = await _consume_transaction(redis, payload.transaction_id)
    if record.get('kind') != 'email-link-confirmation':
        raise HTTPException(
            status_code=401,
            detail='native_social_link_expired',
        )
    local_user_id = record.get('local_user_id')
    if not isinstance(local_user_id, int) or local_user_id != user.id:
        raise HTTPException(
            status_code=401,
            detail='native_social_link_expired',
        )
    if not hmac.compare_digest(
        _required_string(record, 'keycloak_subject'),
        current_subject,
    ):
        raise HTTPException(
            status_code=401,
            detail='native_social_link_expired',
        )
    provider = _record_provider(record)
    provider_subject = _provider_subject_from_record(record)
    status = await NativeIdentityLink(
        user.id,
        current_subject,
        VerifiedSocialIdentity(provider, provider_subject),
    ).execute(db)
    return NativeSocialLinkResponse(provider=provider, status=status)
