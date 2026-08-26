from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import Sequence
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from functools import lru_cache
from typing import Literal

import httpx
import jwt
from fastapi import HTTPException
from fastapi import Request
from pydantic import ValidationError
from redis.asyncio import Redis
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from examples.auth.config import Settings
from examples.auth.deployment_context import DeploymentBinding
from examples.auth.deployment_context import resolve_request_deployment
from examples.auth.models import User
from examples.auth.models import USER_STATUS_ACTIVE
from examples.auth.models import USER_STATUS_EMAIL_UNVERIFIED
from examples.auth.models import USER_STATUS_PENDING_ADMIN_APPROVAL
from examples.auth.models import USER_STATUS_REJECTED
from examples.auth.models import USER_STATUS_SUSPENDED
from examples.auth.models import UserIdentity
from examples.auth.models import UserProfile
from examples.db_management.schemas.auth import AppleTokenExchangeResponse
from examples.db_management.schemas.auth import IdentityListResponse
from examples.db_management.schemas.auth import IdentityRead
from examples.db_management.schemas.auth import ProviderClaims
from examples.db_management.schemas.auth import TokenPairData
from examples.db_management.services.auth_services import (
    issue_token_pair_for_user,
)
from examples.db_management.services.legal_services import record_user_consent
from examples.db_management.services.legal_services import SignupConsentPayload
from examples.db_management.services.legal_services import (
    validate_signup_consents,
)
from src.http_client_pool import get_application_http_client

Provider = Literal['google', 'apple']
OAUTH_DISABLED_PASSWORD_HASH = 'oauth_disabled:provider-only'

settings = Settings()

GOOGLE_ISSUERS = ('accounts.google.com', 'https://accounts.google.com')
GOOGLE_JWKS_URL = 'https://www.googleapis.com/oauth2/v3/certs'
APPLE_ISSUER = 'https://appleid.apple.com'
APPLE_JWKS_URL = 'https://appleid.apple.com/auth/keys'
APPLE_TOKEN_URL = 'https://appleid.apple.com/auth/token'


def _configured_google_client_ids() -> list[str]:
    """Return configured Google OAuth audience identifiers.

    Returns:
        Non-empty client IDs accepted in Google identity tokens.
    """
    return [
        value.strip()
        for value in settings.google_client_ids.split(',')
        if value.strip()
    ]


def _configured_apple_client_ids() -> list[str]:
    """Return configured Apple OAuth audience identifiers.

    Returns:
        Non-empty client IDs accepted in Apple identity tokens.
    """
    return [
        value.strip()
        for value in settings.apple_client_ids.split(',')
        if value.strip()
    ]


def _normalise_email(email: str | None) -> str | None:
    """Return a trimmed, case-normalised optional email address.

    Args:
        email: Optional email address supplied by an identity provider.

    Returns:
        Normalised email address, or ``None`` when absent.
    """
    if email is None:
        return None
    normalized = email.strip().lower()
    return normalized or None


def _provider_claims(payload: object) -> ProviderClaims:
    """Validate provider claims at the OpenID Connect boundary.

    Args:
        payload: Decoded provider-token claims.

    Returns:
        Strictly validated provider claims.
    """
    try:
        return ProviderClaims.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(
            status_code=401,
            detail='Invalid provider token',
        ) from exc


@lru_cache(maxsize=4)
def _jwks_client(jwks_url: str) -> jwt.PyJWKClient:
    """Return the process-wide JWKS client for an identity provider."""
    return jwt.PyJWKClient(jwks_url)


def _verify_jwt_with_jwks(
    token: str,
    jwks_url: str,
    audiences: Sequence[str],
    issuers: Sequence[str],
) -> ProviderClaims:
    """Verify provider-token claims against JWKS metadata.

    Args:
        token: Raw OpenID Connect identity token.
        jwks_url: Provider JSON Web Key Set URL.
        audiences: Accepted client audiences.
        issuers: Accepted token issuers.

    Returns:
        Verified provider claims.

    Raises:
        HTTPException: If token signature or registered claims are invalid.
    """
    if not audiences:
        raise HTTPException(
            status_code=500,
            detail='OAuth client not configured',
        )

    try:
        signing_key = _jwks_client(jwks_url).get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=['RS256'],
            audience=list(audiences),
            issuer=list(issuers),
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=401,
            detail='Invalid provider token',
        ) from exc

    return _provider_claims(payload)


async def verify_google_id_token(
    id_token: str,
    *,
    expected_nonce: str | None = None,
    require_verified_email: bool = True,
) -> ProviderClaims:
    """Verify Google identity-token signature and registered claims.

    Args:
        id_token: Raw Google OpenID Connect identity token.
        expected_nonce: Optional one-use nonce bound to the native request.
        require_verified_email: Whether this caller requires a verified email.
            Account linking and the Keycloak exchange use the immutable
            provider subject rather than an email address, so they do not
            require an email claim.

    Returns:
        Verified Google provider claims.
    """
    payload = await asyncio.to_thread(
        _verify_jwt_with_jwks,
        id_token,
        GOOGLE_JWKS_URL,
        _configured_google_client_ids(),
        GOOGLE_ISSUERS,
    )
    claims = _provider_claims(payload)
    if expected_nonce is not None and claims.nonce != expected_nonce:
        raise HTTPException(
            status_code=401,
            detail='Invalid provider token',
        )
    if require_verified_email and not claims.email_verified:
        raise HTTPException(
            status_code=401,
            detail='Google email is not verified',
        )
    if require_verified_email and not _normalise_email(claims.email):
        raise HTTPException(
            status_code=401,
            detail='Google account did not return an email address',
        )
    return claims


def _load_apple_private_key() -> str:
    """Load the configured Apple signing private key.

    Returns:
        PEM-encoded Apple signing key.

    Raises:
        HTTPException: If the required Apple key configuration is unavailable.
    """
    if settings.apple_private_key:
        return settings.apple_private_key.replace('\\n', '\n')
    if settings.apple_private_key_path:
        with open(settings.apple_private_key_path, encoding='utf-8') as file:
            return file.read()
    raise HTTPException(
        status_code=500,
        detail='Apple client secret is not configured',
    )


def _build_apple_client_secret(client_id: str) -> str:
    """Build a signed Apple client-secret JWT for a client.

    Args:
        client_id: Apple Services ID or bundle identifier.

    Returns:
        Signed short-lived Apple client-secret JWT.
    """
    if not settings.apple_team_id or not settings.apple_key_id:
        raise HTTPException(
            status_code=500,
            detail='Apple client secret is not configured',
        )

    now = datetime.now(timezone.utc)
    return jwt.encode(
        {
            'iss': settings.apple_team_id,
            'iat': int(now.timestamp()),
            'exp': int((now + timedelta(days=180)).timestamp()),
            'aud': APPLE_ISSUER,
            'sub': client_id,
        },
        _load_apple_private_key(),
        algorithm='ES256',
        headers={'kid': settings.apple_key_id},
    )


async def verify_apple_identity_token(
    identity_token: str | None,
    authorization_code: str,
    expected_nonce: str | None = None,
) -> ProviderClaims:
    """Verify Apple identity token and validate its authorisation code.

    Args:
        identity_token: Optional Apple OpenID Connect identity token.
        authorization_code: Apple authorisation code to validate.
        expected_nonce: Optional nonce expected in token claims.

    Returns:
        Verified Apple provider claims.
    """
    client_ids = _configured_apple_client_ids()
    payload: ProviderClaims | None = None
    if identity_token:
        payload = _provider_claims(
            await asyncio.to_thread(
                _verify_jwt_with_jwks,
                identity_token,
                APPLE_JWKS_URL,
                client_ids,
                (APPLE_ISSUER,),
            ),
        )
        client_id = payload.aud
        if client_id not in client_ids:
            raise HTTPException(
                status_code=401,
                detail='Invalid provider token',
            )
        token_response = await _exchange_apple_authorization_code(
            authorization_code,
            [client_id],
        )
    else:
        token_response = await _exchange_apple_authorization_code(
            authorization_code,
            _apple_exchange_client_id_candidates(),
        )

    token_response = AppleTokenExchangeResponse.model_validate(
        token_response,
    )
    if token_response.id_token is not None:
        exchanged_payload = _provider_claims(
            await asyncio.to_thread(
                _verify_jwt_with_jwks,
                token_response.id_token,
                APPLE_JWKS_URL,
                client_ids,
                (APPLE_ISSUER,),
            ),
        )
        if payload is None:
            payload = exchanged_payload
        elif exchanged_payload.sub != payload.sub:
            raise HTTPException(
                status_code=401,
                detail='Invalid provider token',
            )
    if payload is None:
        raise HTTPException(status_code=401, detail='Invalid provider token')
    if expected_nonce and payload.nonce != expected_nonce:
        raise HTTPException(status_code=401, detail='Invalid provider token')
    return payload


def _apple_exchange_client_id_candidates() -> list[str]:
    """Try web/service ID first, then native bundle ID for Apple code
    exchange."""
    candidates = [
        settings.apple_service_id,
        settings.apple_bundle_id,
        *_configured_apple_client_ids(),
    ]
    deduped: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in deduped:
            deduped.append(candidate)
    return deduped


async def _exchange_apple_authorization_code(
    authorization_code: str,
    client_ids: Sequence[str],
) -> AppleTokenExchangeResponse:
    """Validate an Apple authorisation code against an allowed client ID.

    Args:
        authorization_code: Apple authorisation code to exchange.
        client_ids: Candidate allowed Apple client IDs.

    Returns:
        Validated Apple token-exchange response.
    """
    last_error: HTTPException | None = None
    for client_id in client_ids:
        try:
            return AppleTokenExchangeResponse.model_validate(
                await _exchange_apple_authorization_code_once(
                    authorization_code,
                    client_id,
                ),
            )
        except HTTPException as exc:
            last_error = exc
        except ValidationError as exc:
            raise HTTPException(
                status_code=401,
                detail='Invalid provider token',
            ) from exc
    if last_error is not None:
        raise last_error
    raise HTTPException(status_code=500, detail='Apple client not configured')


async def _exchange_apple_authorization_code_once(
    authorization_code: str,
    client_id: str,
) -> AppleTokenExchangeResponse:
    """Exchange an Apple authorisation code for provider tokens once.

    Args:
        authorization_code: Apple authorisation code to exchange.
        client_id: Apple client ID bound to the code.

    Returns:
        Validated Apple token-exchange response.
    """
    data = {
        'client_id': client_id,
        'client_secret': _build_apple_client_secret(client_id),
        'code': authorization_code,
        'grant_type': 'authorization_code',
    }
    if client_id == settings.apple_service_id:
        data['redirect_uri'] = settings.apple_redirect_uri

    client = await get_application_http_client(
        'apple-token-exchange',
        timeout=10.0,
    )
    if client is not None:
        response = await client.post(
            APPLE_TOKEN_URL,
            data=data,
        )
    else:
        async with httpx.AsyncClient(timeout=10.0) as ephemeral_client:
            response = await ephemeral_client.post(
                APPLE_TOKEN_URL,
                data=data,
            )
    if response.status_code >= 400:
        raise HTTPException(status_code=401, detail='Invalid provider token')
    try:
        return AppleTokenExchangeResponse.model_validate(response.json())
    except (ValueError, ValidationError) as exc:
        raise HTTPException(
            status_code=401,
            detail='Invalid provider token',
        ) from exc


def _status_error(status: str) -> HTTPException:
    """Return the HTTP error corresponding to an account status.

    Args:
        status: Account lifecycle status.

    Returns:
        Appropriate authentication HTTP exception.
    """
    if status == USER_STATUS_EMAIL_UNVERIFIED:
        return HTTPException(
            status_code=403,
            detail={'code': 'email_unverified', 'status': status},
        )
    if status == USER_STATUS_PENDING_ADMIN_APPROVAL:
        return HTTPException(
            status_code=403,
            detail={'code': 'pending_admin_approval', 'status': status},
        )
    if status == USER_STATUS_SUSPENDED:
        return HTTPException(
            status_code=403,
            detail={'code': 'account_suspended', 'status': status},
        )
    if status == USER_STATUS_REJECTED:
        return HTTPException(
            status_code=403,
            detail={'code': 'account_rejected', 'status': status},
        )
    return HTTPException(
        status_code=403,
        detail={'code': 'user_not_active', 'status': status},
    )


def _ensure_active_user(user: User) -> None:
    """Reject a user account that is not active.

    Args:
        user: Account whose lifecycle status is checked.

    Raises:
        HTTPException: If the account is not active.
    """
    if user.status != USER_STATUS_ACTIVE:
        raise _status_error(user.status)


def _username_from_claims(provider: Provider, claims: ProviderClaims) -> str:
    """Derive a stable local username from provider claims.

    Args:
        provider: Identity-provider name.
        claims: Verified identity-provider claims.

    Returns:
        Bounded local username candidate.
    """
    email = _normalise_email(claims.email)
    source = email.split('@', 1)[0] if email else f"{provider}_{claims.sub}"
    username = re.sub(r'[^A-Za-z0-9_.-]+', '_', source).strip('._-')
    return username[:64] or f"{provider}_user"


def _identity_username(
    provider: Provider,
    claims: ProviderClaims,
) -> str:
    """Return a deterministic, collision-resistant OAuth username.

    Args:
        provider: Identity-provider name.
        claims: Verified identity-provider claims.

    Returns:
        A stable username at most 80 characters long.
    """
    base = _username_from_claims(provider, claims)
    # Provider subjects are already stable identity keys.  A short digest keeps
    # a human-readable prefix while avoiding serial SELECT/retry loops for
    # common display names such as "john".
    digest = hashlib.blake2s(
        f"{provider}:{claims.sub}".encode(),
        digest_size=6,
    ).hexdigest()
    return f"{base[:67]}_{digest}"


def _profile_names(
    provider: Provider,
    claims: ProviderClaims,
) -> tuple[str, str]:
    """Derive bounded family and given names from provider claims.

    Args:
        provider: Identity-provider name.
        claims: Verified identity-provider claims.

    Returns:
        Bounded family-name and given-name pair.
    """
    given_name = (claims.given_name or '').strip()
    family_name = (claims.family_name or '').strip()
    if given_name or family_name:
        return (family_name or provider.title())[:50], (given_name or 'User')[
            :50
        ]

    name = (claims.name or '').strip()
    if name:
        parts = name.split()
        if len(parts) > 1:
            return parts[0][:50], ' '.join(parts[1:])[:50]
        return provider.title(), parts[0][:50]
    email = _normalise_email(claims.email)
    if email:
        return provider.title(), email.split('@', 1)[0][:50]
    return provider.title(), 'User'


def _new_identity(
    user: User,
    provider: Provider,
    claims: ProviderClaims,
) -> UserIdentity:
    """Build an unpersisted provider identity for a user.

    Args:
        user: Local user receiving the identity.
        provider: Identity-provider name.
        claims: Verified identity-provider claims.

    Returns:
        New unpersisted identity model.
    """
    email = _normalise_email(claims.email)
    return UserIdentity(
        user=user,
        provider=provider,
        provider_user_id=claims.sub,
        email=email,
        email_verified=claims.email_verified,
        display_name=str(
            claims.name
            or ' '.join(
                part
                for part in [
                    (claims.given_name or '').strip(),
                    (claims.family_name or '').strip(),
                ]
                if part
            )
            or '',
        )
        or None,
        raw_profile=claims.model_dump(),
        raw_email_is_private=(
            claims.is_private_email
            or bool(email and email.endswith('@privaterelay.appleid.com'))
        ),
    )


def _display_name_from_claims(claims: ProviderClaims) -> str | None:
    """Return the optional display name supplied by an identity provider.

    Args:
        claims: Verified identity-provider claims.

    Returns:
        Bounded display name, or ``None`` when absent.
    """
    display_name = str(
        claims.name
        or ' '.join(
            part
            for part in [
                (claims.given_name or '').strip(),
                (claims.family_name or '').strip(),
            ]
            if part
        )
        or '',
    )
    return display_name or None


async def _find_identity_user(
    db: AsyncSession,
    provider: Provider,
    provider_user_id: str,
) -> User | None:
    """Find the local user associated with a provider subject.

    Args:
        db: Database session used to load identity and user data.
        provider: Identity-provider name.
        provider_user_id: Stable external provider subject.

    Returns:
        Associated user, or ``None`` when no identity exists.
    """
    identity = await db.scalar(
        select(UserIdentity).where(
            UserIdentity.provider == provider,
            UserIdentity.provider_user_id == provider_user_id,
        ),
    )
    if identity is None:
        return None
    return await db.get(User, identity.user_id)


async def _find_user_by_email(db: AsyncSession, email: str) -> User | None:
    """Find the local user with a matching profile email address.

    Args:
        db: Database session used to search profiles.
        email: Normalised profile email address.

    Returns:
        Matching local user, or ``None`` when absent.
    """
    return await db.scalar(
        select(User)
        .join(UserProfile, UserProfile.user_id == User.id)
        .where(UserProfile.email == email.strip().lower()),
    )


async def _create_pending_user_with_identity(
    db: AsyncSession,
    provider: Provider,
    claims: ProviderClaims,
    deployment: DeploymentBinding | None = None,
) -> User:
    """Create a pending account and identity from provider claims.

    Args:
        db: Database session used to persist account and identity.
        provider: Identity-provider name.
        claims: Verified identity-provider claims.

    Returns:
        Newly created pending user.
    """
    email = _normalise_email(claims.email)
    if not email:
        raise HTTPException(
            status_code=400,
            detail='Provider account did not return an email address',
        )

    family_name, given_name = _profile_names(provider, claims)
    user = User(
        username=_identity_username(provider, claims),
        password_hash=OAUTH_DISABLED_PASSWORD_HASH,
        role='user',
        status=USER_STATUS_PENDING_ADMIN_APPROVAL,
        email_verified_at=datetime.now(timezone.utc),
        group_id=None,
        **({'tenant_id': deployment.tenant_id} if deployment else {}),
    )
    try:
        db.add(user)
        await db.flush()
        db.add(
            UserProfile(
                user_id=user.id,
                family_name=family_name,
                given_name=given_name,
                middle_name=None,
                email=email,
                mobile_number=None,
            ),
        )
        db.add(_new_identity(user, provider, claims))
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        # Concurrent callbacks for the same provider subject should converge
        # on the account that won the unique identity constraint.
        existing = await _find_identity_user(db, provider, claims.sub)
        if existing is not None:
            return existing
        raise HTTPException(
            status_code=409,
            detail='OAuth account creation conflict',
        ) from exc
    await db.refresh(user, attribute_names=['profile', 'group'])
    return user


async def authenticate_provider_user(
    provider: Provider,
    claims: ProviderClaims,
    db: AsyncSession,
    redis_pool: Redis,
    consent_payload: SignupConsentPayload | None = None,
    hash_refresh_token: bool = False,
    deployment: DeploymentBinding | None = None,
    request: Request | None = None,
) -> TokenPairData:
    """Resolve a verified provider identity to a local user and issue
    tokens."""
    provider_user_id = claims.sub
    if deployment is None and isinstance(request, Request):
        deployment = await resolve_request_deployment(request, db)
    if deployment is None and isinstance(request, Request):
        raise HTTPException(
            status_code=409,
            detail='deployment_required',
        )

    user = await _find_identity_user(db, provider, provider_user_id)
    if user is not None:
        _ensure_active_user(user)
        if deployment is not None:
            return await issue_token_pair_for_user(
                user,
                db,
                redis_pool,
                hash_refresh_token=hash_refresh_token,
                deployment=deployment,
            )
        return await issue_token_pair_for_user(
            user,
            db,
            redis_pool,
            hash_refresh_token=hash_refresh_token,
        )

    email = _normalise_email(claims.email)
    if email:
        existing_user = await _find_user_by_email(db, email)
        if existing_user is not None:
            raise HTTPException(
                status_code=409,
                detail={
                    'code': 'account_link_required',
                    'message': (
                        'Please login with your existing account and link '
                        'this provider.'
                    ),
                },
            )

    if consent_payload is None:
        raise HTTPException(
            status_code=400,
            detail={
                'code': 'legal_consent_required',
                'message': 'Legal consent is required for new accounts.',
            },
        )

    await validate_signup_consents(consent_payload, db)
    user = await _create_pending_user_with_identity(
        db,
        provider,
        claims,
        deployment,
    )
    await record_user_consent(user.id, consent_payload, db, request=None)
    raise _status_error(user.status)


async def login_with_google(
    id_token: str,
    db: AsyncSession,
    redis_pool: Redis,
    email: str | None = None,
    display_name: str | None = None,
    device_lang: str | None = None,
    consent_payload: SignupConsentPayload | None = None,
    hash_refresh_token: bool = False,
    deployment: DeploymentBinding | None = None,
    request: Request | None = None,
) -> TokenPairData:
    """Authenticate a Google identity token and issue local tokens.

    Args:
        id_token: Google OpenID Connect identity token.
        db: Database session used to resolve the account.
        redis_pool: Redis connection used to issue token state.
        email: Optional client-supplied email assertion.
        display_name: Optional client-supplied display name.
        device_lang: Optional client device language.
        consent_payload: Required consents for a new account.
        hash_refresh_token: Whether browser refresh tokens are cache-hashed.

    Returns:
        Locally issued token-pair data.
    """
    claims = await verify_google_id_token(id_token)
    if display_name and not claims.name:
        claims.name = display_name
    if device_lang:
        claims.device_lang = device_lang
    return await authenticate_provider_user(
        'google',
        claims,
        db,
        redis_pool,
        consent_payload=consent_payload,
        hash_refresh_token=hash_refresh_token,
        deployment=deployment,
        request=request,
    )


async def login_with_apple(
    identity_token: str | None,
    authorization_code: str,
    db: AsyncSession,
    redis_pool: Redis,
    email: str | None = None,
    given_name: str | None = None,
    family_name: str | None = None,
    nonce: str | None = None,
    device_lang: str | None = None,
    consent_payload: SignupConsentPayload | None = None,
    hash_refresh_token: bool = False,
    deployment: DeploymentBinding | None = None,
    request: Request | None = None,
) -> TokenPairData:
    """Authenticate an Apple identity and issue local tokens.

    Args:
        identity_token: Optional Apple identity token.
        authorization_code: Apple authorisation code.
        db: Database session used to resolve the account.
        redis_pool: Redis connection used to issue token state.
        email: Optional client-supplied email assertion.
        given_name: Optional client-supplied given name.
        family_name: Optional client-supplied family name.
        nonce: Optional expected identity-token nonce.
        device_lang: Optional client device language.
        consent_payload: Required consents for a new account.
        hash_refresh_token: Whether browser refresh tokens are cache-hashed.

    Returns:
        Locally issued token-pair data.
    """
    claims = await verify_apple_identity_token(
        identity_token,
        authorization_code,
        expected_nonce=nonce,
    )
    if email and not claims.email:
        claims.email = email
    if given_name and not claims.given_name:
        claims.given_name = given_name
    if family_name and not claims.family_name:
        claims.family_name = family_name
    if device_lang:
        claims.device_lang = device_lang
    return await authenticate_provider_user(
        'apple',
        claims,
        db,
        redis_pool,
        consent_payload=consent_payload,
        hash_refresh_token=hash_refresh_token,
        deployment=deployment,
        request=request,
    )


def _user_has_password(user: User) -> bool:
    """Return whether the user has a configured password credential.

    Args:
        user: User whose password credential is inspected.

    Returns:
        ``True`` when the user has a usable password hash.
    """
    return not str(user.password_hash).startswith('oauth_disabled:')


def _identity_read(identity: UserIdentity) -> IdentityRead:
    """Convert a persisted provider identity into its public schema.

    Args:
        identity: Persisted provider identity.

    Returns:
        Safe public identity response.
    """
    linked_at = identity.linked_at
    return IdentityRead(
        id=identity.id,
        provider=identity.provider,
        email=identity.email,
        display_name=identity.display_name,
        linked_at=linked_at.isoformat().replace('+00:00', 'Z'),
    )


async def list_user_identities(
    user: User,
    db: AsyncSession,
) -> IdentityListResponse:
    """Return the current user's linked provider identities.

    Args:
        user: Current authenticated user.
        db: Database session used to refresh identity data.

    Returns:
        Public identities and password-credential availability.
    """
    identities = (
        (
            await db.execute(
                select(UserIdentity)
                .where(UserIdentity.user_id == user.id)
                .order_by(UserIdentity.linked_at.asc()),
            )
        )
        .scalars()
        .all()
    )
    return IdentityListResponse(
        identities=[_identity_read(identity) for identity in identities],
        has_password=_user_has_password(user),
    )


async def _find_identity(
    db: AsyncSession,
    provider: Provider,
    provider_user_id: str,
) -> UserIdentity | None:
    """Find a provider identity matching an external subject.

    Args:
        db: Database session used to search identities.
        provider: Identity-provider name.
        provider_user_id: Stable external provider subject.

    Returns:
        Matching identity, or ``None`` when absent.
    """
    return await db.scalar(
        select(UserIdentity).where(
            UserIdentity.provider == provider,
            UserIdentity.provider_user_id == provider_user_id,
        ),
    )


async def _find_current_user_provider_identity(
    db: AsyncSession,
    user: User,
    provider: Provider,
) -> UserIdentity | None:
    """Find the user's existing identity for one provider.

    Args:
        db: Database session used to search identities.
        user: Local user that owns the identity.
        provider: Identity-provider name.

    Returns:
        Matching identity, or ``None`` when absent.
    """
    return await db.scalar(
        select(UserIdentity).where(
            UserIdentity.user_id == user.id,
            UserIdentity.provider == provider,
        ),
    )


def _update_identity_from_claims(
    identity: UserIdentity,
    claims: ProviderClaims,
) -> None:
    """Synchronise mutable provider claims onto a persisted identity.

    Args:
        identity: Persisted provider identity to update.
        claims: Newly verified provider claims.
    """
    email = _normalise_email(claims.email)
    if email:
        identity.email = email
    identity.email_verified = claims.email_verified
    display_name = _display_name_from_claims(claims)
    if display_name:
        identity.display_name = display_name
    identity.raw_profile = claims.model_dump()
    identity.raw_email_is_private = claims.is_private_email or bool(
        identity.email
        and identity.email.endswith(
            '@privaterelay.appleid.com',
        ),
    )


async def link_provider_identity(
    user: User,
    provider: Provider,
    claims: ProviderClaims,
    db: AsyncSession,
) -> IdentityRead:
    """Link a verified provider identity to the current user.

    Args:
        user: Authenticated local user receiving the identity.
        provider: Identity-provider name.
        claims: Verified identity-provider claims.
        db: Database session used to persist the link.

    Returns:
        Public linked identity response.
    """
    provider_user_id = claims.sub

    existing = await _find_identity(db, provider, provider_user_id)
    if existing is not None and existing.user_id != user.id:
        raise HTTPException(
            status_code=409,
            detail={
                'code': 'identity_already_linked_to_another_user',
                'message': (
                    'This provider account is already linked to another user.'
                ),
            },
        )
    if existing is not None:
        _update_identity_from_claims(existing, claims)
        await db.commit()
        await db.refresh(existing)
        return _identity_read(existing)

    current_provider = await _find_current_user_provider_identity(
        db,
        user,
        provider,
    )
    if current_provider is not None:
        raise HTTPException(
            status_code=409,
            detail={
                'code': 'provider_already_linked',
                'message': (
                    'This user already has a different account linked for '
                    f"{provider}."
                ),
            },
        )

    identity = _new_identity(user, provider, claims)
    db.add(identity)
    await db.commit()
    await db.refresh(identity)
    return _identity_read(identity)


async def link_google_identity(
    user: User,
    id_token: str,
    db: AsyncSession,
) -> IdentityRead:
    """Verify and link a Google identity token.

    Args:
        user: Authenticated local user receiving the identity.
        id_token: Google OpenID Connect identity token.
        db: Database session used to persist the link.

    Returns:
        Public linked Google identity.
    """
    claims = await verify_google_id_token(id_token)
    return await link_provider_identity(user, 'google', claims, db)


async def link_apple_identity(
    user: User,
    identity_token: str | None,
    authorization_code: str,
    db: AsyncSession,
    nonce: str | None = None,
) -> IdentityRead:
    """Verify and link an Apple identity token.

    Args:
        user: Authenticated local user receiving the identity.
        identity_token: Optional Apple identity token.
        authorization_code: Apple authorisation code.
        db: Database session used to persist the link.
        nonce: Optional expected identity-token nonce.

    Returns:
        Public linked Apple identity.
    """
    claims = await verify_apple_identity_token(
        identity_token,
        authorization_code,
        expected_nonce=nonce,
    )
    return await link_provider_identity(user, 'apple', claims, db)


async def unlink_identity(
    user: User,
    identity_id: int,
    db: AsyncSession,
) -> dict[str, str]:
    """Unlink a provider identity while retaining a login method.

    Args:
        user: Authenticated local user that owns the identity.
        identity_id: Identifier of the identity to remove.
        db: Database session used to remove the identity.

    Returns:
        Confirmation message after unlinking.

    Raises:
        HTTPException: If the identity is absent or is the last login method.
    """
    identity = await db.get(UserIdentity, identity_id)
    if identity is None or identity.user_id != user.id:
        raise HTTPException(status_code=404, detail='Identity not found')

    identity_count = await db.scalar(
        select(func.count(UserIdentity.id)).where(
            UserIdentity.user_id == user.id,
        ),
    )
    if not _user_has_password(user) and int(identity_count or 0) <= 1:
        raise HTTPException(
            status_code=400,
            detail={
                'code': 'last_login_method',
                'message': 'Cannot unlink the last login method.',
            },
        )

    await db.delete(identity)
    await db.commit()
    return {'message': 'Identity unlinked successfully.'}
