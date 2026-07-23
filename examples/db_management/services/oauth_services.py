from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import Any
from typing import Literal

import httpx
import jwt
from fastapi import HTTPException
from redis.asyncio import Redis
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from examples.auth.config import Settings
from examples.auth.models import User
from examples.auth.models import USER_STATUS_ACTIVE
from examples.auth.models import USER_STATUS_EMAIL_UNVERIFIED
from examples.auth.models import USER_STATUS_INACTIVE
from examples.auth.models import USER_STATUS_PENDING
from examples.auth.models import USER_STATUS_REJECTED
from examples.auth.models import USER_STATUS_SUSPENDED
from examples.auth.models import UserIdentity
from examples.auth.models import UserProfile
from examples.db_management.schemas.auth import IdentityListResponse
from examples.db_management.schemas.auth import IdentityRead
from examples.db_management.schemas.auth import TokenPairData
from examples.db_management.services.auth_services import (
    issue_token_pair_for_user,
)
from examples.db_management.services.legal_services import record_user_consent
from examples.db_management.services.legal_services import SignupConsentPayload
from examples.db_management.services.legal_services import (
    validate_signup_consents,
)

Provider = Literal['google', 'apple']
OAUTH_DISABLED_PASSWORD_HASH = 'oauth_disabled:provider-only'

settings = Settings()

GOOGLE_ISSUERS = ('accounts.google.com', 'https://accounts.google.com')
GOOGLE_JWKS_URL = 'https://www.googleapis.com/oauth2/v3/certs'
APPLE_ISSUER = 'https://appleid.apple.com'
APPLE_JWKS_URL = 'https://appleid.apple.com/auth/keys'
APPLE_TOKEN_URL = 'https://appleid.apple.com/auth/token'


def _configured_google_client_ids() -> list[str]:
    return [
        value.strip()
        for value in settings.google_client_ids.split(',')
        if value.strip()
    ]


def _configured_apple_client_ids() -> list[str]:
    return [
        value.strip()
        for value in settings.apple_client_ids.split(',')
        if value.strip()
    ]


def _bool_claim(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == 'true'
    return False


def _normalise_email(email: object) -> str | None:
    if not isinstance(email, str):
        return None
    normalized = email.strip().lower()
    return normalized or None


def _verify_jwt_with_jwks(
    token: str,
    jwks_url: str,
    audiences: Sequence[str],
    issuers: Sequence[str],
) -> dict[str, Any]:
    if not audiences:
        raise HTTPException(
            status_code=500, detail='OAuth client not configured',
        )

    try:
        signing_key = jwt.PyJWKClient(jwks_url).get_signing_key_from_jwt(token)
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

    if not payload.get('sub'):
        raise HTTPException(
            status_code=401,
            detail='Invalid provider token',
        )
    return dict(payload)


async def verify_google_id_token(id_token: str) -> dict[str, Any]:
    """Verify Google ID token signature, audience, issuer, and expiry."""
    payload = await asyncio.to_thread(
        _verify_jwt_with_jwks,
        id_token,
        GOOGLE_JWKS_URL,
        _configured_google_client_ids(),
        GOOGLE_ISSUERS,
    )
    if not _bool_claim(payload.get('email_verified')):
        raise HTTPException(
            status_code=401,
            detail='Google email is not verified',
        )
    if not _normalise_email(payload.get('email')):
        raise HTTPException(
            status_code=401,
            detail='Google account did not return an email address',
        )
    return payload


def _load_apple_private_key() -> str:
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
) -> dict[str, Any]:
    """Verify Apple identity token and validate authorization code."""
    client_ids = _configured_apple_client_ids()
    payload: dict[str, Any] | None = None
    if identity_token:
        payload = await asyncio.to_thread(
            _verify_jwt_with_jwks,
            identity_token,
            APPLE_JWKS_URL,
            client_ids,
            (APPLE_ISSUER,),
        )
        client_id = str(payload.get('aud', ''))
        if client_id not in client_ids:
            raise HTTPException(
                status_code=401, detail='Invalid provider token',
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

    exchanged_id_token = token_response.get('id_token')
    if isinstance(exchanged_id_token, str) and exchanged_id_token:
        exchanged_payload = await asyncio.to_thread(
            _verify_jwt_with_jwks,
            exchanged_id_token,
            APPLE_JWKS_URL,
            client_ids,
            (APPLE_ISSUER,),
        )
        if payload is None:
            payload = exchanged_payload
        elif exchanged_payload.get('sub') != payload.get('sub'):
            raise HTTPException(
                status_code=401,
                detail='Invalid provider token',
            )
    if payload is None:
        raise HTTPException(status_code=401, detail='Invalid provider token')
    if expected_nonce and payload.get('nonce') != expected_nonce:
        raise HTTPException(status_code=401, detail='Invalid provider token')
    return payload


def _apple_exchange_client_id_candidates() -> list[str]:
    """Try web/service ID first, then native bundle ID for Apple code exchange."""
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
) -> dict[str, Any]:
    """Validate an Apple authorization code against one allowed client id."""
    last_error: HTTPException | None = None
    for client_id in client_ids:
        try:
            return await _exchange_apple_authorization_code_once(
                authorization_code,
                client_id,
            )
        except HTTPException as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise HTTPException(status_code=500, detail='Apple client not configured')


async def _exchange_apple_authorization_code_once(
    authorization_code: str,
    client_id: str,
) -> dict[str, Any]:
    data = {
        'client_id': client_id,
        'client_secret': _build_apple_client_secret(client_id),
        'code': authorization_code,
        'grant_type': 'authorization_code',
    }
    if client_id == settings.apple_service_id:
        data['redirect_uri'] = settings.apple_redirect_uri

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            APPLE_TOKEN_URL,
            data=data,
        )
    if response.status_code >= 400:
        raise HTTPException(status_code=401, detail='Invalid provider token')
    try:
        token_response = response.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=401,
            detail='Invalid provider token',
        ) from exc
    return dict(token_response)


def _status_error(status: str) -> HTTPException:
    if status == USER_STATUS_EMAIL_UNVERIFIED:
        return HTTPException(
            status_code=403,
            detail={'code': 'email_unverified', 'status': status},
        )
    if status == USER_STATUS_PENDING:
        return HTTPException(
            status_code=403,
            detail={'code': 'pending_admin_approval', 'status': status},
        )
    if status in {USER_STATUS_INACTIVE, USER_STATUS_SUSPENDED}:
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
    if user.status != USER_STATUS_ACTIVE:
        raise _status_error(user.status)


def _username_from_claims(provider: Provider, claims: dict[str, Any]) -> str:
    email = _normalise_email(claims.get('email'))
    source = email.split('@', 1)[0] if email else f'{provider}_{claims["sub"]}'
    username = re.sub(r'[^A-Za-z0-9_.-]+', '_', source).strip('._-')
    return username[:64] or f'{provider}_user'


async def _unique_username(
    db: AsyncSession,
    provider: Provider,
    claims: dict[str, Any],
) -> str:
    base = _username_from_claims(provider, claims)
    candidate = base
    suffix = 1
    while await db.scalar(select(User.id).where(User.username == candidate)):
        suffix += 1
        candidate = f'{base[:70]}_{suffix}'
    return candidate


def _profile_names(provider: Provider, claims: dict[str, Any]) -> tuple[str, str]:
    given_name = str(claims.get('given_name') or '').strip()
    family_name = str(claims.get('family_name') or '').strip()
    if given_name or family_name:
        return (family_name or provider.title())[:50], (given_name or 'User')[:50]

    name = str(claims.get('name') or '').strip()
    if name:
        parts = name.split()
        if len(parts) > 1:
            return parts[0][:50], ' '.join(parts[1:])[:50]
        return provider.title(), parts[0][:50]
    email = _normalise_email(claims.get('email'))
    if email:
        return provider.title(), email.split('@', 1)[0][:50]
    return provider.title(), 'User'


def _new_identity(
    user: User,
    provider: Provider,
    claims: dict[str, Any],
) -> UserIdentity:
    email = _normalise_email(claims.get('email'))
    return UserIdentity(
        user=user,
        provider=provider,
        provider_user_id=str(claims['sub']),
        email=email,
        email_verified=_bool_claim(claims.get('email_verified')),
        display_name=str(
            claims.get('name')
            or ' '.join(
                part
                for part in [
                    str(claims.get('given_name') or '').strip(),
                    str(claims.get('family_name') or '').strip(),
                ]
                if part
            )
            or '',
        ) or None,
        raw_profile=dict(claims),
        raw_email_is_private=(
            _bool_claim(claims.get('is_private_email'))
            or bool(email and email.endswith('@privaterelay.appleid.com'))
        ),
    )


def _display_name_from_claims(claims: dict[str, Any]) -> str | None:
    display_name = str(
        claims.get('name')
        or ' '.join(
            part
            for part in [
                str(claims.get('given_name') or '').strip(),
                str(claims.get('family_name') or '').strip(),
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
    return await db.scalar(
        select(User)
        .join(UserProfile, UserProfile.user_id == User.id)
        .where(func.lower(UserProfile.email) == email.lower()),
    )


async def _create_pending_user_with_identity(
    db: AsyncSession,
    provider: Provider,
    claims: dict[str, Any],
) -> User:
    email = _normalise_email(claims.get('email'))
    if not email:
        raise HTTPException(
            status_code=400,
            detail='Provider account did not return an email address',
        )

    family_name, given_name = _profile_names(provider, claims)
    user = User(
        username=await _unique_username(db, provider, claims),
        password_hash=OAUTH_DISABLED_PASSWORD_HASH,
        role='user',
        status=USER_STATUS_PENDING,
        email_verified_at=datetime.now(timezone.utc),
        group_id=None,
    )
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
    await db.refresh(user, attribute_names=['profile', 'group'])
    return user


async def authenticate_provider_user(
    provider: Provider,
    claims: dict[str, Any],
    db: AsyncSession,
    redis_pool: Redis,
    consent_payload: SignupConsentPayload | None = None,
    hash_refresh_token: bool = False,
) -> TokenPairData:
    """Resolve a verified provider identity to a local user and issue tokens."""
    provider_user_id = str(claims.get('sub') or '')
    if not provider_user_id:
        raise HTTPException(status_code=401, detail='Invalid provider token')

    user = await _find_identity_user(db, provider, provider_user_id)
    if user is not None:
        _ensure_active_user(user)
        return await issue_token_pair_for_user(
            user,
            db,
            redis_pool,
            hash_refresh_token=hash_refresh_token,
        )

    email = _normalise_email(claims.get('email'))
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
    user = await _create_pending_user_with_identity(db, provider, claims)
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
) -> TokenPairData:
    claims = await verify_google_id_token(id_token)
    if display_name and not claims.get('name'):
        claims['name'] = display_name
    if device_lang:
        claims['device_lang'] = device_lang
    return await authenticate_provider_user(
        'google',
        claims,
        db,
        redis_pool,
        consent_payload=consent_payload,
        hash_refresh_token=hash_refresh_token,
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
) -> TokenPairData:
    claims = await verify_apple_identity_token(
        identity_token,
        authorization_code,
        expected_nonce=nonce,
    )
    if email and not claims.get('email'):
        claims['email'] = email
    if given_name and not claims.get('given_name'):
        claims['given_name'] = given_name
    if family_name and not claims.get('family_name'):
        claims['family_name'] = family_name
    if device_lang:
        claims['device_lang'] = device_lang
    return await authenticate_provider_user(
        'apple',
        claims,
        db,
        redis_pool,
        consent_payload=consent_payload,
        hash_refresh_token=hash_refresh_token,
    )


def _user_has_password(user: User) -> bool:
    return not str(user.password_hash).startswith('oauth_disabled:')


def _identity_read(identity: UserIdentity) -> IdentityRead:
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
    identities = (
        await db.execute(
            select(UserIdentity)
            .where(UserIdentity.user_id == user.id)
            .order_by(UserIdentity.linked_at.asc()),
        )
    ).scalars().all()
    return IdentityListResponse(
        identities=[_identity_read(identity) for identity in identities],
        has_password=_user_has_password(user),
    )


async def _find_identity(
    db: AsyncSession,
    provider: Provider,
    provider_user_id: str,
) -> UserIdentity | None:
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
    return await db.scalar(
        select(UserIdentity).where(
            UserIdentity.user_id == user.id,
            UserIdentity.provider == provider,
        ),
    )


def _update_identity_from_claims(
    identity: UserIdentity,
    claims: dict[str, Any],
) -> None:
    email = _normalise_email(claims.get('email'))
    if email:
        identity.email = email
    identity.email_verified = _bool_claim(claims.get('email_verified'))
    display_name = _display_name_from_claims(claims)
    if display_name:
        identity.display_name = display_name
    identity.raw_profile = dict(claims)
    identity.raw_email_is_private = (
        _bool_claim(claims.get('is_private_email'))
        or bool(
            identity.email and identity.email.endswith(
                '@privaterelay.appleid.com',
            ),
        )
    )


async def link_provider_identity(
    user: User,
    provider: Provider,
    claims: dict[str, Any],
    db: AsyncSession,
) -> IdentityRead:
    provider_user_id = str(claims.get('sub') or '')
    if not provider_user_id:
        raise HTTPException(status_code=401, detail='Invalid provider token')

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
                    f'{provider}.'
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
    claims = await verify_google_id_token(id_token)
    return await link_provider_identity(user, 'google', claims, db)


async def link_apple_identity(
    user: User,
    identity_token: str | None,
    authorization_code: str,
    db: AsyncSession,
    nonce: str | None = None,
) -> IdentityRead:
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
