"""Secure native Google/Apple assertion exchange and account linking.

The public Keycloak Token Exchange v1 grant is deprecated and unsuitable for
mobile public clients.  This module therefore treats a provider assertion as a
short-lived, one-use *proof* only.  Keycloak receives the proof through its
private authenticator, resolves an already linked federated identity, and then
issues its normal Authorization Code + PKCE response (including refresh-token
support) to the Flutter client.

No provider token, Apple code, email address, or Keycloak administrative token
is ever written to Redis or returned to a client.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import json
import re
import secrets
import time
from collections.abc import Mapping
from typing import Literal
from urllib.parse import quote
from urllib.parse import urlencode

import httpx
from fastapi import HTTPException
from fastapi import Request
from redis.asyncio import Redis

from examples.auth.config import Settings
from examples.auth.jwt_config import JwtAuthorizationCredentials
from examples.db_management.schemas.auth import NativeSocialCredential
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
from examples.db_management.services.oauth_services import ProviderClaims
from examples.db_management.services.oauth_services import (
    verify_apple_identity_token,
)
from examples.db_management.services.oauth_services import (
    verify_google_id_token,
)

NativeSocialProvider = Literal['google', 'apple']

settings = Settings()

_TRANSACTION_RE = re.compile(r'^[A-Za-z0-9_-]{43,128}$')
_KEYCLOAK_SUBJECT_RE = re.compile(r'^[A-Za-z0-9._:-]{1,255}$')
_REDEEM_HMAC_MAX_AGE_SECONDS = 30
_BEGIN_RATE_LIMIT_MAX = 20
_BEGIN_RATE_LIMIT_WINDOW_SECONDS = 300


def _not_enabled() -> HTTPException:
    """Return the consistent disabled-feature response."""
    return HTTPException(
        status_code=404,
        detail='native_social_exchange_not_enabled',
    )


def _require_enabled() -> None:
    """Fail closed unless the server has every native-exchange secret."""
    if not settings.native_social_exchange_enabled:
        raise _not_enabled()


def _opaque_identifier() -> str:
    """Create an unguessable URL-safe transaction identifier."""
    return secrets.token_urlsafe(32)


def _transaction_key(transaction_id: str) -> str:
    """Hash public transaction IDs before using them as Redis keys."""
    digest = hashlib.sha256(transaction_id.encode('ascii')).hexdigest()
    return f'native-social:transaction:{digest}'


def _rate_limit_key(request: Request) -> str:
    """Build a privacy-preserving anonymous begin rate-limit key."""
    host = request.client.host if request.client else 'unknown'
    digest = hashlib.sha256(host.encode('utf-8')).hexdigest()
    return f'native-social:begin-rate:{digest}'


async def _enforce_begin_rate_limit(
    request: Request,
    redis: Redis,
) -> None:
    """Bound unauthenticated transaction allocation per trusted peer."""
    key = _rate_limit_key(request)
    count = await redis.incr(key)
    if count == 1:
        await redis.expire(key, _BEGIN_RATE_LIMIT_WINDOW_SECONDS)
    if count > _BEGIN_RATE_LIMIT_MAX:
        retry_after = await redis.ttl(key)
        raise HTTPException(
            status_code=429,
            detail='native_social_exchange_rate_limited',
            headers={
                'Retry-After': str(
                    max(1, retry_after)
                    if isinstance(retry_after, int)
                    else _BEGIN_RATE_LIMIT_WINDOW_SECONDS,
                ),
            },
        )


def _validate_client_binding(
    client_id: str,
    redirect_uri: str,
) -> None:
    """Require one of the server-configured Keycloak public clients."""
    allowed = settings.native_social_allowed_clients
    if redirect_uri not in allowed.get(client_id, ()):
        raise HTTPException(status_code=400, detail='invalid_oidc_client')


def _decode_record(raw: bytes | str | None) -> dict[str, object]:
    """Decode a Redis transaction record without exposing malformed state."""
    if raw is None:
        raise HTTPException(
            status_code=401, detail='native_social_exchange_expired',
        )
    try:
        value = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=401,
            detail='native_social_exchange_expired',
        ) from exc
    if not isinstance(value, dict):
        raise HTTPException(
            status_code=401, detail='native_social_exchange_expired',
        )
    return value


async def _consume_transaction(
    redis: Redis,
    transaction_id: str,
) -> dict[str, object]:
    """Atomically consume a one-use transaction before validation work."""
    if not _TRANSACTION_RE.fullmatch(transaction_id):
        raise HTTPException(
            status_code=401, detail='native_social_exchange_expired',
        )
    return _decode_record(await redis.getdel(_transaction_key(transaction_id)))


async def _store_transaction(
    redis: Redis,
    transaction_id: str,
    payload: Mapping[str, object],
    ttl_seconds: int,
) -> None:
    """Store compact, non-secret transaction state with an absolute TTL."""
    await redis.set(
        _transaction_key(transaction_id),
        json.dumps(payload, separators=(',', ':')).encode('utf-8'),
        ex=ttl_seconds,
    )


def _required_string(record: Mapping[str, object], name: str) -> str:
    """Return a bounded string field from trusted transaction state."""
    value = record.get(name)
    if not isinstance(value, str) or not value:
        raise HTTPException(
            status_code=401, detail='native_social_exchange_expired',
        )
    return value


def _record_provider(record: Mapping[str, object]) -> NativeSocialProvider:
    """Return the fixed provider stored in a transaction."""
    provider = record.get('provider')
    if provider not in {'google', 'apple'}:
        raise HTTPException(
            status_code=401, detail='native_social_exchange_expired',
        )
    return provider


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
                status_code=401, detail='invalid_provider_credential',
            )
        return await verify_google_id_token(
            credential.id_token,
            expected_nonce=nonce,
            require_verified_email=False,
        )
    if not credential.authorization_code:
        raise HTTPException(
            status_code=401, detail='invalid_provider_credential',
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
            status_code=401, detail='invalid_provider_credential',
        )
    return subject


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
) -> NativeSocialExchangeCompleteResponse:
    """Validate native credentials and hand a one-use proof to Keycloak."""
    _require_enabled()
    record = await _consume_transaction(redis, payload.transaction_id)
    if record.get('kind') != 'login-begin':
        raise HTTPException(
            status_code=401, detail='native_social_exchange_expired',
        )
    provider = _record_provider(record)
    nonce = _required_string(record, 'nonce')
    claims = await _verify_credential(provider, payload, nonce)
    subject = _provider_subject(claims)

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
            ).rstrip(b'=').decode('ascii'),
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
            status_code=401, detail='keycloak_reauthentication_required',
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
            status_code=401, detail='keycloak_reauthentication_required',
        )
    sid = payload.get('sid') or payload.get('session_state')
    if sid is not None and not isinstance(sid, str):
        raise HTTPException(
            status_code=401, detail='keycloak_reauthentication_required',
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
) -> NativeSocialLinkResponse:
    """Validate provider proof and attach it to the current Keycloak user."""
    _require_enabled()
    current_subject, current_session_id = _keycloak_identity_from_credentials(
        credentials,
    )
    record = await _consume_transaction(redis, payload.transaction_id)
    if record.get('kind') != 'link-begin':
        raise HTTPException(
            status_code=401, detail='native_social_link_expired',
        )
    if not hmac.compare_digest(
        _required_string(record, 'keycloak_subject'),
        current_subject,
    ):
        raise HTTPException(
            status_code=401, detail='native_social_link_expired',
        )
    stored_session_id = record.get('keycloak_session_id')
    if isinstance(stored_session_id, str):
        if (
            not isinstance(current_session_id, str)
            or not hmac.compare_digest(stored_session_id, current_session_id)
        ):
            raise HTTPException(
                status_code=401,
                detail='native_social_link_expired',
            )
    provider = _record_provider(record)
    claims = await _verify_credential(
        provider,
        payload,
        _required_string(record, 'nonce'),
    )
    status = await _link_keycloak_federated_identity(
        keycloak_subject=current_subject,
        provider=provider,
        provider_subject=_provider_subject(claims),
    )
    return NativeSocialLinkResponse(provider=provider, status=status)


def _is_loopback_request(request: Request) -> bool:
    """Return whether a Keycloak redemption request came from loopback."""
    if request.client is None:
        return False
    try:
        return ipaddress.ip_address(request.client.host).is_loopback
    except ValueError:
        return request.client.host.lower() == 'localhost'


def _hmac_signature(timestamp: str, body: bytes) -> str:
    """Build the canonical HMAC for a private Keycloak redemption request."""
    digest = hmac.digest(
        settings.native_social_exchange_shared_secret.encode('utf-8'),
        timestamp.encode('ascii') + b'.' + body,
        'sha256',
    )
    return base64.urlsafe_b64encode(digest).rstrip(b'=').decode('ascii')


def _parse_redeem_body(body: bytes) -> dict[str, str]:
    """Parse the custom Keycloak request as a strict, string-only object."""
    try:
        data = json.loads(body)
    except (UnicodeDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=401, detail='native_social_exchange_invalid',
        ) from exc
    if not isinstance(data, dict):
        raise HTTPException(
            status_code=401, detail='native_social_exchange_invalid',
        )
    fields = ('transaction_id', 'client_id', 'redirect_uri', 'code_challenge')
    if set(data) != set(fields) or any(
        not isinstance(data.get(field), str) or not data[field]
        for field in fields
    ):
        raise HTTPException(
            status_code=401, detail='native_social_exchange_invalid',
        )
    return {field: data[field] for field in fields}


async def redeem_keycloak_native_social_exchange(
    request: Request,
    redis: Redis,
) -> dict[str, str]:
    """Redeem one proof for the Keycloak authenticator over loopback HMAC."""
    _require_enabled()
    if not _is_loopback_request(request):
        raise HTTPException(
            status_code=403, detail='native_social_exchange_forbidden',
        )
    timestamp = request.headers.get('X-Visionnaire-Timestamp', '')
    signature = request.headers.get('X-Visionnaire-Signature', '')
    if not timestamp.isdecimal():
        raise HTTPException(
            status_code=401, detail='native_social_exchange_invalid',
        )
    if abs(time.time() - int(timestamp)) > _REDEEM_HMAC_MAX_AGE_SECONDS:
        raise HTTPException(
            status_code=401, detail='native_social_exchange_invalid',
        )
    body = await request.body()
    if not hmac.compare_digest(_hmac_signature(timestamp, body), signature):
        raise HTTPException(
            status_code=401, detail='native_social_exchange_invalid',
        )
    payload = _parse_redeem_body(body)
    record = await _consume_transaction(redis, payload['transaction_id'])
    if record.get('kind') != 'login-redeem':
        raise HTTPException(
            status_code=401, detail='native_social_exchange_invalid',
        )
    for name in ('client_id', 'redirect_uri', 'code_challenge'):
        if not hmac.compare_digest(
            _required_string(record, name),
            payload[name],
        ):
            raise HTTPException(
                status_code=401, detail='native_social_exchange_invalid',
            )
    encoded_subject = _required_string(record, 'provider_subject_b64')
    try:
        padded = encoded_subject + '=' * (-len(encoded_subject) % 4)
        subject = base64.urlsafe_b64decode(
            padded.encode('ascii'),
        ).decode('utf-8')
    except (UnicodeDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=401, detail='native_social_exchange_invalid',
        ) from exc
    if not subject or len(subject.encode('utf-8')) > 512:
        raise HTTPException(
            status_code=401, detail='native_social_exchange_invalid',
        )
    return {
        'provider': _record_provider(record),
        'provider_subject_b64': encoded_subject,
    }


async def _keycloak_service_access_token() -> str:
    """Acquire a short-lived Admin API token without storing it in Redis."""
    token_url = (
        f'{settings.resolved_keycloak_admin_base_url}/realms/'
        f'{quote(settings.keycloak_realm, safe="")}/'
        'protocol/openid-connect/token'
    )
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                token_url,
                data={
                    'grant_type': 'client_credentials',
                    'client_id': settings.keycloak_user_linker_client_id,
                    'client_secret': (
                        settings.keycloak_user_linker_client_secret
                    ),
                },
            )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=503,
            detail='keycloak_identity_service_unavailable',
        ) from exc
    if response.status_code != 200:
        raise HTTPException(
            status_code=503,
            detail='keycloak_identity_service_unavailable',
        )
    try:
        token = response.json().get('access_token')
    except ValueError as exc:
        raise HTTPException(
            status_code=503,
            detail='keycloak_identity_service_unavailable',
        ) from exc
    if not isinstance(token, str) or not token:
        raise HTTPException(
            status_code=503,
            detail='keycloak_identity_service_unavailable',
        )
    return token


async def _keycloak_admin_request(
    method: str,
    path: str,
    *,
    json_body: Mapping[str, object] | None = None,
) -> httpx.Response:
    """Perform one authenticated Keycloak Admin API request."""
    token = await _keycloak_service_access_token()
    url = (
        f'{settings.resolved_keycloak_admin_base_url}/admin/realms/'
        f'{quote(settings.keycloak_realm, safe="")}{path}'
    )
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            return await client.request(
                method,
                url,
                headers={'Authorization': f'Bearer {token}'},
                json=json_body,
            )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=503,
            detail='keycloak_identity_service_unavailable',
        ) from exc


async def _link_keycloak_federated_identity(
    *,
    keycloak_subject: str,
    provider: NativeSocialProvider,
    provider_subject: str,
) -> Literal['linked', 'already_linked']:
    """Attach an immutable provider subject using Keycloak's Admin API.

    The backend selects the target solely from the verified current Keycloak
    token.  It never accepts a user ID or an email in the client payload.
    """
    user_path = f'/users/{quote(keycloak_subject, safe="")}/federated-identity'
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
            or identity.get('identityProvider') != provider
        ):
            continue
        if identity.get('userId') == provider_subject:
            return 'already_linked'
        raise HTTPException(status_code=409, detail='provider_already_linked')

    response = await _keycloak_admin_request(
        'POST',
        f'{user_path}/{provider}',
        json_body={
            'identityProvider': provider,
            'userId': provider_subject,
            # Keep the mutable email/name out of Keycloak's broker record.
            'userName': provider_subject,
        },
    )
    if response.status_code == 204:
        return 'linked'
    if response.status_code == 409:
        # A race can mean this user finished the same link in another window.
        # Re-read only once; do not retry POST and risk a broad admin action.
        latest = await _keycloak_admin_request('GET', user_path)
        if latest.status_code == 200:
            try:
                identities = latest.json()
            except ValueError:
                identities = []
            if isinstance(identities, list) and any(
                isinstance(identity, Mapping)
                and identity.get('identityProvider') == provider
                and identity.get('userId') == provider_subject
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
