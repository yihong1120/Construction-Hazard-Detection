from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
from urllib.parse import parse_qsl
from urllib.parse import urlencode
from urllib.parse import urlsplit
from urllib.parse import urlunsplit

from fastapi import HTTPException
from fastapi import Request
from fastapi.responses import RedirectResponse
from pydantic import ValidationError
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from examples.auth.models import User
from examples.auth.session_store import get_auth_session
from examples.auth.session_store import revoke_media_for_parent
from examples.db_management.schemas.auth import RefreshRequest
from examples.db_management.schemas.auth import TokenPairData
from examples.db_management.schemas.oauth import AuthSession
from examples.db_management.schemas.oauth import MeResponse
from examples.db_management.schemas.oauth import NativeOAuthClients
from examples.db_management.schemas.oauth import OAuthAuthorizationCode
from examples.db_management.schemas.oauth import (
    OAuthAuthorizationCodeRequest,
)
from examples.db_management.schemas.oauth import OAuthRefreshTokenRequest
from examples.db_management.schemas.oauth import OAuthRequestParameters
from examples.db_management.schemas.oauth import OAuthRevocationRequest
from examples.db_management.schemas.oauth import OAuthTokenRequest
from examples.db_management.schemas.oauth import OAuthTokenResponse
from examples.db_management.services.auth_services import _load_feature_names
from examples.db_management.services.auth_services import ACCESS_TTL
from examples.db_management.services.auth_services import (
    issue_token_pair_for_user,
)
from examples.db_management.services.auth_services import logout_user
from examples.db_management.services.auth_services import refresh_tokens

SESSION_COOKIE = os.getenv('BFF_SESSION_COOKIE_NAME', '__Host-vn_session')
AUTH_CODE_TTL_SECONDS = int(os.getenv('OAUTH_CODE_TTL_SECONDS', '120'))
PKCE_VERIFIER_RE = re.compile(r'^[A-Za-z0-9._~-]{43,128}$')


def native_clients() -> dict[str, set[str]]:
    """Load configured native OAuth clients and redirect URIs.

    Returns:
        Client identifiers mapped to exact permitted redirect URIs.
    """
    configured: dict[str, list[str]] = {
        'visionnaire-ios': ['com.changdar.visionnaire:/oauth2redirect'],
        'visionnaire-android': ['com.changdar.visionnaire:/oauth2redirect'],
    }
    raw = os.getenv('OAUTH_NATIVE_CLIENTS_JSON', '').strip()
    if raw:
        try:
            configured = NativeOAuthClients.model_validate(
                json.loads(raw),
            ).root
        except (json.JSONDecodeError, ValidationError) as exc:
            raise RuntimeError('Invalid OAUTH_NATIVE_CLIENTS_JSON') from exc
    return {
        client_id: set(redirect_uris)
        for client_id, redirect_uris in configured.items()
    }


def validate_client(client_id: str, redirect_uri: str) -> None:
    """Require an exact registered redirect URI for a native client.

    Args:
        client_id: Native OAuth client identifier.
        redirect_uri: Callback URI supplied by the client.

    Raises:
        HTTPException: If the client or redirect URI is not configured.
    """
    if redirect_uri not in native_clients().get(client_id, set()):
        raise HTTPException(status_code=400, detail='invalid_oauth_client')


def _code_key(code: str) -> str:
    """Build the Redis key for a one-use authorisation code.

    Args:
        code: Raw authorisation code.

    Returns:
        Redis code-record key.
    """
    return f'oauth:authorization-code:{hashlib.sha256(code.encode()).hexdigest()}'


def _append_query(uri: str, **values: str) -> str:
    """Append encoded query values while retaining an existing query string.

    Args:
        uri: Base URI that may already include a query string.

    Returns:
        Function accepting additional query parameters and returning a URI.
    """
    parsed = urlsplit(uri)
    query = parse_qsl(parsed.query, keep_blank_values=True)
    query.extend(values.items())
    return urlunsplit(parsed._replace(query=urlencode(query)))


def pkce_challenge(verifier: str) -> str:
    """Build the S256 PKCE challenge for a verifier.

    Args:
        verifier: Client-generated PKCE verifier.

    Returns:
        URL-safe base64 S256 challenge without padding.
    """
    digest = hashlib.sha256(verifier.encode('ascii')).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b'=').decode()


async def request_data(request: Request) -> dict[str, str]:
    """Read strict string-only OAuth body parameters.

    Args:
        request: Form or JSON OAuth request.

    Returns:
        Validated mapping of OAuth parameter names to string values.

    Raises:
        HTTPException: If the body cannot be parsed as string-only data.
    """
    payload = (
        await request.json()
        if 'application/json' in request.headers.get(
            'content-type', '',
        ).lower()
        else dict(await request.form())
    )
    try:
        return OAuthRequestParameters.model_validate(payload).root
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail='invalid_request') from exc


def _token_response(result: TokenPairData) -> OAuthTokenResponse:
    """Convert issued token data into the OAuth response schema.

    Args:
        result: Token-pair data issued by the authentication service.

    Returns:
        OAuth-compatible token response.
    """
    return OAuthTokenResponse(
        access_token=result['access_token'],
        refresh_token=result['refresh_token'],
        expires_in=int(ACCESS_TTL.total_seconds()),
    )


async def authorize_native_app(
    request: Request,
    response_type: str,
    client_id: str,
    redirect_uri: str,
    code_challenge: str,
    code_challenge_method: str,
    state: str,
    redis: Redis,
) -> RedirectResponse:
    """Issue a one-use PKCE code for an authenticated native app.

    Args:
        request: Authenticated browser request carrying the BFF session.
        response_type: Requested OAuth response type.
        client_id: Native OAuth client identifier.
        redirect_uri: Native-app callback URI.
        code_challenge: PKCE challenge bound to the code.
        code_challenge_method: PKCE challenge method.
        state: Opaque client state returned to the callback.
        redis: Redis connection used to store code state.

    Returns:
        Redirect to the client's validated callback URI.
    """
    validate_client(client_id, redirect_uri)
    if response_type != 'code' or code_challenge_method != 'S256':
        raise HTTPException(status_code=400, detail='pkce_s256_required')
    if not re.fullmatch(r'[A-Za-z0-9_-]{43,128}', code_challenge):
        raise HTTPException(status_code=400, detail='invalid_code_challenge')
    session = await get_auth_session(redis, request.cookies.get(SESSION_COOKIE))
    if session is None:
        raise HTTPException(status_code=401, detail='login_required')
    try:
        auth_session = AuthSession.model_validate(session)
    except ValidationError:
        raise HTTPException(status_code=401, detail='login_required')
    code = secrets.token_urlsafe(32)
    record = OAuthAuthorizationCode(
        user_id=auth_session.user.id,
        client_id=client_id,
        redirect_uri=redirect_uri,
        code_challenge=code_challenge,
    )
    await redis.set(
        _code_key(code),
        record.model_dump_json().encode('utf-8'),
        ex=AUTH_CODE_TTL_SECONDS,
        nx=True,
    )
    values = {'code': code}
    if state:
        values['state'] = state
    return RedirectResponse(_append_query(redirect_uri, **values), 302)


async def exchange_native_token(
    request: Request,
    db: AsyncSession,
    redis: Redis,
) -> OAuthTokenResponse:
    """Exchange a PKCE code or rotate a native refresh token.

    Args:
        request: OAuth token request with form or JSON grant data.
        db: Database session used for authorised-user lookup.
        redis: Redis connection holding grant and token state.

    Returns:
        Newly issued native OAuth token pair.
    """
    data = await request_data(request)
    try:
        grant_request = OAuthTokenRequest.model_validate(data)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail='invalid_request') from exc
    if grant_request.grant_type == 'authorization_code':
        try:
            exchange = OAuthAuthorizationCodeRequest.model_validate(data)
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail='invalid_grant') from exc
        return await _exchange_authorization_code(exchange, db, redis)
    if grant_request.grant_type == 'refresh_token':
        if grant_request.client_id not in native_clients():
            raise HTTPException(status_code=400, detail='invalid_oauth_client')
        try:
            exchange = OAuthRefreshTokenRequest.model_validate(data)
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail='invalid_grant') from exc
        return await _exchange_refresh_token(exchange, redis)
    raise HTTPException(status_code=400, detail='unsupported_grant_type')


async def _exchange_authorization_code(
    data: OAuthAuthorizationCodeRequest,
    db: AsyncSession,
    redis: Redis,
) -> OAuthTokenResponse:
    """Exchange a validated PKCE authorisation-code request.

    Args:
        data: Parsed OAuth authorisation-code grant fields.
        db: Database session used to load the authorised user.
        redis: Redis connection holding one-use code state.

    Returns:
        Newly issued native OAuth token pair.
    """
    client_id = data.client_id
    if client_id is None:
        raise HTTPException(status_code=400, detail='invalid_client')
    validate_client(client_id, data.redirect_uri)
    if not PKCE_VERIFIER_RE.fullmatch(data.code_verifier):
        raise HTTPException(status_code=400, detail='invalid_grant')
    stored = await _load_authorization_code(redis, data.code)
    if (
        stored.client_id != client_id
        or stored.redirect_uri != data.redirect_uri
        or not hmac.compare_digest(
            stored.code_challenge,
            pkce_challenge(data.code_verifier),
        )
    ):
        raise HTTPException(status_code=400, detail='invalid_grant')
    user = await db.scalar(select(User).where(User.id == stored.user_id))
    if user is None or user.status != 'active':
        raise HTTPException(status_code=400, detail='invalid_grant')
    return _token_response(await issue_token_pair_for_user(user, db, redis))


async def _load_authorization_code(
    redis: Redis,
    code: str,
) -> OAuthAuthorizationCode:
    """Load and consume a one-use authorisation code from Redis.

    Args:
        redis: Redis connection holding code state.
        code: Raw one-use authorisation code.

    Returns:
        Validated stored authorisation-code record.

    Raises:
        HTTPException: If the code is unknown, expired, or malformed.
    """
    raw = await redis.getdel(_code_key(code))
    if raw is None:
        raise HTTPException(status_code=400, detail='invalid_grant')
    try:
        return OAuthAuthorizationCode.model_validate_json(raw)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail='invalid_grant') from exc


async def _exchange_refresh_token(
    data: OAuthRefreshTokenRequest,
    redis: Redis,
) -> OAuthTokenResponse:
    """Rotate a native OAuth refresh token into a new token pair.

    Args:
        data: Parsed refresh-token grant fields.
        redis: Redis connection holding token-rotation state.

    Returns:
        Newly issued native OAuth token pair.
    """
    result = await refresh_tokens(
        RefreshRequest(refresh_token=data.refresh_token),
        redis,
    )
    return _token_response(result)


async def current_user_profile(
    user: User,
    db: AsyncSession,
) -> MeResponse:
    """Load an active user and construct their native-app profile response.

    Args:
        user: Authenticated user from the access-token dependency.
        db: Database session used to load profile details.

    Returns:
        Public native OAuth profile.
    """
    loaded = await db.scalar(
        select(User)
        .options(selectinload(User.profile))
        .where(User.id == user.id),
    )
    if loaded is None or loaded.status != 'active':
        raise HTTPException(status_code=401, detail='invalid_user')
    display_name = (
        f'{loaded.profile.given_name} {loaded.profile.family_name}'.strip()
    )
    return MeResponse(
        id=loaded.id,
        username=loaded.username,
        display_name=display_name,
        role=loaded.role,
        group_id=loaded.group_id,
        status=loaded.status,
        feature_names=await _load_feature_names(db, loaded.group_id),
    )


async def revoke_native_token(request: Request, redis: Redis) -> None:
    """Best-effort revoke a native token from body or authorisation header.

    Args:
        request: OAuth revocation request.
        redis: Redis connection holding token state.
    """
    try:
        data = OAuthRevocationRequest.model_validate(await request_data(request))
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail='invalid_request') from exc
    token_value = data.token or ''
    if token_value:
        if data.token_type_hint == 'access_token':
            await logout_user(None, f'Bearer {token_value}', redis)
        else:
            await logout_user(token_value, None, redis)
        await revoke_media_for_parent(
            redis,
            f'native:{hashlib.sha256(token_value.encode()).hexdigest()}',
        )
    authorization = request.headers.get('authorization')
    if authorization:
        _, _, bearer = authorization.partition(' ')
        if bearer:
            await logout_user(None, authorization, redis)
            await revoke_media_for_parent(
                redis,
                f'native:{hashlib.sha256(bearer.encode()).hexdigest()}',
            )
