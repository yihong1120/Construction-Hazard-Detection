"""OIDC authorization-code + PKCE flow for the browser BFF."""
from __future__ import annotations

import base64
import hashlib
import json
import secrets
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlencode
from urllib.parse import urlsplit

import httpx
from fastapi import HTTPException
from fastapi import Request
from fastapi.responses import RedirectResponse
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from examples.auth.config import Settings
from examples.auth.deployment_context import resolve_request_deployment
from examples.auth.models import User
from examples.auth.oidc import OidcTokenVerifier
from examples.auth.oidc_identity import subject_from_oidc_identity
from examples.auth.session_store import create_auth_session
from examples.bff.schemas import UserSummary
from examples.bff.security import set_session_cookie
from src.http_client_pool import get_application_http_client

settings = Settings()
_access_verifier = OidcTokenVerifier.from_settings(settings)
_STATE_PREFIX = 'bff:oidc:state'


def _state_key(state: str) -> str:
    """Return a Redis key that never contains the raw browser state value."""
    digest = hashlib.sha256(state.encode('utf-8')).hexdigest()
    return f'{_STATE_PREFIX}:{digest}'


def _code_verifier() -> str:
    """Create an RFC 7636-compliant high-entropy PKCE verifier."""
    return secrets.token_urlsafe(64)


def _code_challenge(verifier: str) -> str:
    """Return the S256 PKCE challenge for a verifier."""
    digest = hashlib.sha256(verifier.encode('ascii')).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b'=').decode('ascii')


def _safe_return_to(value: str | None) -> str:
    """Allow only an application-relative post-login browser route."""
    candidate = (value or '/').strip()
    parsed = urlsplit(candidate)
    if (
        not candidate.startswith('/')
        or candidate.startswith('//')
        or parsed.scheme
        or parsed.netloc
    ):
        return '/'
    return candidate


def _require_web_client() -> None:
    """Fail closed until every web authorization-code setting is configured."""
    if not settings.oidc_web_client_configured or _access_verifier is None:
        raise HTTPException(
            status_code=503,
            detail='oidc_web_login_not_configured',
        )


async def _user_summary(
    db: AsyncSession,
    user_id: int,
) -> UserSummary:
    """Load the local profile data stored in the BFF session response."""
    user = await db.scalar(
        select(User)
        .options(selectinload(User.profile))
        .where(User.id == user_id),
    )
    if user is None:
        raise HTTPException(status_code=401, detail='user_not_found')
    profile = user.profile
    display_name = user.username
    if profile is not None:
        display_name = (
            ' '.join(
                part
                for part in (profile.given_name, profile.family_name)
                if part
            )
            or user.username
        )
    return UserSummary(
        id=user.id,
        username=user.username,
        display_name=display_name,
        role=user.role,
        group_id=user.group_id,
        status=user.status,
    )


async def oidc_login_redirect(
    request: Request,
    redis: Redis,
    db: AsyncSession,
    *,
    return_to: str | None,
) -> RedirectResponse:
    """Store single-use state and redirect the browser to the OIDC provider."""
    _require_web_client()
    binding = await resolve_request_deployment(request, db)
    state = secrets.token_urlsafe(32)
    verifier = _code_verifier()
    record = {
        'code_verifier': verifier,
        'deployment': binding.as_response(),
        'return_to': _safe_return_to(return_to),
    }
    await redis.set(
        _state_key(state),
        json.dumps(record, separators=(',', ':')).encode('utf-8'),
        ex=settings.oidc_state_ttl_seconds,
    )
    query = urlencode(
        {
            'client_id': settings.oidc_web_client_id,
            'code_challenge': _code_challenge(verifier),
            'code_challenge_method': 'S256',
            'redirect_uri': settings.oidc_web_redirect_uri,
            'response_type': 'code',
            'scope': 'openid profile email offline_access',
            'state': state,
        },
    )
    return RedirectResponse(
        f'{settings.oidc_web_authorization_endpoint}?{query}',
        status_code=307,
    )


async def _consume_state(redis: Redis, state: str | None) -> dict[str, object]:
    """Read and delete the opaque login state exactly once."""
    if not state:
        raise HTTPException(status_code=400, detail='oidc_state_missing')
    raw = await redis.getdel(_state_key(state))
    if raw is None:
        raise HTTPException(status_code=400, detail='oidc_state_invalid')
    try:
        record = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail='oidc_state_invalid',
        ) from exc
    if not isinstance(record, dict):
        raise HTTPException(status_code=400, detail='oidc_state_invalid')
    return record


async def _post_token_form(form: Mapping[str, str]) -> dict[str, Any]:
    """Exchange an OIDC code or refresh token server-side."""
    _require_web_client()
    client = await get_application_http_client(
        'bff-oidc',
        timeout=httpx.Timeout(10.0, connect=5.0),
    )
    try:
        if client is None:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(10.0, connect=5.0),
                follow_redirects=False,
            ) as transient_client:
                response = await transient_client.post(
                    settings.oidc_web_token_endpoint,
                    data=form,
                    auth=(
                        settings.oidc_web_client_id,
                        settings.oidc_web_client_secret,
                    ),
                )
        else:
            response = await client.post(
                settings.oidc_web_token_endpoint,
                data=form,
                auth=(
                    settings.oidc_web_client_id,
                    settings.oidc_web_client_secret,
                ),
            )
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPStatusError as exc:
        status_code = (
            401
            if 400 <= exc.response.status_code < 500
            else 503
        )
        raise HTTPException(
            status_code=status_code,
            detail='oidc_token_exchange_failed',
        ) from exc
    except (httpx.RequestError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail='oidc_token_exchange_failed',
        ) from exc
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get('access_token'), str)
        or not payload['access_token']
        or not isinstance(payload.get('refresh_token'), str)
        or not payload['refresh_token']
    ):
        raise HTTPException(
            status_code=401,
            detail='oidc_token_exchange_failed',
        )
    return payload


async def refresh_oidc_tokens(refresh_token: str) -> dict[str, Any]:
    """Refresh an OIDC BFF session and revalidate its API access token."""
    payload = await _post_token_form(
        {
            'grant_type': 'refresh_token',
            'refresh_token': refresh_token,
        },
    )
    assert _access_verifier is not None
    await _access_verifier.decode_access_token(str(payload['access_token']))
    return payload


async def complete_oidc_login(
    request: Request,
    redis: Redis,
    db: AsyncSession,
    *,
    code: str | None,
    state: str | None,
) -> RedirectResponse:
    """Exchange a verified callback code for a token-private BFF session."""
    _require_web_client()
    record = await _consume_state(redis, state)
    verifier = record.get('code_verifier')
    stored_deployment = record.get('deployment')
    if (
        not isinstance(verifier, str)
        or not isinstance(stored_deployment, dict)
    ):
        raise HTTPException(status_code=400, detail='oidc_state_invalid')
    if not code:
        raise HTTPException(
            status_code=400,
            detail='oidc_authorization_failed',
        )
    binding = await resolve_request_deployment(request, db)
    if binding.as_response() != stored_deployment:
        raise HTTPException(
            status_code=409,
            detail='deployment_configuration_changed',
        )
    token_pair = await _post_token_form(
        {
            'code': code,
            'code_verifier': verifier,
            'grant_type': 'authorization_code',
            'redirect_uri': settings.oidc_web_redirect_uri,
        },
    )
    assert _access_verifier is not None
    claims = await _access_verifier.decode_access_token(
        str(token_pair['access_token']),
    )
    subject = await subject_from_oidc_identity(
        db,
        claims,
        provider=settings.oidc_identity_provider,
        binding=binding,
    )
    summary: UserSummary = await _user_summary(db, int(subject['user_id']))
    session_id, _ = await create_auth_session(
        redis,
        {
            'access_token': token_pair['access_token'],
            'auth_provider': 'oidc',
            'deployment': binding.as_response(),
            'feature_names': subject['features'],
            'refresh_token': token_pair['refresh_token'],
        },
        summary.model_dump(),
    )
    stored_return_to = record.get('return_to')
    response = RedirectResponse(
        _safe_return_to(
            stored_return_to if isinstance(stored_return_to, str) else None,
        ),
        status_code=303,
    )
    set_session_cookie(response, session_id)
    response.headers['Cache-Control'] = 'no-store'
    return response
