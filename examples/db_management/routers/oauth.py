from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
from collections.abc import Mapping
from typing import Any
from urllib.parse import parse_qsl
from urllib.parse import urlencode
from urllib.parse import urlsplit
from urllib.parse import urlunsplit

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Request
from fastapi.responses import RedirectResponse
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from examples.auth.database import get_db
from examples.auth.models import User
from examples.auth.redis_pool import get_redis_pool
from examples.auth.session_store import get_auth_session
from examples.auth.session_store import revoke_media_for_parent
from examples.db_management.deps import get_current_user
from examples.db_management.schemas.auth import RefreshRequest
from examples.db_management.schemas.oauth import MeResponse
from examples.db_management.schemas.oauth import OAuthTokenResponse
from examples.db_management.services.auth_services import _load_feature_names
from examples.db_management.services.auth_services import ACCESS_TTL
from examples.db_management.services.auth_services import (
    issue_token_pair_for_user,
)
from examples.db_management.services.auth_services import logout_user
from examples.db_management.services.auth_services import refresh_tokens

router = APIRouter(tags=['oauth'])
SESSION_COOKIE = os.getenv('BFF_SESSION_COOKIE_NAME', '__Host-vn_session')
AUTH_CODE_TTL_SECONDS = int(os.getenv('OAUTH_CODE_TTL_SECONDS', '120'))
PKCE_VERIFIER_RE = re.compile(r'^[A-Za-z0-9._~-]{43,128}$')


def _native_clients() -> dict[str, set[str]]:
    default = {
        'visionnaire-ios': [
            'com.changdar.visionnaire:/oauth2redirect',
        ],
        'visionnaire-android': [
            'com.changdar.visionnaire:/oauth2redirect',
        ],
    }
    raw = os.getenv('OAUTH_NATIVE_CLIENTS_JSON', '').strip()
    if raw:
        try:
            configured = json.loads(raw)
            if isinstance(configured, dict):
                default = configured
        except json.JSONDecodeError as exc:
            raise RuntimeError('Invalid OAUTH_NATIVE_CLIENTS_JSON') from exc
    return {
        str(client_id): {str(uri) for uri in redirect_uris}
        for client_id, redirect_uris in default.items()
        if isinstance(redirect_uris, list)
    }


def _validate_client(client_id: str, redirect_uri: str) -> None:
    if redirect_uri not in _native_clients().get(client_id, set()):
        raise HTTPException(status_code=400, detail='invalid_oauth_client')


def _code_key(code: str) -> str:
    digest = hashlib.sha256(code.encode()).hexdigest()
    return f'oauth:authorization-code:{digest}'


def _append_query(uri: str, **values: str) -> str:
    parsed = urlsplit(uri)
    query = parse_qsl(parsed.query, keep_blank_values=True)
    query.extend(values.items())
    return urlunsplit(parsed._replace(query=urlencode(query)))


def _pkce_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode('ascii')).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b'=').decode()


async def _request_data(request: Request) -> dict[str, Any]:
    content_type = request.headers.get('content-type', '').lower()
    if 'application/json' in content_type:
        payload = await request.json()
        return payload if isinstance(payload, dict) else {}
    form = await request.form()
    return {key: str(value) for key, value in form.items()}


def _token_response(result: Mapping[str, object]) -> OAuthTokenResponse:
    return OAuthTokenResponse(
        access_token=str(result['access_token']),
        refresh_token=str(result['refresh_token']),
        expires_in=int(ACCESS_TTL.total_seconds()),
    )


@router.get('/oauth/authorize')
async def authorize(
    request: Request,
    response_type: str,
    client_id: str,
    redirect_uri: str,
    code_challenge: str,
    code_challenge_method: str,
    state: str = '',
    redis: Redis = Depends(get_redis_pool),
) -> RedirectResponse:
    """Issue a one-use code to an authenticated system browser."""
    _validate_client(client_id, redirect_uri)
    if response_type != 'code' or code_challenge_method != 'S256':
        raise HTTPException(status_code=400, detail='pkce_s256_required')
    if not re.fullmatch(r'[A-Za-z0-9_-]{43,128}', code_challenge):
        raise HTTPException(status_code=400, detail='invalid_code_challenge')

    session = await get_auth_session(
        redis,
        request.cookies.get(SESSION_COOKIE),
    )
    if session is None:
        raise HTTPException(status_code=401, detail='login_required')

    user = session.get('user') or {}
    if not isinstance(user, dict) or not user.get('id'):
        raise HTTPException(status_code=401, detail='login_required')
    code = secrets.token_urlsafe(32)
    data = {
        'user_id': int(user['id']),
        'client_id': client_id,
        'redirect_uri': redirect_uri,
        'code_challenge': code_challenge,
    }
    await redis.set(
        _code_key(code),
        json.dumps(data, separators=(',', ':')),
        ex=AUTH_CODE_TTL_SECONDS,
        nx=True,
    )
    values = {'code': code}
    if state:
        values['state'] = state
    return RedirectResponse(_append_query(redirect_uri, **values), 302)


@router.post('/oauth/token', response_model=OAuthTokenResponse)
async def token(
    request: Request,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis_pool),
) -> OAuthTokenResponse:
    """Exchange a PKCE code or rotate a Native refresh token."""
    data = await _request_data(request)
    grant_type = str(data.get('grant_type') or '')
    client_id = str(data.get('client_id') or '')

    if grant_type == 'authorization_code':
        code = str(data.get('code') or '')
        redirect_uri = str(data.get('redirect_uri') or '')
        verifier = str(data.get('code_verifier') or '')
        _validate_client(client_id, redirect_uri)
        if not code or not PKCE_VERIFIER_RE.fullmatch(verifier):
            raise HTTPException(status_code=400, detail='invalid_grant')

        raw = await redis.getdel(_code_key(code))
        if isinstance(raw, bytes):
            raw = raw.decode()
        if not raw:
            raise HTTPException(status_code=400, detail='invalid_grant')
        try:
            stored = json.loads(raw)
        except (TypeError, json.JSONDecodeError) as exc:
            raise HTTPException(
                status_code=400,
                detail='invalid_grant',
            ) from exc
        if (
            stored.get('client_id') != client_id
            or stored.get('redirect_uri') != redirect_uri
            or not hmac.compare_digest(
                str(stored.get('code_challenge') or ''),
                _pkce_challenge(verifier),
            )
        ):
            raise HTTPException(status_code=400, detail='invalid_grant')

        user = await db.scalar(
            select(User).where(User.id == int(stored['user_id'])),
        )
        if user is None or user.status != 'active':
            raise HTTPException(status_code=400, detail='invalid_grant')
        result = await issue_token_pair_for_user(user, db, redis)
        return _token_response(result)

    if grant_type == 'refresh_token':
        if client_id not in _native_clients():
            raise HTTPException(status_code=400, detail='invalid_oauth_client')
        result = await refresh_tokens(
            RefreshRequest(
                refresh_token=str(data.get('refresh_token') or ''),
            ),
            redis,
        )
        return _token_response(result)

    raise HTTPException(status_code=400, detail='unsupported_grant_type')


@router.get('/me', response_model=MeResponse)
async def me(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> MeResponse:
    loaded = await db.scalar(
        select(User)
        .options(selectinload(User.profile))
        .where(User.id == user.id),
    )
    if loaded is None or loaded.status != 'active':
        raise HTTPException(status_code=401, detail='invalid_user')
    display_name = ' '.join(
        part
        for part in (
            getattr(loaded.profile, 'given_name', ''),
            getattr(loaded.profile, 'family_name', ''),
        )
        if part
    ) or loaded.username
    return MeResponse(
        id=loaded.id,
        username=loaded.username,
        display_name=display_name,
        role=loaded.role,
        group_id=loaded.group_id,
        status=loaded.status,
        feature_names=await _load_feature_names(db, loaded.group_id),
    )


@router.post('/oauth/revoke', status_code=204)
async def revoke(
    request: Request,
    redis: Redis = Depends(get_redis_pool),
) -> None:
    """Revoke the token; the response intentionally reveals nothing."""
    data = await _request_data(request)
    token_value = str(data.get('token') or '')
    hint = str(data.get('token_type_hint') or '')
    authorization = request.headers.get('authorization')
    if token_value:
        if hint == 'access_token':
            await logout_user(None, f'Bearer {token_value}', redis)
        else:
            await logout_user(token_value, None, redis)
        await revoke_media_for_parent(
            redis,
            'native:'
            f'{hashlib.sha256(token_value.encode()).hexdigest()}',
        )
    if authorization:
        _, _, bearer = authorization.partition(' ')
        if bearer:
            await logout_user(None, authorization, redis)
            await revoke_media_for_parent(
                redis,
                'native:'
                f'{hashlib.sha256(bearer.encode()).hexdigest()}',
            )
