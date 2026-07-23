from __future__ import annotations

import hmac
import os

from fastapi import HTTPException
from fastapi import Request
from fastapi import Response

from examples.auth.config import Settings
from examples.auth.session_store import AUTH_SESSION_TTL_SECONDS

settings = Settings()
SESSION_COOKIE = os.getenv('BFF_SESSION_COOKIE_NAME', '__Host-vn_session')
SESSION_COOKIE_SECURE = os.getenv(
    'BFF_SESSION_COOKIE_SECURE',
    'true',
).lower() in {'1', 'true', 'yes', 'on'}


def allowed_origins() -> set[str]:
    return {
        value.strip().rstrip('/')
        for value in settings.cors_allowed_origins.split(',')
        if value.strip()
    }


def require_trusted_origin(request: Request) -> None:
    origin = (request.headers.get('origin') or '').rstrip('/')
    if not origin or origin not in allowed_origins():
        raise HTTPException(status_code=403, detail='invalid_origin')


def check_csrf(
    request: Request,
    session: dict[str, object],
    csrf_token: str | None,
) -> None:
    require_trusted_origin(request)
    expected = str(session.get('csrf_secret') or '')
    if not csrf_token or not hmac.compare_digest(csrf_token, expected):
        raise HTTPException(status_code=403, detail='invalid_csrf_token')


def set_session_cookie(response: Response, session_id: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE,
        value=session_id,
        max_age=AUTH_SESSION_TTL_SECONDS,
        secure=SESSION_COOKIE_SECURE,
        httponly=True,
        samesite='lax',
        path='/',
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(
        key=SESSION_COOKIE,
        secure=SESSION_COOKIE_SECURE,
        httponly=True,
        samesite='lax',
        path='/',
    )
