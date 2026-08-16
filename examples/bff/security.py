from __future__ import annotations

import hmac
import os
from typing import Literal
from typing import TypedDict

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


class _SessionCookieOptions(TypedDict):
    """Define shared security attributes accepted by Starlette cookies.

    Attributes:
        secure: Whether browsers must send the cookie only over HTTPS.
        httponly: Whether browser scripts are barred from reading the cookie.
        samesite: Cross-site sending policy for the cookie.
        path: URL path that receives the cookie.
    """

    secure: bool
    httponly: bool
    samesite: Literal['lax', 'strict', 'none']
    path: str


def allowed_origins() -> set[str]:
    """Return normalised browser origins permitted to use the BFF.

    Returns:
        Origins explicitly configured for credentialed browser requests.
    """
    return {
        value.strip().rstrip('/')
        for value in settings.cors_allowed_origins.split(',')
        if value.strip()
    }


def require_trusted_origin(request: Request) -> None:
    """Reject a request whose origin is not trusted by the BFF.

    Args:
        request: Browser request whose ``Origin`` header is validated.

    Raises:
        HTTPException: If the origin is absent or not configured.
    """
    origin = (request.headers.get('origin') or '').rstrip('/')
    if not origin or origin not in allowed_origins():
        raise HTTPException(status_code=403, detail='invalid_origin')


def check_csrf(
    request: Request,
    session: dict[str, object],
    csrf_token: str | None,
) -> None:
    """Validate CSRF protection for a mutating browser request.

    Args:
        request: Browser request whose origin is validated.
        session: Active server-side BFF session containing the CSRF secret.
        csrf_token: Token supplied by the browser request.

    Raises:
        HTTPException: If origin or CSRF-token validation fails.
    """
    require_trusted_origin(request)
    expected = str(session.get('csrf_secret') or '')
    if not csrf_token or not hmac.compare_digest(csrf_token, expected):
        raise HTTPException(status_code=403, detail='invalid_csrf_token')


def _session_cookie_options() -> _SessionCookieOptions:
    """Return the common security attributes for BFF session cookies.

    Returns:
        Cookie options shared by creation and removal responses.
    """
    return {
        'secure': SESSION_COOKIE_SECURE,
        'httponly': True,
        'samesite': 'lax',
        'path': '/',
    }


def set_session_cookie(response: Response, session_id: str) -> None:
    """Set the opaque HTTP-only BFF session cookie.

    Args:
        response: Response on which to set the session cookie.
        session_id: Opaque server-side session identifier.
    """
    response.set_cookie(
        key=SESSION_COOKIE,
        value=session_id,
        max_age=AUTH_SESSION_TTL_SECONDS,
        **_session_cookie_options(),
    )


def clear_session_cookie(response: Response) -> None:
    """Clear the opaque HTTP-only BFF session cookie.

    Args:
        response: Response on which to expire the session cookie.
    """
    response.delete_cookie(
        key=SESSION_COOKIE,
        **_session_cookie_options(),
    )
