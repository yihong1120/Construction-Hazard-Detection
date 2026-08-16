from __future__ import annotations

import os
from typing import cast
from typing import Literal
from urllib.parse import urlencode

from fastapi import HTTPException
from fastapi import Request
from fastapi import Response
from fastapi.responses import RedirectResponse

from examples.auth.config import Settings
from examples.db_management.schemas.auth import TokenPair
from examples.db_management.schemas.auth import TokenPairData

settings = Settings()
LEGACY_WEB_TOKEN_ENDPOINTS_ENABLED = os.getenv(
    'LEGACY_WEB_TOKEN_ENDPOINTS_ENABLED',
    'false',
).lower() in {'1', 'true', 'yes', 'on'}


def cookie_samesite() -> Literal['lax', 'strict', 'none']:
    """Return the configured validated SameSite value for refresh cookies.

    Returns:
        Cookie SameSite policy accepted by Starlette.
    """
    return cast(
        Literal['lax', 'strict', 'none'],
        settings.web_refresh_cookie_samesite,
    )


def is_web_auth_request(request: Request) -> bool:
    """Return whether a request should use an HTTP-only refresh cookie.

    Args:
        request: Request whose platform, auth mode, and browser headers are
            inspected.

    Returns:
        ``True`` when the request is from a browser-oriented client.
    """
    platform = request.headers.get('x-client-platform', '').strip().lower()
    auth_mode = request.headers.get('x-auth-mode', '').strip().lower()
    return (
        platform in {'web', 'flutter-web', 'browser'}
        or auth_mode in {'cookie', 'web-cookie', 'web_cookie'}
        or bool(
            request.headers.get('origin')
            or request.headers.get('sec-fetch-site'),
        )
    )


def reject_legacy_web_token_request(request: Request) -> None:
    """Block browser clients from receiving a refresh token in a response body.

    Args:
        request: Request checked for browser-oriented authentication headers.

    Raises:
        HTTPException: If legacy browser token endpoints are disabled.
    """
    if is_web_auth_request(request) and not LEGACY_WEB_TOKEN_ENDPOINTS_ENABLED:
        raise HTTPException(status_code=410, detail='use_bff_auth_endpoint')


def set_web_refresh_cookie(response: Response, refresh_token: str) -> None:
    """Set the browser-only HTTP-only refresh-token cookie.

    Args:
        response: Response on which to set the cookie.
        refresh_token: Newly issued refresh token to protect from JavaScript.
    """
    # Cookie attributes are centrally configured to keep every web-auth flow
    # subject to the same browser security policy.
    response.set_cookie(
        key=settings.web_refresh_cookie_name,
        value=refresh_token,
        max_age=settings.web_refresh_cookie_max_age_seconds,
        httponly=True,
        secure=settings.web_refresh_cookie_secure,
        samesite=cookie_samesite(),
        path=settings.web_refresh_cookie_path,
        domain=settings.web_refresh_cookie_domain or None,
    )


def clear_web_refresh_cookie(response: Response) -> None:
    """Clear the browser-only refresh-token cookie.

    Args:
        response: Response on which to expire the cookie.
    """
    response.delete_cookie(
        key=settings.web_refresh_cookie_name,
        path=settings.web_refresh_cookie_path,
        domain=settings.web_refresh_cookie_domain or None,
        secure=settings.web_refresh_cookie_secure,
        samesite=cookie_samesite(),
    )


def refresh_token_from_cookie(request: Request) -> str | None:
    """Read the refresh token from the browser-only cookie.

    Args:
        request: Request containing browser cookies.

    Returns:
        Refresh token value, or ``None`` when the cookie is absent.
    """
    return request.cookies.get(settings.web_refresh_cookie_name)


def token_pair_response(
    result: TokenPairData,
    *,
    omit_refresh_token: bool,
) -> TokenPair:
    """Build an API token response without exposing browser refresh tokens.

    Args:
        result: Complete token data produced by the authentication service.
        omit_refresh_token: Whether the refresh token is held only in a cookie.

    Returns:
        Validated token response safe to serialise to the client.
    """
    data = dict(result)
    if omit_refresh_token:
        # Do not serialise a browser refresh token after it is placed in a
        # secure HTTP-only cookie.
        del data['refresh_token']
    return TokenPair.model_validate(data)


async def apple_callback_redirect(request: Request) -> RedirectResponse:
    """Convert an Apple callback payload into the native-app deep link.

    Args:
        request: Apple callback request containing query or form parameters.

    Returns:
        Redirect response targeting the application's Apple callback scheme.
    """
    params: list[tuple[str, str]] = list(request.query_params.multi_items())
    if request.method == 'POST':
        # Apple may submit callback values in a form rather than the query.
        form = await request.form()
        params.extend((key, str(value)) for key, value in form.multi_items())
    query = urlencode(params)
    suffix = f'?{query}' if query else ''
    return RedirectResponse(
        (
            f'intent://callback{suffix}'
            '#Intent;package=com.changdar.visionnaire;'
            'scheme=signinwithapple;end'
        ),
        status_code=302,
    )
