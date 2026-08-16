from __future__ import annotations

from fastapi import APIRouter
from fastapi import Depends
from fastapi import Header
from fastapi import Request
from fastapi import Response
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.database import get_db
from examples.auth.redis_pool import get_redis_pool
from examples.bff.schemas import BffLoginRequest
from examples.bff.schemas import BffSessionResponse
from examples.bff.schemas import CsrfResponse
from examples.bff.session_services import csrf_response
from examples.bff.session_services import current_bff_session
from examples.bff.session_services import login_bff_session
from examples.bff.session_services import logout_bff_session
from examples.bff.session_services import proxy_bff_request

router = APIRouter(prefix='/bff', tags=['bff'])


@router.post('/auth/login', response_model=BffSessionResponse)
async def login(
    payload: BffLoginRequest,
    request: Request,
    response: Response,
    x_hcaptcha_bypass_key: str | None = Header(
        None,
        alias='X-HCaptcha-Bypass-Key',
    ),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis_pool),
) -> BffSessionResponse:
    """Create a BFF session without exposing JWTs to the browser.

    Args:
        payload: Browser login credentials.
        request: HTTP request used for origin and client-IP validation.
        response: HTTP response receiving the opaque session cookie.
        x_hcaptcha_bypass_key: Optional trusted server-side hCaptcha bypass key.
        db: Database session used by the authentication service.
        redis: Redis connection holding token and BFF session state.

    Returns:
        Token-free browser session response.
    """
    return await login_bff_session(
        payload,
        request,
        response,
        x_hcaptcha_bypass_key,
        db,
        redis,
    )


@router.get('/auth/session', response_model=BffSessionResponse)
async def get_session(
    request: Request,
    response: Response,
    redis: Redis = Depends(get_redis_pool),
) -> BffSessionResponse:
    """Return the current BFF session without exposing JWTs.

    Args:
        request: HTTP request containing the opaque session cookie.
        response: HTTP response receiving the renewed session cookie.
        redis: Redis connection holding session and token state.

    Returns:
        Token-free browser session response.
    """
    return await current_bff_session(request, response, redis)


@router.get('/auth/csrf', response_model=CsrfResponse)
async def get_csrf(
    request: Request,
    response: Response,
    redis: Redis = Depends(get_redis_pool),
) -> CsrfResponse:
    """Return the CSRF token for an active BFF session.

    Args:
        request: HTTP request containing the opaque session cookie.
        response: HTTP response receiving the renewed session cookie.
        redis: Redis connection holding BFF session state.

    Returns:
        CSRF token required by mutating BFF requests.
    """
    return await csrf_response(request, response, redis)


@router.post('/auth/logout', status_code=204)
async def logout(
    request: Request,
    response: Response,
    x_csrf_token: str | None = Header(None, alias='X-CSRF-Token'),
    redis: Redis = Depends(get_redis_pool),
) -> None:
    """End the BFF session and revoke its credentials.

    Args:
        request: HTTP request containing the opaque session cookie.
        response: HTTP response on which to clear the session cookie.
        x_csrf_token: CSRF token authorising the logout request.
        redis: Redis connection holding session and token state.
    """
    await logout_bff_session(request, response, x_csrf_token, redis)


@router.api_route(
    '/{service}/{path:path}',
    methods=['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'HEAD'],
    include_in_schema=False,
)
async def api_proxy(
    service: str,
    path: str,
    request: Request,
    x_csrf_token: str | None = Header(None, alias='X-CSRF-Token'),
    redis: Redis = Depends(get_redis_pool),
) -> Response:
    """Forward an authenticated browser request to an allow-listed service.

    Args:
        service: Allow-listed upstream service name.
        path: Remaining upstream request path.
        request: Original browser request.
        x_csrf_token: CSRF token required by mutating requests.
        redis: Redis connection holding BFF session state.

    Returns:
        Sanitised non-cacheable upstream response.
    """
    return await proxy_bff_request(
        service,
        path,
        request,
        x_csrf_token,
        redis,
    )
