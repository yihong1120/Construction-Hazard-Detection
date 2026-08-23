from __future__ import annotations

from fastapi import HTTPException
from fastapi import Request
from fastapi import Response
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from examples.auth.deployment_context import DeploymentBinding
from examples.auth.deployment_context import resolve_request_deployment
from examples.auth.models import User
from examples.auth.session_store import auth_tokens
from examples.auth.session_store import create_auth_session
from examples.auth.session_store import delete_auth_session
from examples.auth.session_store import get_auth_session
from examples.auth.session_store import touch_auth_session
from examples.bff.proxy import get_proxy_access_token
from examples.bff.proxy import proxy_request
from examples.bff.schemas import BffLoginRequest
from examples.bff.schemas import BffSessionResponse
from examples.bff.schemas import CsrfResponse
from examples.bff.schemas import UserSummary
from examples.bff.security import check_csrf
from examples.bff.security import clear_session_cookie
from examples.bff.security import require_trusted_origin
from examples.bff.security import SESSION_COOKIE
from examples.bff.security import set_session_cookie
from examples.db_management.schemas.auth import DeploymentInfo
from examples.db_management.services.auth_services import login_user
from examples.db_management.services.auth_services import logout_user


async def _session(
    request: Request,
    redis: Redis,
) -> tuple[str, dict[str, object]]:
    """Load the active BFF session identified by a browser cookie.

    Args:
        request: HTTP request containing the BFF session cookie.
        redis: Redis connection holding BFF session records.

    Returns:
        Session identifier and its server-side record.

    Raises:
        HTTPException: If the session cookie is absent or its server-side
            record
            has expired.
    """
    session_id = request.cookies.get(SESSION_COOKIE)
    data = await get_auth_session(redis, session_id)
    if not session_id or data is None:
        raise HTTPException(status_code=401, detail='app_session_expired')
    return session_id, data


async def _require_session_deployment(
    request: Request,
    db: AsyncSession,
    session: dict[str, object],
) -> DeploymentBinding:
    """Require a BFF session to match the current active deployment."""
    try:
        stored = DeploymentInfo.model_validate(session['deployment'])
    except (KeyError, ValueError, TypeError):
        raise HTTPException(
            status_code=409,
            detail={
                'code': 'deployment_configuration_changed',
                'message': 'Deployment configuration changed; sign in again.',
            },
        )
    binding = await resolve_request_deployment(request, db)
    if binding.as_response() != stored.model_dump():
        raise HTTPException(
            status_code=409,
            detail={
                'code': 'deployment_configuration_changed',
                'message': 'Deployment configuration changed; sign in again.',
            },
        )
    return binding


async def _roll_session(
    response: Response,
    redis: Redis,
    session_id: str,
) -> None:
    """Renew a BFF session's server-side and browser idle timeouts.

    Args:
        response: Response on which to renew the browser cookie.
        redis: Redis connection holding BFF session records.
        session_id: Active BFF session identifier.
    """
    await touch_auth_session(redis, session_id)
    set_session_cookie(response, session_id)


def _session_response(session: dict[str, object]) -> BffSessionResponse:
    """Build a public BFF session response without token material.

    Args:
        session: Server-side BFF session record.

    Returns:
        Authenticated user summary and granted feature names.
    """
    # The session schema deliberately ignores encrypted server-only fields.
    return BffSessionResponse.model_validate(session)


async def _user_summary(
    db: AsyncSession,
    user_id: int,
) -> UserSummary:
    """Load a user summary suitable for server-side BFF session storage.

    Args:
        db: Database session used to load the account profile.
        user_id: Identifier of the authenticated user.

    Returns:
        Public user summary used by BFF session responses.

    Raises:
        HTTPException: If the authenticated user no longer exists.
    """
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
                for part in (
                    profile.given_name,
                    profile.family_name,
                )
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


async def login_bff_session(
    payload: BffLoginRequest,
    request: Request,
    response: Response,
    hcaptcha_bypass_key: str | None,
    db: AsyncSession,
    redis: Redis,
) -> BffSessionResponse:
    """Authenticate a browser user and create a token-private BFF session.

    Args:
        payload: Browser login credentials.
        request: HTTP request checked against trusted origins.
        response: HTTP response receiving the session cookie.
        hcaptcha_bypass_key: Optional trusted server-side hCaptcha bypass key.
        db: Database session used for credential authentication and profile
            data.
        redis: Redis connection holding token and BFF session state.

    Returns:
        Public BFF session response without access or refresh tokens.

    Raises:
        HTTPException: If origin or credential validation fails.
    """
    require_trusted_origin(request)
    result = await login_user(
        payload,
        db,
        redis,
        hcaptcha_bypass_key=hcaptcha_bypass_key,
        client_ip=request.client.host if request.client else None,
        hash_refresh_token=True,
        request=request,
    )
    summary = await _user_summary(db, int(result['user_id']))
    session_id, session = await create_auth_session(
        redis,
        result,
        summary.model_dump(),
    )
    # Web clients receive only an opaque HTTP-only session identifier.
    set_session_cookie(response, session_id)
    response.headers['Cache-Control'] = 'no-store'
    return _session_response(session)


async def current_bff_session(
    request: Request,
    response: Response,
    redis: Redis,
    db: AsyncSession,
) -> BffSessionResponse:
    """Return an active BFF session and renew its idle timeout.

    Args:
        request: HTTP request containing the BFF session cookie.
        response: HTTP response receiving the renewed session cookie.
        redis: Redis connection holding session and token state.

    Returns:
        Public session response without token material.
    """
    session_id, stored = await _session(request, redis)
    deployment = await _require_session_deployment(request, db, stored)
    _, session = await get_proxy_access_token(
        redis,
        session_id,
        deployment=deployment,
    )
    await _roll_session(response, redis, session_id)
    response.headers['Cache-Control'] = 'no-store'
    return _session_response(session)


async def csrf_response(
    request: Request,
    response: Response,
    redis: Redis,
    db: AsyncSession,
) -> CsrfResponse:
    """Return a session's CSRF secret and renew its idle timeout.

    Args:
        request: HTTP request containing the BFF session cookie.
        response: HTTP response receiving the renewed session cookie.
        redis: Redis connection holding the BFF session.

    Returns:
        CSRF response for use with subsequent mutating requests.
    """
    session_id, session = await _session(request, redis)
    await _require_session_deployment(request, db, session)
    await _roll_session(response, redis, session_id)
    response.headers['Cache-Control'] = 'no-store'
    return CsrfResponse(csrf_token=str(session['csrf_secret']))


async def logout_bff_session(
    request: Request,
    response: Response,
    csrf_token: str | None,
    redis: Redis,
) -> None:
    """Revoke BFF credentials and remove the browser session.

    Args:
        request: HTTP request containing the BFF session cookie.
        response: HTTP response on which to clear the session cookie.
        csrf_token: Token required to authorise the logout request.
        redis: Redis connection holding token and BFF session state.

    Raises:
        HTTPException: If the session is absent or CSRF validation fails.
    """
    session_id, session = await _session(request, redis)
    check_csrf(request, session, csrf_token)
    access_token, refresh_token = auth_tokens(session)
    await logout_user(refresh_token, f"Bearer {access_token}", redis)
    await delete_auth_session(redis, session_id)
    clear_session_cookie(response)
    response.headers['Cache-Control'] = 'no-store'


async def proxy_bff_request(
    service: str,
    path: str,
    request: Request,
    csrf_token: str | None,
    redis: Redis,
    db: AsyncSession,
) -> Response:
    """Proxy an authenticated BFF request and renew its idle timeout.

    Args:
        service: Allow-listed upstream service name.
        path: Path to forward to the upstream service.
        request: Original browser request.
        csrf_token: Token required for mutating requests.
        redis: Redis connection holding BFF session state.

    Returns:
        Upstream response after BFF security and session processing.

    Raises:
        HTTPException: If session, CSRF, routing, or upstream processing fails.
    """
    session_id, session = await _session(request, redis)
    deployment = await _require_session_deployment(request, db, session)
    # A metadata/SSE proxy can live indefinitely.  Deployment verification is
    # the only database work in this request path, so release its connection
    # before constructing the upstream response rather than holding a pool
    # slot for the lifetime of the browser stream.
    await db.close()
    if request.method not in {'GET', 'HEAD', 'OPTIONS'}:
        check_csrf(request, session, csrf_token)
    response = await proxy_request(
        request,
        redis,
        session_id,
        f"{service}/{path}",
        deployment=deployment,
    )
    await _roll_session(response, redis, session_id)
    return response
