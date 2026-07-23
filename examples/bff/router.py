from __future__ import annotations

from fastapi import APIRouter
from fastapi import Depends
from fastapi import Header
from fastapi import HTTPException
from fastapi import Request
from fastapi import Response
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from examples.auth.database import get_db
from examples.auth.models import User
from examples.auth.redis_pool import get_redis_pool
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
from examples.db_management.services.auth_services import login_user
from examples.db_management.services.auth_services import logout_user

router = APIRouter(prefix='/bff', tags=['bff'])


async def _session(
    request: Request,
    redis: Redis,
) -> tuple[str, dict[str, object]]:
    session_id = request.cookies.get(SESSION_COOKIE)
    data = await get_auth_session(redis, session_id)
    if not session_id or data is None:
        raise HTTPException(status_code=401, detail='app_session_expired')
    return session_id, data


async def _roll_session(
    response: Response,
    redis: Redis,
    session_id: str,
) -> None:
    """Renew the server-side idle timeout and the browser session cookie."""
    await touch_auth_session(redis, session_id)
    set_session_cookie(response, session_id)


def _session_response(session: dict[str, object]) -> BffSessionResponse:
    raw_features = session.get('feature_names')
    feature_names = (
        [str(value) for value in raw_features]
        if isinstance(raw_features, (list, tuple))
        else []
    )
    return BffSessionResponse(
        user=UserSummary.model_validate(session['user']),
        feature_names=feature_names,
    )


async def _user_summary(
    db: AsyncSession,
    user_id: int,
) -> UserSummary:
    user = await db.scalar(
        select(User)
        .options(selectinload(User.profile))
        .where(User.id == user_id),
    )
    if user is None:
        raise HTTPException(status_code=401, detail='user_not_found')
    profile = user.profile
    display_name = ' '.join(
        part
        for part in (
            getattr(profile, 'given_name', ''),
            getattr(profile, 'family_name', ''),
        )
        if part
    ) or user.username
    return UserSummary(
        id=user.id,
        username=user.username,
        display_name=display_name,
        role=user.role,
        group_id=user.group_id,
        status=user.status,
    )


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
    """Create a Full-BFF session without exposing either JWT to Web."""
    require_trusted_origin(request)
    result = await login_user(
        payload,
        db,
        redis,
        hcaptcha_bypass_key=x_hcaptcha_bypass_key,
        client_ip=request.client.host if request.client else None,
        hash_refresh_token=True,
    )
    summary = await _user_summary(db, int(result['user_id']))
    session_id, session = await create_auth_session(
        redis,
        result,
        summary.model_dump(),
    )
    set_session_cookie(response, session_id)
    response.headers['Cache-Control'] = 'no-store'
    return _session_response(session)


@router.get('/auth/session', response_model=BffSessionResponse)
async def get_session(
    request: Request,
    response: Response,
    redis: Redis = Depends(get_redis_pool),
) -> BffSessionResponse:
    session_id, _ = await _session(request, redis)
    _, session = await get_proxy_access_token(redis, session_id)
    await _roll_session(response, redis, session_id)
    response.headers['Cache-Control'] = 'no-store'
    return _session_response(session)


@router.get('/auth/csrf', response_model=CsrfResponse)
async def get_csrf(
    request: Request,
    response: Response,
    redis: Redis = Depends(get_redis_pool),
) -> CsrfResponse:
    session_id, session = await _session(request, redis)
    await _roll_session(response, redis, session_id)
    response.headers['Cache-Control'] = 'no-store'
    return CsrfResponse(csrf_token=str(session['csrf_secret']))


@router.post('/auth/logout', status_code=204)
async def logout(
    request: Request,
    response: Response,
    x_csrf_token: str | None = Header(None, alias='X-CSRF-Token'),
    redis: Redis = Depends(get_redis_pool),
) -> None:
    session_id, session = await _session(request, redis)
    check_csrf(request, session, x_csrf_token)
    access_token, refresh_token = auth_tokens(session)
    await logout_user(refresh_token, f'Bearer {access_token}', redis)
    await delete_auth_session(redis, session_id)
    clear_session_cookie(response)
    response.headers['Cache-Control'] = 'no-store'


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
    session_id, session = await _session(request, redis)
    if request.method not in {'GET', 'HEAD', 'OPTIONS'}:
        check_csrf(request, session, x_csrf_token)
    response = await proxy_request(
        request,
        redis,
        session_id,
        f'{service}/{path}',
    )
    await _roll_session(response, redis, session_id)
    return response
