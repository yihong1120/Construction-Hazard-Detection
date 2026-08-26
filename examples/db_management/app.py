from __future__ import annotations

from collections.abc import Awaitable
from collections.abc import Callable
from uuid import uuid4

import uvicorn
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.responses import Response

from examples.auth.config import Settings
from examples.auth.lifespan import global_lifespan
from examples.bff.router import router as bff_router
from examples.db_management.routers.auth import router as auth_router
from examples.db_management.routers.deployment_enrollment_codes import (
    router as deployment_enrollment_code_router,
)
from examples.db_management.routers.deployments import (
    router as deployment_router,
)
from examples.db_management.routers.features import router as feature_router
from examples.db_management.routers.groups import router as group_router
from examples.db_management.routers.legal import router as legal_router
from examples.db_management.routers.native_social import (
    router as native_social_router,
)
from examples.db_management.routers.oauth import router as oauth_router
from examples.db_management.routers.password_reset import (
    router as password_reset_router,
)
from examples.db_management.routers.playback import router as playback_router
from examples.db_management.routers.sites import router as site_router
from examples.db_management.routers.streams import router as stream_cfg_router
from examples.db_management.routers.users import (
    router as user_management_router,
)
from examples.deployment_registry.router import router as registry_router

settings = Settings()


def _allowed_cors_origins() -> list[str]:
    """Return the configured non-empty CORS origins.

    Returns:
        A trimmed list of explicitly configured allowed origins.
    """
    # Drop empty entries so an accidental trailing comma grants no origin.
    return [
        origin.strip()
        for origin in settings.cors_allowed_origins.split(',')
        if origin.strip()
    ]


# Initialise the FastAPI app with a custom lifespan handler
app: FastAPI = FastAPI(lifespan=global_lifespan)


def _log_server_traceback(
    request_id: str,
    exc: BaseException,
) -> None:
    """Record type and traceback without serialising exception arguments."""
    import logging
    import traceback

    # Some exceptions (notably SQL and provider errors) include submitted
    # values in ``str(exc)``.  Log stack locations and the exception class, but
    # never their message, request body, query string, or headers.
    trace = ''.join(traceback.format_tb(exc.__traceback__))
    logging.getLogger('uvicorn.error').error(
        'Server error request_id=%s error_type=%s traceback=%s',
        request_id,
        type(exc).__name__,
        trace,
    )


@app.exception_handler(HTTPException)
async def safe_http_exception(
    request: Request,
    exc: HTTPException,
) -> JSONResponse:
    """Prevent handled 500s from leaking database/provider exception text."""
    # The public Deployment Registry deliberately fails closed with a
    # sanitised 503 when signing, its enrollment pepper, Redis, or PostgreSQL
    # cannot safely serve an anonymous request.  Preserve that actionable
    # status rather than misreporting it as an internal 500.
    is_safe_registry_unavailable = (
        exc.status_code == 503
        and isinstance(exc.detail, dict)
        and exc.detail.get('code')
        in {
            'registry_signing_unavailable',
            'enrollment_unavailable',
        }
    )
    if exc.status_code < 500 or is_safe_registry_unavailable:
        return JSONResponse(
            status_code=exc.status_code,
            content={'detail': exc.detail},
            headers=exc.headers,
        )
    request_id = getattr(request.state, 'request_id', str(uuid4()))
    _log_server_traceback(request_id, exc)
    return JSONResponse(
        status_code=500,
        content={
            'detail': {
                'code': 'internal_server_error',
                'request_id': request_id,
            },
        },
    )


@app.middleware('http')
async def request_id_and_unhandled_error_logging(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Return a traceable safe 500 without logging credential-bearing data."""
    request_id = str(uuid4())
    request.state.request_id = request_id
    try:
        response = await call_next(request)
    except Exception as exc:
        # Do not log request bodies, query strings, headers, or exception
        # messages: they can carry passwords, JWTs, refresh tokens, hCaptcha
        # responses, or provider payloads.
        _log_server_traceback(request_id, exc)
        response = JSONResponse(
            status_code=500,
            content={
                'detail': {
                    'code': 'internal_server_error',
                    'request_id': request_id,
                },
            },
        )
    response.headers['X-Request-ID'] = request_id
    return response


@app.middleware('http')
async def prevent_sensitive_response_caching(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Prevent authenticated and token responses being cached.

    Args:
        request: The inbound HTTP request.
        call_next: The application middleware chain.

    Returns:
        The downstream response, with ``Cache-Control: no-store`` added for
        sensitive endpoints.
    """
    response = await call_next(request)
    # Authentication and signed-media data must not be stored by browsers or
    # intermediary caches after the response leaves the application.
    if request.url.path.startswith(
        (
            '/bff/',
            '/oauth/',
            '/auth/',
            '/me',
            '/api/playback/',
        ),
    ) or request.url.path in {'/login', '/refresh'}:
        response.headers['Cache-Control'] = 'no-store'
    return response


allowed_origins = _allowed_cors_origins()
if allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )

# Keep routing composition here; endpoint implementations remain in routers.
app.include_router(auth_router)
app.include_router(deployment_router)
app.include_router(deployment_enrollment_code_router)
app.include_router(registry_router)
app.include_router(oauth_router)
app.include_router(native_social_router)
app.include_router(playback_router)
app.include_router(bff_router)
app.include_router(legal_router)
app.include_router(password_reset_router)
app.include_router(user_management_router)
app.include_router(site_router)
app.include_router(feature_router)
app.include_router(group_router)
app.include_router(stream_cfg_router)


def main() -> None:
    """Run the database-management ASGI application with Uvicorn.

    This entry point is intended for local execution. Production deployments
    should invoke the ASGI application through their process manager.
    """
    uvicorn.run(app, host='127.0.0.1', port=8005, workers=4)


if __name__ == '__main__':
    main()


"""
uvicorn examples.db_management.app:app \
    --host 127.0.0.1 --port 8005 --workers 4

uv run uvicorn examples.db_management.app:app \
    --host 127.0.0.1 --port 8005 --workers 4
"""
