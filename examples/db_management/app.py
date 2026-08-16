from __future__ import annotations

from collections.abc import Awaitable
from collections.abc import Callable

import uvicorn
from fastapi import FastAPI
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response

from examples.auth.config import Settings
from examples.auth.lifespan import global_lifespan
from examples.bff.router import router as bff_router
from examples.db_management.routers.auth import router as auth_router
from examples.db_management.routers.features import router as feature_router
from examples.db_management.routers.groups import router as group_router
from examples.db_management.routers.legal import router as legal_router
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
    if request.url.path.startswith((
        '/bff/',
        '/oauth/',
        '/me',
        '/api/playback/',
    )):
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
app.include_router(oauth_router)
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
uvicorn examples.db_management.app:app\
    --host 127.0.0.1\
    --port 8005 --workers 4

uv run uvicorn examples.db_management.app:app\
    --host 127.0.0.1\
    --port 8005 --workers 4
"""
