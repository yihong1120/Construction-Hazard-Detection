from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
    return [
        origin.strip()
        for origin in settings.cors_allowed_origins.split(',')
        if origin.strip()
    ]


# Initialise the FastAPI app with a custom lifespan handler
app: FastAPI = FastAPI(lifespan=global_lifespan)


@app.middleware('http')
async def prevent_sensitive_response_caching(request, call_next):
    """Keep browser/session/token responses out of intermediary caches."""
    response = await call_next(request)
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

# Include routers for authentication, user management,
# site management, feature management, group management,
# and stream configuration management
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
    """
    Main function to run the FastAPI application using Uvicorn.
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
