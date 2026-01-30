from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from examples.auth.lifespan import global_lifespan
from examples.db_management.routers.auth import router as auth_router
from examples.db_management.routers.features import router as feature_router
from examples.db_management.routers.groups import router as group_router
from examples.db_management.routers.sites import router as site_router
from examples.db_management.routers.streams import router as stream_cfg_router
from examples.db_management.routers.users import (
    router as user_management_router,
)

# Initialise the FastAPI app with a custom lifespan handler
app: FastAPI = FastAPI(lifespan=global_lifespan)

# Add Cross-Origin Resource Sharing (CORS) middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  # Allow all origins (adjust this in production)
    allow_credentials=True,
    allow_methods=['*'],  # Allow all HTTP methods
    allow_headers=['*'],  # Allow all headers
)

# Include routers for authentication, user management,
# site management, feature management, group management,
# and stream configuration management
app.include_router(auth_router)
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
