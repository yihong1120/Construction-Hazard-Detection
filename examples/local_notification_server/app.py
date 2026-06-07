from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from examples.auth.lifespan import global_lifespan
from examples.local_notification_server.fcm_service import init_firebase_app
from examples.local_notification_server.routers import (
    router as notification_router,
)
load_dotenv()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def notification_lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Lifespan event handler for FastAPI app.
    Initialise global resources (DB/Redis) and Firebase Admin SDK at startup.
    """
    cred_path = os.getenv(
        'FIREBASE_CRED_PATH',
        'path/to/your/firebase/credentials.json',
    )
    project_id = os.getenv('FIREBASE_PROJECT_ID', 'your-firebase-project-id')

    async with global_lifespan(app):
        init_firebase_app(cred_path=cred_path, project_id=project_id)
        yield

app: FastAPI = FastAPI(lifespan=notification_lifespan)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Log request validation failures with enough context to fix callers."""
    body = await request.body()
    body_preview = body[:2000].decode('utf-8', errors='replace')
    logger.warning(
        'Request validation failed path=%s errors=%s body=%s',
        request.url.path,
        exc.errors(),
        body_preview,
    )
    return JSONResponse(status_code=422, content={'detail': exc.errors()})

# Add Cross-Origin Resource Sharing (CORS) middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  # Allow all origins (adjust this in production)
    allow_credentials=True,
    allow_methods=['*'],  # Allow all HTTP methods
    allow_headers=['*'],  # Allow all headers
)

# Include routers for  notification services
app.include_router(notification_router)


def main() -> None:
    """
    Main function to run the FastAPI application using Uvicorn.
    """
    uvicorn.run(app, host='127.0.0.1', port=8003)


if __name__ == '__main__':
    main()

"""
uvicorn examples.local_notification_server.app:app\
    --host 127.0.0.1 \
    --port 8003 \
    --workers 4

uv run uvicorn examples.local_notification_server.app:app\
    --host 127.0.0.1 \
    --port 8003 \
    --workers 4
"""
