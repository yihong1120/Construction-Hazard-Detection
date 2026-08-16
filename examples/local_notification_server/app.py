from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from cryptography.fernet import Fernet
from dotenv import load_dotenv
from fastapi import FastAPI

from examples.auth.lifespan import global_lifespan
from examples.local_notification_server.fcm_service import init_firebase_app
from examples.local_notification_server.routers import (
    router as notification_router,
)
load_dotenv()


@asynccontextmanager
async def notification_lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialise notification-server resources during application lifespan.

    Args:
        app: FastAPI application instance receiving shared resources.

    Yields:
        Control to FastAPI while shared database, Redis, and Firebase resources
        are available.
    """
    # Validate mandatory secret material before accepting any notification work.
    cred_path = os.environ['FIREBASE_CRED_PATH']
    project_id = os.environ['FIREBASE_PROJECT_ID']
    Fernet(os.environ['FCM_TOKEN_ENCRYPTION_KEY'].encode('utf-8'))

    async with global_lifespan(app):
        init_firebase_app(cred_path=cred_path, project_id=project_id)
        yield

app: FastAPI = FastAPI(lifespan=notification_lifespan)

# Keep HTTP route registration at the application composition boundary.
app.include_router(notification_router)


def main() -> None:
    """Run the notification FastAPI application with Uvicorn."""
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
