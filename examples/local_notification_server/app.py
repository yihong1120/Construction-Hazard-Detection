from __future__ import annotations

import os
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from examples.auth.lifespan import global_lifespan
from examples.local_notification_server.fcm_service import init_firebase_app
from examples.local_notification_server.routers import (
    router as notification_router,
)
load_dotenv()


@asynccontextmanager
async def notification_lifespan(app: FastAPI):
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
