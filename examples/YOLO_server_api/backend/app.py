# examples/YOLO_server_api/backend/app.py
from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import socketio
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_jwt import JwtAccessBearer
from fastapi_limiter import FastAPILimiter

from .config import Settings
from .models import Base
from .models import engine
from .redis_pool import RedisClient
from .routers import auth_router
from .routers import detection_router
from .routers import model_management_router
from .routers import user_management_router
from .security import update_secret_key

# Instantiate the FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Register API routers for different functionalities
app.include_router(auth_router)
app.include_router(detection_router)
app.include_router(model_management_router)
app.include_router(user_management_router)

# Set up JWT authentication
jwt_access = JwtAccessBearer(secret_key=Settings().authjwt_secret_key)

# Configure background scheduler to refresh secret keys every 30 days
scheduler = BackgroundScheduler()
scheduler.add_job(
    func=lambda: update_secret_key(app),
    trigger='interval',
    days=30,
)
scheduler.start()

# Initialise Socket.IO server for real-time events
sio = socketio.AsyncServer(async_mode='asgi')
sio_app = socketio.ASGIApp(sio, app)

# Define Socket.IO events for real-time communication


@sio.event
async def connect(sid: str, environ: dict) -> None:
    """
    Handles client connection event to the Socket.IO server.

    Args:
        sid (str): The session ID of the connected client.
        environ (dict): The environment dictionary for the connection.
    """
    print('Client connected:', sid)


# Define Socket.IO event for client disconnection from the server
@sio.event
async def disconnect(sid: str) -> None:
    """
    Handles client disconnection from the Socket.IO server.

    Args:
        sid (str): The session ID of the disconnected client.
    """
    print('Client disconnected:', sid)

# Define lifespan event to manage the application startup and shutdown


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """
    Context manager to handle application startup and shutdown tasks.

    Args:
        app (FastAPI): The FastAPI application instance.
    """
    # Initialise Redis connection pool for rate limiting
    redis_host = os.getenv('REDIS_HOST', '127.0.0.1')
    redis_port = os.getenv('REDIS_PORT', '6379')
    redis_password = os.getenv('REDIS_PASSWORD', '')

    redis_url = f"redis://:{redis_password}@{redis_host}:{redis_port}/0"

    app.state.redis_client = RedisClient(redis_url)
    redis_conn = await app.state.redis_client.connect()
    await FastAPILimiter.init(redis_conn)

    # Create database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Yield control to allow application operation
    yield

    # Shutdown the scheduler and Redis connection pool
    # upon application termination
    scheduler.shutdown()
    await app.state.redis_client.close()

# Assign lifespan context to the FastAPI app
app.router.lifespan_context = lifespan

# Main entry point for running the app
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        sio_app,
        host='0.0.0.0',
        port=8000,
        workers=2,
    )
