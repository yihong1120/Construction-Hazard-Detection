from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import redis.asyncio as redis
import socketio
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI
from fastapi_jwt import JwtAccessBearer
from fastapi_limiter import FastAPILimiter

from .auth import auth_router
from .config import Settings
from .detection import detection_router
from .model_downloader import models_router
from .models import Base
from .models import engine
from .security import update_secret_key

# Instantiate the FastAPI app
app = FastAPI()

# Register API routers for different functionalities
app.include_router(auth_router)
app.include_router(detection_router)
app.include_router(models_router)

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

# Define Socket.IO events
@sio.event
async def connect(sid: str, environ: dict) -> None:
    """
    Handles client connection event to the Socket.IO server.

    Args:
        sid (str): The session ID of the connected client.
        environ (dict): The environment dictionary for the connection.
    """
    print('Client connected:', sid)


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

    # Initialise Redis connection pool for rate limiting
    app.state.redis_pool = await redis.from_url(
        redis_url,
        encoding='utf-8',
        decode_responses=True,
    )
    await FastAPILimiter.init(app.state.redis_pool)

    # Create database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Yield control to allow application operation
    yield

    # Shutdown the scheduler and Redis connection pool
    # upon application termination
    scheduler.shutdown()
    await app.state.redis_pool.close()

# Assign lifespan context to the FastAPI app
app.router.lifespan_context = lifespan

# Main entry point for running the app
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(sio_app, host='0.0.0.0', port=5000)
