from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
import os

import socketio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi_limiter import FastAPILimiter

from .routes import register_routes
from .sockets import register_sockets
from .utils import RedisManager

# Define Redis connection settings with default values and initialise RedisManager
redis_host = os.getenv('REDIS_HOST', 'localhost')
redis_port = int(os.getenv('REDIS_PORT', 6379))
redis_password = os.getenv('REDIS_PASSWORD', None)

try:
    redis_manager = RedisManager(redis_host, redis_port, redis_password)
    print("Redis connection initialised successfully.")
except Exception as e:
    print(f"Failed to initialise Redis connection: {e}")
    raise SystemExit("Exiting application due to Redis connection failure.")

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """
    Initialises resources at startup and performs cleanup on shutdown.

    Args:
        app (FastAPI): The FastAPI application instance.

    Yields:
        None
    """
    # Initialises rate limiter with the Redis client
    await FastAPILimiter.init(redis_manager.client)
    try:
        yield
    finally:
        # Cleanup code: Closes the Redis connection after application shutdown
        await redis_manager.client.close()
        print('Redis connection closed.')


# Create the FastAPI application with a lifespan manager for setup and cleanup
app = FastAPI(lifespan=lifespan)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Mount the static files directory to serve static assets
app.mount(
    '/static',
    StaticFiles(directory='examples/streaming_web/static'),
    name='static',
)

# Initialise Socket.IO server with ASGI support
sio = socketio.AsyncServer(async_mode='asgi')
sio_app = socketio.ASGIApp(sio, app)

# Register application routes and Socket.IO events
register_routes(app)
register_sockets(sio, redis_manager)

# Run the application using Uvicorn ASGI server
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        'examples.streaming_web.app:sio_app',
        host='127.0.0.1',
        port=8000,
        log_level='info',
    )
