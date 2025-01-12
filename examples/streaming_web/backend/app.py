from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import socketio
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_limiter import FastAPILimiter

from .routes import register_routes
from .sockets import register_sockets
from .utils import RedisManager
# from fastapi.staticfiles import StaticFiles

redis_manager = RedisManager()


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

# Uncomment and use the following endpoint for file uploads if needed
# Mount the static files directory to serve static assets
# app.mount(
#     '/static',
#     StaticFiles(directory='examples/streaming_web/backend/static'),
#     name='static',
# )

# Initialise Socket.IO server with ASGI support
sio = socketio.AsyncServer(async_mode='asgi')
sio_app = socketio.ASGIApp(sio, app)

# Register application routes and Socket.IO events
register_routes(app)
register_sockets(sio, redis_manager)


def run_server():
    """
    Run the application using Uvicorn ASGI server.
    """
    uvicorn.run(
        'examples.streaming_web.backend.app:sio_app',
        host='127.0.0.1',
        port=8000,
        log_level='info',
    )


if __name__ == '__main__':
    run_server()
