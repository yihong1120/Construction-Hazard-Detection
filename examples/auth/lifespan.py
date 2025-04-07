from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi_limiter import FastAPILimiter

from examples.auth.database import engine
from examples.auth.jwt_scheduler import start_jwt_scheduler
from examples.auth.models import Base
from examples.auth.redis_pool import RedisClient


@asynccontextmanager
async def global_lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """
    Provide a global lifespan manager for the FastAPI application.

    Args:
        app (FastAPI): The FastAPI application instance to
            manage resources for.

    Yields:
        None: Control is yielded back to the application
            after performing startup tasks.
    """
    # Step 1: Start the scheduler (e.g., for rotating JWT secret keys).
    scheduler = start_jwt_scheduler(app)

    # Step 2: Initialise Redis connection and rate limiter.
    redis_host: str = os.getenv('REDIS_HOST', '127.0.0.1')
    redis_port: str = os.getenv('REDIS_PORT', '6379')
    redis_password: str = os.getenv('REDIS_PASSWORD', '')
    redis_url: str = f"redis://:{redis_password}@{redis_host}:{redis_port}/0"

    app.state.redis_client = RedisClient(redis_url)
    redis_conn = await app.state.redis_client.connect()
    await FastAPILimiter.init(redis_conn)

    # Step 3: Optionally create database tables on startup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # -- All startup logic completed --
    yield  # Provide control back to the application

    # -- Shutdown logic --
    scheduler.shutdown()
    await app.state.redis_client.close()
    # Close database engine if needed
    await engine.dispose()
