from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from examples.auth import models
from examples.auth.cache import rate_limiter_service
from examples.auth.database import AsyncSessionLocal
from examples.auth.database import engine
from examples.auth.jwt_scheduler import start_jwt_scheduler
from examples.auth.redis_pool import RedisClient
from examples.db_management.services.site_media_cleanup import (
    drain_site_media_cleanup_jobs,
)
from src.http_client_pool import HttpClientPool
from src.http_client_pool import set_application_http_clients


logger = logging.getLogger(__name__)
_site_media_cleanup_interval_seconds = 30


async def _run_site_media_cleanup_worker(
    stop_event: asyncio.Event,
) -> None:
    """Periodically drain durable media-cleanup jobs outside HTTP requests."""
    while not stop_event.is_set():
        try:
            async with AsyncSessionLocal() as db:
                await drain_site_media_cleanup_jobs(db)
        except Exception:
            logger.warning(
                'Deferred site media cleanup worker failed',
                exc_info=True,
            )
        try:
            await asyncio.wait_for(
                stop_event.wait(),
                timeout=_site_media_cleanup_interval_seconds,
            )
        except TimeoutError:
            continue


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
    # Start the scheduler (e.g., for rotating JWT secret keys).
    scheduler = start_jwt_scheduler(app)

    # Initialise Redis connection for auth cache scripts.
    redis_host: str = os.getenv('REDIS_HOST', '127.0.0.1')
    redis_port: str = os.getenv('REDIS_PORT', '6379')
    redis_password: str = os.getenv('REDIS_PASSWORD', '')
    redis_url: str = f"redis://:{redis_password}@{redis_host}:{redis_port}/0"

    app.state.redis_client = RedisClient(redis_url)
    redis_conn = await app.state.redis_client.connect()
    app.state.http_clients = HttpClientPool()
    set_application_http_clients(app.state.http_clients)

    # Preload Lua scripts into Redis (if any).
    try:
        await rate_limiter_service.preload_script(redis_conn)
    except Exception:
        # Log the error or handle it as needed
        pass

    # Schema is owned by the versioned PostgreSQL migrations.  Local scratch
    # environments can opt into metadata creation explicitly; production
    # workers never issue startup DDL or race one another for schema locks.
    if os.getenv('AUTO_CREATE_SCHEMA', '').strip().lower() in {
        '1', 'true', 'yes', 'on',
    }:
        async with engine.begin() as conn:
            await conn.run_sync(models.Base.metadata.create_all)

    # A successful site deletion may leave an outbox row when the file system
    # was unavailable.  Retry a small bounded batch without delaying startup.
    cleanup_worker_enabled = False
    try:
        async with AsyncSessionLocal() as db:
            await drain_site_media_cleanup_jobs(db)
        cleanup_worker_enabled = True
    except Exception:
        logger.warning(
            'Deferred site media cleanup unavailable at startup',
            exc_info=True,
        )

    cleanup_stop_event = asyncio.Event()
    cleanup_task = (
        asyncio.create_task(_run_site_media_cleanup_worker(cleanup_stop_event))
        if cleanup_worker_enabled
        else None
    )

    # -- All startup logic completed --
    yield  # Provide control back to the application

    # -- Shutdown logic --
    cleanup_stop_event.set()
    if cleanup_task is not None:
        await cleanup_task
    scheduler.shutdown()
    await app.state.http_clients.close()
    set_application_http_clients(None)
    await app.state.redis_client.close()
    # Close database engine if needed
    await engine.dispose()
