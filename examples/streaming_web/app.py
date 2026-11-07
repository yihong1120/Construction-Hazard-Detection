from __future__ import annotations

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from examples.auth.lifespan import global_lifespan
from examples.streaming_web.metadata_fanout import metadata_fanout
from examples.streaming_web.routers import (
    router as streaming_web_router,
)
from src.settings.overlay_language import get_overlay_language_policy


@asynccontextmanager
async def streaming_lifespan(app: FastAPI):
    get_overlay_language_policy().default_language()
    async with global_lifespan(app), metadata_fanout.lifespan():
        yield


app = FastAPI(lifespan=streaming_lifespan)

# Route declarations remain in the router module; this file only composes them.
app.include_router(streaming_web_router)


def main() -> None:
    """Run the local streaming-web application with Uvicorn.

    The development entry point deliberately binds to the loopback interface,
    leaving public deployment configuration to the process manager.
    """
    uvicorn.run(
        app,
        host='127.0.0.1',
        port=8800,
    )


if __name__ == '__main__':
    main()

"""
uvicorn examples.streaming_web.app:app \
    --host 127.0.0.1 --port 8800 --workers 2

uv run uvicorn examples.streaming_web.app:app \
    --host 127.0.0.1 --port 8800 --workers 2
"""
