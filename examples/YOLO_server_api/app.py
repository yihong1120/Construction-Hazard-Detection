from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from examples.auth.lifespan import global_lifespan
from examples.YOLO_server_api.config import log_configuration
from examples.YOLO_server_api.routers import detection_router
from examples.YOLO_server_api.routers import model_management_router


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start shared services and log YOLO configuration once per worker."""
    log_configuration()
    async with global_lifespan(app):
        yield


app = FastAPI(lifespan=_lifespan)

app.include_router(detection_router)
app.include_router(model_management_router)


def main() -> None:
    """Run the YOLO FastAPI server for local development."""
    uvicorn.run(app, host='127.0.0.1', port=8000, workers=2)


if __name__ == '__main__':
    main()


"""
uvicorn examples.YOLO_server_api.app:app \
    --host 127.0.0.1 --port 8000 --workers 2

uv run uvicorn examples.YOLO_server_api.app:app \
    --host 127.0.0.1 --port 8000 --workers 2
"""
