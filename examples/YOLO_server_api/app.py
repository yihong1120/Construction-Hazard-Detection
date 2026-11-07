from __future__ import annotations

import argparse
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from examples.auth.lifespan import global_lifespan
from examples.YOLO_server_api.config import log_configuration
from examples.YOLO_server_api.routers import detection_router
from examples.YOLO_server_api.routers import get_inference_runtime
from examples.YOLO_server_api.routers import get_overlay_runtime
from examples.YOLO_server_api.routers import model_management_router
from src.settings.inference import ApiInferenceSettings


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start shared services and log YOLO configuration once per worker."""
    log_configuration()
    runtime = get_inference_runtime()
    overlays = get_overlay_runtime()
    try:
        async with global_lifespan(app):
            yield
    finally:
        try:
            await overlays.close()
        finally:
            await runtime.close()


app = FastAPI(lifespan=_lifespan)

app.include_router(detection_router)
app.include_router(model_management_router)


def main() -> None:
    """Run the YOLO FastAPI server for local development."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    uvicorn.run(
        'examples.YOLO_server_api.app:app',
        host=args.host,
        port=args.port,
        workers=args.workers,
        ws_max_size=ApiInferenceSettings().max_upload_bytes,
    )


if __name__ == '__main__':
    main()
