from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from examples.auth.lifespan import global_lifespan
from examples.YOLO_server_api.backend.routers import detection_router
from examples.YOLO_server_api.backend.routers import model_management_router


app = FastAPI(lifespan=global_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(detection_router)
app.include_router(model_management_router)


def main() -> None:
    uvicorn.run(app, host='127.0.0.1', port=8000, workers=2)


if __name__ == '__main__':
    main()

"""
uvicorn examples.YOLO_server_api.backend.app:app \
    --host 127.0.0.1 --port 8000 --workers 8
"""
