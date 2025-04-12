from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from examples.auth.lifespan import global_lifespan
from examples.auth.routers import auth_router
from examples.auth.routers import user_management_router
from examples.YOLO_server_api.backend.routers import detection_router
from examples.YOLO_server_api.backend.routers import model_management_router

# Initialise the FastAPI app with a custom lifespan handler
app = FastAPI(lifespan=global_lifespan)

# Add Cross-Origin Resource Sharing (CORS) middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  # Allow all origins (adjust this in production)
    allow_credentials=True,
    allow_methods=['*'],  # Allow all HTTP methods
    allow_headers=['*'],  # Allow all headers
)

# Include routers for authentication, user management,
# object detection services and model management
app.include_router(auth_router)
app.include_router(detection_router)
app.include_router(model_management_router)
app.include_router(user_management_router)


def main():
    """
    Main function to run the FastAPI application using Uvicorn.
    """
    uvicorn.run(
        app,
        host='127.0.0.1',
        port=8000,
        workers=2,
    )


if __name__ == '__main__':
    main()

"""
uvicorn examples.YOLO_server_api.backend.app:app \
    --host 127.0.0.1 --port 8000 --workers 1
"""
