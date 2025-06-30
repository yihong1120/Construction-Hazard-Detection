from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from examples.auth.lifespan import global_lifespan
from examples.streaming_web.backend.routers import (
    router as streaming_web_router,
)

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

# Include routers for authentication and user management
# and streaming web services
app.include_router(streaming_web_router)


def main() -> None:
    """
    Main function to run the FastAPI application using Uvicorn.
    """
    uvicorn.run(
        app, host='127.0.0.1', port=8800,
    )


if __name__ == '__main__':
    main()

'''
uvicorn examples.streaming_web.backend.app:app \
    --host 127.0.0.1 --port 8800 --workers 4
'''
