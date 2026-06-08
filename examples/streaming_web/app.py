from __future__ import annotations

import os

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from examples.auth.lifespan import global_lifespan
from examples.streaming_web.routers import (
    router as streaming_web_router,
)

# Initialise the FastAPI app with a custom lifespan handler
app = FastAPI(lifespan=global_lifespan)


def _cors_origins() -> list[str]:
    """Return concrete origins allowed to make credentialed browser calls."""
    configured = os.getenv('STREAMING_WEB_CORS_ORIGINS', '')
    if configured.strip():
        return [
            origin.strip()
            for origin in configured.split(',')
            if origin.strip()
        ]
    return [
        'https://changdar-server.mooo.com',
        'http://changdar-server.mooo.com',
        'http://localhost',
        'http://localhost:3000',
        'http://localhost:5000',
        'http://localhost:8080',
        'http://127.0.0.1:3000',
        'http://127.0.0.1:5000',
        'http://127.0.0.1:8080',
    ]


def _cors_origin_regex() -> str | None:
    """Return an optional regex for local Flutter Web development origins."""
    configured = os.getenv('STREAMING_WEB_CORS_ORIGIN_REGEX')
    if configured is not None:
        return configured or None
    return r'https?://(localhost|127\.0\.0\.1)(:\d+)?'


# Add Cross-Origin Resource Sharing (CORS) middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_origin_regex=_cors_origin_regex(),
    allow_credentials=True,
    allow_methods=['*'],  # Allow all HTTP methods
    allow_headers=['*'],  # Allow all headers
)

# Include routers for authentication and user management
# and streaming web services
app.include_router(streaming_web_router)
app.include_router(streaming_web_router, prefix='/hazard')
app.include_router(streaming_web_router, prefix='/api/hazard')


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
uvicorn examples.streaming_web.app:app \
    --host 127.0.0.1 --port 8800 --workers 4

uv run uvicorn examples.streaming_web.app:app\
    --host 127.0.0.1 --port 8800 --workers 4
'''
