from __future__ import annotations

import uvicorn
from fastapi import FastAPI

from examples.auth.lifespan import global_lifespan
from examples.streaming_web.routers import (
    router as streaming_web_router,
)

# Initialise the FastAPI app with a custom lifespan handler
app = FastAPI(lifespan=global_lifespan)

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
uvicorn examples.streaming_web.app:app \
    --host 127.0.0.1 --port 8800 --workers 2

uv run uvicorn examples.streaming_web.app:app\
    --host 127.0.0.1 --port 8800 --workers 2
'''
