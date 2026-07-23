from __future__ import annotations

import uvicorn
from fastapi import FastAPI

from examples.auth.lifespan import global_lifespan
from examples.violation_records.routers import router as violation_router

# Initialise the FastAPI app with a custom lifespan handler
app: FastAPI = FastAPI(lifespan=global_lifespan)

# Include routers for violation records
app.include_router(violation_router)


def main() -> None:
    """
    Main function to run the FastAPI application using Uvicorn.
    """
    uvicorn.run(
        'examples.violation_records.app:app',
        host='0.0.0.0',
        port=8081,
        reload=True,
    )


if __name__ == '__main__':
    main()

"""
uvicorn examples.violation_records.app:app\
    --host 127.0.0.1\
    --port 8002 --workers 4

uv run uvicorn examples.violation_records.app:app\
    --host 127.0.0.1\
    --port 8002 --workers 4
"""
