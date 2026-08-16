from __future__ import annotations

import uvicorn
from fastapi import FastAPI

from examples.auth.lifespan import global_lifespan
from examples.streaming_web.routers import (
    router as streaming_web_router,
)

# Reuse the common lifespan so worker resources have one lifecycle contract.
app = FastAPI(lifespan=global_lifespan)

# Route declarations remain in the router module; this file only composes them.
app.include_router(streaming_web_router)


def main() -> None:
    """Run the local streaming-web application with Uvicorn.

    The development entry point deliberately binds to the loopback interface,
    leaving public deployment configuration to the process manager.
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
