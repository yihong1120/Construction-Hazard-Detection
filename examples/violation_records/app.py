from __future__ import annotations

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from examples.auth.lifespan import global_lifespan
from examples.violation_records.evidence_storage import (
    EvidenceStorageMaintenance,
)
from examples.violation_records.response_debug import ViolationResponseDebug
from examples.violation_records.routers import router as violation_router


# Share database and Redis lifecycle management with the other API services.
@asynccontextmanager
async def evidence_lifespan(app: FastAPI):
    async with global_lifespan(app), EvidenceStorageMaintenance().lifespan():
        yield


app: FastAPI = FastAPI(lifespan=evidence_lifespan)

# Keep endpoint implementation in the router and service modules.
app.include_router(violation_router)
app.add_middleware(ViolationResponseDebug)


def main() -> None:
    """Run the violation-record ASGI application with Uvicorn.

    This entry point is intended for local development; deployed processes
    should be managed by the production process supervisor.
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
uvicorn examples.violation_records.app:app \
    --host 127.0.0.1 --port 8002 --workers 4

uv run uvicorn examples.violation_records.app:app \
    --host 127.0.0.1 --port 8002 --workers 4
"""
