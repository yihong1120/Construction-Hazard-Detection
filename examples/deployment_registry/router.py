from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter
from fastapi import Depends
from fastapi import Request
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.database import get_db
from examples.auth.redis_pool import get_redis_pool
from examples.deployment_registry.schemas import DeploymentRegistryDocument
from examples.deployment_registry.schemas import EnrollmentExchangeRequest
from examples.deployment_registry.schemas import EnrollmentExchangeResponse
from examples.deployment_registry.service import exchange_enrollment_code
from examples.deployment_registry.service import get_deployment_registry_document

# The application mounts this public router outside user-facing API routes.
router: APIRouter = APIRouter(
    prefix='/deployment-registry',
    tags=['deployment-registry'],
)


@router.post(
    '/v1/enrollments/exchange',
    response_model=EnrollmentExchangeResponse,
)
# The response model performs the sole HTTP-boundary serialisation check.
async def exchange_enrollment_code_endpoint(
    payload: EnrollmentExchangeRequest,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
    redis: Annotated[Redis, Depends(get_redis_pool)],
) -> dict[str, UUID]:
    """Exchange one anonymous enrollment code for a deployment identifier.

    Args:
        payload: Validated one-time enrollment code request.
        request: Incoming request used to obtain the client address.
        db: Database session for atomic code redemption.
        redis: Redis client for the enrollment rate limit.

    Returns:
        Deployment identifier associated with the redeemed code.
    """
    return {
        'deployment_id': await exchange_enrollment_code(
            payload.enrollment_code,
            request.client.host if request.client is not None else None,
            db,
            redis,
        ),
    }


@router.get(
    '/v1/deployments/{deployment_id}',
    response_model=DeploymentRegistryDocument,
)
# The service returns plain data; FastAPI validates it exactly once here.
async def get_deployment_registry_document_endpoint(
    deployment_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict[str, object]:
    """Return a freshly signed public Registry document.

    Args:
        deployment_id: Deployment identifier from the public route.
        db: Database session used to load the deployment.

    Returns:
        Signed Registry document validated by the route response model.
    """
    return await get_deployment_registry_document(deployment_id, db)
