from __future__ import annotations

from uuid import UUID

from fastapi import HTTPException
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.config import Settings
from examples.auth.models import Deployment
from examples.auth.models import DEPLOYMENT_STATUS_ACTIVE
from examples.auth.models import TENANT_STATUS_ACTIVE
from examples.deployment_registry.enrollments import (
    enforce_enrollment_exchange_rate_limit,
)
from examples.deployment_registry.enrollments import (
    enrollment_code_verifier_hash,
)
from examples.deployment_registry.enrollments import redeem_enrollment_code
from examples.deployment_registry.signing import build_registry_document

# Construct settings once because the module serves a fixed application config.
settings: Settings = Settings()


async def exchange_enrollment_code(
    raw_code: str,
    client_ip: str | None,
    db: AsyncSession,
    redis: Redis,
) -> UUID:
    """Atomically exchange one enrollment code for a deployment identifier.

    Args:
        raw_code: Validated raw enrollment code.
        client_ip: Optional client address used by the rate limiter.
        db: Database session used for the atomic redemption transaction.
        redis: Redis client used for anonymous rate limits.

    Returns:
        Deployment identifier issued by the redeemed code.

    Raises:
        HTTPException: If the code is rate-limited, invalid, or terminal.
    """
    # Hash before rate limiting so Redis never receives the raw enrollment
    # code.
    verifier_hash = enrollment_code_verifier_hash(
        raw_code,
        settings.deployment_enrollment_code_pepper,
    )
    retry_after: int | None = await enforce_enrollment_exchange_rate_limit(
        redis,
        client_ip=client_ip,
        verifier_hash=verifier_hash,
        maximum=settings.deployment_enrollment_rate_limit_max,
        window_seconds=(
            settings.deployment_enrollment_rate_limit_window_seconds
        ),
    )
    if retry_after is not None:
        # The client can wait for the Redis fixed window rather than retrying.
        raise HTTPException(
            status_code=429,
            detail={'code': 'enrollment_rate_limited'},
            headers={'Retry-After': str(max(1, retry_after))},
        )

    # Redemption holds the database lock and consumes a valid code once.
    result = await redeem_enrollment_code(db, verifier_hash=verifier_hash)
    if result.status == 'invalid':
        raise HTTPException(
            status_code=403,
            detail={'code': 'invalid_enrollment_code'},
        )
    if result.status != 'redeemed' or result.deployment_id is None:
        raise HTTPException(
            status_code=410,
            detail={'code': 'enrollment_unavailable_or_expired'},
        )
    return result.deployment_id


async def get_deployment_registry_document(
    deployment_id: UUID,
    db: AsyncSession,
) -> dict[str, object]:
    """Build a fresh signed Registry document for one active deployment.

    Args:
        deployment_id: Canonical deployment identifier from the route path.
        db: Database session used to load deployment state.

    Returns:
        Signed deployment registry document.

    Raises:
        HTTPException: If the deployment is unavailable or revoked.
    """
    # Load one deployment; the tenant relationship supplies its active status.
    deployment = await db.scalar(
        select(Deployment).where(Deployment.id == deployment_id),
    )
    if deployment is None:
        raise HTTPException(
            status_code=404,
            detail={'code': 'deployment_not_found'},
        )
    if (
        deployment.status != DEPLOYMENT_STATUS_ACTIVE
        or deployment.tenant.status != TENANT_STATUS_ACTIVE
    ):
        raise HTTPException(
            status_code=410,
            detail={'code': 'deployment_revoked'},
        )
    # Signing happens only after deployment and tenant status are confirmed.
    document = build_registry_document(
        deployment,
        private_key_pem=settings.deployment_registry_ed25519_private_key,
        key_id=settings.deployment_registry_key_id,
        ttl_seconds=settings.deployment_registry_ttl_seconds,
    )
    return document
