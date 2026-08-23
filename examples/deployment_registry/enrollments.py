from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.models import Deployment
from examples.auth.models import DEPLOYMENT_STATUS_ACTIVE
from examples.auth.models import DeploymentEnrollmentCode
from examples.auth.models import Tenant
from examples.auth.models import TENANT_STATUS_ACTIVE
from examples.auth.models import utc_now

# Keep the counter increment and its expiry in one Redis operation.  A worker
# failure between separate commands would otherwise leave a permanent counter.
_RATE_LIMIT_LUA: str = """
local current = redis.call('INCR', KEYS[1])
local ttl = redis.call('TTL', KEYS[1])
if ttl == -1 or ttl == -2 then
    redis.call('EXPIRE', KEYS[1], tonumber(ARGV[1]))
    ttl = tonumber(ARGV[1])
end
return { current, ttl }
"""


@dataclass(frozen=True, slots=True)
class EnrollmentExchangeResult:
    """Represent the non-secret outcome of one enrollment-code redemption.

    Attributes:
        status: Stable result state: ``invalid``, ``terminal``, or
            ``redeemed``.
        deployment_id: Redeemed deployment identifier when the status is
            ``redeemed``.
    """

    status: str
    deployment_id: UUID | None = None


@dataclass(frozen=True, slots=True)
class ProvisionedEnrollmentCode:
    """Hold a raw code only until its successful management response.

    Attributes:
        raw_code: One-time code returned to the authorised caller.
        enrollment: Newly staged database row containing only its verifier.
    """

    raw_code: str
    enrollment: DeploymentEnrollmentCode


def enrollment_code_verifier_hash(raw_code: str, pepper: str) -> str:
    """Create the persisted HMAC verifier for an enrollment code.

    Args:
        raw_code: One-time code supplied by the caller.
        pepper: Server-only HMAC key.

    Returns:
        Hexadecimal SHA-256 HMAC verifier.

    Raises:
        ValueError: If the server pepper is too short.
    """
    secret: bytes = pepper.encode('utf-8')
    if len(secret) < 32:
        raise ValueError(
            'deployment enrollment code pepper is not configured',
        )
    # A keyed verifier remains unusable without the server-only pepper.
    return hmac.new(
        secret,
        raw_code.encode('utf-8'),
        hashlib.sha256,
    ).hexdigest()


async def enforce_enrollment_exchange_rate_limit(
    redis: Redis,
    client_ip: str | None,
    verifier_hash: str,
    maximum: int,
    window_seconds: int,
) -> int | None:
    """Rate-limit redemption attempts by source IP and code verifier.

    Args:
        redis: Asynchronous Redis client used for fixed-window counters.
        client_ip: Optional address of the requesting client.
        verifier_hash: HMAC verifier for the submitted enrollment code.
        maximum: Maximum attempts within a window.
        window_seconds: Fixed-window duration in seconds.

    Returns:
        Retry delay in seconds when over quota; otherwise ``None``.

    Raises:
        ValueError: If the rate-limit configuration is out of range.
    """
    # Bounds prevent a configuration error from disabling protection entirely.
    if not 1 <= maximum <= 1000 or not 1 <= window_seconds <= 24 * 60 * 60:
        raise ValueError('deployment enrollment rate limit is not configured')
    client_identifier: str = client_ip or 'unknown-client'
    # Redis keys retain only hashes, never raw client addresses or verifiers.
    keys: list[str] = [
        'construction-hazard-detection:deployment-enrollment:'
        f"{scope}:{hashlib.sha256(identifier.encode('utf-8')).hexdigest()}"
        for scope, identifier in (
            ('ip', client_identifier),
            ('code', verifier_hash),
        )
    ]
    # Queue both independent atomic scripts to share one Redis round trip.
    async with redis.pipeline(transaction=False) as pipeline:
        for key in keys:
            pipeline.eval(_RATE_LIMIT_LUA, 1, key, window_seconds)
        replies = await pipeline.execute()
    for current, ttl in replies:
        if int(current) > maximum:
            return max(1, int(ttl))
    return None


async def redeem_enrollment_code(
    db: AsyncSession,
    verifier_hash: str,
    now: datetime | None = None,
) -> EnrollmentExchangeResult:
    """Redeem one enrollment code once under a database row lock.

    Args:
        db: Database session used for the redemption transaction.
        verifier_hash: Stored HMAC verifier derived from the supplied code.
        now: Optional redemption timestamp for deterministic callers.

    Returns:
        Stable redemption outcome without exposing the raw code.
    """
    redeemed_at: datetime = now or utc_now()
    async with db.begin():
        # The row lock makes simultaneous redemptions observe one outcome.
        enrollment = await db.scalar(
            select(DeploymentEnrollmentCode)
            .where(
                DeploymentEnrollmentCode.code_verifier_hash == verifier_hash,
            )
            .with_for_update(),
        )
        if enrollment is None:
            return EnrollmentExchangeResult(status='invalid')
        # Expired, revoked, and previously redeemed codes can never be reused.
        if (
            enrollment.redeemed_at is not None
            or enrollment.revoked_at is not None
            or enrollment.expires_at <= redeemed_at
        ):
            return EnrollmentExchangeResult(status='terminal')

        # Lock the active deployment and tenant before consuming the code.
        deployment_id = await db.scalar(
            select(Deployment.id)
            .join(Tenant, Tenant.id == Deployment.tenant_id)
            .where(
                Deployment.id == enrollment.deployment_id,
                Deployment.status == DEPLOYMENT_STATUS_ACTIVE,
                Tenant.status == TENANT_STATUS_ACTIVE,
            )
            .with_for_update(of=(Deployment, Tenant)),
        )
        if deployment_id is None:
            return EnrollmentExchangeResult(status='terminal')

        enrollment.redeemed_at = redeemed_at
        return EnrollmentExchangeResult(
            status='redeemed',
            deployment_id=deployment_id,
        )


async def provision_enrollment_code(
    db: AsyncSession,
    deployment_id: UUID,
    expires_at: datetime,
    created_by: str,
    pepper: str,
    tenant_id: UUID | None = None,
) -> ProvisionedEnrollmentCode:
    """Create one enrollment verifier and retain its raw code for one response.

    Args:
        db: Database session staging the enrollment row.
        deployment_id: Active deployment receiving the code.
        expires_at: UTC expiry timestamp for the code.
        created_by: Authorised operator identifier retained for audit.
        pepper: Server-only HMAC key used to derive the verifier.
        tenant_id: Optional tenant scope for management requests.

    Returns:
        Raw code and staged enrollment row for the caller to commit.

    Raises:
        ValueError: If the input is invalid or the deployment is inactive.
    """
    if not 1 <= len(created_by) <= 160:
        raise ValueError('created_by is required')
    created_at = utc_now()
    if expires_at <= created_at:
        raise ValueError('expires_at must be in the future')
    # The raw code is never persisted; only its HMAC verifier reaches the row.
    value: str = secrets.token_urlsafe(32)
    verifier_hash = enrollment_code_verifier_hash(value, pepper)
    # Tenant scoping is optional only for the trusted operator command.
    criteria = [
        Deployment.id == deployment_id,
        Deployment.status == DEPLOYMENT_STATUS_ACTIVE,
        Tenant.status == TENANT_STATUS_ACTIVE,
    ]
    if tenant_id is not None:
        criteria.append(Deployment.tenant_id == tenant_id)
    deployment = await db.scalar(
        select(Deployment)
        .join(Tenant, Tenant.id == Deployment.tenant_id)
        .where(*criteria),
    )
    if deployment is None:
        raise ValueError('deployment must be active before creating a code')
    enrollment = DeploymentEnrollmentCode(
        deployment_id=deployment_id,
        public_id=UUID(bytes=secrets.token_bytes(16), version=4),
        code_verifier_hash=verifier_hash,
        expires_at=expires_at,
        created_by=created_by,
    )
    # The caller commits this staged row with its matching audit record.
    db.add(enrollment)
    return ProvisionedEnrollmentCode(raw_code=value, enrollment=enrollment)
