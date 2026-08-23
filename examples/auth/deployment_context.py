from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass
from urllib.parse import urlsplit
from urllib.parse import urlunsplit
from uuid import UUID

from fastapi import HTTPException
from fastapi import Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.models import Deployment
from examples.auth.models import Tenant


@dataclass(frozen=True, slots=True)
class DeploymentBinding:
    """Immutable security contract for one active tenant deployment."""

    deployment_id: UUID
    tenant_id: UUID
    api_base_url: str
    config_revision: int

    @property
    def issuer(self) -> str:
        """Return the issuer bound to this deployment's canonical API URL."""
        return self.api_base_url

    @property
    def audience(self) -> str:
        """Return the audience unique to this deployment."""
        return f'construction-hazard-detection:deployment:{self.deployment_id}'

    def as_response(self) -> dict[str, object]:
        """Return the public, non-secret deployment response object."""
        return {
            'deployment_id': str(self.deployment_id),
            'tenant_id': str(self.tenant_id),
            'config_revision': self.config_revision,
        }


def _is_loopback_host(value: str | None) -> bool:
    """Return whether a host value is a loopback address or localhost."""
    if not value:
        return False
    if value.lower() == 'localhost':
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def trusted_local_development_deployment_id(request: Request) -> UUID | None:
    """Return the server-configured deployment for an explicit loopback mode.

    This is a development-only exception for a client talking directly to a
    local Uvicorn port.  It does not trust a client-provided tenant or origin:
    both the socket peer and Host must be loopback, and the deployment ID is
    supplied only by the backend environment.
    """
    enabled = os.getenv('LOCAL_DEVELOPMENT_AUTH_ENABLED', '').strip().lower()
    if enabled not in {'1', 'true', 'yes', 'on'}:
        return None
    client = request.client
    if not (
        client
        and _is_loopback_host(client.host)
        and _is_loopback_host(request.url.hostname)
    ):
        return None
    value = os.getenv('LOCAL_DEVELOPMENT_DEPLOYMENT_ID', '').strip()
    try:
        return UUID(value)
    except ValueError as exc:
        raise HTTPException(
            status_code=503,
            detail={'code': 'local_development_deployment_not_configured'},
        ) from exc


def canonical_api_base_url(value: str) -> str:
    """Validate and canonicalise an HTTPS API root URL.

    The URL must be an absolute HTTPS URL with the server-configured public
    API path, without credentials, query, or fragment.  For this deployment
    topology the public root is normally ``https://<host>/hazard/api``.
    Service paths such as ``/db_management`` are appended by trusted client
    code and are never part of the deployment configuration.
    """
    raw = value.strip()
    parsed = urlsplit(raw)
    if (
        parsed.scheme.lower() != 'https'
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError('api_base_url must be an absolute HTTPS URL')
    try:
        host = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ValueError('api_base_url must have a valid HTTPS port') from exc
    if not host:
        raise ValueError('api_base_url must include a hostname')
    normalised_host = host.rstrip('.').lower()
    # URL parsing removes IPv6 brackets from ``hostname``; restore them when
    # rebuilding the authority component.
    if ':' in normalised_host:
        normalised_host = f'[{normalised_host}]'
    normalised_netloc = normalised_host
    if port is not None and port != 443:
        normalised_netloc = f'{normalised_netloc}:{port}'

    path = parsed.path.rstrip('/')
    configured_path = os.getenv(
        'DEPLOYMENT_API_BASE_PATH',
        '/hazard/api',
    ).strip().rstrip('/')
    path_segments = path.split('/')
    if (
        path in {'', '/'}
        or not path.startswith('/')
        or '%' in path
        or any(segment in {'', '.', '..'} for segment in path_segments[1:])
        or not configured_path.startswith('/')
        or configured_path in {'', '/'}
        or path != configured_path
    ):
        raise ValueError(
            'api_base_url must match the configured public API root',
        )
    return urlunsplit(('https', normalised_netloc, path, '', ''))


def request_api_base_url(request: Request) -> str:
    """Return the canonical external API root represented by a request.

    Deployment selection comes from ASGI's resolved external scheme and host
    plus a server-configured public API path, rather than an application-
    provided tenant, deployment, or API-origin header.  The reverse proxy
    removes the public path before forwarding, so it cannot be reconstructed
    from the ASGI request path.
    """
    api_path = os.getenv('DEPLOYMENT_API_BASE_PATH', '/hazard/api').strip()
    return canonical_api_base_url(
        f'{request.url.scheme}://{request.url.netloc}{api_path}',
    )


def binding_from_deployment(deployment: Deployment) -> DeploymentBinding:
    """Convert an active ORM deployment into its token/security contract."""
    return DeploymentBinding(
        deployment_id=deployment.id,
        tenant_id=deployment.tenant_id,
        api_base_url=deployment.api_base_url,
        config_revision=deployment.config_revision,
    )


async def resolve_request_deployment(
    request: Request,
    db: AsyncSession,
) -> DeploymentBinding:
    """Resolve one active deployment solely from the request's API origin.

    A deployed API never falls back to a default tenant.  Such a fallback
    would let a host/header mistake silently cross a tenant boundary.
    """
    local_development_id = trusted_local_development_deployment_id(request)
    if local_development_id is not None:
        deployment = await db.scalar(
            select(Deployment)
            .join(Tenant, Tenant.id == Deployment.tenant_id)
            .where(Deployment.id == local_development_id),
        )
    else:
        try:
            api_base_url = request_api_base_url(request)
        except ValueError as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    'code': 'deployment_origin_mismatch',
                    'message': (
                        'The request did not use a canonical HTTPS API origin.'
                    ),
                },
            ) from exc

        deployment = await db.scalar(
            select(Deployment)
            .join(Tenant, Tenant.id == Deployment.tenant_id)
            .where(Deployment.api_base_url == api_base_url),
        )
    if deployment is None:
        raise HTTPException(
            status_code=409,
            detail={
                'code': 'unknown_deployment_origin',
                'message': 'This API origin is not registered as a deployment.',
            },
        )
    if deployment.tenant.status != 'active':
        raise HTTPException(
            status_code=409,
            detail={
                'code': 'tenant_disabled',
                'message': 'This tenant is disabled; sign in is unavailable.',
            },
        )
    if deployment.status != 'active':
        raise HTTPException(
            status_code=409,
            detail={
                'code': 'deployment_revoked',
                'message': 'This deployment is revoked; sign in is unavailable.',
            },
        )
    return binding_from_deployment(deployment)


async def require_deployment_match(
    request: Request,
    db: AsyncSession,
    *,
    tenant_id: str,
    deployment_id: str,
    config_revision: int,
) -> DeploymentBinding:
    """Resolve and compare a token/session deployment contract.

    Any mismatch is deliberately a conflict rather than a generic server
    error: the client must re-read the signed Registry profile and sign in again.
    """
    binding = await resolve_request_deployment(request, db)
    if (
        str(binding.tenant_id) != tenant_id
        or str(binding.deployment_id) != deployment_id
        or binding.config_revision != config_revision
    ):
        raise HTTPException(
            status_code=409,
            detail={
                'code': 'deployment_configuration_changed',
                'message': 'Deployment configuration changed; sign in again.',
            },
        )
    return binding
