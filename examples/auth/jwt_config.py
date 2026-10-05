from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Annotated
from typing import Any
from typing import cast

from fastapi import HTTPException
from fastapi import Request
from fastapi import Security
from fastapi import status
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.security import HTTPBearer
from jwt.exceptions import InvalidTokenError
from redis.asyncio import Redis
from redis.exceptions import RedisError
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.config import Settings
from examples.auth.database import AsyncSessionLocal
from examples.auth.deployment_context import DeploymentBinding
from examples.auth.deployment_context import resolve_request_deployment
from examples.auth.oidc import OidcTokenVerifier
from examples.auth.oidc_identity import subject_from_oidc_identity
from examples.auth.token_revocation import is_access_token_revoked
from examples.db_management.schemas.auth import AccessTokenSubject

_bearer_scheme = HTTPBearer(auto_error=False)


@dataclass(slots=True)
class JwtAuthorizationCredentials:
    """Verified OIDC credentials mapped to local business permissions."""
    subject: AccessTokenSubject
    payload: dict[str, Any] = field(default_factory=dict)
    token: str = ''
    is_oidc: bool = True

    def __getitem__(self, key: str) -> Any:
        return cast(Any, self.subject)[key]

    def get(self, key: str, default: Any = None) -> Any:
        return cast(Any, self.subject).get(key, default)


class PyJWTBearer:
    """Verify identity-provider access tokens for the current deployment."""

    def __init__(
        self, *, oidc_verifier: OidcTokenVerifier | None = None,
        oidc_identity_provider: str = 'keycloak',
    ) -> None:
        self.oidc_verifier = oidc_verifier
        self.oidc_identity_provider = oidc_identity_provider
        self.bearer_scheme = _bearer_scheme

    async def decode_access_token_for_deployment(
        self,
        token: str,
        db: AsyncSession,
        binding: DeploymentBinding,
    ) -> JwtAuthorizationCredentials:
        """Verify one access token and map it to local deployment grants.

        Both regular API dependencies and lower-level services (such as media
        playback) need the same OIDC subject mapping.  Keeping it here avoids
        using different verification policies across transports.
        The caller remains responsible for checking the mapped JTI against the
        revocation store.
        """
        if self.oidc_verifier is None:
            raise HTTPException(503, detail='Identity provider unavailable')
        if not self.oidc_verifier.matches_configured_issuer(token):
            raise InvalidTokenError('Invalid identity-provider issuer')
        payload = await self.oidc_verifier.decode_access_token(token)
        if self.oidc_identity_provider == 'keycloak' and payload.get(
            'typ',
        ) != 'Bearer':
            raise InvalidTokenError('Keycloak API access token required')
        subject = await subject_from_oidc_identity(
            db, payload, provider=self.oidc_identity_provider, binding=binding,
        )
        tenant_id = subject.get('tenant_id')
        deployment_id = subject.get('deployment_id')
        config_revision = subject.get('config_revision')
        if (
            not isinstance(tenant_id, str)
            or not isinstance(deployment_id, str)
            or not isinstance(config_revision, int)
            or str(binding.tenant_id) != tenant_id
            or str(binding.deployment_id) != deployment_id
            or binding.config_revision != config_revision
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    'code': 'deployment_configuration_changed',
                    'message': (
                        'Deployment configuration changed; sign in again.'
                    ),
                },
            )
        return JwtAuthorizationCredentials(
            subject=subject,
            payload=payload,
            token=token,
            is_oidc=True,
        )

    async def __call__(
        self,
        request: Request,
        bearer: Annotated[
            HTTPAuthorizationCredentials | None, Security(_bearer_scheme),
        ] = None,
    ) -> JwtAuthorizationCredentials:
        """Perform call.

        Args:
            request: Value used by this callable.

        Returns:
            The callable result.
        """
        if bearer is None:
            bearer = await self.bearer_scheme(request)
        token = bearer.credentials if bearer is not None else None
        redis_client = getattr(request.app.state, 'redis_client', None)
        redis = getattr(redis_client, 'client', None)
        return await self.authenticate_token(request, token, redis)

    async def authenticate_token(
        self,
        request: Request,
        token: str | None,
        redis: Redis | None,
    ) -> JwtAuthorizationCredentials:
        """Validate HTTP or WebSocket credentials with one security policy."""
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Could not validate credentials',
            headers={'WWW-Authenticate': 'Bearer'},
        )
        try:
            if token is None:
                raise credentials_exception
            async with AsyncSessionLocal() as db:
                binding = await resolve_request_deployment(request, db)
                credentials = await self.decode_access_token_for_deployment(
                    token, db, binding,
                )
                payload = credentials.payload
                subject = credentials.subject
            if redis is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail='Authentication revocation service unavailable',
                )
            if await is_access_token_revoked(
                redis,
                {'jti': subject['jti']},
            ):
                raise credentials_exception
        except InvalidTokenError:
            raise credentials_exception
        except RedisError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail='Authentication revocation service unavailable',
            ) from exc

        return JwtAuthorizationCredentials(
            subject=subject,
            payload=payload,
            token=token,
            is_oidc=True,
        )


settings = Settings()
oidc_access_verifier = OidcTokenVerifier.from_settings(settings)
jwt_access = PyJWTBearer(
    oidc_verifier=oidc_access_verifier,
    oidc_identity_provider=settings.oidc_identity_provider,
)
