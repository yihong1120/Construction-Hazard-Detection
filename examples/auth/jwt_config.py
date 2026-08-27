from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import Any
from typing import cast
from uuid import uuid4

import jwt
from fastapi import HTTPException
from fastapi import Request
from fastapi import status
from fastapi.security import OAuth2PasswordBearer
from jwt.exceptions import InvalidTokenError
from pydantic import ValidationError
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
from examples.db_management.schemas.auth import AccessTokenSubjectModel
from examples.db_management.schemas.auth import RefreshTokenSubject
from examples.db_management.schemas.auth import RefreshTokenSubjectModel


def access_token_subject_from_payload(
    payload: Mapping[str, object],
) -> AccessTokenSubject:
    """Return the validated, canonical access-token subject."""
    try:
        subject = cast(
            AccessTokenSubject,
            AccessTokenSubjectModel.model_validate(
                payload['subject'],
            ).model_dump(
                exclude_none=True,
            ),
        )
        if payload['jti'] != subject['jti']:
            raise InvalidTokenError('Access-token JTI does not match subject')
        return subject
    except (KeyError, ValidationError) as exc:
        raise InvalidTokenError('Invalid token subject') from exc


def refresh_token_subject_from_payload(
    payload: Mapping[str, object],
) -> RefreshTokenSubject:
    """Return the validated, canonical refresh-token subject."""
    try:
        return cast(
            RefreshTokenSubject,
            RefreshTokenSubjectModel.model_validate(
                payload['subject'],
            ).model_dump(exclude_none=True),
        )
    except (KeyError, ValidationError) as exc:
        raise InvalidTokenError('Invalid token subject') from exc


@dataclass(slots=True)
class JwtAuthorizationCredentials:
    """Decoded JWT credentials exposed to FastAPI security dependencies."""

    subject: AccessTokenSubject
    payload: dict[str, Any] = field(default_factory=dict)
    token: str = ''

    def __getitem__(self, key: str) -> Any:
        """Support existing handlers that read claims as a mapping."""
        return cast(Any, self.subject)[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Support existing handlers that use mapping-style claim access."""
        return cast(Any, self.subject).get(key, default)


class PyJWTBearer:
    """FastAPI security dependency backed directly by PyJWT."""

    def __init__(
        self,
        secret_key: str,
        algorithm: str = 'HS256',
        token_url: str = '/api/auth/login',
        token_use: str = 'access',
        oidc_verifier: OidcTokenVerifier | None = None,
        oidc_identity_provider: str = 'keycloak',
    ) -> None:
        """Initialise the JWT bearer dependency.

        Args:
            secret_key: Secret used to verify JWT signatures.
            algorithm: JWT signing algorithm.
            token_url: OAuth2 token URL exposed in OpenAPI.
        """
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.token_use = token_use
        self.oidc_verifier = oidc_verifier
        self.oidc_identity_provider = oidc_identity_provider
        self.oauth2_scheme = OAuth2PasswordBearer(tokenUrl=token_url)

    def create_access_token(
        self,
        subject: Mapping[str, object],
        expires_delta: timedelta | None = None,
        *,
        issuer: str | None = None,
        audience: str | None = None,
    ) -> str:
        """Create a signed JWT using PyJWT."""
        now = datetime.now(timezone.utc)
        expire = now + (expires_delta or timedelta(minutes=15))
        subject_data: AccessTokenSubject | RefreshTokenSubject
        if self.token_use == 'access':
            raw_subject = dict(subject)
            raw_subject['jti'] = raw_subject.get('jti') or str(uuid4())
            subject_data = cast(
                AccessTokenSubject,
                AccessTokenSubjectModel.model_validate(
                    raw_subject,
                ).model_dump(),
            )
        else:
            subject_data = cast(
                RefreshTokenSubject,
                RefreshTokenSubjectModel.model_validate(subject).model_dump(),
            )
        to_encode: dict[str, Any] = {
            'sub': subject_data['username'],
            'subject': subject_data,
            'token_use': self.token_use,
            # Deployment-issued tokens receive an explicit API-origin issuer
            # and deployment audience.  The legacy defaults only preserve
            # programmatic token construction for maintenance tooling; HTTP
            # authentication rejects subjects without deployment claims.
            'aud': audience or f"docformify:{self.token_use}",
            'iss': issuer or 'docformify',
            'iat': now,
            'exp': expire,
        }
        if self.token_use == 'access':
            to_encode['jti'] = cast(
                AccessTokenSubject,
                subject_data,
            )['jti']
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)

    def decode_token(
        self,
        token: str,
        verify_exp: bool = True,
        *,
        expected_issuer: str | None = None,
        expected_audience: str | None = None,
    ) -> dict[str, Any]:
        """Decode and validate a JWT with the shared token contract."""
        payload = jwt.decode(
            token,
            self.secret_key,
            algorithms=[self.algorithm],
            audience=expected_audience or f"docformify:{self.token_use}",
            issuer=expected_issuer or 'docformify',
            options={
                'verify_exp': verify_exp,
                'require': [
                    'exp',
                    'iat',
                    'sub',
                    'subject',
                    'token_use',
                    'aud',
                    'iss',
                ],
            },
        )
        if payload.get('token_use') != self.token_use:
            raise InvalidTokenError('Invalid token use')
        if self.token_use == 'access':
            payload['subject'] = access_token_subject_from_payload(payload)
        else:
            payload['subject'] = refresh_token_subject_from_payload(payload)
        return payload

    def decode_token_for_lifecycle(
        self,
        token: str,
        verify_exp: bool = True,
    ) -> dict[str, Any]:
        """Verify a token for cleanup or revocation without authorising it.

        Normal request authentication always supplies the deployment selected
        from the current API origin. Cache cleanup and logout instead need to
        recognise a previously issued token solely to expire or revoke it. In
        that narrow case, parse its issuer/audience without trust, then verify
        the signature and both claims using those values. Callers must never
        use this method to grant access.
        """
        try:
            unsigned = jwt.decode(
                token,
                options={
                    'verify_signature': False,
                    'verify_exp': False,
                    'verify_aud': False,
                    'verify_iss': False,
                },
            )
            issuer = unsigned.get('iss')
            audience = unsigned.get('aud')
        except jwt.PyJWTError:
            # Preserve the regular decoder's canonical error handling.
            if verify_exp:
                return self.decode_token(token)
            return self.decode_token(token, verify_exp=False)
        if isinstance(issuer, str) and isinstance(audience, str):
            if not verify_exp:
                return self.decode_token(
                    token,
                    verify_exp=False,
                    expected_issuer=issuer,
                    expected_audience=audience,
                )
            return self.decode_token(
                token,
                expected_issuer=issuer,
                expected_audience=audience,
            )
        if verify_exp:
            return self.decode_token(token)
        return self.decode_token(token, verify_exp=False)

    async def decode_access_token_for_deployment(
        self,
        token: str,
        db: AsyncSession,
        binding: DeploymentBinding,
    ) -> JwtAuthorizationCredentials:
        """Verify one access token and map it to local deployment grants.

        Both regular API dependencies and lower-level services (such as media
        playback) need the same OIDC subject mapping.  Keeping it here avoids
        accidentally applying the legacy HMAC JWT decoder to a Keycloak token.
        The caller remains responsible for checking the mapped JTI against the
        revocation store.
        """
        if self.token_use != 'access':
            raise InvalidTokenError('Invalid token use')

        oidc_token = bool(
            self.oidc_verifier is not None
            and self.oidc_verifier.matches_configured_issuer(token),
        )
        if settings.oidc_passwords_managed_externally and not oidc_token:
            raise InvalidTokenError('Legacy access tokens are disabled')

        if oidc_token:
            assert self.oidc_verifier is not None
            payload = await self.oidc_verifier.decode_access_token(token)
            subject = await subject_from_oidc_identity(
                db,
                payload,
                provider=self.oidc_identity_provider,
                binding=binding,
            )
        else:
            payload = self.decode_token(
                token,
                expected_issuer=binding.issuer,
                expected_audience=binding.audience,
            )
            subject = access_token_subject_from_payload(payload)

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
        )

    async def __call__(self, request: Request) -> JwtAuthorizationCredentials:
        """Perform call.

        Args:
            request: Value used by this callable.

        Returns:
            The callable result.
        """
        token = await self.oauth2_scheme(request)
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Could not validate credentials',
            headers={'WWW-Authenticate': 'Bearer'},
        )
        try:
            if token is None:
                raise credentials_exception
            binding = None
            if isinstance(request, Request):
                async with AsyncSessionLocal() as db:
                    # Resolve from the actual API origin before token decoding;
                    # neither a client header nor a body field can select it.
                    binding = await resolve_request_deployment(request, db)
                    credentials = (
                        await self.decode_access_token_for_deployment(
                            token,
                            db,
                            binding,
                        )
                    )
                    payload = credentials.payload
                    subject = credentials.subject
            else:
                payload = self.decode_token(token)
                subject = access_token_subject_from_payload(payload)
            if self.token_use != 'access':
                raise credentials_exception
            redis_client = getattr(request.app.state, 'redis_client', None)
            redis = getattr(redis_client, 'client', None)
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
        )


settings: Settings = Settings()

oidc_access_verifier = OidcTokenVerifier.from_settings(settings)

jwt_access: PyJWTBearer = PyJWTBearer(
    secret_key=settings.authjwt_secret_key,
    algorithm=settings.ALGORITHM,
    token_use='access',
    oidc_verifier=oidc_access_verifier,
    oidc_identity_provider=settings.oidc_identity_provider,
)

jwt_refresh: PyJWTBearer = PyJWTBearer(
    secret_key=settings.authjwt_secret_key,
    algorithm=settings.ALGORITHM,
    token_use='refresh',
)
