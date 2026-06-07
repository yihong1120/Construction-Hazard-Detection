from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import Any

import jwt
from fastapi import HTTPException
from fastapi import Request
from fastapi import status
from fastapi.security import OAuth2PasswordBearer
from jwt.exceptions import InvalidTokenError

from examples.auth.config import Settings


@dataclass(slots=True)
class JwtAuthorizationCredentials:
    """Decoded JWT credentials exposed to FastAPI security dependencies."""

    subject: dict[str, Any]
    payload: dict[str, Any] = field(default_factory=dict)
    token: str = ''


class PyJWTBearer:
    """FastAPI security dependency backed directly by PyJWT."""

    def __init__(
        self,
        secret_key: str,
        algorithm: str = 'HS256',
        token_url: str = '/api/auth/login',
    ) -> None:
        """Initialise the JWT bearer dependency.

        Args:
            secret_key: Secret used to verify JWT signatures.
            algorithm: JWT signing algorithm.
            token_url: OAuth2 token URL exposed in OpenAPI.
        """
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.oauth2_scheme = OAuth2PasswordBearer(tokenUrl=token_url)

    def create_access_token(
        self,
        subject: dict[str, Any],
        expires_delta: timedelta | None = None,
    ) -> str:
        """Create a signed JWT using PyJWT."""
        now = datetime.now(timezone.utc)
        expire = now + (expires_delta or timedelta(minutes=15))
        to_encode: dict[str, Any] = {
            'sub': subject.get('username'),
            'subject': subject,
            'iat': now,
            'exp': expire,
        }
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)

    def decode_token(
        self,
        token: str,
        verify_exp: bool = True,
    ) -> dict[str, Any]:
        """Decode and validate a JWT using PyJWT."""
        return jwt.decode(
            token,
            self.secret_key,
            algorithms=[self.algorithm],
            options={'verify_exp': verify_exp},
        )

    async def __call__(self, request: Request) -> JwtAuthorizationCredentials:
        token = await self.oauth2_scheme(request)
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Could not validate credentials',
            headers={'WWW-Authenticate': 'Bearer'},
        )
        try:
            if token is None:
                raise credentials_exception
            payload = self.decode_token(token)
        except InvalidTokenError:
            raise credentials_exception

        subject = payload.get('subject')
        if not isinstance(subject, dict):
            sub = payload.get('sub')
            subject = {'username': sub} if isinstance(sub, str) else {}

        if not subject:
            raise credentials_exception

        return JwtAuthorizationCredentials(
            subject=subject,
            payload=payload,
            token=token,
        )


settings: Settings = Settings()

jwt_access: PyJWTBearer = PyJWTBearer(
    secret_key=settings.authjwt_secret_key,
    algorithm=settings.ALGORITHM,
)

jwt_refresh: PyJWTBearer = PyJWTBearer(
    secret_key=settings.authjwt_secret_key,
    algorithm=settings.ALGORITHM,
)
