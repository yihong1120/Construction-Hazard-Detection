from __future__ import annotations

from typing import Annotated

from fastapi import Depends
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.security import HTTPBearer
from jwt.exceptions import InvalidTokenError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from examples.auth.config import Settings
from examples.auth.database import get_db
from examples.auth.models import User
from examples.auth.models import USER_STATUS_ACTIVE
from examples.auth.models import UserIdentity
from examples.auth.oidc import OidcTokenVerifier
from examples.db_management.deps import is_super_admin

settings = Settings()
bearer = HTTPBearer(auto_error=False)
verifier = OidcTokenVerifier(
    issuer=settings.oidc_issuer_url,
    jwks_url=settings.oidc_jwks_url,
    audiences=('account',),
    algorithms=settings.oidc_algorithms,
    jwks_cache_seconds=settings.oidc_jwks_cache_seconds,
    jwks_timeout_seconds=settings.oidc_jwks_timeout_seconds,
)
Database = Annotated[AsyncSession, Depends(get_db)]
BearerCredential = Annotated[
    HTTPAuthorizationCredentials | None,
    Depends(bearer),
]


async def openwebui_user(
    credentials: BearerCredential,
    db: Database,
) -> User:
    """Resolve a verified Open WebUI OIDC subject to its active user."""
    if credentials is None or credentials.scheme.lower() != 'bearer':
        raise HTTPException(401, 'Open WebUI sign-in required.')
    try:
        claims = await verifier.decode_access_token(credentials.credentials)
    except InvalidTokenError as exc:
        raise HTTPException(401, 'Invalid Open WebUI identity.') from exc
    if claims.get('azp') != 'open-webui':
        raise HTTPException(401, 'Invalid Open WebUI client.')
    identity = await db.scalar(
        select(UserIdentity)
        .options(selectinload(UserIdentity.user))
        .where(
            UserIdentity.provider == settings.oidc_identity_provider,
            UserIdentity.provider_user_id == claims['sub'],
        ),
    )
    if identity is None:
        raise HTTPException(401, 'Visionnaire identity is not linked.')
    user = identity.user
    if user.status != USER_STATUS_ACTIVE:
        raise HTTPException(403, 'Visionnaire account is not active.')
    return user


async def openwebui_admin(
    credentials: BearerCredential,
    db: Database,
) -> User:
    """Resolve a verified Open WebUI OIDC subject to a Visionnaire admin."""
    user = await openwebui_user(credentials, db)
    if user.role != 'admin' and not is_super_admin(user):
        raise HTTPException(403, 'Visionnaire administrator required.')
    return user


OpenWebUIUser = Annotated[User, Depends(openwebui_user)]
OpenWebUIAdmin = Annotated[User, Depends(openwebui_admin)]
