from __future__ import annotations

import datetime
from typing import Any


def deployment_claims(deployment: Any | None) -> dict[str, object]:
    """Return deployment-bound claims without coupling callers to its type."""
    if deployment is None:
        return {}
    return {
        'tenant_id': str(deployment.tenant_id),
        'deployment_id': str(deployment.deployment_id),
        'config_revision': deployment.config_revision,
    }


def issue_access_token(
    issuer: Any,
    *,
    username: str,
    user_id: int,
    role: str,
    jti: str,
    feature_names: list[str],
    expires_delta: datetime.timedelta,
    deployment: Any | None,
) -> str:
    """Issue an access token with the common user and deployment claims."""
    return str(
        issuer.create_access_token(
            subject={
                'username': username,
                'user_id': user_id,
                'role': role,
                'jti': jti,
                'features': feature_names,
                **deployment_claims(deployment),
            },
            expires_delta=expires_delta,
            issuer=deployment.issuer if deployment else None,
            audience=deployment.audience if deployment else None,
        ),
    )


def issue_refresh_token(
    issuer: Any,
    *,
    username: str,
    family_id: str,
    token_id: str,
    expires_delta: datetime.timedelta,
    deployment: Any | None,
) -> str:
    """Issue a rotating refresh token with deployment-bound claims."""
    return str(
        issuer.create_access_token(
            subject={
                'username': username,
                'family_id': family_id,
                'token_id': token_id,
                **deployment_claims(deployment),
            },
            expires_delta=expires_delta,
            issuer=deployment.issuer if deployment else None,
            audience=deployment.audience if deployment else None,
        ),
    )
