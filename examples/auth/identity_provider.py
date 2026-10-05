"""Identity-provider-owned account-management boundaries."""
from __future__ import annotations

from fastapi import HTTPException

from examples.auth.config import Settings

settings = Settings()


def identity_provider_account_url() -> str:
    """Return the account console, or fail closed when it is unavailable."""
    if not settings.oidc_enabled or not settings.oidc_account_url:
        raise HTTPException(
            status_code=404,
            detail='identity_provider_unavailable',
        )
    return settings.oidc_account_url
