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


def require_local_password_management() -> None:
    """Reject legacy password writes after the OIDC password cutover.

    Returning a structured account-console URL lets Visionnaire keep ownership
    of its account UI while delegating the actual password update to Keycloak.
    """
    if not settings.oidc_passwords_managed_externally:
        return
    raise HTTPException(
        status_code=409,
        detail={
            'code': 'password_managed_by_identity_provider',
            'account_url': identity_provider_account_url(),
        },
    )


def require_local_login() -> None:
    """Reject legacy local login and refresh after the OIDC cutover.

    Keeping legacy password endpoints active after Keycloak takes ownership
    would leave a second credential authority and let stale local hashes keep
    granting API access.
    """
    if not settings.oidc_passwords_managed_externally:
        return
    raise HTTPException(
        status_code=409,
        detail={
            'code': 'login_managed_by_identity_provider',
            'login_url': '/bff/auth/oidc/login',
        },
    )
