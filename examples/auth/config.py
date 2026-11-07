from __future__ import annotations

import json
import re
from collections.abc import Mapping
from urllib.parse import urlsplit

from dotenv import load_dotenv

from examples.auth.settings_fields import DatabaseSettings
from examples.auth.settings_fields import DeploymentSettings
from examples.auth.settings_fields import OidcSettings
from examples.auth.settings_fields import SecuritySettings
from examples.auth.settings_fields import SocialSettings

# Load environment variables from .env file
load_dotenv()


_OIDC_CLIENT_ID_RE = re.compile(r'^[A-Za-z0-9._-]{1,128}$')


class Settings(
    OidcSettings,
    SocialSettings,
    SecuritySettings,
    DeploymentSettings,
    DatabaseSettings,
):
    """Configuration settings for the application.

    Attributes:
        sqlalchemy_database_uri (str): The database connection URI (async).
            Defaults to the value of the DATABASE_URL environment variable
            or 'postgresql+asyncpg://user:password@localhost/dbname' if
            not set.
        sqlalchemy_track_modifications (bool): Indicates whether SQLAlchemy
            should track modifications. Defaults to False.
    """

    def __init__(self) -> None:
        """Construct the Settings object."""
        super().__init__()
        if self.oidc_enabled:
            missing = [
                name
                for name, value in (
                    ('OIDC_ISSUER_URL', self.oidc_issuer_url),
                    ('OIDC_JWKS_URL', self.oidc_jwks_url),
                    ('OIDC_AUDIENCE', self.oidc_audience),
                    ('OIDC_IDENTITY_PROVIDER', self.oidc_identity_provider),
                )
                if not value
            ]
            if missing:
                raise RuntimeError(
                    'OIDC is enabled but required settings are missing: '
                    + ', '.join(missing),
                )
            if len(self.oidc_identity_provider) > 20:
                raise RuntimeError(
                    'OIDC_IDENTITY_PROVIDER must be at most 20 characters',
                )
            if not self.oidc_account_url:
                raise RuntimeError(
                    'OIDC_ACCOUNT_URL is required when Keycloak manages '
                    'passwords',
                )
        if self.native_social_exchange_enabled:
            missing = [
                name
                for name, value in (
                    (
                        'NATIVE_SOCIAL_EXCHANGE_SHARED_SECRET',
                        self.native_social_exchange_shared_secret,
                    ),
                    (
                        'KEYCLOAK_USER_LINKER_CLIENT_SECRET',
                        self.keycloak_user_linker_client_secret,
                    ),
                    ('KEYCLOAK_REALM', self.keycloak_realm),
                )
                if not value
            ]
            if missing:
                raise RuntimeError(
                    'Native social exchange is enabled but required settings '
                    'are missing: ' + ', '.join(missing),
                )
            if len(self.native_social_exchange_shared_secret) < 32:
                raise RuntimeError(
                    'NATIVE_SOCIAL_EXCHANGE_SHARED_SECRET must be at least '
                    '32 characters',
                )
            if not self.oidc_enabled:
                raise RuntimeError(
                    'OIDC_ENABLED is required for native social exchange',
                )
            # Parse once at startup, rather than accepting a malformed client
            # allow-list at the unauthenticated exchange endpoint.
            _ = self.native_social_allowed_clients
            _ = self.resolved_keycloak_admin_base_url

    @property
    def oidc_audiences(self) -> tuple[str, ...]:
        """Return the configured API audiences as non-empty values."""
        return tuple(
            value.strip()
            for value in self.oidc_audience.split(',')
            if value.strip()
        )

    @property
    def oidc_algorithms(self) -> tuple[str, ...]:
        """Return the explicit asymmetric JWT algorithms accepted for OIDC."""
        return tuple(
            value.strip()
            for value in self.oidc_jwt_algorithms.split(',')
            if value.strip()
        )

    @property
    def oidc_web_client_configured(self) -> bool:
        """Return whether this deployment can start BFF OIDC sign-in."""
        return self.oidc_enabled and all(
            (
                self.oidc_web_client_id,
                self.oidc_web_client_secret,
                self.oidc_web_authorization_endpoint,
                self.oidc_web_token_endpoint,
                self.oidc_web_redirect_uri,
            ),
        )

    @property
    def oidc_web_logout_configured(self) -> bool:
        """Return whether the BFF can end the Keycloak browser session."""
        return self.oidc_web_client_configured and all(
            (
                self.oidc_web_end_session_endpoint,
                self.oidc_web_post_logout_redirect_uri,
            ),
        )

    @property
    def native_social_allowed_clients(self) -> dict[str, tuple[str, ...]]:
        """Return the strict Keycloak client/redirect allow-list.

        This is not an OAuth dynamic-registration mechanism.  Every native
        callback is a deployment-owned, exact URI and is subsequently checked
        again by Keycloak before it can issue an authorisation code.
        """
        try:
            raw = json.loads(self.native_social_allowed_clients_json)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                'NATIVE_SOCIAL_ALLOWED_CLIENTS_JSON must be valid JSON',
            ) from exc
        if not isinstance(raw, Mapping) or not raw:
            raise RuntimeError(
                'NATIVE_SOCIAL_ALLOWED_CLIENTS_JSON must be a non-empty map',
            )
        allowed: dict[str, tuple[str, ...]] = {}
        for client_id, redirect_uris in raw.items():
            if (
                not isinstance(client_id, str)
                or not _OIDC_CLIENT_ID_RE.fullmatch(client_id)
                or not isinstance(redirect_uris, list)
                or not redirect_uris
            ):
                raise RuntimeError(
                    'NATIVE_SOCIAL_ALLOWED_CLIENTS_JSON contains an invalid '
                    'client entry',
                )
            validated_uris: list[str] = []
            for redirect_uri in redirect_uris:
                if not isinstance(redirect_uri, str) or not redirect_uri:
                    raise RuntimeError(
                        'NATIVE_SOCIAL_ALLOWED_CLIENTS_JSON contains an '
                        'invalid redirect URI',
                    )
                parsed = urlsplit(redirect_uri)
                # Native custom schemes and HTTPS browser callbacks are both
                # valid; opaque / relative URLs, credentials, fragments, and
                # wildcard redirects are not.
                if (
                    not parsed.scheme
                    or parsed.scheme.lower() in {'data', 'file', 'javascript'}
                    or not (
                        parsed.netloc
                        or redirect_uri.startswith(f'{parsed.scheme}:/')
                    )
                    or parsed.username
                    or parsed.password
                    or parsed.fragment
                    or '*' in redirect_uri
                ):
                    raise RuntimeError(
                        'NATIVE_SOCIAL_ALLOWED_CLIENTS_JSON contains an '
                        'unsafe redirect URI',
                    )
                if parsed.scheme.lower() == 'https' and not parsed.netloc:
                    raise RuntimeError(
                        'NATIVE_SOCIAL_ALLOWED_CLIENTS_JSON contains an '
                        'unsafe HTTPS redirect URI',
                    )
                if redirect_uri not in validated_uris:
                    validated_uris.append(redirect_uri)
            allowed[client_id] = tuple(validated_uris)
        return allowed

    @property
    def resolved_keycloak_admin_base_url(self) -> str:
        """Return trusted Keycloak base used by backend-only Admin calls."""
        candidate = self.keycloak_admin_base_url
        if not candidate:
            marker = '/realms/'
            if marker not in self.oidc_issuer_url:
                raise RuntimeError(
                    'OIDC_ISSUER_URL must contain /realms/ for Keycloak '
                    'native social exchange',
                )
            candidate = self.oidc_issuer_url.split(marker, maxsplit=1)[0]
        parsed = urlsplit(candidate)
        if (
            parsed.scheme not in {'http', 'https'}
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise RuntimeError('KEYCLOAK_ADMIN_BASE_URL is invalid')
        return candidate.rstrip('/')
