from __future__ import annotations

import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

from examples.deployment_registry.schemas import MAX_REGISTRY_TTL_SECONDS

# Load environment variables from .env file
load_dotenv()


def _join_env_values(*names: str) -> str:
    """Return comma-separated non-empty environment values."""
    return ','.join(
        value for name in names if (value := os.getenv(name, '').strip())
    )


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse common boolean environment variable values."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


DEFAULT_GOOGLE_CLIENT_IDS = (
    '860473757501-c1gtkrqr4lsa52vgoq7vclprm8atjvtv.apps.googleusercontent.com,'
    '860473757501-s53qldp7i294qbg1ia8aq822oa0rudj2.apps.googleusercontent.com,'
    '860473757501-088t4flpgv0kdds6pu4a5m1fntamf1ht.apps.googleusercontent.com'
)

DEFAULT_APPLE_BUNDLE_ID = 'com.changdar.visionnaire'
DEFAULT_APPLE_SERVICE_ID = 'com.changdar.visionnaire.signin'


class Settings(BaseSettings):
    """Configuration settings for the application.

    Attributes:
        authjwt_secret_key (str): The secret key for signing JWT tokens.
            Defaults to the value of the JWT_SECRET_KEY environment variable.
        sqlalchemy_database_uri (str): The database connection URI (async).
            Defaults to the value of the DATABASE_URL environment variable
            or 'postgresql+asyncpg://user:password@localhost/dbname' if
            not set.
        sqlalchemy_track_modifications (bool): Indicates whether SQLAlchemy
            should track modifications. Defaults to False.
    """

    authjwt_secret_key: str = os.getenv('JWT_SECRET_KEY', '')
    # OIDC resource-server settings. They are deliberately generic so the
    # application can use a standards-compliant provider, while the deployment
    # guide supplies Keycloak values. Existing application JWTs remain enabled
    # during the migration.
    oidc_enabled: bool = _env_bool('OIDC_ENABLED', False)
    oidc_issuer_url: str = os.getenv('OIDC_ISSUER_URL', '').strip().rstrip('/')
    oidc_jwks_url: str = os.getenv('OIDC_JWKS_URL', '').strip()
    oidc_audience: str = os.getenv('OIDC_AUDIENCE', '').strip()
    oidc_identity_provider: str = os.getenv(
        'OIDC_IDENTITY_PROVIDER',
        'keycloak',
    ).strip()
    oidc_jwt_algorithms: str = os.getenv(
        'OIDC_JWT_ALGORITHMS',
        'RS256',
    ).strip()
    oidc_jwks_cache_seconds: int = max(
        60,
        int(os.getenv('OIDC_JWKS_CACHE_SECONDS', '300')),
    )
    oidc_jwks_timeout_seconds: float = max(
        1.0,
        float(os.getenv('OIDC_JWKS_TIMEOUT_SECONDS', '5')),
    )
    # Browser BFF client settings. OIDC may be enabled for API bearer-token
    # validation before the BFF itself is switched to the authorization-code
    # flow, so these are validated only by ``oidc_web_client_configured``.
    oidc_web_client_id: str = os.getenv('OIDC_WEB_CLIENT_ID', '').strip()
    oidc_web_client_secret: str = os.getenv(
        'OIDC_WEB_CLIENT_SECRET',
        '',
    ).strip()
    oidc_web_authorization_endpoint: str = os.getenv(
        'OIDC_WEB_AUTHORIZATION_ENDPOINT',
        '',
    ).strip()
    oidc_web_token_endpoint: str = os.getenv(
        'OIDC_WEB_TOKEN_ENDPOINT',
        '',
    ).strip()
    oidc_web_redirect_uri: str = os.getenv(
        'OIDC_WEB_REDIRECT_URI',
        '',
    ).strip()
    oidc_account_url: str = os.getenv('OIDC_ACCOUNT_URL', '').strip()
    oidc_passwords_managed_externally: bool = _env_bool(
        'OIDC_PASSWORDS_MANAGED_EXTERNALLY',
        False,
    )
    oidc_state_ttl_seconds: int = max(
        60,
        min(600, int(os.getenv('OIDC_STATE_TTL_SECONDS', '300'))),
    )
    hcaptcha_enabled: bool = _env_bool('HCAPTCHA_ENABLED', True)
    hcaptcha_secret_key: str = os.getenv('HCAPTCHA_SECRET_KEY', '')
    hcaptcha_site_key: str = os.getenv('HCAPTCHA_SITE_KEY', '')
    hcaptcha_bypass_key: str = os.getenv('HCAPTCHA_BYPASS_KEY', '')
    web_refresh_cookie_name: str = os.getenv(
        'WEB_REFRESH_COOKIE_NAME',
        'refresh_session',
    )
    web_refresh_cookie_path: str = os.getenv(
        'WEB_REFRESH_COOKIE_PATH',
        '/hazard/api/db_management',
    )
    web_refresh_cookie_domain: str = os.getenv(
        'WEB_REFRESH_COOKIE_DOMAIN',
        '',
    )
    web_refresh_cookie_secure: bool = _env_bool(
        'WEB_REFRESH_COOKIE_SECURE',
        True,
    )
    web_refresh_cookie_samesite: str = os.getenv(
        'WEB_REFRESH_COOKIE_SAMESITE',
        'lax',
    ).lower()
    web_refresh_cookie_max_age_seconds: int = int(
        os.getenv('WEB_REFRESH_COOKIE_MAX_AGE_SECONDS', str(30 * 24 * 3600)),
    )
    fcm_token_encryption_key: str = os.getenv(
        'FCM_TOKEN_ENCRYPTION_KEY',
        '',
    )
    brevo_api_key: str = os.getenv('BREVO_API_KEY', '')
    mail_from: str = os.getenv('MAIL_FROM', '')
    mail_from_name: str = os.getenv('MAIL_FROM_NAME', 'Visionnaire')
    app_public_url: str = os.getenv(
        'APP_PUBLIC_URL',
        'https://changdar-server.mooo.com',
    )
    password_reset_token_ttl_seconds: int = int(
        os.getenv('PASSWORD_RESET_TOKEN_TTL_SECONDS', '1800'),
    )
    password_reset_email_rate_limit_seconds: int = int(
        os.getenv('PASSWORD_RESET_EMAIL_RATE_LIMIT_SECONDS', '60'),
    )
    password_reset_ip_rate_limit_window_seconds: int = int(
        os.getenv('PASSWORD_RESET_IP_RATE_LIMIT_WINDOW_SECONDS', '600'),
    )
    password_reset_ip_rate_limit_max: int = int(
        os.getenv('PASSWORD_RESET_IP_RATE_LIMIT_MAX', '5'),
    )
    email_verification_token_ttl_seconds: int = int(
        os.getenv('EMAIL_VERIFICATION_TOKEN_TTL_SECONDS', '86400'),
    )
    email_verification_resend_rate_limit_seconds: int = int(
        os.getenv('EMAIL_VERIFICATION_RESEND_RATE_LIMIT_SECONDS', '60'),
    )
    email_verification_daily_limit: int = int(
        os.getenv('EMAIL_VERIFICATION_DAILY_LIMIT', '5'),
    )
    email_verification_daily_limit_window_seconds: int = int(
        os.getenv('EMAIL_VERIFICATION_DAILY_LIMIT_WINDOW_SECONDS', '86400'),
    )
    brevo_email_verification_template_id: int = int(
        os.getenv('BREVO_EMAIL_VERIFICATION_TEMPLATE_ID', '0') or '0',
    )
    login_failure_window_seconds: int = int(
        os.getenv('LOGIN_FAILURE_WINDOW_SECONDS', '1800'),
    )
    login_cooldown_threshold: int = int(
        os.getenv('LOGIN_COOLDOWN_THRESHOLD', '5'),
    )
    login_cooldown_seconds: int = int(
        os.getenv('LOGIN_COOLDOWN_SECONDS', '300'),
    )
    login_lock_threshold: int = int(
        os.getenv('LOGIN_LOCK_THRESHOLD', '10'),
    )
    login_lock_seconds: int = int(
        os.getenv('LOGIN_LOCK_SECONDS', '1800'),
    )
    google_client_ids: str = (
        os.getenv('GOOGLE_CLIENT_IDS')
        or _join_env_values(
            'GOOGLE_WEB_CLIENT_ID',
            'GOOGLE_IOS_CLIENT_ID',
            'GOOGLE_ANDROID_CLIENT_ID',
        )
        or DEFAULT_GOOGLE_CLIENT_IDS
    )
    apple_client_ids: str = (
        os.getenv('APPLE_CLIENT_IDS')
        or _join_env_values('APPLE_BUNDLE_ID', 'APPLE_SERVICE_ID')
        or f"{DEFAULT_APPLE_BUNDLE_ID},{DEFAULT_APPLE_SERVICE_ID}"
    )
    apple_team_id: str = os.getenv('APPLE_TEAM_ID', '5DU8R27949')
    apple_key_id: str = os.getenv('APPLE_KEY_ID', 'NGC4QBS7ZY')
    apple_private_key: str = os.getenv('APPLE_PRIVATE_KEY', '')
    apple_private_key_path: str = os.getenv(
        'APPLE_PRIVATE_KEY_PATH',
        'config/secrets/apple/AuthKey_NGC4QBS7ZY.p8',
    )
    apple_service_id: str = os.getenv(
        'APPLE_SERVICE_ID',
        DEFAULT_APPLE_SERVICE_ID,
    )
    apple_bundle_id: str = os.getenv(
        'APPLE_BUNDLE_ID',
        DEFAULT_APPLE_BUNDLE_ID,
    )
    apple_redirect_uri: str = os.getenv(
        'APPLE_REDIRECT_URI',
        (
            'https://changdar-server.mooo.com/'
            'hazard/api/db_management/auth/apple/callback'
        ),
    )
    cors_allowed_origins: str = os.getenv(
        'CORS_ALLOWED_ORIGINS',
        (
            'https://changdar-server.mooo.com,'
            'http://localhost:3000,http://127.0.0.1:3000,'
            'http://localhost:5000,http://127.0.0.1:5000,'
            'http://localhost:8080,http://127.0.0.1:8080'
        ),
    )
    # The reverse proxy removes this prefix before forwarding to individual
    # services.  It is still part of the public deployment contract and token
    # issuer, so it must never be supplied by a client request.
    deployment_api_base_path: str = os.getenv(
        'DEPLOYMENT_API_BASE_PATH',
        '/hazard/api',
    )
    # Explicit local-only development escape hatch.  The resolver requires a
    # loopback peer and host, then maps it to this server-controlled ID; a
    # request can never select a tenant/deployment itself.
    local_development_auth_enabled: bool = _env_bool(
        'LOCAL_DEVELOPMENT_AUTH_ENABLED',
        False,
    )
    local_development_deployment_id: str = os.getenv(
        'LOCAL_DEVELOPMENT_DEPLOYMENT_ID',
        '',
    )
    # This value is injected by the backend secret store (or a KMS-backed
    # deployment secret) at runtime.  It is intentionally empty by default:
    # the public registry must fail closed rather than emit an unsigned config.
    deployment_registry_ed25519_private_key: str = os.getenv(
        'DEPLOYMENT_REGISTRY_ED25519_PRIVATE_KEY',
        '',
    )
    deployment_registry_key_id: str = os.getenv(
        'DEPLOYMENT_REGISTRY_KEY_ID',
        'registry-ed25519-2026-01',
    )
    deployment_registry_ttl_seconds: int = int(
        os.getenv(
            'DEPLOYMENT_REGISTRY_TTL_SECONDS',
            str(MAX_REGISTRY_TTL_SECONDS),
        ),
    )
    # An independent secret used to make enrollment-code verifiers
    # non-reversible even if the registry database is disclosed.  It is
    # intentionally empty by default so enrollment fails closed until the
    # deployment secret store supplies it.
    deployment_enrollment_code_pepper: str = os.getenv(
        'DEPLOYMENT_ENROLLMENT_CODE_PEPPER',
        '',
    )
    deployment_enrollment_rate_limit_max: int = int(
        os.getenv('DEPLOYMENT_ENROLLMENT_RATE_LIMIT_MAX', '5'),
    )
    deployment_enrollment_rate_limit_window_seconds: int = int(
        os.getenv('DEPLOYMENT_ENROLLMENT_RATE_LIMIT_WINDOW_SECONDS', '300'),
    )
    sqlalchemy_database_uri: str = os.getenv(
        'DATABASE_URL',
        'postgresql+asyncpg://user:password@localhost/dbname',
    )
    sqlalchemy_track_modifications: bool = False

    ALGORITHM: str = 'HS256'

    def __init__(self) -> None:
        """Construct the Settings object."""
        super().__init__()
        if not self.authjwt_secret_key:
            raise RuntimeError('JWT_SECRET_KEY is required')
        if (
            self.oidc_passwords_managed_externally
            and not self.oidc_enabled
        ):
            raise RuntimeError(
                'OIDC_ENABLED is required when Keycloak manages passwords',
            )
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
            if (
                self.oidc_passwords_managed_externally
                and not self.oidc_account_url
            ):
                raise RuntimeError(
                    'OIDC_ACCOUNT_URL is required when Keycloak manages '
                    'passwords',
                )

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
