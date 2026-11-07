from __future__ import annotations

import os

from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator

from examples.deployment_registry.schemas import MAX_REGISTRY_TTL_SECONDS
from src.settings.base import EnvironmentSettings


DEFAULT_GOOGLE_CLIENT_IDS = (
    '860473757501-c1gtkrqr4lsa52vgoq7vclprm8atjvtv.apps.googleusercontent.com,'
    '860473757501-s53qldp7i294qbg1ia8aq822oa0rudj2.apps.googleusercontent.com,'
    '860473757501-088t4flpgv0kdds6pu4a5m1fntamf1ht.apps.googleusercontent.com'
)

DEFAULT_APPLE_BUNDLE_ID = 'com.changdar.visionnaire'

DEFAULT_APPLE_SERVICE_ID = 'com.changdar.visionnaire.signin'

DEFAULT_NATIVE_SOCIAL_ALLOWED_CLIENTS_JSON = (
    '{"visionnaire-mobile":['
    '"com.changdar.visionnaire:/oauthredirect"]}'
)


class OidcSettings(EnvironmentSettings):
    oidc_enabled: bool = False
    oidc_issuer_url: str = ''
    oidc_jwks_url: str = ''
    oidc_audience: str = ''
    oidc_identity_provider: str = 'keycloak'
    oidc_jwt_algorithms: str = 'RS256'
    oidc_jwks_cache_seconds: int = Field(300, ge=60)
    oidc_jwks_timeout_seconds: float = Field(5.0, ge=1.0)
    oidc_web_client_id: str = ''
    oidc_web_client_secret: str = ''
    oidc_web_authorization_endpoint: str = ''
    oidc_web_token_endpoint: str = ''
    oidc_web_redirect_uri: str = ''
    oidc_web_end_session_endpoint: str = ''
    oidc_web_post_logout_redirect_uri: str = ''
    oidc_account_url: str = ''
    keycloak_admin_base_url: str = ''
    keycloak_realm: str = 'visionnaire'
    keycloak_user_linker_client_id: str = 'visionnaire-user-linker'
    keycloak_user_linker_client_secret: str = ''
    oidc_state_ttl_seconds: int = Field(300, ge=60, le=600)

    @field_validator('oidc_issuer_url', 'keycloak_admin_base_url')
    @classmethod
    def canonical_base_url(cls, value: str) -> str:
        return value.rstrip('/')


class SocialSettings(EnvironmentSettings):
    native_social_exchange_enabled: bool = False
    native_social_exchange_shared_secret: str = ''
    native_social_allowed_clients_json: str = (
        DEFAULT_NATIVE_SOCIAL_ALLOWED_CLIENTS_JSON
    )
    native_social_exchange_ttl_seconds: int = Field(90, ge=30, le=300)
    native_social_link_ttl_seconds: int = Field(120, ge=30, le=300)
    native_social_link_max_auth_age_seconds: int = Field(300, ge=60, le=900)
    google_client_ids: str = DEFAULT_GOOGLE_CLIENT_IDS
    apple_client_ids: str = (
        f'{DEFAULT_APPLE_BUNDLE_ID},{DEFAULT_APPLE_SERVICE_ID}'
    )
    apple_team_id: str = '5DU8R27949'
    apple_key_id: str = 'NGC4QBS7ZY'
    apple_private_key: str = ''
    apple_private_key_path: str = 'config/secrets/apple/AuthKey_NGC4QBS7ZY.p8'
    apple_service_id: str = DEFAULT_APPLE_SERVICE_ID
    apple_bundle_id: str = DEFAULT_APPLE_BUNDLE_ID
    apple_redirect_uri: str = (
        'https://changdar-server.mooo.com/'
        'hazard/api/db_management/auth/apple/callback'
    )

    @model_validator(mode='before')
    @classmethod
    def provider_client_ids(cls, values: dict) -> dict:
        for field, variable, components in (
            (
                'google_client_ids', 'GOOGLE_CLIENT_IDS', (
                    'GOOGLE_WEB_CLIENT_ID',
                    'GOOGLE_IOS_CLIENT_ID', 'GOOGLE_ANDROID_CLIENT_ID',
                ),
            ),
            (
                'apple_client_ids', 'APPLE_CLIENT_IDS',
                ('APPLE_BUNDLE_ID', 'APPLE_SERVICE_ID'),
            ),
        ):
            if values.get(variable) or values.get(field):
                continue
            combined = ','.join(
                value for component in components if (
                    value := os.getenv(component, '').strip()
                )
            )
            if combined:
                values[field] = combined
        return values


class SecuritySettings(EnvironmentSettings):
    fcm_token_encryption_key: str = ''
    login_failure_window_seconds: int = Field(1800, ge=1)
    login_cooldown_threshold: int = Field(5, ge=1)
    login_cooldown_seconds: int = Field(300, ge=1)
    login_lock_threshold: int = Field(10, ge=1)
    login_lock_seconds: int = Field(1800, ge=1)


class DeploymentSettings(EnvironmentSettings):
    cors_allowed_origins: str = (
        'https://changdar-server.mooo.com,'
        'http://localhost:3000,http://127.0.0.1:3000,'
        'http://localhost:5000,http://127.0.0.1:5000,'
        'http://localhost:8080,http://127.0.0.1:8080'
    )
    deployment_api_base_path: str = '/hazard/api'
    local_development_auth_enabled: bool = False
    local_development_deployment_id: str = ''
    deployment_registry_ed25519_private_key: str = ''
    deployment_registry_key_id: str = 'registry-ed25519-2026-01'
    deployment_registry_ttl_seconds: int = Field(
        MAX_REGISTRY_TTL_SECONDS, ge=1, le=MAX_REGISTRY_TTL_SECONDS,
    )
    deployment_enrollment_code_pepper: str = ''
    deployment_enrollment_rate_limit_max: int = Field(5, ge=1)
    deployment_enrollment_rate_limit_window_seconds: int = Field(300, ge=1)


class DatabaseSettings(EnvironmentSettings):
    sqlalchemy_database_uri: str = Field(
        default='postgresql+asyncpg://user:password@localhost/dbname',
        validation_alias='DATABASE_URL',
    )
    sqlalchemy_track_modifications: bool = False

    db_pool_size: int = Field(2, ge=1)
    db_max_overflow: int = Field(1, ge=0)
    db_pool_timeout_seconds: float = Field(10, ge=0.1, allow_inf_nan=False)
    db_pool_recycle_seconds: int = Field(1800, ge=1)
