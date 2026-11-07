from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator
from pydantic import SecretStr

from src.settings.base import EnvironmentSettings


class SessionPolicy(BaseModel):
    model_config = {'frozen': True, 'extra': 'forbid'}
    idle_seconds: int = Field(gt=0)
    maximum_seconds: int = Field(gt=0)
    access_token_seconds: int = Field(gt=0)
    other_client_idle_seconds: int = Field(gt=0)
    other_client_maximum_seconds: int = Field(gt=0)
    clients: tuple[str, ...]
    optional_clients: tuple[str, ...] = ()

    @model_validator(mode='after')
    def valid_limits(self) -> SessionPolicy:
        if (
            not self.access_token_seconds
            <= self.idle_seconds
            <= self.maximum_seconds
        ):
            raise ValueError(
                (
                    'Require access token lifetime <= idle timeout <= '
                    'maximum lifetime'
                ),
            )
        if (
            self.other_client_idle_seconds > self.other_client_maximum_seconds
            or self.other_client_idle_seconds > self.idle_seconds
            or self.other_client_maximum_seconds > self.maximum_seconds
        ):
            raise ValueError('Client defaults cannot exceed their SSO limits')
        if not self.clients or len(set(self.clients)) != len(self.clients):
            raise ValueError('Session clients must be unique and nonempty')
        if not set(self.optional_clients).issubset(self.clients):
            raise ValueError('Optional clients must belong to the policy')
        return self

    def realm_settings(self) -> dict[str, int]:
        return {
            'ssoSessionIdleTimeout': self.idle_seconds,
            'ssoSessionMaxLifespan': self.maximum_seconds,
            'accessTokenLifespan': self.access_token_seconds,
            'clientSessionIdleTimeout': self.other_client_idle_seconds,
            'clientSessionMaxLifespan': self.other_client_maximum_seconds,
        }

    def client_attributes(self) -> dict[str, str]:
        return {
            'client.session.idle.timeout': str(self.idle_seconds),
            'client.session.max.lifespan': str(self.maximum_seconds),
            'access.token.lifespan': str(self.access_token_seconds),
        }


SESSION_POLICY = SessionPolicy.model_validate_json(
    Path(__file__).with_name('session_policy.json').read_text(),
)


class SessionStoreSettings(EnvironmentSettings):
    session_ttl_seconds: int = Field(
        SESSION_POLICY.idle_seconds,
        gt=0,
        le=SESSION_POLICY.maximum_seconds,
        validation_alias='BFF_SESSION_TTL_SECONDS',
    )
    media_session_ttl_seconds: int = Field(600, ge=300, le=900)


class SessionCryptoSettings(EnvironmentSettings):
    token_encryption_key: SecretStr = Field(
        '',
        validation_alias='BFF_TOKEN_ENCRYPTION_KEY',
        repr=False,
        exclude=True,
    )

    @field_validator('token_encryption_key')
    @classmethod
    def legacy_key_normalization(cls, value: SecretStr) -> SecretStr:
        return SecretStr(value.get_secret_value().strip())
