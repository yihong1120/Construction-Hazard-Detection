from __future__ import annotations

from urllib.parse import quote
from urllib.parse import urlsplit

from pydantic import Field
from pydantic import field_validator
from pydantic import SecretStr

from src.settings.base import EnvironmentSettings


def http_endpoint(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {'http', 'https'}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.fragment
    ):
        raise ValueError('Expected an HTTP(S) URL without credentials')
    if parsed.port is not None and parsed.port < 1:
        raise ValueError('Invalid HTTP port')
    return value.rstrip('/')


class RedisSettings(EnvironmentSettings):
    model_config = {'env_prefix': 'REDIS_'}
    host: str = Field('127.0.0.1', min_length=1)
    port: int = Field(6379, ge=1, le=65535)
    password: SecretStr = Field('', repr=False, exclude=True)

    @property
    def url(self) -> str:
        host = f'[{self.host}]' if ':' in self.host else self.host
        password = quote(self.password.get_secret_value(), safe='')
        return f'redis://:{password}@{host}:{self.port}/0'


class StartupSettings(EnvironmentSettings):
    auto_create_schema: bool = False
    site_media_cleanup_worker_enabled: bool = True


class ModelFetchSettings(EnvironmentSettings):
    model_config = {'env_prefix': 'MODEL_FETCH_'}
    api_url: str = 'http://your-server-address/get_new_model'
    bearer_token: SecretStr = Field('', repr=False, exclude=True)
    max_bytes: int = Field(6 * 1024**3, ge=1)

    @field_validator('api_url')
    @classmethod
    def complete_endpoint(cls, value: str) -> str:
        http_endpoint(value)
        # Keep the complete download endpoint, including its trailing slash.
        return value


class FcmTransportSettings(EnvironmentSettings):
    api_url: str = Field(
        'http://127.0.0.1:8003',
        validation_alias='FCM_API_URL',
    )
    _endpoint = field_validator('api_url')(http_endpoint)


class ViolationTransportSettings(EnvironmentSettings):
    api_url: str = Field(
        'http://127.0.0.1:8002',
        validation_alias='VIOLATION_RECORD_API_URL',
    )
    _endpoint = field_validator('api_url')(http_endpoint)


class PlaybackTransportSettings(EnvironmentSettings):
    api_url: str = Field(
        'http://127.0.0.1:8800',
        validation_alias='PLAYBACK_STREAMING_API_URL',
    )
    timeout_seconds: float = Field(
        20,
        gt=0,
        allow_inf_nan=False,
        validation_alias='PLAYBACK_UPSTREAM_TIMEOUT_SECONDS',
    )
    _endpoint = field_validator('api_url')(http_endpoint)
