from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic import field_validator
from pydantic import SecretStr

from src.settings.base import EnvironmentSettings
from src.settings.connections import http_endpoint


class McpTransportSettings(EnvironmentSettings):
    model_config = {'env_prefix': 'MCP_'}
    transport: Literal['stdio', 'streamable-http', 'sse'] = 'streamable-http'
    host: str = Field('127.0.0.1', min_length=1)
    port: int = Field(8092, ge=1, le=65535)
    path: str = '/mcp'
    sse_path: str = '/sse'
    debug: bool = False

    @field_validator('path', 'sse_path')
    @classmethod
    def route(cls, value):
        if not value.startswith('/') or value.startswith('//'):
            raise ValueError('MCP routes must be absolute local paths')
        return value


class McpImageSettings(EnvironmentSettings):
    remote_image_timeout_seconds: float = Field(
        20,
        ge=1,
        allow_inf_nan=False,
        validation_alias='MCP_REMOTE_IMAGE_TIMEOUT_SECONDS',
    )
    max_remote_image_bytes: int = Field(
        10 * 1024**2,
        ge=1,
        validation_alias='MCP_MAX_REMOTE_IMAGE_BYTES',
    )
    target_fps: int = Field(1, ge=1)
    max_id_keep: int = Field(10, ge=0)


class McpQuerySettings(EnvironmentSettings):
    upstream_timeout_seconds: float = Field(
        20,
        ge=1,
        allow_inf_nan=False,
        validation_alias='MCP_UPSTREAM_TIMEOUT_SECONDS',
    )
    static_bearer: SecretStr = Field(
        '',
        validation_alias='MCP_STATIC_BEARER',
        repr=False,
        exclude=True,
    )

    @field_validator('static_bearer')
    @classmethod
    def legacy_bearer_normalization(cls, value: SecretStr) -> SecretStr:
        return SecretStr(value.get_secret_value().strip())


class BroadcastSettings(EnvironmentSettings):
    url: str = Field(
        'http://localhost:8080/broadcast',
        validation_alias='BROADCAST_URL',
    )
    _endpoint = field_validator('url')(http_endpoint)
