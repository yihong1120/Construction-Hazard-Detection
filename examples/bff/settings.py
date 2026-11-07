from __future__ import annotations

from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator
from pydantic import SecretStr

from src.settings.base import EnvironmentSettings
from src.settings.connections import http_endpoint


class ProxySettings(EnvironmentSettings):
    model_config = {'env_prefix': 'BFF_'}
    chat_api_url: str = 'http://127.0.0.1:8001'
    files_api_url: str = 'http://127.0.0.1:8004'
    detection_api_url: str = 'http://127.0.0.1:8000'
    management_api_url: str = 'http://127.0.0.1:8005'
    fcm_api_url: str = 'http://127.0.0.1:8003'
    violations_api_url: str = 'http://127.0.0.1:8002'
    streaming_api_url: str = 'http://127.0.0.1:8800'
    ai_usage_api_url: str = 'http://127.0.0.1:8012'
    ai_access_api_url: str = 'http://127.0.0.1:8013'
    refresh_before_seconds: int = Field(30, ge=0)
    upstream_timeout_seconds: float = Field(20, gt=0, allow_inf_nan=False)
    company_ai_network_proxy_key: SecretStr = Field(
        '', repr=False, exclude=True,
    )

    @field_validator(
        'chat_api_url',
        'files_api_url',
        'detection_api_url',
        'management_api_url',
        'fcm_api_url',
        'violations_api_url',
        'streaming_api_url',
        'ai_usage_api_url',
        'ai_access_api_url',
    )
    @classmethod
    def endpoint(cls, value):
        return http_endpoint(value)

    @property
    def upstreams(self) -> dict[str, str]:
        return {
            'ai_access': self.ai_access_api_url,
            'chat': self.chat_api_url,
            'db_management': self.management_api_url,
            'detect': self.detection_api_url,
            'detection': self.detection_api_url,
            'fcm': self.fcm_api_url,
            'files': self.files_api_url,
            'file_manage': self.files_api_url,
            'streaming': self.streaming_api_url,
            'streaming_web': self.streaming_api_url,
            'violations': self.violations_api_url,
            'ai_usage': self.ai_usage_api_url,
        }


class CookieSettings(EnvironmentSettings):
    model_config = {'env_prefix': 'BFF_'}
    session_cookie_name: str = Field('__Host-vn_session', min_length=1)
    session_cookie_secure: bool = True

    @model_validator(mode='after')
    def host_cookie(self):
        if (
            self.session_cookie_name.startswith('__Host-')
            and not self.session_cookie_secure
        ):
            raise ValueError('__Host- cookies require secure transport')
        return self
