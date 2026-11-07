from __future__ import annotations

from urllib.parse import urlsplit

from pydantic import Field
from pydantic import model_validator
from pydantic import SecretStr

from src.settings.base import EnvironmentSettings


class WorkerOidcSettings(EnvironmentSettings):
    issuer_url: str = Field('', validation_alias='OIDC_ISSUER_URL')
    token_endpoint: str = Field(
        '',
        validation_alias='WORKER_OIDC_TOKEN_ENDPOINT',
    )
    client_id: str = Field('', validation_alias='WORKER_OIDC_CLIENT_ID')
    client_secret: SecretStr = Field(
        '',
        validation_alias='WORKER_OIDC_CLIENT_SECRET',
        repr=False,
        exclude=True,
    )

    @property
    def endpoint(self) -> str:
        return self.token_endpoint or (
            self.issuer_url.rstrip('/') + '/protocol/openid-connect/token'
            if self.issuer_url
            else ''
        )

    @model_validator(mode='after')
    def configured_credentials(self):
        if (
            not self.endpoint
            or not self.client_id
            or not self.client_secret.get_secret_value()
        ):
            raise ValueError(
                'Configure the worker OIDC endpoint, client ID and secret',
            )
        url = urlsplit(self.endpoint)
        if (
            not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
            or (url.port is not None and url.port < 1)
            or (
                url.scheme != 'https'
                and not (
                    url.scheme == 'http'
                    and url.hostname in {'localhost', '127.0.0.1', '::1'}
                )
            )
        ):
            raise ValueError(
                'Worker token endpoint must use HTTPS or loopback HTTP',
            )
        return self
