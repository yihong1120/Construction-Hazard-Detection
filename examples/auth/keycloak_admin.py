from __future__ import annotations

from collections.abc import AsyncIterator
from collections.abc import Mapping
from contextlib import asynccontextmanager
from urllib.parse import quote
from weakref import WeakKeyDictionary

import httpx
from fastapi import HTTPException

from examples.auth.client_credentials import ClientCredentialsTokenCache
from examples.auth.config import Settings
from src.http_client_pool import get_application_http_client


def identity_service_unavailable() -> HTTPException:
    return HTTPException(503, detail='keycloak_identity_service_unavailable')


class KeycloakAdminClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings if settings is not None else Settings()
        self._tokens: WeakKeyDictionary[
            httpx.AsyncClient, ClientCredentialsTokenCache,
        ] = WeakKeyDictionary()

    def _require_configuration(self) -> None:
        if not (
            self.settings.oidc_enabled
            and self.settings.keycloak_realm
            and self.settings.keycloak_user_linker_client_id
            and self.settings.keycloak_user_linker_client_secret
        ):
            raise identity_service_unavailable()

    @asynccontextmanager
    async def _http_client(self) -> AsyncIterator[httpx.AsyncClient]:
        client = await get_application_http_client('keycloak-admin', timeout=5)
        if client is not None:
            yield client
        else:
            async with httpx.AsyncClient(timeout=5) as client:
                yield client

    async def _access_token(self, client: httpx.AsyncClient) -> str:
        config = self.settings
        token_url = (
            f'{config.resolved_keycloak_admin_base_url}/realms/'
            f'{quote(config.keycloak_realm, safe="")}/'
            'protocol/openid-connect/token'
        )
        cache = self._tokens.get(client)
        if cache is None:
            cache = ClientCredentialsTokenCache()
            self._tokens[client] = cache
        try:
            return await cache.get(
                client, token_url=token_url,
                client_id=config.keycloak_user_linker_client_id,
                client_secret=config.keycloak_user_linker_client_secret,
            )
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            raise identity_service_unavailable() from exc

    async def request(
        self,
        method: str,
        path: str,
        *,
        json_body: Mapping[
            str,
            object,
        ] | list[Mapping[str, object]] | None = None,
    ) -> httpx.Response:
        self._require_configuration()
        url = (
            f'{self.settings.resolved_keycloak_admin_base_url}/admin/realms/'
            f'{quote(self.settings.keycloak_realm, safe="")}{path}'
        )
        try:
            async with self._http_client() as client:
                token = await self._access_token(client)
                response = await client.request(
                    method, url,
                    headers={'Authorization': f'Bearer {token}'},
                    json=json_body,
                )
                if response.status_code == 401:
                    self._tokens[client].invalidate(token)
                    token = await self._access_token(client)
                    response = await client.request(
                        method, url,
                        headers={'Authorization': f'Bearer {token}'},
                        json=json_body,
                    )
                return response
        except httpx.HTTPError as exc:
            raise identity_service_unavailable() from exc


keycloak_admin = KeycloakAdminClient()
