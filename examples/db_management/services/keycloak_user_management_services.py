"""Minimal, server-only Keycloak user-management operations.

Visionnaire remains authoritative for business roles, group membership, sites,
and features.  This module owns only the corresponding Keycloak identity
lifecycle and is deliberately reachable only from administrator-protected
routes.  Flutter clients never receive its service-account credentials.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from collections.abc import Mapping
from contextlib import asynccontextmanager
from urllib.parse import quote
from urllib.parse import urlsplit
from weakref import WeakKeyDictionary

import httpx
from fastapi import HTTPException

from examples.auth.client_credentials import ClientCredentialsTokenCache
from examples.auth.config import Settings
from src.http_client_pool import get_application_http_client

settings = Settings()
_token_caches: WeakKeyDictionary[
    httpx.AsyncClient,
    ClientCredentialsTokenCache,
] = (WeakKeyDictionary())


def _identity_service_unavailable() -> HTTPException:
    """Return the single safe error exposed for Keycloak Admin API failures."""
    return HTTPException(
        status_code=503,
        detail='keycloak_identity_service_unavailable',
    )


def _require_keycloak_admin_configuration() -> None:
    """Fail closed without the server-held Keycloak Admin API credential."""
    if (
        not settings.oidc_enabled
        or not settings.keycloak_realm
        or not settings.keycloak_user_linker_client_id
        or not settings.keycloak_user_linker_client_secret
    ):
        raise _identity_service_unavailable()


@asynccontextmanager
async def _keycloak_http_client() -> AsyncIterator[httpx.AsyncClient]:
    """Use the lifespan-owned pool, closing clients created outside an app."""
    client = await get_application_http_client('keycloak-admin', timeout=5.0)
    if client is not None:
        yield client
    else:
        async with httpx.AsyncClient(timeout=5.0) as ephemeral:
            yield ephemeral


async def _keycloak_service_access_token(client: httpx.AsyncClient) -> str:
    """Reuse a provider token within this HTTP client's lifetime."""
    _require_keycloak_admin_configuration()
    token_url = (
        f'{settings.resolved_keycloak_admin_base_url}/realms/'
        f'{quote(settings.keycloak_realm, safe="")}/'
        'protocol/openid-connect/token'
    )
    cache = _token_caches.get(client)
    if cache is None:
        cache = ClientCredentialsTokenCache()
        _token_caches[client] = cache
    try:
        return await cache.get(
            client,
            token_url=token_url,
            client_id=settings.keycloak_user_linker_client_id,
            client_secret=settings.keycloak_user_linker_client_secret,
        )
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        raise _identity_service_unavailable() from exc


async def keycloak_admin_request(
    method: str,
    path: str,
    *,
    json_body: Mapping[str, object] | list[Mapping[str, object]] | None = None,
) -> httpx.Response:
    """Call the realm-scoped API, retrying one rejected service token."""
    _require_keycloak_admin_configuration()
    url = (
        f'{settings.resolved_keycloak_admin_base_url}/admin/realms/'
        f'{quote(settings.keycloak_realm, safe="")}{path}'
    )
    try:
        async with _keycloak_http_client() as client:
            token = await _keycloak_service_access_token(client)
            response = await client.request(
                method, url,
                headers={'Authorization': f'Bearer {token}'},
                json=json_body,
            )
            if response.status_code == 401:
                _token_caches[client].invalidate(token)
                token = await _keycloak_service_access_token(client)
                response = await client.request(
                    method, url,
                    headers={'Authorization': f'Bearer {token}'},
                    json=json_body,
                )
            return response
    except httpx.HTTPError as exc:
        raise _identity_service_unavailable() from exc


def _created_subject(response: httpx.Response) -> str:
    """Return the Keycloak subject from a successful create response."""
    location = response.headers.get('Location')
    if not isinstance(location, str) or not location:
        raise _identity_service_unavailable()
    subject = urlsplit(location).path.rsplit('/', maxsplit=1)[-1]
    if not subject:
        raise _identity_service_unavailable()
    return subject


async def provision_keycloak_user(
    *,
    username: str,
    password: str,
    email: str,
    given_name: str,
    family_name: str,
    force_password_change: bool,
) -> str:
    """Create a Keycloak account and set its initial password.

    Passwords are passed only to Keycloak over the server-to-server request;
    Visionnaire stores only the immutable provider identity link.
    """
    response = await keycloak_admin_request(
        'POST',
        '/users',
        json_body={
            'username': username,
            'email': email,
            'emailVerified': True,
            'firstName': given_name,
            'lastName': family_name,
            'enabled': True,
        },
    )
    if response.status_code == 409:
        raise HTTPException(
            status_code=409,
            detail='keycloak_username_or_email_already_exists',
        )
    if response.status_code != 201:
        raise _identity_service_unavailable()

    subject = _created_subject(response)
    try:
        await reset_keycloak_password(
            subject,
            password=password,
            temporary=force_password_change,
        )
    except HTTPException:
        # Do not leave a sign-in-capable account after provisioning fails.
        await delete_keycloak_user(subject, suppress_errors=True)
        raise
    return subject


async def reset_keycloak_password(
    subject: str,
    *,
    password: str,
    temporary: bool,
) -> None:
    """Replace a Keycloak password without ever writing it to Visionnaire."""
    response = await keycloak_admin_request(
        'PUT',
        f'/users/{quote(subject, safe="")}/reset-password',
        json_body={
            'type': 'password',
            'value': password,
            'temporary': temporary,
        },
    )
    if response.status_code == 404:
        raise HTTPException(status_code=404, detail='keycloak_user_not_found')
    if response.status_code == 400:
        raise HTTPException(
            status_code=400,
            detail='keycloak_password_policy_rejected',
        )
    if response.status_code != 204:
        raise _identity_service_unavailable()


async def logout_keycloak_user(subject: str) -> None:
    """Revoke every active Keycloak session after a credential change."""
    response = await keycloak_admin_request(
        'POST', f'/users/{quote(subject, safe="")}/logout',
    )
    if response.status_code == 404:
        raise HTTPException(status_code=404, detail='keycloak_user_not_found')
    if response.status_code != 204:
        raise _identity_service_unavailable()


async def set_keycloak_user_enabled(subject: str, *, enabled: bool) -> None:
    """Enable or disable a Keycloak identity before changing local status."""
    await update_keycloak_user(subject, enabled=enabled)


async def assign_keycloak_client_role(
    subject: str,
    *,
    client_id: str,
    role_name: str,
) -> None:
    """Assign one explicit application role to a provisioned identity."""
    clients = await keycloak_admin_request(
        'GET',
        f'/clients?clientId={quote(client_id, safe="")}',
    )
    if clients.status_code != 200:
        raise _identity_service_unavailable()
    matches = clients.json()
    if len(matches) != 1:
        raise _identity_service_unavailable()
    internal_id = matches[0]['id']
    role = await keycloak_admin_request(
        'GET',
        f'/clients/{quote(internal_id, safe="")}/roles/'
        f'{quote(role_name, safe="")}',
    )
    if role.status_code != 200:
        raise _identity_service_unavailable()
    assigned = await keycloak_admin_request(
        'POST',
        f'/users/{quote(subject, safe="")}/role-mappings/clients/'
        f'{quote(internal_id, safe="")}',
        json_body=[role.json()],
    )
    if assigned.status_code != 204:
        raise _identity_service_unavailable()


async def remove_keycloak_client_role(
    subject: str,
    *,
    client_id: str,
    role_name: str,
) -> None:
    """Remove one application role without changing the identity itself."""
    clients = await keycloak_admin_request(
        'GET',
        f'/clients?clientId={quote(client_id, safe="")}',
    )
    if clients.status_code != 200 or len(clients.json()) != 1:
        raise _identity_service_unavailable()
    internal_id = clients.json()[0]['id']
    role = await keycloak_admin_request(
        'GET',
        f'/clients/{quote(internal_id, safe="")}/roles/'
        f'{quote(role_name, safe="")}',
    )
    if role.status_code != 200:
        raise _identity_service_unavailable()
    removed = await keycloak_admin_request(
        'DELETE',
        f'/users/{quote(subject, safe="")}/role-mappings/clients/'
        f'{quote(internal_id, safe="")}',
        json_body=[role.json()],
    )
    if removed.status_code != 204:
        raise _identity_service_unavailable()


async def update_keycloak_user(
    subject: str,
    *,
    username: str | None = None,
    email: str | None = None,
    given_name: str | None = None,
    family_name: str | None = None,
    enabled: bool | None = None,
) -> None:
    """Update only the identity fields Visionnaire is allowed to manage."""
    payload: dict[str, object] = {}
    if username is not None:
        payload['username'] = username
    if email is not None:
        payload['email'] = email
    if given_name is not None:
        payload['firstName'] = given_name
    if family_name is not None:
        payload['lastName'] = family_name
    if enabled is not None:
        payload['enabled'] = enabled
    if not payload:
        return
    response = await keycloak_admin_request(
        'PUT',
        f'/users/{quote(subject, safe="")}',
        json_body=payload,
    )
    if response.status_code == 404:
        raise HTTPException(status_code=404, detail='keycloak_user_not_found')
    if response.status_code != 204:
        raise _identity_service_unavailable()


async def delete_keycloak_user(
    subject: str,
    *,
    suppress_errors: bool = False,
) -> None:
    """Permanently remove an identity only after explicit admin action."""
    try:
        response = await keycloak_admin_request(
            'DELETE',
            f'/users/{quote(subject, safe="")}',
        )
    except HTTPException:
        if suppress_errors:
            return
        raise
    if response.status_code in {204, 404}:
        return
    if not suppress_errors:
        raise _identity_service_unavailable()


async def find_keycloak_user_subject(username: str) -> str | None:
    """Find an existing Keycloak user by exact canonical username."""
    response = await keycloak_admin_request(
        'GET',
        '/users?username='
        f'{quote(username, safe="")}&exact=true',
    )
    if response.status_code != 200:
        raise _identity_service_unavailable()
    try:
        candidates = response.json()
    except ValueError as exc:
        raise _identity_service_unavailable() from exc
    if not isinstance(candidates, list):
        raise _identity_service_unavailable()
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        candidate_username = candidate.get('username')
        candidate_subject = candidate.get('id')
        if (
            isinstance(candidate_username, str)
            and candidate_username.casefold() == username.casefold()
            and isinstance(candidate_subject, str)
            and candidate_subject
        ):
            return candidate_subject
    return None
