"""Minimal, server-only Keycloak user-management operations.

Visionnaire remains authoritative for business roles, group membership, sites,
and features.  This module owns only the corresponding Keycloak identity
lifecycle and is deliberately reachable only from administrator-protected
routes.  Flutter clients never receive its service-account credentials.
"""
from __future__ import annotations

from collections.abc import Mapping
from urllib.parse import quote
from urllib.parse import urlsplit

import httpx
from fastapi import HTTPException

from examples.auth.config import Settings

settings = Settings()


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
        or not settings.oidc_passwords_managed_externally
        or not settings.keycloak_realm
        or not settings.keycloak_user_linker_client_id
        or not settings.keycloak_user_linker_client_secret
    ):
        raise _identity_service_unavailable()


async def _keycloak_service_access_token() -> str:
    """Acquire one short-lived Keycloak Admin API access token."""
    _require_keycloak_admin_configuration()
    token_url = (
        f'{settings.resolved_keycloak_admin_base_url}/realms/'
        f'{quote(settings.keycloak_realm, safe="")}/'
        'protocol/openid-connect/token'
    )
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                token_url,
                data={
                    'grant_type': 'client_credentials',
                    'client_id': settings.keycloak_user_linker_client_id,
                    'client_secret': (
                        settings.keycloak_user_linker_client_secret
                    ),
                },
            )
    except httpx.HTTPError as exc:
        raise _identity_service_unavailable() from exc

    if response.status_code != 200:
        raise _identity_service_unavailable()
    try:
        token = response.json().get('access_token')
    except ValueError as exc:
        raise _identity_service_unavailable() from exc
    if not isinstance(token, str) or not token:
        raise _identity_service_unavailable()
    return token


async def keycloak_admin_request(
    method: str,
    path: str,
    *,
    json_body: Mapping[str, object] | None = None,
) -> httpx.Response:
    """Perform one authenticated, realm-scoped Keycloak Admin API request."""
    token = await _keycloak_service_access_token()
    url = (
        f'{settings.resolved_keycloak_admin_base_url}/admin/realms/'
        f'{quote(settings.keycloak_realm, safe="")}{path}'
    )
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            return await client.request(
                method,
                url,
                headers={'Authorization': f'Bearer {token}'},
                json=json_body,
            )
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
    Visionnaire stores a disabled local-password marker for this account.
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
    if response.status_code != 204:
        raise _identity_service_unavailable()


async def set_keycloak_user_enabled(subject: str, *, enabled: bool) -> None:
    """Enable or disable a Keycloak identity before changing local status."""
    await update_keycloak_user(subject, enabled=enabled)


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
