from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from urllib.parse import quote
from urllib.parse import urlsplit

import httpx
from fastapi import HTTPException

from examples.auth.keycloak_admin import (
    identity_service_unavailable as _identity_service_unavailable,
)
from examples.auth.keycloak_admin import keycloak_admin

keycloak_admin_request = keycloak_admin.request


def _created_subject(response: httpx.Response) -> str:
    """Return the Keycloak subject from a successful create response."""
    location = response.headers.get('Location')
    if not isinstance(location, str) or not location:
        raise _identity_service_unavailable()
    subject = urlsplit(location).path.rsplit('/', maxsplit=1)[-1]
    if not subject:
        raise _identity_service_unavailable()
    return subject


@dataclass(frozen=True)
class KeycloakAccount:
    username: str
    password: str = field(repr=False)
    email: str
    given_name: str
    family_name: str
    temporary: bool
    operation_id: str | None = None


async def create_keycloak_user(account: KeycloakAccount) -> str:
    response = await keycloak_admin_request(
        'POST',
        '/users',
        json_body={
            'username': account.username,
            'email': account.email,
            'emailVerified': True,
            'firstName': account.given_name,
            'lastName': account.family_name,
            'enabled': False,
            **(
                {
                    'attributes': {
                        'visionnaire_operation_id': [account.operation_id],
                    },
                }
                if account.operation_id
                else {}
            ),
        },
    )
    if response.status_code == 409:
        raise HTTPException(409, 'keycloak_username_or_email_already_exists')
    if response.status_code != 201:
        raise _identity_service_unavailable()
    return _created_subject(response)


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
        'POST',
        f'/users/{quote(subject, safe="")}/logout',
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
        '/users?username=' f'{quote(username, safe="")}&exact=true',
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


async def read_keycloak_user(subject: str) -> dict[str, object]:
    response = await keycloak_admin_request(
        'GET',
        f'/users/{quote(subject, safe="")}',
    )
    if response.status_code != 200:
        raise _identity_service_unavailable()
    data = response.json()
    if not isinstance(data, dict):
        raise _identity_service_unavailable()
    return data


async def keycloak_client_role_names(subject: str, client_id: str) -> set[str]:
    clients = await keycloak_admin_request(
        'GET',
        f'/clients?clientId={quote(client_id, safe="")}',
    )
    if clients.status_code != 200 or len(clients.json()) != 1:
        raise _identity_service_unavailable()
    internal_id = clients.json()[0]['id']
    response = await keycloak_admin_request(
        'GET',
        f'/users/{quote(subject, safe="")}/role-mappings/clients/'
        f'{quote(internal_id, safe="")}',
    )
    if response.status_code != 200 or not isinstance(response.json(), list):
        raise _identity_service_unavailable()
    return {role['name'] for role in response.json()}


async def restore_openwebui_roles(subject: str, previous: set[str]) -> None:
    for role in ('openwebui-user', 'openwebui-admin'):
        operation = (
            assign_keycloak_client_role
            if role in previous
            else remove_keycloak_client_role
        )
        await operation(subject, client_id='open-webui', role_name=role)


class KeycloakClientRoles:
    """Resolve one client and apply only changes to its managed role subset."""

    def __init__(self, client_id: str, managed: frozenset[str]):
        self.client_id = client_id
        self.managed = managed

    async def synchronize(self, subject: str, desired: set[str]):
        if not desired <= self.managed:
            raise ValueError('Role is outside the managed application subset')
        response = await keycloak_admin_request(
            'GET',
            f'/clients?clientId={quote(self.client_id, safe="")}',
        )
        if response.status_code != 200:
            raise _identity_service_unavailable()
        clients = response.json()
        if not isinstance(clients, list) or len(clients) != 1:
            raise _identity_service_unavailable()
        client = quote(str(clients[0]['id']), safe='')
        path = (
            f'/users/{quote(subject, safe="")}/role-mappings/clients/{client}'
        )
        assigned = await keycloak_admin_request('GET', path)
        if assigned.status_code != 200:
            raise _identity_service_unavailable()
        roles = assigned.json()
        if not isinstance(roles, list):
            raise _identity_service_unavailable()
        current = {item['name']: item for item in roles}
        additions = []
        for name in sorted(desired - current.keys()):
            definition = await keycloak_admin_request(
                'GET',
                f'/clients/{client}/roles/{quote(name, safe="")}',
            )
            if definition.status_code != 200:
                raise _identity_service_unavailable()
            additions.append(definition.json())
        removals = [
            role
            for name, role in current.items()
            if name in self.managed - desired
        ]
        for method, values in (('POST', additions), ('DELETE', removals)):
            if values:
                result = await keycloak_admin_request(
                    method,
                    path,
                    json_body=values,
                )
                if result.status_code != 204:
                    raise _identity_service_unavailable()


async def synchronize_openwebui_roles(subject: str, role: str):
    if role not in ('user', 'admin'):
        raise ValueError('Unsupported OpenWebUI role')
    await KeycloakClientRoles(
        'open-webui',
        frozenset({'openwebui-user', 'openwebui-admin'}),
    ).synchronize(subject, {'openwebui-' + role})
