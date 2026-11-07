from __future__ import annotations

import json
from pathlib import Path

import httpx

from src.settings.sessions import SESSION_POLICY

BASE = 'https://changdar-server.mooo.com/keycloak'
ORIGIN = 'https://dev.changdar-server.mooo.com'
CLIENT = 'visionnaire-web-dev'
AUDIENCE = 'visionnaire-dev-api'
MARKER = 'construction-dev-v1'


def provision():
    credentials = json.loads(
        (
            Path.home() / 'service-migration-phase5/source-keycloak-env.json'
        ).read_text(),
    )
    with httpx.Client(timeout=30, trust_env=False) as client:
        token = client.post(
            BASE + '/realms/master/protocol/openid-connect/token',
            data={
                'client_id': 'admin-cli',
                'grant_type': 'password',
                'username': credentials['KC_BOOTSTRAP_ADMIN_USERNAME'],
                'password': credentials['KC_BOOTSTRAP_ADMIN_PASSWORD'],
            },
        )
        token.raise_for_status()
        client.headers['Authorization'] = (
            'Bearer ' + token.json()['access_token']
        )
        prefix = BASE + '/admin/realms/visionnaire'

        api = _RealmRequests(client, prefix).request

        existing = api('GET', '/clients', params={'clientId': CLIENT})
        desired = {
            'clientId': CLIENT,
            'name': 'Visionnaire DEV (A)',
            'enabled': True,
            'protocol': 'openid-connect',
            'publicClient': False,
            'standardFlowEnabled': True,
            'directAccessGrantsEnabled': False,
            'serviceAccountsEnabled': False,
            'implicitFlowEnabled': False,
            'fullScopeAllowed': False,
            'redirectUris': [ORIGIN + '/bff/auth/oidc/callback'],
            'webOrigins': [ORIGIN],
            'attributes': {
                'construction-managed': MARKER,
                **SESSION_POLICY.client_attributes(),
                'pkce.code.challenge.method': 'S256',
                'post.logout.redirect.uris': ORIGIN + '/*',
            },
        }
        if existing:
            if (
                len(existing) != 1
                or existing[0]
                .get('attributes', {})
                .get('construction-managed')
                != MARKER
            ):
                raise RuntimeError(
                    'Refusing to replace an unowned OIDC client.',
                )
            cid = existing[0]['id']
            api('PUT', '/clients/' + cid, json={**existing[0], **desired})
        else:
            api('POST', '/clients', json=desired)
            cid = api('GET', '/clients', params={'clientId': CLIENT})[0]['id']
        path = '/clients/' + cid
        mapper = {
            'name': 'dev-api-audience',
            'protocol': 'openid-connect',
            'protocolMapper': 'oidc-audience-mapper',
            'config': {
                'included.custom.audience': AUDIENCE,
                'access.token.claim': 'true',
                'id.token.claim': 'false',
            },
        }
        old = next(
            (
                m
                for m in api('GET', path + '/protocol-mappers/models')
                if m['name'] == mapper['name']
            ),
            None,
        )
        if old:
            api(
                'PUT',
                path + '/protocol-mappers/models/' + old['id'],
                json={**mapper, 'id': old['id']},
            )
        else:
            api('POST', path + '/protocol-mappers/models', json=mapper)
        users = api(
            'GET',
            '/users',
            params={'username': 'ChangDar', 'exact': 'true'},
        )
        if len(users) != 1 or not users[0]['enabled']:
            raise RuntimeError('Expected one enabled ChangDar operator.')
        user = users[0]
        return api('GET', path + '/client-secret')['value'], {
            'id': user['id'],
            'username': 'ChangDar',
            'email': user.get('email') or f"oidc@{user['id']}.local",
        }


class _RealmRequests:
    def __init__(self, client, prefix):
        self.client = client
        self.prefix = prefix

    def request(self, method, path, **kwargs):
        response = self.client.request(method, self.prefix + path, **kwargs)
        response.raise_for_status()
        return response.json() if response.content else None
