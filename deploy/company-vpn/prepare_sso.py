from __future__ import annotations

import json
import os
import sys
from contextlib import closing
from pathlib import Path

import httpx

BASE = 'https://changdar-server.mooo.com/keycloak'
ISSUER = BASE + '/realms/visionnaire'
ORIGIN = 'https://vpn.changdar-server.mooo.com'
CLIENT = 'company-vpn'
MARKER = 'company-vpn-v1'
GROUPS = ('vpn-admins', 'vpn-employees')
OUTPUT = Path.home() / 'company-vpn-build/keycloak-sso.json'


def admin_client():
    credentials = json.loads(
        (
            Path.home() / 'service-migration-phase5/source-keycloak-env.json'
        ).read_text(),
    )
    client = httpx.Client(timeout=30, trust_env=False, follow_redirects=False)
    response = client.post(
        BASE + '/realms/master/protocol/openid-connect/token',
        data={
            'client_id': 'admin-cli',
            'grant_type': 'password',
            'username': credentials['KC_BOOTSTRAP_ADMIN_USERNAME'],
            'password': credentials['KC_BOOTSTRAP_ADMIN_PASSWORD'],
        },
    )
    if response.status_code != 200:
        client.close()
        raise RuntimeError('Keycloak administrator authentication failed.')
    client.headers['Authorization'] = (
        'Bearer ' + response.json()['access_token']
    )
    return client


def api(client, method, path, **kwargs):
    response = client.request(
        method,
        BASE + '/admin/realms/visionnaire' + path,
        **kwargs,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            'Keycloak operation failed: '
            + method
            + ' '
            + path.split('?')[0]
            + ' HTTP '
            + str(response.status_code),
        )
    return response.json() if response.content else None


def provision_client(client, client_id=CLIENT, origin=ORIGIN, marker=MARKER):
    existing = api(client, 'GET', '/clients', params={'clientId': client_id})
    desired = {
        'clientId': client_id,
        'name': 'Company VPN',
        'enabled': True,
        'protocol': 'openid-connect',
        'publicClient': False,
        'standardFlowEnabled': True,
        'directAccessGrantsEnabled': False,
        'serviceAccountsEnabled': False,
        'implicitFlowEnabled': False,
        'fullScopeAllowed': False,
        'redirectUris': [origin + '/oauth2/callback'],
        'webOrigins': [origin],
        'attributes': {
            'company-vpn-managed': marker,
            'post.logout.redirect.uris': origin + '/oauth2/logout/callback',
            'access.token.lifespan': '300',
        },
    }
    if existing:
        if (
            len(existing) != 1
            or existing[0].get('attributes', {}).get('company-vpn-managed')
            != marker
        ):
            raise RuntimeError('Refusing to change an unmanaged VPN client.')
        cid = existing[0]['id']
        api(client, 'PUT', '/clients/' + cid, json={**existing[0], **desired})
    else:
        api(client, 'POST', '/clients', json=desired)
        cid = api(
            client,
            'GET',
            '/clients',
            params={'clientId': client_id},
        )[
            0
        ]['id']
    path = '/clients/' + cid
    mapper = {
        'name': 'company-vpn-groups',
        'protocol': 'openid-connect',
        'protocolMapper': 'oidc-group-membership-mapper',
        'config': {
            'claim.name': 'groups',
            'full.path': 'false',
            'id.token.claim': 'true',
            'access.token.claim': 'true',
            'userinfo.token.claim': 'true',
        },
    }
    old = next(
        (
            m
            for m in api(client, 'GET', path + '/protocol-mappers/models')
            if m['name'] == mapper['name']
        ),
        None,
    )
    if old:
        api(
            client,
            'PUT',
            path + '/protocol-mappers/models/' + old['id'],
            json={**mapper, 'id': old['id']},
        )
    else:
        api(client, 'POST', path + '/protocol-mappers/models', json=mapper)
    current = api(client, 'GET', path)
    for key in (
        'publicClient',
        'standardFlowEnabled',
        'directAccessGrantsEnabled',
        'serviceAccountsEnabled',
        'implicitFlowEnabled',
        'redirectUris',
        'fullScopeAllowed',
    ):
        if current.get(key) != desired[key]:
            raise RuntimeError('VPN client verification differs: ' + key)
    return cid, api(client, 'GET', path + '/client-secret')['value']


def provision_groups(client):
    ids = {}
    for name in GROUPS:
        found = [
            g
            for g in api(
                client,
                'GET',
                '/groups',
                params={'search': name, 'exact': 'true'},
            )
            if g['name'] == name and g.get('path') == '/' + name
        ]
        if not found:
            api(
                client,
                'POST',
                '/groups',
                json={
                    'name': name,
                    'attributes': {'company-vpn-managed': [MARKER]},
                },
            )
            found = [
                g
                for g in api(
                    client,
                    'GET',
                    '/groups',
                    params={'search': name, 'exact': 'true'},
                )
                if g['name'] == name and g.get('path') == '/' + name
            ]
        if len(found) != 1:
            raise RuntimeError('Expected one VPN group: ' + name)
        group = api(client, 'GET', '/groups/' + found[0]['id'])
        if group.get('attributes', {}).get('company-vpn-managed') != [MARKER]:
            raise RuntimeError('Refusing to reuse an unmanaged group: ' + name)
        ids[name] = group['id']
    operator = json.loads(
        (Path.home() / '.config/construction-dev/operator.json').read_text(),
    )
    user = api(client, 'GET', '/users/' + operator['id'])
    if not user['enabled'] or user['username'].casefold() != 'changdar':
        raise RuntimeError('Expected the existing enabled ChangDar operator.')
    api(client, 'PUT', '/users/' + user['id'] + '/groups/' + ids['vpn-admins'])
    membership = api(client, 'GET', '/users/' + user['id'] + '/groups')
    if ids['vpn-admins'] not in {g['id'] for g in membership}:
        raise RuntimeError('Operator VPN group assignment did not persist.')
    return ids, {'id': user['id'], 'username': user['username']}


def main():
    os.umask(0o077)
    OUTPUT.parent.mkdir(mode=0o700, exist_ok=True)
    with closing(admin_client()) as client:
        cid, secret = provision_client(client)
        groups, operator = provision_groups(client)
        bundle = {
            'marker': MARKER,
            'client_id': CLIENT,
            'client_secret': secret,
            'client_uuid': cid,
            'issuer': ISSUER,
            'redirect_uri': ORIGIN + '/oauth2/callback',
            'allowed_groups': list(GROUPS),
            'operator': operator,
            'keycloak_groups': groups,
        }
    temporary = OUTPUT.with_suffix('.tmp')
    temporary.write_text(json.dumps(bundle, indent=2) + '\n')
    temporary.chmod(0o600)
    temporary.replace(OUTPUT)
    print(
        (
            'Prepared dedicated VPN client, exact callback, groups claim and '
            'operator vpn-admins '
            'membership.'
        ),
    )
    print('Private bundle:', str(OUTPUT), '(client secret is not displayed)')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('Stopped:', str(error), file=sys.stderr)
        raise SystemExit(1)
