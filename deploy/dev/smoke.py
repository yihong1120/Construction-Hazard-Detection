# Exercise a temporary operator login, isolated CRUD and service reads on A.
#
# The existing Keycloak administrator creates one temporary SSO session for the
# operator. The normal authorization-code/PKCE callback verifies the actual
# provider signature and audience. Only that session is removed afterwards.
# No browser password or reusable authentication token is written to disk.
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from urllib.parse import parse_qs
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
import jwt
import redis
from cryptography.fernet import Fernet

from deploy.dev.identity import AUDIENCE
from deploy.dev.identity import BASE
from deploy.dev.identity import ORIGIN
from deploy.dev.run import CONFIG
from deploy.dev.run import validate


def main():
    env = json.loads((CONFIG / 'environment.json').read_text())
    validate(env)
    credentials = json.loads(
        (
            Path.home() / 'service-migration-phase5/source-keycloak-env.json'
        ).read_text(),
    )
    operator = json.loads((CONFIG / 'operator.json').read_text())
    store = redis.Redis(
        host='127.0.0.1',
        port=16380,
        password=env['REDIS_PASSWORD'],
        socket_timeout=10,
    )
    cipher = Fernet(
        base64.urlsafe_b64encode(
            hashlib.sha256(env['BFF_TOKEN_ENCRYPTION_KEY'].encode()).digest(),
        ),
    )
    site_id = None
    session_key = None
    sid = None
    with httpx.Client(
        timeout=30,
        trust_env=False,
        follow_redirects=False,
    ) as client:
        result = client.post(
            BASE + '/realms/master/protocol/openid-connect/token',
            data={
                'client_id': 'admin-cli',
                'grant_type': 'password',
                'username': credentials['KC_BOOTSTRAP_ADMIN_USERNAME'],
                'password': credentials['KC_BOOTSTRAP_ADMIN_PASSWORD'],
            },
        )
        if result.status_code != 200:
            raise RuntimeError(
                (
                    'Operator test could not authenticate the '
                    'private Keycloak administrator.'
                ),
            )
        admin = {'Authorization': 'Bearer ' + result.json()['access_token']}
        prefix = BASE + '/admin/realms/visionnaire'
        result = client.post(
            prefix + '/users/' + operator['id'] + '/impersonation',
            headers=admin,
        )
        if result.status_code != 200:
            raise RuntimeError(
                'Temporary operator SSO session was not created.',
            )
        base = 'http://127.0.0.1:18090'
        headers = {'Host': 'dev.changdar-server.mooo.com'}

        request = _DevRequests(client, base, headers).request

        try:
            start = request('GET', '/bff/auth/oidc/login', 307)
            authorized = client.get(start.headers['location'])
            target = urlsplit(authorized.headers.get('location', ''))
            if (
                target.hostname != 'dev.changdar-server.mooo.com'
                or 'code' not in parse_qs(target.query)
            ):
                raise RuntimeError(
                    'The dev OIDC client did not return a valid dev callback.',
                )
            callback = request('GET', target.path + '?' + target.query, 303)
            cookie = callback.cookies.get('__Host-vn_dev_session')
            if not cookie:
                raise RuntimeError(
                    'Dev callback did not set its own secure session cookie.',
                )
            headers['Cookie'] = '__Host-vn_dev_session=' + cookie
            session_key = (
                'bff:session:' + hashlib.sha256(cookie.encode()).hexdigest()
            )
            saved = json.loads(store.get(session_key))
            access = cipher.decrypt(
                saved['access_token_encrypted'].encode(),
            ).decode()
            claims = jwt.decode(access, options={'verify_signature': False})
            sid = claims.get('sid')
            audience = claims['aud']
            if AUDIENCE not in (
                [audience] if isinstance(audience, str) else audience
            ):
                raise RuntimeError(
                    'Dev access token is missing its isolated audience.',
                )
            summary = request('GET', '/bff/auth/session').json()
            expected = json.loads(
                (CONFIG / 'authorization-baseline.json').read_text(),
            )['operator']
            if (
                set(summary['feature_names']) != set(expected['features'])
                or summary['user']['role'] != expected['role']
            ):
                raise RuntimeError(
                    (
                        'Dev session is missing the operator '
                        'baseline permissions.'
                    ),
                )
            csrf = request('GET', '/bff/auth/csrf').json()['csrf_token']
            headers.update({'Origin': ORIGIN, 'X-CSRF-Token': csrf})
            request('GET', '/bff/db_management/list_sites')
            request('GET', '/bff/file_manage/sign/tasks')
            request('GET', '/bff/fcm/notifications')
            request('GET', '/bff/streaming_web/labels')
            models = request('GET', '/bff/detect/models').json()
            assert len(models['models']) == 5
            selected = next(
                m
                for m in models['models']
                if m['id'] == models['default_model_id']
            )
            image = (
                Path(__file__).resolve().parents[2]
                / 'assets/images/demo/persons_in_restricted_zones.jpg'
            )
            overlay = request(
                'POST',
                '/bff/detect/v2/detect',
                timeout=120,
                data={
                    'model': selected['id'],
                    'model_version': selected['version'],
                },
                files={
                    'image': (image.name, image.read_bytes(), 'image/jpeg'),
                },
            ).json()
            assert (
                overlay['model_id'] == selected['id']
                and overlay['model_version'] == selected['version']
            )
            assert len(overlay['class_metadata']) == len(selected['classes'])
            assert overlay['overlay_objects']
            site = request(
                'POST',
                '/bff/db_management/create_site',
                json={
                    'name': 'DEV smoke ' + uuid4().hex,
                    'group_ids': [summary['user']['group_id']],
                },
            ).json()
            site_id = site['id']
            request(
                'DELETE',
                '/bff/db_management/delete_site',
                json={'site_id': site_id},
            )
            site_id = None
            production = client.get(
                (
                    'https://changdar-server.mooo.com/hazard/api/db_m'
                    'anagement/list_sites'
                ),
                headers={'Authorization': 'Bearer ' + access},
            )
            if production.status_code not in {401, 403}:
                raise RuntimeError('Production accepted a dev access token.')
            headers['Origin'] = 'https://changdar-server.mooo.com'
            request(
                'POST',
                '/bff/db_management/create_site',
                403,
                json={'name': 'must not exist'},
            )
            headers['Origin'] = ORIGIN
            request('POST', '/bff/auth/logout')
            print(
                (
                    'Passed: real OIDC/PKCE callback, secure dev '
                    'session, files/signatures, notifications, '
                    'streaming catalog,'
                ),
            )
            print(
                (
                    'dev site create/delete, CSRF origin isolation '
                    'and production rejection of dev tokens.'
                ),
            )
            print(
                (
                    'Five dev models and real YOLO image inference '
                    'with versioned class metadata and overlays '
                    'verified.'
                ),
            )
        finally:
            if site_id is not None:
                request(
                    'DELETE',
                    '/bff/db_management/delete_site',
                    json={'site_id': site_id},
                )
            if session_key is not None:
                store.delete(session_key)
            if sid:
                response = client.delete(
                    prefix + '/sessions/' + sid,
                    headers=admin,
                )
                if response.status_code not in {204, 404}:
                    raise RuntimeError('Temporary SSO session cleanup failed.')
    store.close()


class _DevRequests:
    def __init__(self, client, base, headers):
        self.client = client
        self.base = base
        self.headers = headers

    def request(self, method, path, expected=200, **kwargs):
        response = self.client.request(
            method,
            self.base + path,
            headers=self.headers,
            **kwargs,
        )
        if response.status_code != expected:
            raise RuntimeError(
                (
                    'Dev request failed: '
                    f'{method}'
                    ' '
                    f"{path.split('?')[0]}"
                    ' HTTP '
                    f'{response.status_code}'
                ),
            )
        return response


if __name__ == '__main__':
    main()
