# Verify the public dev BFF with a temporary real OIDC/PKCE login.
#
# No passwords, access tokens, session cookies or device keys are printed or
# saved. Only the SSO session and disposable enrollment created here are
# retired.
# This check never calls a model or changes an employee's budget or roles.
from __future__ import annotations

import base64
import hashlib
import json
import secrets
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
    prefix = BASE + '/admin/realms/visionnaire'
    api = '/bff/ai_access/api/company-ai'
    checks = {}
    session_key = sid = device_id = None
    label = 'DEV-BFF-CHECK-' + uuid4().hex[:12]
    with httpx.Client(
        timeout=30,
        trust_env=False,
        follow_redirects=False,
    ) as client:
        request = _DevRequests(client).request

        request('GET', api + '/policy', 401)
        checks['anonymous_rejected'] = True
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
                'Private Keycloak administrator authentication failed.',
            )
        admin = {'Authorization': 'Bearer ' + result.json()['access_token']}
        result = client.post(
            prefix + '/users/' + operator['id'] + '/impersonation',
            headers=admin,
        )
        if result.status_code != 200:
            raise RuntimeError(
                'Temporary operator SSO session was not created.',
            )
        # Record this new SSO session before callback, so failure cleanup does
        # not remove unrelated operator/browser sessions.
        for cookie in result.cookies.jar:
            if cookie.name == 'KEYCLOAK_SESSION':
                sid = cookie.value.rsplit('/', 1)[-1]
        try:
            start = request('GET', '/bff/auth/oidc/login', 307)
            authorized = client.get(start.headers['location'])
            target = urlsplit(authorized.headers.get('location', ''))
            if (
                target.hostname != 'dev.changdar-server.mooo.com'
                or 'code' not in parse_qs(target.query)
            ):
                raise RuntimeError(
                    'OIDC did not return the expected dev callback.',
                )
            callback = request('GET', target.path + '?' + target.query, 303)
            cookie = callback.cookies.get('__Host-vn_dev_session')
            if not cookie:
                raise RuntimeError(
                    'Dev callback did not set its secure session cookie.',
                )
            session_key = (
                'bff:session:' + hashlib.sha256(cookie.encode()).hexdigest()
            )
            saved = json.loads(store.get(session_key))
            access = cipher.decrypt(
                saved['access_token_encrypted'].encode(),
            ).decode()
            claims = jwt.decode(access, options={'verify_signature': False})
            sid = claims.get('sid') or sid
            audience = claims['aud']
            assert AUDIENCE in (
                [audience] if isinstance(audience, str) else audience
            )
            assert claims['sub'] == operator['id']
            checks['public_https_real_oidc_pkce_session'] = True

            policy_response = request('GET', api + '/policy')
            assert policy_response.headers['content-type'].startswith(
                'application/json',
            )
            assert 'no-store' in policy_response.headers[
                'cache-control'
            ].split(', ')
            policy = policy_response.json()
            assert policy['environment'] == 'dev' and policy['models']
            checks['authenticated_policy_json'] = True
            listing = request('GET', api + '/devices').json()
            assert isinstance(listing['devices'], list)
            assert all(
                not {'key', 'key_box', 'key_hash'}.intersection(d)
                for d in listing['devices']
            )
            checks['authenticated_own_devices_without_secrets'] = True
            request('GET', api + '/admin/devices')
            checks['current_directory_admin_role'] = True

            csrf = request('GET', '/bff/auth/csrf').json()['csrf_token']
            browser = {
                'Origin': ORIGIN,
                'X-CSRF-Token': csrf,
                'X-Company-AI-Action': '1',
            }
            missing = api + '/devices/' + str(uuid4()) + '/revoke'
            request(
                'POST',
                missing,
                403,
                headers={**browser, 'X-CSRF-Token': 'wrong'},
                json={},
            )
            request(
                'POST',
                missing,
                403,
                headers={**browser, 'Origin': 'https://evil.example'},
                json={},
            )
            denied = request(
                'POST',
                missing,
                403,
                headers={**browser, 'X-Company-AI-Action': ''},
                json={},
            )
            assert (
                denied.json()['detail']['code']
                == 'ai_access_same_origin_required'
            )
            reached = request('POST', missing, 404, headers=browser, json={})
            assert (
                reached.json()['detail']['code']
                == 'ai_access_device_not_found'
            )
            checks['bff_csrf_and_backend_origin_action_enforced'] = True

            verifier = secrets.token_urlsafe(48)
            challenge = (
                base64.urlsafe_b64encode(
                    hashlib.sha256(verifier.encode()).digest(),
                )
                .decode()
                .rstrip('=')
            )
            started = client.post(
                (
                    'https://ai-dev.changdar-server.mooo.com/api/comp'
                    'any-ai/device/start'
                ),
                json={'device_name': label, 'code_challenge': challenge},
            )
            if started.status_code != 200:
                raise RuntimeError(
                    'Disposable native enrollment could not be started.',
                )
            enrollment = started.json()
            path = api + '/enrollments/' + enrollment['request_id']
            preview = request('GET', path).json()
            assert (
                preview['label'] == label
                and preview['code'] == enrollment['user_code']
            )
            approved = request(
                'POST',
                path + '/approve',
                headers=browser,
                json={'user_code': enrollment['user_code'], 'confirm': True},
            ).json()
            device_id = approved['device_id']
            request(
                'POST',
                path + '/approve',
                409,
                headers=browser,
                json={'user_code': enrollment['user_code'], 'confirm': True},
            )
            # Never poll: this BFF check creates no virtual key.
            revoked = request(
                'POST',
                api + '/devices/' + device_id + '/revoke',
                headers=browser,
                json={},
            ).json()
            assert revoked['status'] == 'revoked'
            checks['preview_approve_duplicate_denial_and_revoke'] = True
            device_id = None

            # A validly signed token for the directory service must fail the
            # configured application audience before membership is evaluated.
            config = json.loads(
                (
                    Path.home() / '.config/ai-gateway-dev/ai-access.json'
                ).read_text(),
            )
            other = client.post(
                config['AI_ACCESS_ISSUER'] + '/protocol/openid-connect/token',
                data={
                    'grant_type': 'client_credentials',
                    'client_id': config['AI_ACCESS_DIRECTORY_CLIENT'],
                    'client_secret': config['AI_ACCESS_DIRECTORY_SECRET'],
                },
            )
            if other.status_code != 200:
                raise RuntimeError(
                    (
                        'Directory audience isolation check could '
                        'not obtain its own token.'
                    ),
                )
            invalid = client.get(
                'http://127.0.0.1:18083/api/company-ai/policy',
                headers={
                    'Authorization': 'Bearer ' + other.json()['access_token'],
                },
            )
            assert invalid.status_code == 401
            checks['different_audience_rejected'] = True

            request('POST', '/bff/auth/logout', headers=browser)
            request('GET', api + '/policy', 401)
            checks['logout_invalidates_access'] = True
        finally:
            if device_id is not None:
                request(
                    'POST',
                    api + '/devices/' + device_id + '/revoke',
                    headers=browser,
                    json={},
                )
            if session_key is not None:
                store.delete(session_key)
            if sid:
                cleaned = client.delete(
                    prefix + '/sessions/' + sid,
                    headers=admin,
                )
                if cleaned.status_code not in {204, 404}:
                    raise RuntimeError('Temporary SSO session cleanup failed.')
            store.close()
    print(
        json.dumps(
            {'environment': 'dev', 'origin': ORIGIN, 'checks': checks},
            ensure_ascii=False,
            indent=2,
        ),
    )


class _DevRequests:
    def __init__(self, client):
        self.client = client

    def request(self, method, path, expected=200, headers=None, **kwargs):
        response = self.client.request(
            method,
            ORIGIN + path,
            headers=headers,
            **kwargs,
        )
        if response.status_code != expected:
            raise RuntimeError(
                (
                    f'{method}'
                    ' '
                    f"{path.split('?')[0]}"
                    ' returned HTTP '
                    f'{response.status_code}'
                    '; expected '
                    f'{expected}'
                ),
            )
        return response


if __name__ == '__main__':
    main()
