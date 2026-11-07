# Test the deployed frontend/web OIDC flow and session restore over HTTPS.
#
# Uses one temporary operator SSO session, never an operator password. The
# first
# HTTP callback records its session identity so cleanup also works if Chromium
# fails before completing its own callback. No credentials are saved to disk.
# Requires Playwright and an installed Chromium; pass --chrome for its
# executable.
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from functools import partial
from pathlib import Path
from urllib.parse import parse_qs
from urllib.parse import urlsplit

import httpx
import jwt
import redis
from cryptography.fernet import Fernet
from playwright.sync_api import sync_playwright

from deploy.dev.identity import BASE
from deploy.dev.identity import CLIENT
from deploy.dev.identity import ORIGIN
from deploy.dev.run import CONFIG
from deploy.dev.run import validate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chrome', required=True)
    args = parser.parse_args()
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
    recorded = _RecordedSessions(store, cipher, operator['id'])
    session_keys = recorded.keys

    record_session = recorded.record

    with httpx.Client(
        timeout=30,
        trust_env=False,
        follow_redirects=False,
    ) as client, sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=args.chrome,
            headless=True,
            args=[
                '--disable-dev-shm-usage',
                '--use-angle=swiftshader',
                '--enable-unsafe-swiftshader',
            ],
        )
        context = browser.new_context(viewport={'width': 1920, 'height': 1080})
        page = context.new_page()
        page.set_default_timeout(30000)
        admin = None
        phase = 'anonymous sign-in flow'
        prefix = BASE + '/admin/realms/visionnaire'
        try:
            page.goto(ORIGIN + '/files', wait_until='networkidle')
            # The repo artifact redirects directly to OIDC. A future artifact
            # may present a sign-in button instead; exercise that if present.
            if urlsplit(page.url).hostname == urlsplit(ORIGIN).hostname:
                page.locator('flt-semantics-placeholder').evaluate(
                    '(el) => el.click()',
                )
                button = page.get_by_role(
                    'button',
                    name='Login',
                    exact=True,
                ).last
                button.wait_for()
                assert page.get_by_role('textbox').count() == 0
                button.click()
            page.wait_for_url(
                (
                    '**/keycloak/realms/visionnaire/protocol/openid-c'
                    'onnect/auth**'
                ),
            )
            query = parse_qs(urlsplit(page.url).query)
            assert query['client_id'] == [CLIENT]
            assert query['redirect_uri'] == [
                ORIGIN + '/bff/auth/oidc/callback',
            ]
            page.locator('input[name="username"]').wait_for()
            page.locator('input[name="password"]').wait_for()

            phase = 'temporary provider session'
            result = client.post(
                BASE + '/realms/master/protocol/openid-connect/token',
                data={
                    'client_id': 'admin-cli',
                    'grant_type': 'password',
                    'username': credentials['KC_BOOTSTRAP_ADMIN_USERNAME'],
                    'password': credentials['KC_BOOTSTRAP_ADMIN_PASSWORD'],
                },
            )
            assert result.status_code == 200
            admin = {
                'Authorization': 'Bearer ' + result.json()['access_token'],
            }
            result = client.post(
                prefix + '/users/' + operator['id'] + '/impersonation',
                headers=admin,
            )
            assert result.status_code == 200
            start = client.get(
                ORIGIN + '/bff/auth/oidc/login?return_to=/files',
            )
            assert start.status_code == 307
            authorized = client.get(start.headers['location'])
            target = urlsplit(authorized.headers.get('location', ''))
            assert target.hostname == urlsplit(ORIGIN).hostname
            assert target.path == '/bff/auth/oidc/callback'
            assert 'code' in parse_qs(target.query)
            callback = client.get(ORIGIN + target.path + '?' + target.query)
            assert callback.status_code == 303
            record_session(callback.cookies['__Host-vn_dev_session'])

            # Transfer only the temporary provider session; the browser must
            # acquire its own dev cookie through the real public callback.
            phase = 'browser callback'
            provider_cookies = [
                {
                    'name': c.name,
                    'value': c.value,
                    'domain': c.domain,
                    'path': c.path,
                    'secure': c.secure,
                }
                for c in client.cookies.jar
                if c.domain.lstrip('.') == urlsplit(BASE).hostname
            ]
            assert provider_cookies
            context.add_cookies(provider_cookies)
            page.reload(wait_until='networkidle')
            page.wait_for_url(ORIGIN + '/files')
            cookies = context.cookies(ORIGIN)
            cookie = next(
                c['value']
                for c in cookies
                if c['name'] == '__Host-vn_dev_session'
            )
            record_session(cookie)
            phase = 'authenticated documents page'
            page.locator('flt-semantics-placeholder').evaluate(
                '(el) => el.click()',
            )
            page.get_by_text('Documents', exact=True).first.wait_for()
            assert (
                context.request.get(ORIGIN + '/bff/auth/session').status == 200
            )
            assert (
                context.request.get(
                    ORIGIN + '/bff/file_manage/sign/tasks',
                ).status
                == 200
            )
            phase = 'session restoration'
            # Reproduce an operator cookie created before grants were seeded.
            # Change only this test's metadata, retaining the same credentials
            # and TTL; the endpoint must recover permissions from the dev DB.
            key = 'bff:session:' + hashlib.sha256(cookie.encode()).hexdigest()
            saved = json.loads(store.get(key))
            saved['feature_names'] = ['yolo_api']
            saved['user']['display_name'] = 'Stale test profile'
            store.set(key, json.dumps(saved), xx=True, keepttl=True)
            page.reload(wait_until='networkidle')
            assert urlsplit(page.url).path == '/files'
            restored = context.request.get(ORIGIN + '/bff/auth/session')
            assert restored.status == 200
            assert set(restored.json()['feature_names']) == {
                'doc_chat',
                'file_manage',
                'yolo_api',
            }
            page.locator('flt-semantics-placeholder').evaluate(
                '(el) => el.click()',
            )
            for label in ('Chat List', 'File Management', 'AI Usage'):
                phase = 'restored navigation: ' + label
                page.get_by_text(label, exact=True).first.wait_for()
            phase = 'detection model selector'
            with page.expect_response(
                partial(_response_path, suffix='/models'),
            ) as model_response:
                page.goto(ORIGIN + '/detection', wait_until='networkidle')
            assert model_response.value.status == 200
            assert len(model_response.value.json()['models']) == 5
            page.locator('flt-semantics-placeholder').evaluate(
                '(el) => el.click()',
            )
            page.get_by_text('YOLO26N', exact=True).first.wait_for()
            phase = 'browser image upload and inference'
            # Flutter scrolls its canvas rather than the DOM document.
            page.mouse.move(600, 650)
            page.mouse.wheel(0, 800)
            with page.expect_file_chooser() as chooser:
                page.get_by_role(
                    'button',
                    name='Photo Library',
                    exact=True,
                ).click()
            image = (
                Path(__file__).resolve().parents[2]
                / 'assets/images/demo/persons_in_restricted_zones.jpg'
            )
            chooser.value.set_files(str(image))
            with page.expect_response(
                partial(_response_path, suffix='/v2/detect'),
                timeout=120000,
            ) as inference:
                page.get_by_role(
                    'button',
                    name='Start Detection',
                    exact=True,
                ).click()
            assert (
                inference.value.status == 200
                and inference.value.json()['overlay_objects']
            )
            page.get_by_role(
                'button',
                name='Start Detection',
                exact=True,
            ).wait_for()
            phase = 'local dev logout'
            csrf = context.request.get(ORIGIN + '/bff/auth/csrf').json()[
                'csrf_token'
            ]
            logout = context.request.post(
                ORIGIN + '/bff/auth/logout',
                headers={
                    'Origin': ORIGIN,
                    'X-CSRF-Token': csrf,
                },
            )
            assert logout.status == 200
            bridge = logout.json()['global_logout_url']
            assert urlsplit(bridge).path == '/bff/auth/oidc/logout'
            local = context.request.get(ORIGIN + bridge, max_redirects=0)
            assert (
                local.status == 303
                and local.headers['location'] == ORIGIN + '/login'
            )
            assert (
                context.request.get(ORIGIN + bridge, max_redirects=0).status
                == 400
            )
            assert (
                context.request.get(ORIGIN + '/bff/auth/session').status == 401
            )
            print(
                (
                    'Passed: deployed frontend/web sign-in, Keycloak '
                    'dev client, browser OIDC callback,'
                ),
            )
            print(
                (
                    'authenticated files/signatures, stale-session '
                    'grant recovery, complete navigation and local '
                    'dev logout.'
                ),
            )
            print(
                (
                    'Detection page model selector, photo upload and '
                    'real image inference passed over public HTTPS.'
                ),
            )
        except Exception:
            # Browser diagnostics can contain single-use callback query
            # strings.
            raise RuntimeError(
                'Public dev browser verification failed at '
                + phase
                + '; sensitive diagnostics suppressed.',
            ) from None
        finally:
            for cookie in context.cookies(ORIGIN):
                if cookie['name'] == '__Host-vn_dev_session':
                    session_keys.add(
                        'bff:session:'
                        + hashlib.sha256(cookie['value'].encode()).hexdigest(),
                    )
            for key in session_keys:
                store.delete(key)
            if recorded.sid and admin:
                response = client.delete(
                    prefix + '/sessions/' + recorded.sid,
                    headers=admin,
                )
                if response.status_code not in {204, 404}:
                    raise RuntimeError('Temporary browser SSO cleanup failed.')
            browser.close()
            store.close()


class _RecordedSessions:
    def __init__(self, store, cipher, subject):
        self.store = store
        self.cipher = cipher
        self.subject = subject
        self.keys = set()
        self.sid = None

    def record(self, cookie):
        key = 'bff:session:' + hashlib.sha256(cookie.encode()).hexdigest()
        self.keys.add(key)
        saved = json.loads(self.store.get(key))
        access = self.cipher.decrypt(
            saved['access_token_encrypted'].encode(),
        ).decode()
        claims = jwt.decode(access, options={'verify_signature': False})
        assert claims['azp'] == CLIENT
        assert claims['sub'] == self.subject
        assert self.sid is None or self.sid == claims['sid']
        self.sid = claims['sid']


def _response_path(response, *, suffix):
    return urlsplit(response.url).path.endswith(suffix)


if __name__ == '__main__':
    main()
