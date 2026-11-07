# Guard against accidental use of production resources or TLS fallback.
from __future__ import annotations

import asyncio
from functools import partial
from subprocess import CompletedProcess
from unittest.mock import patch
from urllib.parse import parse_qs
from urllib.parse import urlsplit

import pytest

from deploy.dev import gateway
from deploy.dev.apps import local_logout_response
from deploy.dev.apps import local_logout_url
from deploy.dev.run import validate


async def _dev_logout_bridge_is_local_exercise(Store, /):
    store = Store()
    target = urlsplit(await local_logout_url(store))
    assert not target.netloc and target.path == '/bff/auth/oidc/logout'
    assert store.ttl == 120
    state = parse_qs(target.query)['state'][0]
    response = await local_logout_response(store, state)
    assert response.status_code == 303
    assert (
        response.headers['location']
        == 'https://dev.changdar-server.mooo.com/login'
    )
    assert (await local_logout_response(store, state)).status_code == 400
    assert (await local_logout_response(store, None)).status_code == 400


class _DevLogoutBridgeIsLocalStore:
    def __init__(self):
        self.values = {}
        self.ttl = None

    async def set(self, key, value, ex):
        self.values[key] = value
        self.ttl = ex

    async def getdel(self, key):
        return self.values.pop(key, None)


@pytest.fixture
def values():
    from deploy.dev.run import STATE

    fixture = {
        'DATABASE_URL': (
            'postgresql+asyncpg://construction_dev:test@'
            '127.0.0.1:15433/construction_dev'
        ),
        'REDIS_HOST': '127.0.0.1',
        'REDIS_PORT': '16380',
        'STORAGE_ROOT': str(STATE / 'documents'),
        'SIGNATURE_ROOT': str(STATE / 'documents/signatures'),
        'WIN_UPLOAD_ROOT': str(STATE / 'documents'),
        'BFF_SESSION_COOKIE_NAME': '__Host-vn_dev_session',
        'OIDC_AUDIENCE': 'visionnaire-dev-api',
    }
    for name, port in {
        'MANAGEMENT': 19005,
        'FILES': 19004,
        'FCM': 19003,
        'VIOLATIONS': 19002,
        'STREAMING': 19800,
        'CHAT': 19009,
        'DETECTION': 19000,
        'AI_USAGE': 19009,
        'AI_ACCESS': 18083,
    }.items():
        fixture[f'BFF_{name}_API_URL'] = f'http://127.0.0.1:{port}'
    fixture['STREAMING_PLAYBACK_API_URL'] = 'http://127.0.0.1:19800'
    return fixture


def test_accepts_the_isolated_dev_contract(values):
    validate(values)


@pytest.mark.parametrize(
    ('key', 'wrong'),
    [
        (
            'DATABASE_URL',
            (
                'postgresql+asyncpg://user:pw@127.0.0.1:5432/'
                'construction_hazard_detection'
            ),
        ),
        ('REDIS_PORT', '6379'),
        ('STORAGE_ROOT', '/mnt/data'),
        ('BFF_SESSION_COOKIE_NAME', '__Host-vn_session'),
        ('OIDC_AUDIENCE', 'construction-hazard-detection'),
        ('KEYCLOAK_USER_LINKER_CLIENT_SECRET', 'production-credential'),
        ('BFF_FILES_API_URL', 'http://127.0.0.1:8004'),
        ('BFF_MANAGEMENT_API_URL', 'http://127.0.0.1:8005'),
        ('BFF_CHAT_API_URL', 'http://127.0.0.1:8001'),
        ('BFF_DETECTION_API_URL', 'http://127.0.0.1:8000'),
        ('BFF_AI_ACCESS_API_URL', 'http://127.0.0.1:8013'),
        ('SIGNATURE_ROOT', '/mnt/data/signatures'),
        ('FIREBASE_CRED_PATH', '/production/firebase.json'),
    ],
)
def test_refuses_shared_resources(values, key, wrong):
    values[key] = wrong
    with pytest.raises(RuntimeError):
        validate(values)


def test_async_reload_keeps_tls_verification():
    ready = CompletedProcess(
        [],
        0,
        'HTTP/2 200\nX-Construction-Gateway: dev-b\n'
        'X-Construction-Environment: dev\n\n'
        '{"status":true,"environment":"dev"}',
        '',
    )
    with patch.object(
        gateway.subprocess,
        'run',
        side_effect=[CompletedProcess([], 60, '', 'wrong certificate'), ready],
    ) as call:
        with patch.object(gateway.time, 'sleep'):
            gateway.wait_gateway()
    assert call.call_count == 2
    for args in call.call_args_list:
        assert '--insecure' not in args.args[0] and '-k' not in args.args[0]


def test_production_response_cannot_pass_readiness():
    production = CompletedProcess([], 0, 'HTTP/2 200\n\n{"status":true}', '')
    with patch.object(gateway.subprocess, 'run', return_value=production):
        with pytest.raises(RuntimeError):
            gateway.wait_gateway(timeout=0)


def test_wrong_hostname_certificate_remains_a_failure():
    with patch.object(
        gateway.subprocess,
        'run',
        return_value=CompletedProcess([], 60, '', 'hostname mismatch'),
    ):
        with pytest.raises(RuntimeError, match='hostname mismatch'):
            gateway.wait_gateway(timeout=0)


def test_dev_logout_bridge_is_local_single_use_and_expires():
    Store = _DevLogoutBridgeIsLocalStore

    exercise = partial(_dev_logout_bridge_is_local_exercise, Store)
    asyncio.run(exercise())
