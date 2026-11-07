from __future__ import annotations

import copy
import json
import tempfile
import unittest
from functools import partial
from pathlib import Path
from unittest.mock import patch

import httpx
from pydantic import ValidationError

from deploy import checks
from deploy.keycloak.apply_session_policy import SessionPolicyDeployment
from deploy.keycloak.session_policy import generated_files
from src.settings.sessions import SESSION_POLICY
from src.settings.sessions import SessionPolicy


def _server_handle(testcase, state, /, request):
    path = request.url.path
    if request.method == 'GET':
        if path.endswith('/clients'):
            name = request.url.params['clientId']
            return httpx.Response(
                200,
                json=[
                    c
                    for c in state['clients'].values()
                    if c['clientId'] == name
                ],
            )
        if '/clients/' in path:
            return httpx.Response(
                200,
                json=state['clients'][path.rsplit('/', 1)[-1]],
            )
        return httpx.Response(200, json=state['realm'])
    if testcase.fail_next and path.endswith(
        testcase.fail_next,
    ):
        testcase.fail_next = None
        return httpx.Response(503)
    value = json.loads(request.content)
    if '/clients/' in path:
        state['clients'][path.rsplit('/', 1)[-1]] = value
    else:
        state['realm'] = value
    return httpx.Response(204)


class SessionPolicyTest(unittest.TestCase):
    def test_limits_reject_inconsistent_or_duplicate_configuration(
        self,
    ) -> None:
        values = SESSION_POLICY.model_dump()
        for changes in (
            {'maximum_seconds': 1},
            {'access_token_seconds': 0},
            {'clients': ['duplicate', 'duplicate']},
            {'other_client_maximum_seconds': 1},
            {'optional_clients': ['unknown']},
        ):
            with self.assertRaises(ValidationError):
                SessionPolicy.model_validate({**values, **changes})

    def test_generation_preserves_unrelated_client_configuration(self) -> None:
        files = generated_files(SESSION_POLICY)
        realm = json.loads(
            next(
                value
                for path, value in files.items()
                if path.name == 'visionnaire-realm.template.json'
            ),
        )
        self.assertEqual(realm['ssoSessionIdleTimeout'], 604800)
        self.assertEqual(realm['ssoSessionMaxLifespan'], 7776000)
        self.assertEqual(realm['accessTokenLifespan'], 300)
        mobile = next(
            c
            for c in realm['clients']
            if c['clientId'] == 'visionnaire-mobile'
        )
        self.assertTrue(mobile['publicClient'])
        self.assertEqual(
            mobile['attributes']['pkce.code.challenge.method'],
            'S256',
        )
        linker = next(
            c
            for c in realm['clients']
            if c['clientId'] == 'visionnaire-user-linker'
        )
        self.assertNotIn(
            'client.session.max.lifespan',
            linker.get('attributes', {}),
        )

    def server(self):
        state = {
            'realm': {
                'realm': 'example',
                'displayName': 'Preserve this',
                'ssoSessionIdleTimeout': 1800,
            },
            'clients': {},
        }
        for index, name in enumerate(SESSION_POLICY.clients):
            state['clients'][str(index)] = {
                'id': str(index),
                'clientId': name,
                'attributes': {
                    'pkce.code.challenge.method': 'S256',
                },
            }
        self.fail_next = None

        handle = partial(_server_handle, self, state)

        client = httpx.Client(transport=httpx.MockTransport(handle))
        self.addCleanup(client.close)
        return (
            SessionPolicyDeployment(
                client,
                'https://identity.test/admin/realms/example',
            ),
            state,
        )

    def test_apply_merges_settings_and_readback_is_idempotent(self) -> None:
        deployment, state = self.server()
        self.assertTrue(deployment.differences(deployment.snapshot()))
        with tempfile.TemporaryDirectory() as root:
            backup = Path(root) / 'before.json'
            self.assertTrue(deployment.apply(backup)['matched'])
            self.assertEqual(backup.stat().st_mode & 0o777, 0o600)
            self.assertEqual(state['realm']['displayName'], 'Preserve this')
            self.assertEqual(state['realm']['ssoSessionMaxLifespan'], 7776000)
            self.assertEqual(
                state['clients']['0']['attributes'][
                    'pkce.code.challenge.method'
                ],
                'S256',
            )
            self.assertEqual(deployment.differences(deployment.snapshot()), [])
            with self.assertRaises(FileExistsError):
                deployment.apply(backup)
            self.assertTrue(
                deployment.apply(
                    Path(root) / 'second.json',
                )['matched'],
            )

    def test_only_optional_clients_may_be_missing(self) -> None:
        deployment, state = self.server()
        optional = next(
            key
            for key, client in state['clients'].items()
            if client['clientId'] == 'visionnaire-web-dev'
        )
        del state['clients'][optional]
        self.assertEqual(len(deployment.snapshot()['clients']), 2)
        del state['clients']['0']
        with self.assertRaisesRegex(ValueError, 'configured policy client'):
            deployment.snapshot()

    def test_partial_client_failure_restores_all_changed_configuration(
        self,
    ) -> None:
        deployment, state = self.server()
        before = copy.deepcopy(state)
        self.fail_next = '/clients/1'
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(httpx.HTTPStatusError):
                deployment.apply(Path(root) / 'before.json')
        self.assertEqual(state, before)

    def test_deployment_checks_reject_invalid_json_and_stale_generated_policy(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            file = directory / 'bad.json'
            file.write_text('{invalid')
            with patch(
                'deploy.checks.generated_files',
                return_value={file: '{}'},
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    'Regenerate session policy',
                ):
                    checks.check_paths([])
