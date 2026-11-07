from __future__ import annotations

import json
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

from src.settings.sessions import SESSION_POLICY


class SessionStartupTest(unittest.TestCase):
    def run_startup(self, drift=''):
        root = Path(__file__).resolve().parents[2]
        source = (root / 'deploy/keycloak/start.sh').read_text()
        function = source.split('configure_session_lifetimes() {', 1)[1]
        function = function.split('configure_visionnaire_browser_flow()', 1)[0]
        function = 'configure_session_lifetimes() {' + function
        realm = SESSION_POLICY.realm_settings()
        attrs = SESSION_POLICY.client_attributes()
        if drift == 'realm':
            realm['accessTokenLifespan'] = 1
        if drift == 'client':
            attrs['access.token.lifespan'] = '1'
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'updates.log'
            script = '\n'.join(
                [
                    'set -euo pipefail',
                    'source deploy/keycloak/session-policy.env',
                    'realm=visionnaire',
                    'kcadm=fake_admin',
                    'fake_admin() {',
                    '  if [[ $1 == update ]]; then',
                    f'    printf "%s\\n" "$*" >> {shlex.quote(str(log))}',
                    '  elif [[ $2 == realms/* ]]; then',
                    '    printf "%s" ' + shlex.quote(json.dumps(realm)),
                    '  elif [[ $2 == clients ]]; then',
                    '    if [[ $* == *clientId=visionnaire-web-dev* ]]; then',
                    "      printf '[]'",
                    '    else',
                    '      printf \'[{"id":"known-client"}]\'',
                    '    fi',
                    '  else',
                    '    printf "%s" '
                    + shlex.quote(
                        json.dumps({'attributes': attrs}),
                    ),
                    '  fi',
                    '}',
                    function,
                    'configure_session_lifetimes',
                ],
            )
            result = subprocess.run(
                ['bash', '-c', script],
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
            )
            return result, log.read_text()

    def test_startup_updates_existing_clients_and_verifies_policy(self):
        result, updates = self.run_startup()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(updates.count('update clients/known-client'), 2)
        self.assertIn(
            'attributes."client.session.idle.timeout"="604800"',
            updates,
        )
        self.assertNotIn('pkce.code.challenge.method', updates)

    def test_startup_rejects_realm_or_client_readback_drift(self):
        for drift in ('realm', 'client'):
            with self.subTest(drift=drift):
                result, _ = self.run_startup(drift)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('readback failed', result.stderr)
