from __future__ import annotations

import argparse
import json
import os
from functools import partial
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from src.settings.sessions import SESSION_POLICY
from src.settings.sessions import SessionPolicy


class SessionPolicyDeployment:
    def __init__(
        self,
        client: httpx.Client,
        realm_url: str,
        policy: SessionPolicy = SESSION_POLICY,
    ) -> None:
        self.client, self.url, self.policy = client, realm_url, policy

    def _get(self, path: str = '') -> dict:
        response = self.client.get(self.url + path)
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict):
            raise ValueError('Expected a configuration object')
        return result

    def snapshot(self) -> dict:
        realm = self._get()
        clients = []
        for name in self.policy.clients:
            response = self.client.get(
                self.url + '/clients',
                params={'clientId': name},
            )
            response.raise_for_status()
            matches = response.json()
            if not matches and name in self.policy.optional_clients:
                continue
            if len(matches) != 1:
                raise ValueError(
                    'Expected one configured policy client: ' + name,
                )
            clients.append(self._get('/clients/' + matches[0]['id']))
        return {'realm': realm, 'clients': clients}

    def differences(self, snapshot: dict) -> list[str]:
        drift = [
            f'realm.{key}'
            for key, value in self.policy.realm_settings().items()
            if snapshot['realm'].get(key) != value
        ]
        for client in snapshot['clients']:
            drift.extend(
                f"{client['clientId']}.{key}"
                for key, value in self.policy.client_attributes().items()
                if client.get('attributes', {}).get(key) != value
            )
        return drift

    def apply(self, backup: Path) -> dict:
        before = self.snapshot()
        backup.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Preserve the original rollback snapshot on repeated application.
        with open(
            backup,
            'x',
            opener=partial(os.open, mode=0o600),
        ) as output:
            json.dump(before, output, indent=2)
        modified = []
        try:
            modified.append(('', before['realm']))
            self.client.put(
                self.url,
                json={
                    **before['realm'],
                    **self.policy.realm_settings(),
                },
            ).raise_for_status()
            for previous in before['clients']:
                path = '/clients/' + previous['id']
                modified.append((path, previous))
                payload = {
                    **previous,
                    'attributes': {
                        **previous.get('attributes', {}),
                        **self.policy.client_attributes(),
                    },
                }
                self.client.put(
                    self.url + path,
                    json=payload,
                ).raise_for_status()
            after = self.snapshot()
            if self.differences(after):
                raise RuntimeError('Session policy readback differs')
            return {
                'matched': True,
                'clients': [c['clientId'] for c in after['clients']],
            }
        except BaseException:
            failed = []
            for path, previous in reversed(modified):
                try:
                    self.client.put(
                        self.url + path,
                        json=previous,
                    ).raise_for_status()
                except Exception:
                    failed.append(path or '/realm')
            if failed:
                raise RuntimeError(
                    'Session policy rollback requires reconciliation: '
                    + ', '.join(failed),
                ) from None
            raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Inspect or apply the canonical Keycloak session policy',
    )
    parser.add_argument('--server', required=True)
    parser.add_argument('--realm', required=True)
    parser.add_argument('--credentials', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--backup', type=Path)
    args = parser.parse_args()
    url = urlsplit(args.server)
    if url.scheme != 'https' and not (
        url.scheme == 'http'
        and url.hostname in ('localhost', '127.0.0.1', '::1')
    ):
        parser.error('Use HTTPS or a loopback server')
    if args.apply and args.backup is None:
        parser.error('--apply requires a new --backup file')
    credentials = json.loads(args.credentials.read_text())
    username = credentials.get('username') or credentials.get(
        'KC_BOOTSTRAP_ADMIN_USERNAME',
    )
    password = credentials.get('password') or credentials.get(
        'KC_BOOTSTRAP_ADMIN_PASSWORD',
    )
    if not username or not password:
        parser.error('Credentials require username and password')
    with httpx.Client(timeout=20, trust_env=False) as client:
        base = args.server.rstrip('/')
        token = client.post(
            base + '/realms/master/protocol/openid-connect/token',
            data={
                'client_id': 'admin-cli',
                'grant_type': 'password',
                'username': username,
                'password': password,
            },
        )
        token.raise_for_status()
        client.headers['Authorization'] = (
            'Bearer ' + token.json()['access_token']
        )
        deployment = SessionPolicyDeployment(
            client,
            base + '/admin/realms/' + args.realm,
        )
        if args.apply:
            result = deployment.apply(args.backup)
        else:
            drift = deployment.differences(deployment.snapshot())
            result = {'matched': not drift, 'differences': drift}
        print(json.dumps(result))
        if not result['matched']:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
