from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import json
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from contextlib import closing
from pathlib import Path
from urllib.parse import parse_qs
from urllib.parse import urlencode
from urllib.parse import urlsplit

import deploy as d

CLIENT = 'company-vpn'
ISSUER = 'https://changdar-server.mooo.com/keycloak/realms/visionnaire'
GROUPS = ['vpn-admins', 'vpn-employees']
NAME = '公司帳號（Keycloak）'
ROOT = d.ROOT


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def account_domain_ready(database, account_id):
    (
        "Validate the pinned server's initial single-account metadata, "
        'read-only.'
    )
    with closing(
        sqlite3.connect(database.as_uri() + '?mode=ro', uri=True),
    ) as db:
        rows = db.execute(
            (
                'SELECT id, domain, domain_category, '
                'is_domain_primary_account FROM '
                'accounts'
            ),
        ).fetchall()
        if len(rows) != 1 or rows[0][0] != account_id:
            raise RuntimeError(
                (
                    'Expected the one original VPN account; refusing account '
                    'metadata repair.'
                ),
            )
        _, domain, category, primary = rows[0]
        if (
            domain == 'netbird.selfhosted'
            and category == 'private'
            and primary
        ):
            return True
        if domain or category not in ('', 'public', None):
            raise RuntimeError(
                'Unexpected VPN domain metadata; review before changing it.',
            )
        users = db.execute('SELECT id, role FROM users').fetchall()
        owner = json.loads((ROOT / 'bootstrap.json').read_text())
        if len(users) != 1 or users[0] != (owner['user_id'], 'owner'):
            raise RuntimeError(
                (
                    'VPN users already exist; refusing initial account '
                    'metadata repair.'
                ),
            )
        if db.execute('SELECT count(*) FROM peers').fetchone()[0]:
            raise RuntimeError(
                (
                    'VPN devices already exist; refusing initial account '
                    'metadata repair.'
                ),
            )
        return False


def repair_account_domain(database, account_id):
    """Repair only the fresh account created by setup-time PAT provisioning.

    v0.80.0 setup creates the owner account before single-account domain claims
    are supplied. Stop this VPN server first. Preserve all users, policies,
    network ranges and encrypted values; change only three account columns.
    """
    if account_domain_ready(database, account_id):
        return
    backup = ROOT / ('account-domain-backup-' + str(time.time_ns()) + '.db')
    with closing(sqlite3.connect(database)) as db:
        with closing(sqlite3.connect(backup)) as target:
            db.backup(target)
        backup.chmod(0o600)
        db.execute('BEGIN IMMEDIATE')
        count = db.execute(
            (
                'UPDATE accounts SET domain = ?, domain_category = ?, '
                'is_domain_primary_account = ? WHERE id = ? AND (domain IS '
                'NULL OR domain = '
                "'')"
            ),
            ('netbird.selfhosted', 'private', 1, account_id),
        ).rowcount
        if count != 1:
            raise RuntimeError(
                (
                    'VPN account metadata changed concurrently; no repair '
                    'committed.'
                ),
            )
        db.commit()
    if not account_domain_ready(database, account_id):
        raise RuntimeError('VPN account metadata repair verification failed.')


def ensure_account_domain(token):
    account_id = d.api('accounts', token=token)[0]['id']
    container = d.compose('ps', '-q', 'server').strip()
    inspected = json.loads(d.run('docker', 'inspect', container))[0]
    # This repair is tied to the server version that created the account.
    if inspected['Config']['Image'] != d.SERVER:
        raise RuntimeError(
            'VPN server image changed; review the metadata migration first.',
        )
    mounts = [
        m
        for m in inspected['Mounts']
        if m['Destination'] == '/var/lib/netbird' and m['Type'] == 'volume'
    ]
    if len(mounts) != 1:
        raise RuntimeError('Unexpected VPN database mount.')
    database = Path(mounts[0]['Source']) / 'store.db'
    if not database.is_file():
        raise RuntimeError('VPN SQLite store is missing.')
    if not account_domain_ready(database, account_id):
        d.compose('stop', 'server')
        try:
            repair_account_domain(database, account_id)
        finally:
            d.compose('up', '-d', 'server', timeout=180)
        d.wait_api()
    d.save(
        ROOT / 'account-domain-ready.json',
        {'account_id': account_id, 'domain': 'netbird.selfhosted'},
    )
    print(
        (
            'Verified: the original owner account is the single shared VPN '
            'account.'
        ),
    )


def load_bundle(source=None):
    target = ROOT / 'keycloak-sso.json'
    path = source if source and source.exists() else target
    bundle = json.loads(path.read_text())
    if (
        bundle.get(
            'marker',
        )
        != 'company-vpn-v1'
        or bundle['client_id'] != CLIENT
        or bundle['issuer'] != ISSUER
        or bundle['redirect_uri'] != d.ORIGIN + '/oauth2/callback'
        or bundle['allowed_groups'] != GROUPS
        or not bundle['client_secret']
    ):
        raise RuntimeError('The VPN SSO bundle identity differs.')
    return bundle


def get_token():
    value = json.loads(
        (ROOT / 'bootstrap.json').read_text(),
    ).get('personal_access_token')
    if not value:
        raise RuntimeError(
            'Bootstrap PAT is missing; obtain a new owner PAT privately.',
        )
    return value


def pinned_database(name):
    container = d.compose('ps', '-q', 'server').strip()
    inspected = json.loads(d.run('docker', 'inspect', container))[0]
    if inspected['Config']['Image'] != d.SERVER:
        raise RuntimeError(
            'VPN server image changed; review the connector migration first.',
        )
    mounts = [
        m
        for m in inspected['Mounts']
        if m['Destination'] == '/var/lib/netbird' and m['Type'] == 'volume'
    ]
    if len(mounts) != 1:
        raise RuntimeError('Unexpected VPN database mount.')
    database = Path(mounts[0]['Source']) / name
    if not database.is_file():
        raise RuntimeError('VPN database is missing.')
    return database


def connector_config(db, provider, bundle):
    row = db.execute(
        'SELECT type, name, config FROM connector WHERE id = ?',
        (provider['id'],),
    ).fetchone()
    if row is None or row[:2] != ('oidc', NAME):
        raise RuntimeError('Unexpected VPN connector identity.')
    config = json.loads(row[2])
    expected = {
        'issuer': ISSUER,
        'clientID': CLIENT,
        'clientSecret': bundle['client_secret'],
        'redirectURI': d.ORIGIN + '/oauth2/callback',
        'insecureEnableGroups': True,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise RuntimeError('Stored VPN connector configuration differs.')
    if config.get('scopes') not in (
        ['openid', 'profile', 'email'],
        ['openid', 'profile'],
    ):
        raise RuntimeError(
            'Unexpected VPN connector scopes; review before changing them.',
        )
    if config.get('userIDKey', 'sub') != 'sub':
        raise RuntimeError(
            'VPN must identify users by the immutable Keycloak subject.',
        )
    if config.get('userNameKey', 'name') not in ('name', 'preferred_username'):
        raise RuntimeError(
            'Unexpected VPN display-name claim; review before changing it.',
        )
    return row[2], config


def patch_email_optional(database, provider, bundle):
    (
        'Run with this VPN server stopped; make optional profile fields '
        "optional.\n\n    NetBird 0.80.0's connector API doesn't expose "
        'scopes. Its pinned Dex version\n    permits a missing email claim '
        "when the upstream email scope isn't requested.\n    Keycloak "
        'usernames supply the display name for accounts without full names.\n '
        '   Preserve connector ID, subject, secrets, group handling and all '
        'other data.\n    '
    )
    with closing(sqlite3.connect(database)) as db:
        old, config = connector_config(db, provider, bundle)
        if (
            'email' not in config['scopes']
            and config.get('userNameKey') == 'preferred_username'
        ):
            return None
        backup = ROOT / (
            'email-optional-backup-' + str(time.time_ns()) + '.db'
        )
        with closing(sqlite3.connect(backup)) as target:
            db.backup(target)
        backup.chmod(0o600)
        db.execute('BEGIN IMMEDIATE')
        current, _ = connector_config(db, provider, bundle)
        if current != old:
            raise RuntimeError(
                'VPN connector changed concurrently; no change committed.',
            )
        config['scopes'] = ['openid', 'profile']
        config['userNameKey'] = 'preferred_username'
        new = json.dumps(config, separators=(',', ':'))
        if isinstance(old, bytes):
            new = new.encode()
        count = db.execute(
            'UPDATE connector SET config = ? WHERE id = ? AND config = ?',
            (new, provider['id'], old),
        ).rowcount
        if count != 1:
            raise RuntimeError(
                'VPN connector scope update failed; no change committed.',
            )
        db.commit()
        return backup


def email_optional():
    token = get_token()
    bundle = load_bundle()
    provider, _ = verify(token)
    database = pinned_database('idp.db')
    with closing(
        sqlite3.connect(database.as_uri() + '?mode=ro', uri=True),
    ) as db:
        _, config = connector_config(db, provider, bundle)
    backup = None
    if (
        'email' in config['scopes']
        or config.get('userNameKey') != 'preferred_username'
    ):
        d.compose('stop', 'server')
        try:
            backup = patch_email_optional(database, provider, bundle)
        finally:
            d.compose('up', '-d', 'server', timeout=180)
        d.wait_api()
    with closing(
        sqlite3.connect(database.as_uri() + '?mode=ro', uri=True),
    ) as db:
        _, config = connector_config(db, provider, bundle)
        if (
            config['scopes'] != ['openid', 'profile']
            or config.get('userNameKey') != 'preferred_username'
        ):
            raise RuntimeError(
                'Optional-profile connector verification failed.',
            )
    verify(token, require_email_optional=True)
    d.save(
        ROOT / 'email-optional.json',
        {
            'provider_id': provider['id'],
            'upstream_scopes': config['scopes'],
            'display_name_claim': config['userNameKey'],
            'backup': str(backup) if backup else None,
        },
    )
    print(
        (
            'Applied: Keycloak email and full name are optional; username is '
            'the display '
            'name.'
        ),
    )
    print(
        (
            'Immutable Keycloak user IDs, signature validation and VPN group '
            'restrictions are '
            'retained.'
        ),
    )
    if backup:
        print('Private IdP backup:', backup)
    print(
        (
            'Start a new login from the VPN homepage; do not reload an old '
            'callback '
            'URL.'
        ),
    )


def provider_for(token):
    providers = d.api('identity-providers', token=token)
    found = [p for p in providers if p['client_id'] == CLIENT]
    if len(found) > 1:
        raise RuntimeError('Multiple VPN connectors use the same client.')
    if found and (
        found[0]['issuer'] != ISSUER
        or found[0]['name'] != NAME
        or found[0]['type'] != 'keycloak'
    ):
        raise RuntimeError('Refusing to change an unmanaged VPN connector.')
    return found[0] if found else None


def validate_settings(settings):
    expected = {
        'network_range': d.VPN_RANGE,
        'jwt_groups_enabled': True,
        'jwt_groups_claim_name': 'groups',
        'jwt_allow_groups': GROUPS,
        'groups_propagation_enabled': True,
        'peer_expose_enabled': False,
    }
    for key, value in expected.items():
        if settings.get(key) != value:
            raise RuntimeError('VPN settings verification failed: ' + key)


def verify(token, require_email_optional=False):
    provider = provider_for(token)
    if not provider:
        raise RuntimeError('VPN Keycloak connector is missing.')
    accounts = d.api('accounts', token=token)
    if len(accounts) != 1:
        raise RuntimeError('Expected one VPN account.')
    validate_settings(accounts[0]['settings'])
    # Exercise the actual embedded IdP redirect without a password or session.
    query = urlencode(
        {
            'client_id': 'netbird-dashboard',
            'redirect_uri': d.ORIGIN + '/nb-auth',
            'response_type': 'code',
            'scope': 'openid profile email groups',
            'code_challenge': base64.urlsafe_b64encode(
                hashlib.sha256(os.urandom(32)).digest(),
            )
            .decode()
            .rstrip('='),
            'code_challenge_method': 'S256',
            'state': 'company-vpn-configuration-check',
        },
    )
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        NoRedirect(),
    )
    url = d.ORIGIN + '/oauth2/auth/' + provider['id'] + '?' + query
    try:
        response = opener.open(url, timeout=20)
        response.close()
        raise RuntimeError(
            'Expected the VPN connector to redirect to Keycloak.',
        )
    except urllib.error.HTTPError as response:
        if response.code not in (302, 303):
            raise RuntimeError(
                'VPN authorization redirect failed: HTTP '
                + str(response.code),
            ) from None
        location = urlsplit(response.headers.get('Location', ''))
        parameters = parse_qs(location.query)
        expected = urlsplit(ISSUER + '/protocol/openid-connect/auth')
        if (location.scheme, location.netloc, location.path) != (
            expected.scheme,
            expected.netloc,
            expected.path,
        ):
            raise RuntimeError(
                'VPN authorization went to an unexpected identity provider.',
            )
        if parameters.get('client_id') != [CLIENT] or parameters.get(
            'redirect_uri',
        ) != [d.ORIGIN + '/oauth2/callback']:
            raise RuntimeError('VPN authorization client or callback differs.')
        if require_email_optional:
            scopes = parameters.get('scope', [''])[0].split()
            if 'email' in scopes or not {'openid', 'profile'}.issubset(scopes):
                raise RuntimeError(
                    'VPN still requests the upstream email scope.',
                )
        response.close()
    print(
        (
            'Verified: dedicated Keycloak connector, exact callback and VPN '
            'group allowlist.'
        ),
    )
    print(
        (
            'Employee login and device/routing connectivity still require an '
            'end-to-end '
            'test.'
        ),
    )
    return provider, accounts[0]


def apply(source):
    bundle = load_bundle(source)
    token = get_token()
    if (
        not (ROOT / 'installed.json').exists()
        or d.wait_api()['setup_required']
    ):
        raise RuntimeError('Install the NetBird control plane first.')
    ensure_account_domain(token)
    accounts = d.api('accounts', token=token)
    if len(accounts) != 1:
        raise RuntimeError('Expected one existing VPN account.')
    account = accounts[0]
    if account['settings']['network_range'] != d.VPN_RANGE:
        raise RuntimeError(
            'VPN network range changed; review before applying SSO.',
        )
    provider = provider_for(token)
    previous = ROOT / ('sso-backup-' + str(time.time_ns()) + '.json')
    d.save(
        previous,
        {
            'account': account,
            'providers': d.api('identity-providers', token=token),
            'policies': d.api('policies', token=token),
        },
    )
    # Set the allowlist before enabling an external login connector.
    settings = dict(account['settings'])
    settings.update(
        {
            'jwt_groups_enabled': True,
            'jwt_groups_claim_name': 'groups',
            'jwt_allow_groups': GROUPS,
            'groups_propagation_enabled': True,
            'peer_expose_enabled': False,
        },
    )
    d.api('accounts/' + account['id'], 'PUT', {'settings': settings}, token)
    validate_settings(d.api('accounts', token=token)[0]['settings'])
    desired = {
        'type': 'keycloak',
        'name': NAME,
        'issuer': ISSUER,
        'client_id': CLIENT,
        'client_secret': bundle['client_secret'],
    }
    if provider:
        d.api('identity-providers/' + provider['id'], 'PUT', desired, token)
    else:
        d.api('identity-providers', 'POST', desired, token)
    d.save(ROOT / 'keycloak-sso.json', bundle)
    provider, account = verify(token)
    d.save(
        ROOT / 'sso-ready.json',
        {
            'provider_id': provider['id'],
            'client_id': CLIENT,
            'issuer': ISSUER,
            'account_id': account['id'],
            'allowed_groups': GROUPS,
            'employee_login_verified': False,
        },
    )
    if (
        source
        and source.exists()
        and source.resolve() != (ROOT / 'keycloak-sso.json').resolve()
    ):
        source.unlink()
    print(
        (
            'VPN SSO applied. Sign in using 公司帳號（Keycloak） and your existing '
            'company account.'
        ),
    )
    print('No office routes or new peer access policies were enabled.')


def promote_operator():
    token = get_token()
    bundle = load_bundle()
    provider, _ = verify(token)
    # Dex subjects contain the immutable provider user ID in base64 protobuf.
    # and the connector ID. Never promote by email or display name alone.
    user_id = bundle['operator']['id'].encode()
    connector_id = provider['id'].encode()
    if len(user_id) >= 128 or len(connector_id) >= 128:
        raise RuntimeError('Unexpected provider identifier length.')
    subject = (
        base64.b64encode(
            b'\x0a'
            + bytes([len(user_id)])
            + user_id
            + b'\x12'
            + bytes([len(connector_id)])
            + connector_id,
        )
        .decode()
        .rstrip('=')
    )
    users = d.api('users', token=token)
    found = [u for u in users if u['id'] == subject]
    if len(found) != 1:
        raise RuntimeError(
            (
                'The expected ChangDar operator has not logged in through '
                'this connector '
                'yet.'
            ),
        )
    user = found[0]
    if user['is_blocked']:
        if not user.get('pending_approval'):
            raise RuntimeError(
                'Operator is blocked; review before changing the role.',
            )
        d.api('users/' + subject + '/approve', 'POST', token=token)
        user = next(
            u for u in d.api('users', token=token) if u['id'] == subject
        )
        if user['is_blocked']:
            raise RuntimeError('Operator approval did not persist.')
    if user['role'] not in ('admin', 'owner'):
        d.save(
            ROOT / ('operator-role-backup-' + str(time.time_ns()) + '.json'),
            user,
        )
        d.api(
            'users/' + subject,
            'PUT',
            {
                'role': 'admin',
                'auto_groups': user['auto_groups'],
                'is_blocked': False,
            },
            token,
        )
    current = next(
        u
        for u in d.api(
            'users',
            token=token,
        )
        if u['id'] == subject
    )
    if current['role'] not in ('admin', 'owner'):
        raise RuntimeError('Operator role verification failed.')
    print(
        (
            'Verified: the existing ChangDar Keycloak identity is a NetBird '
            'administrator.'
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'action',
        choices=(
            'apply',
            'verify',
            'promote-admin',
            'email-optional',
            'profile-optional',
        ),
        nargs='?',
        default='apply',
    )
    parser.add_argument(
        '--bundle',
        type=Path,
        default=Path('/home/changdar02/company-vpn/keycloak-sso.json'),
    )
    args = parser.parse_args()
    if (
        os.geteuid() != 0
        or ROOT.is_symlink()
        or not ROOT.exists()
        or ROOT.stat().st_uid != 0
    ):
        raise SystemExit(
            'Run with sudo on B after installing the control plane.',
        )
    os.umask(0o077)
    with (ROOT / 'deploy.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == 'apply':
            apply(args.bundle)
        elif args.action == 'promote-admin':
            promote_operator()
        elif args.action in ('email-optional', 'profile-optional'):
            email_optional()
        else:
            verify(
                get_token(),
                require_email_optional=(ROOT / 'email-optional.json').exists(),
            )


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('Stopped:', str(error), file=sys.stderr)
        raise SystemExit(1)
