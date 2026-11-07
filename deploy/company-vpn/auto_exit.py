from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import configure_exit as x
import configure_sso as s

import deploy as d

ROOT = Path(__file__).resolve().parent
PENDING = d.ROOT / 'exit-oidc-transition.json'
RECEIPT = d.ROOT / 'exit-oidc-auto.json'
INSTALLED = d.ROOT / 'oidc-exit-sync'
TIMER = 'company-vpn-oidc-exit.timer'
OBJECTS = [
    ('policies', 'policy_id'),
    ('dns/nameservers', 'dns_id'),
    ('routes', 'route_id'),
]


def account(token):
    accounts = d.api('accounts', token=token)
    if len(accounts) != 1:
        raise RuntimeError('Expected the existing company VPN account.')
    s.validate_settings(accounts[0]['settings'])
    return accounts[0]


def resolve_groups(token):
    """Only groups populated by verified external JWT claims are eligible."""
    groups = d.api('groups', token=token)
    result = {}
    for name in s.GROUPS:
        matches = [g for g in groups if g['name'] == name]
        if not matches:
            # NetBird creates a JWT group at its first member's login. The
            # timer adds its ID to route/DNS/policy when it appears.
            continue
        if len(matches) != 1 or matches[0].get('issued') != 'jwt':
            raise RuntimeError('Expected one native JWT group: ' + name)
        result[name] = matches[0]['id']
    if not result:
        raise RuntimeError(
            (
                'No company OIDC group exists yet; sign in once using the '
                'company account.'
            ),
        )
    return result


def validate_members(token, groups, server_peer_id):
    provider = s.provider_for(token)
    if not provider:
        raise RuntimeError('Company Keycloak connector is missing.')
    users = {u['id']: u for u in d.api('users', token=token)}
    peers = {p['id']: p for p in d.api('peers', token=token)}
    for group_id in groups.values():
        group = d.api('groups/' + group_id, token=token)
        for entry in group.get('peers', []):
            peer_id = entry['id'] if isinstance(entry, dict) else entry
            user = users.get(peers.get(peer_id, {}).get('user_id'))
            if (
                peer_id == server_peer_id
                or not user
                or user.get(
                    'is_service_user',
                )
                or user.get('idp_id') != provider['id']
            ):
                raise RuntimeError(
                    'An OIDC exit group contains a non-company-login device.',
                )


def validate_distribution(token, state):
    current = account(token)
    if (
        current['settings']
        .get('extra', {})
        .get('user_approval_required', True)
    ):
        raise RuntimeError(
            'Company OIDC logins still require a second manual approval.',
        )
    groups = resolve_groups(token)
    if groups != state.get('oidc_distribution_groups'):
        raise RuntimeError(
            'OIDC group IDs changed; run auto_exit.py sync to reconcile them.',
        )
    validate_members(token, groups, state['server_peer_id'])


def transition(token, before, after):
    """Journal route/policy/DNS changes so a killed apply can be resumed."""
    old, new = x.payloads(before, True), x.payloads(after, True)
    for kind, key in OBJECTS:
        current = d.api(kind + '/' + before[key], token=token)
        if x.comparable(current, old[kind]) not in (
            x.comparable(old[kind], old[kind]),
            x.comparable(
                new[kind],
                new[kind],
            ),
        ):
            raise RuntimeError(
                'Managed exit object changed independently: ' + kind,
            )
    d.save(PENDING, {'before': before, 'after': after})
    try:
        for kind, key in OBJECTS:
            current = d.api(kind + '/' + before[key], token=token)
            if x.comparable(current, new[kind]) != x.comparable(
                new[kind],
                new[kind],
            ):
                d.api(kind + '/' + before[key], 'PUT', new[kind], token)
            current = d.api(kind + '/' + before[key], token=token)
            if x.comparable(current, new[kind]) != x.comparable(
                new[kind],
                new[kind],
            ):
                raise RuntimeError(
                    'Exit object update did not persist: ' + kind,
                )
        d.save(x.STATE, after)
        PENDING.unlink()
    except Exception:
        # Restore only our changes, without overwriting concurrent edits.
        for kind, key in reversed(OBJECTS):
            current = d.api(kind + '/' + before[key], token=token)
            if x.comparable(current, new[kind]) == x.comparable(
                new[kind],
                new[kind],
            ):
                d.api(kind + '/' + before[key], 'PUT', old[kind], token)
            elif x.comparable(current, old[kind]) != x.comparable(
                old[kind],
                old[kind],
            ):
                raise RuntimeError(
                    (
                        'Concurrent exit edit prevents rollback; private '
                        'transition retained.'
                    ),
                ) from None
        d.save(x.STATE, before)
        PENDING.unlink()
        raise


def resume(token):
    if PENDING.exists():
        pending = json.loads(PENDING.read_text())
        transition(token, pending['before'], pending['after'])


def approve_pending(token, groups):
    """Approve only pending company identities with verified JWT groups."""
    provider = s.provider_for(token)
    for user in d.api('users', token=token):
        if (
            user.get('pending_approval')
            and user.get('is_blocked')
            and not user.get('is_service_user')
            and user.get('idp_id') == provider['id']
            and set(user.get('auto_groups', [])).intersection(groups.values())
        ):
            d.api('users/' + user['id'] + '/approve', 'POST', token=token)


def sync(token):
    resume(token)
    state = json.loads(x.STATE.read_text())
    if not state.get('complete') or state.get('distribution_mode') != 'oidc':
        raise RuntimeError('Automatic company exit is not enabled.')
    current = account(token)
    if (
        current['settings']
        .get('extra', {})
        .get('user_approval_required', True)
    ):
        raise RuntimeError(
            (
                'Automatic approval settings changed; review before '
                'synchronizing.'
            ),
        )
    groups = resolve_groups(token)
    validate_members(token, groups, state['server_peer_id'])
    if groups != state['oidc_distribution_groups']:
        transition(token, state, {**state, 'oidc_distribution_groups': groups})
    approve_pending(token, groups)
    # Membership and device propagation are performed by NetBird itself.
    # This timer only handles newly appearing JWT group IDs.


def install_timer():
    INSTALLED.mkdir(mode=0o700, exist_ok=True)
    for name in (
        'auto_exit.py',
        'configure_exit.py',
        'configure_sso.py',
        'deploy.py',
        'enroll_b.py',
    ):
        source, target = ROOT / name, INSTALLED / name
        if source.resolve() != target.resolve():
            temporary = target.with_suffix('.tmp')
            shutil.copyfile(source, temporary)
            temporary.chmod(0o600)
            os.replace(temporary, target)
    service = """[Unit]
Description=Reconcile company OIDC VPN exit distribution
After=network-online.target company-vpn.service
Wants=network-online.target
[Service]
Type=oneshot
ExecStart=/usr/bin/python3 /etc/company-vpn/oidc-exit-sync/auto_exit.py sync
UMask=0077
TimeoutStartSec=120
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=/etc/company-vpn
ProtectHome=true
"""
    timer = """[Unit]
Description=Discover new company OIDC groups for VPN exits
[Timer]
OnActiveSec=30s
OnUnitActiveSec=30s
AccuracySec=5s
Unit=company-vpn-oidc-exit.service
[Install]
WantedBy=timers.target
"""
    d.save(Path('/etc/systemd/system/company-vpn-oidc-exit.service'), service)
    d.save(Path('/etc/systemd/system/' + TIMER), timer)
    d.run('systemctl', 'daemon-reload')
    d.run('systemctl', 'enable', '--now', TIMER)
    d.run('systemctl', 'is-active', TIMER)


def enable(token):
    resume(token)
    state = json.loads(x.STATE.read_text())
    if not state.get('complete'):
        raise RuntimeError(
            (
                'Verify the existing B exit before enabling automatic '
                'distribution.'
            ),
        )
    if state.get('distribution_mode') == 'oidc':
        sync(token)
        install_timer()
        x.verify(token, json.loads(x.STATE.read_text()))
        d.save(
            RECEIPT,
            {
                'backup': state['oidc_backup'],
                'phase': 'complete',
                'distribution': 'native-oidc-groups',
                'login_allow_groups': s.GROUPS,
                'auto_apply': True,
            },
        )
        print('Automatic company OIDC exit is already enabled.')
        return
    x.verify(token, state)
    current = account(token)
    groups = resolve_groups(token)
    validate_members(token, groups, state['server_peer_id'])
    receipt = json.loads(RECEIPT.read_text()) if RECEIPT.exists() else {}
    if receipt.get('phase') == 'prepared':
        backup = Path(receipt['backup'])
        original = json.loads(backup.read_text())
        if original['state'] != state:
            raise RuntimeError(
                'Exit state changed during a pending automatic setup.',
            )
    else:
        backup = d.ROOT / ('exit-oidc-backup-' + str(time.time_ns()) + '.json')
        original = {'state': state, 'settings': current['settings']}
        d.save(backup, original)
        d.save(RECEIPT, {'backup': str(backup), 'phase': 'prepared'})
    settings = copy.deepcopy(current['settings'])
    settings['extra'] = {
        **settings.get('extra', {}),
        'user_approval_required': False,
    }
    d.api('accounts/' + current['id'], 'PUT', {'settings': settings}, token)
    try:
        after = {
            **state,
            'distribution_mode': 'oidc',
            'oidc_distribution_groups': groups,
            'oidc_backup': str(backup),
        }
        transition(token, state, after)
        approve_pending(token, groups)
        x.verify(token, after)
        install_timer()
    except Exception:
        subprocess.run(
            [
                'systemctl',
                'disable',
                '--now',
                TIMER,
            ],
            capture_output=True,
        )
        if json.loads(x.STATE.read_text()).get('distribution_mode') == 'oidc':
            transition(token, json.loads(x.STATE.read_text()), state)
        latest = account(token)
        latest_settings = copy.deepcopy(latest['settings'])
        latest_settings.setdefault('extra', {})['user_approval_required'] = (
            original['settings']
            .get(
                'extra',
                {},
            )
            .get('user_approval_required', True)
        )
        d.api(
            'accounts/' + latest['id'],
            'PUT',
            {'settings': latest_settings},
            token,
        )
        raise
    d.save(
        RECEIPT,
        {
            'backup': str(backup),
            'phase': 'complete',
            'distribution': 'native-oidc-groups',
            'login_allow_groups': s.GROUPS,
            'auto_apply': True,
        },
    )
    print(
        (
            'Applied: company OIDC logins automatically receive the B exit '
            'and VPN '
            'DNS.'
        ),
    )
    print(
        (
            'Existing and new employee devices require no manual device group '
            'assignment.'
        ),
    )
    print(
        (
            'Reconnect the iPhone; select company-internet-b if it is not '
            'already selected.'
        ),
    )
    print('Private backup: ' + str(backup))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('enable', 'sync', 'verify'))
    args = parser.parse_args()
    if os.geteuid() != 0:
        raise SystemExit('Run with sudo on B.')
    os.umask(0o077)
    with (d.ROOT / 'client-enrollment.lock').open('a') as lock:
        flags = (
            fcntl.LOCK_EX
            if args.action == 'sync'
            else fcntl.LOCK_EX | fcntl.LOCK_NB
        )
        fcntl.flock(lock, flags)
        token = s.get_token()
        if args.action == 'enable':
            enable(token)
        elif args.action == 'sync':
            sync(token)
        else:
            sync(token)
            x.verify(token, json.loads(x.STATE.read_text()))
            d.run('systemctl', 'is-active', TIMER)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, ValueError) as error:
        raise SystemExit(str(error)) from None
