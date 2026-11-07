from __future__ import annotations

import argparse
import fcntl
import hashlib
import ipaddress
import json
import os
import platform
import shutil
import socket
import subprocess
import tarfile
import time
import urllib.parse
from pathlib import Path

from configure_sso import get_token

import deploy as d

VERSION = '0.80.0'
ARCHIVE = 'netbird_0.80.0_linux_amd64.tar.gz'
ARCHIVE_SHA = (
    '47ffaba4fc3929f31795bd6c5232d6c29744d3169e2c93e6d9c84624f0ef6405'
)
BINARY_SHA = '777ab46ceb0ca1a3e8336e336b076a37b34255eee8c082ccc071ad954a28d2a2'
BINARY = Path('/opt/company-vpn-client/0.80.0/netbird')
LINK = Path('/usr/local/bin/netbird')
UNIT = Path('/etc/systemd/system/netbird.service')
STATE = d.ROOT / 'client-enrollment.json'
CLIENT = d.ROOT / 'client'
CONFIG = CLIENT / 'config.json'
PEER_NAME = 'company-vpn-b'
FLAGS = [
    '--disable-dns',
    '--disable-client-routes',
    '--disable-server-routes',
    '--allow-server-ssh=false',
    '--allow-remote-jobs=false',
    '--disable-ipv6',
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def isolation(token):
    accounts = d.api('accounts', token=token)
    if (
        len(accounts) != 1
        or accounts[0]['settings']['network_range'] != d.VPN_RANGE
    ):
        raise RuntimeError(
            'Expected the existing isolated company VPN account.',
        )
    policies = d.api('policies', token=token)
    routes = d.api('routes', token=token)
    networks = d.api('networks', token=token)
    if any(p['enabled'] for p in policies) or routes or networks:
        raise RuntimeError(
            (
                'Access policies or resource routes exist; review them before '
                'isolated enrollment.'
            ),
        )
    # Group peer counts in GET policy responses change when a device joins.
    # Compare rule configuration by group ID rather than those computed counts.
    for policy in policies:
        for rule in policy.get('rules', []):
            for side in ('sources', 'destinations'):
                rule[side] = [
                    g['id'] if isinstance(g, dict) else g
                    for g in rule.get(side, [])
                ]
    return {'policies': policies, 'routes': routes, 'networks': networks}


def client(*args, key=None, timeout=60):
    # The one-use enrollment secret is passed through a private file, not argv.
    command = [
        str(BINARY),
        '--config',
        str(CONFIG),
        '--log-file',
        'console',
        *args,
    ]
    keyfile = CLIENT / 'setup-key'
    if key:
        d.save(keyfile, key)
        command += ['--setup-key-file', str(keyfile)]
    try:
        return d.run(*command, timeout=timeout)
    finally:
        keyfile.unlink(missing_ok=True)


def status():
    try:
        return json.loads(client('status', '--json', timeout=20))
    except (RuntimeError, ValueError):
        return {}


def connected(value):
    url = urllib.parse.urlsplit(value.get('management', {}).get('url', ''))
    expected = urllib.parse.urlsplit(d.ORIGIN)
    return (
        value.get('management', {}).get('connected')
        and value.get('netbirdIp')
        and url.scheme == expected.scheme
        and url.hostname == expected.hostname
        and (url.port or (443 if url.scheme == 'https' else 80))
        == (expected.port or (443 if expected.scheme == 'https' else 80))
    )


def peer_for(value, token):
    address = str(ipaddress.ip_interface(value['netbirdIp']).ip)
    if ipaddress.ip_address(address) not in ipaddress.ip_network(d.VPN_RANGE):
        raise RuntimeError('B received an unexpected VPN address.')
    peers = [p for p in d.api('peers', token=token) if p['ip'] == address]
    if len(peers) != 1 or peers[0]['name'] != PEER_NAME or peers[0]['user_id']:
        raise RuntimeError(
            (
                'The server does not identify this device as B enrolled by '
                'setup key.'
            ),
        )
    if peers[0]['ssh_enabled'] or peers[0].get('approval_required'):
        raise RuntimeError('Unexpected peer SSH or approval settings.')
    return peers[0]


def revoke(state, token):
    if state.get('setup_key_id'):
        key = d.api('setup-keys/' + str(state['setup_key_id']), token=token)
        if not key['revoked']:
            d.api(
                'setup-keys/' + str(key['id']),
                'PUT',
                {'revoked': True, 'auto_groups': key['auto_groups']},
                token,
            )
    state.pop('setup_key', None)
    d.save(STATE, state)


def verify(token, state):
    exit_receipt = d.ROOT / 'exit-node.json'
    if exit_receipt.exists():
        exit_state = json.loads(exit_receipt.read_text())
        if exit_state.get('complete'):
            import configure_exit

            configure_exit.verify(token, exit_state)
            return
        raise RuntimeError(
            (
                'Exit configuration is staged or disabled; use '
                'configure_exit.py '
                'apply/verify.'
            ),
        )
    if (
        not BINARY.is_file()
        or digest(BINARY) != BINARY_SHA
        or LINK.resolve() != BINARY
    ):
        raise RuntimeError(
            'Installed client binary differs from the pinned release.',
        )
    value = status()
    if not connected(value) or not value.get('signal', {}).get('connected'):
        raise RuntimeError(
            (
                'B VPN client is not connected; inspect journalctl -u netbird '
                'locally.'
            ),
        )
    peer = peer_for(value, token)
    if state.get('peer_id') and state['peer_id'] != peer['id']:
        raise RuntimeError('B peer identity changed.')
    config = json.loads(CONFIG.read_text())
    for field in (
        'DisableDNS',
        'DisableClientRoutes',
        'DisableServerRoutes',
        'DisableIPv6',
    ):
        if config.get(field) is not True:
            raise RuntimeError('Expected isolated client setting: ' + field)
    if config.get('ServerSSHAllowed') or config.get('RemoteJobsAllowed'):
        raise RuntimeError('Unexpected client SSH or remote job setting.')
    if config.get('DisableAutoConnect'):
        raise RuntimeError('Client automatic reconnection is disabled.')
    isolation(token)
    for service in ('netbird', 'tailscaled', 'nginx', 'company-vpn'):
        if d.run('systemctl', 'is-active', service).strip() != 'active':
            raise RuntimeError('Service is not active: ' + service)
    if d.run('systemctl', 'is-enabled', 'netbird').strip() != 'enabled':
        raise RuntimeError('B VPN client is not enabled at boot.')
    for host in (
        'changdar-server.mooo.com',
        'ai.changdar-server.mooo.com',
        'wiki.changdar-server.mooo.com',
    ):
        d.https_check(host)
    revoke(state, token)
    state.update(peer_id=peer['id'], ip=peer['ip'], complete=True)
    d.save(STATE, state)
    print('B VPN client connected: ' + peer['ip'] + '; autostart verified.')
    print(
        (
            'One-use setup key revoked. Resource policies and office routing '
            'remain disabled.'
        ),
    )
    print('Tailscale and the three existing HTTPS websites are healthy.')


def install(token):
    state = json.loads(STATE.read_text()) if STATE.exists() else None
    if state and state.get('complete'):
        verify(token, state)
        return
    isolation(token)
    if platform.machine() != 'x86_64':
        raise RuntimeError('This bundle is for B Linux x86_64 only.')
    addresses = json.loads(d.run('ip', '-j', '-4', 'address', 'show'))
    if not any(a['local'] == d.LAN for i in addresses for a in i['addr_info']):
        raise RuntimeError('Run this command on B (192.168.1.86).')
    if not state:
        if (
            shutil.which('netbird')
            or UNIT.exists()
            or Path(
                '/etc/netbird',
            ).exists()
            or Path('/var/run/netbird.sock').exists()
            or BINARY.exists()
        ):
            raise RuntimeError(
                (
                    'An unmanaged NetBird client already exists; nothing was '
                    'overwritten.'
                ),
            )
        vpn = ipaddress.ip_network(d.VPN_RANGE)
        if any(vpn.overlaps(n) for n in d.local_networks()):
            raise RuntimeError(
                'VPN address range overlaps a local route or Docker network.',
            )
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.bind(('0.0.0.0', 51820))
        d.run('systemctl', 'is-active', 'tailscaled')
        d.https_check(d.HOST)
        state = {
            'version': VERSION,
            'started_at': time.time(),
            'complete': False,
        }
        d.save(STATE, state)
    package = Path(__file__).resolve().parent / ARCHIVE
    if digest(package) != ARCHIVE_SHA:
        raise RuntimeError(
            'Client archive checksum differs from the official release.',
        )
    CLIENT.mkdir(mode=0o700, exist_ok=True)
    if not BINARY.exists():
        with tarfile.open(package) as archive:
            member = archive.getmember('netbird')
            if not member.isfile():
                raise RuntimeError(
                    'Client archive binary is not a regular file.',
                )
            data = archive.extractfile(member).read()
        if hashlib.sha256(data).hexdigest() != BINARY_SHA:
            raise RuntimeError('Client binary checksum differs.')
        BINARY.parent.mkdir(parents=True, exist_ok=True)
        temporary = BINARY.with_suffix('.tmp')
        temporary.write_bytes(data)
        temporary.chmod(0o755)
        temporary.replace(BINARY)
    if digest(BINARY) != BINARY_SHA:
        raise RuntimeError(
            'Existing client binary differs; nothing was overwritten.',
        )
    if LINK.is_symlink() and LINK.resolve() == BINARY:
        pass
    elif LINK.exists() or LINK.is_symlink():
        raise RuntimeError(
            'Client command path already exists; nothing was overwritten.',
        )
    else:
        LINK.symlink_to(BINARY)
    unit_text = (
        '# Managed by company-vpn/enroll_b.py\n[Unit]\nDesc'
        'ription=Company NetBird VPN '
        'client\nWants=network-online.target\nAfter=network'
        '-online.target company-vpn.service '
        'nginx.service\n[Service]\nType=simple\nExecStart='
        f"{BINARY}"
        ' service run --config '
        f"{CONFIG}"
        ' --management-url '
        f"{d.ORIGIN}"
        ' --log-file console\nRestart=on-failure\nRestartSe'
        'c=5\nUMask=0077\n[Install]\nWantedBy=multi-user.ta'
        'rget\n'
    )
    if UNIT.exists() and UNIT.read_text() != unit_text:
        raise RuntimeError('NetBird unit differs; nothing was overwritten.')
    d.save(UNIT, unit_text)
    d.run('systemctl', 'daemon-reload')
    d.run('systemctl', 'enable', '--now', 'netbird')
    print(
        (
            'Pinned NetBird client installed; registering B as an isolated '
            'server peer.'
        ),
        flush=True,
    )
    for _ in range(30):
        if Path('/var/run/netbird.sock').exists():
            break
        time.sleep(1)
    # Existing local keys make interrupted/repeated enrollment resumable.
    if not connected(status()):
        if state.get('setup_key_id'):
            previous = d.api(
                'setup-keys/' + str(state['setup_key_id']),
                token=token,
            )
            if not previous['valid']:
                revoke(state, token)
        if not state.get('setup_key'):
            key = d.api(
                'setup-keys',
                'POST',
                {
                    'name': 'B one-use enrollment',
                    'type': 'one-off',
                    'expires_in': 86400,
                    'usage_limit': 1,
                    'auto_groups': [],
                    'ephemeral': False,
                    'allow_extra_dns_labels': False,
                },
                token,
            )
            state.update(setup_key_id=key['id'], setup_key=key['key'])
            d.save(STATE, state)
        try:
            client(
                'up',
                '--management-url',
                d.ORIGIN,
                '--hostname',
                PEER_NAME,
                *FLAGS,
                key=state['setup_key'],
                timeout=120,
            )
        except (RuntimeError, subprocess.TimeoutExpired):
            revoke(state, token)
            raise RuntimeError(
                (
                    'B registration did not finish; its one-use key was '
                    'revoked. Inspect /etc/company-vpn/private-error.log '
                    'locally and '
                    'rerun.'
                ),
            ) from None
    revoke(state, token)
    for _ in range(30):
        value = status()
        if connected(value) and value.get('signal', {}).get('connected'):
            break
        time.sleep(2)
    verify(token, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['install', 'verify'])
    action = parser.parse_args().action
    if os.geteuid() != 0:
        raise RuntimeError('Run on B using sudo.')
    os.umask(0o077)
    if not (d.ROOT / 'sso-ready.json').exists():
        raise RuntimeError(
            'The existing company VPN SSO setup must be completed first.',
        )
    with (d.ROOT / 'client-enrollment.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        token = get_token()
        if action == 'install':
            install(token)
        else:
            verify(token, json.loads(STATE.read_text()))


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, ValueError) as error:
        raise SystemExit(str(error)) from None
