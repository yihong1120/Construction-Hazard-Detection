from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import ipaddress
import json
import os
import socket
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path('/etc/company-vpn')
INSTALLED = Path('/usr/local/lib/company-vpn/boot_guard.py')
RECEIPT = ROOT / 'boot-guard.json'
RUNTIME = Path('/run/company-vpn-boot-guard')
IMAGE = (
    'netbirdio/netbird-server@sha256:05b3d8d6d056e5062a2965d4d6e5d6d83f09f2bf9'
    '133947a8e5972e0e0e7a261'
)
SUBNET = ipaddress.ip_network('192.168.1.0/24')
COMPOSE = [
    'docker',
    'compose',
    '--project-name',
    'company-vpn',
    '--file',
    str(ROOT / 'compose.json'),
]
DROPIN = """# Managed by company-vpn/boot_guard.py
[Unit]
Wants=network-online.target NetworkManager-wait-online.service
After=network-online.target NetworkManager-wait-online.service docker.service
StartLimitIntervalSec=0

[Service]
ExecStart=
ExecStart=/usr/bin/python3 /usr/local/lib/company-vpn/boot_guard.py ensure
Restart=on-failure
RestartSec=30s
TimeoutStartSec=180s
"""
CHECK_UNIT = """# Managed by company-vpn/boot_guard.py
[Unit]
Description=Check and recover company VPN network and local API
After=docker.service company-vpn.service
Requires=docker.service

[Service]
Type=oneshot
ExecStart=/usr/bin/python3 /usr/local/lib/company-vpn/boot_guard.py ensure
TimeoutStartSec=180s
UMask=0077
"""
TIMER = """# Managed by company-vpn/boot_guard.py
[Unit]
Description=Check company VPN once a minute

[Timer]
OnBootSec=2min
OnUnitInactiveSec=60s
RandomizedDelaySec=10s
Unit=company-vpn-healthcheck.service

[Install]
WantedBy=timers.target
"""


def call(*args, timeout=90):
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode:
        save(ROOT / 'boot-guard-error.log', result.stderr or result.stdout)
        raise RuntimeError(
            args[0] + ' failed; see /etc/company-vpn/boot-guard-error.log',
        )
    return result.stdout


def save(path, contents, mode=0o600):
    if path.is_symlink():
        raise RuntimeError('Refusing a symlink: ' + str(path))
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load(path):
    if (
        path.is_symlink()
        or path.stat().st_uid != 0
        or path.stat().st_mode & 0o022
    ):
        raise RuntimeError(
            'Expected root-owned protected configuration: ' + str(path),
        )
    return json.loads(path.read_text())


def config_for_address(config, address):
    parsed = ipaddress.ip_address(address)
    if parsed not in SUBNET or parsed in (
        SUBNET.network_address,
        SUBNET.broadcast_address,
        ipaddress.ip_address('192.168.1.1'),
    ):
        raise RuntimeError('Address is outside the approved B LAN.')
    result = copy.deepcopy(config)
    server = result['services']['server']
    if (
        server.get('image') != IMAGE
        or server.get('network_mode')
        or server.get('volumes')
        != [
            'data:/var/lib/netbird',
            './config.yaml:/etc/netbird/config.yaml:ro',
        ]
        or result.get('volumes', {}).get('data') != {}
    ):
        raise RuntimeError(
            'Unexpected image, network mode or persistent mounts.',
        )
    ports = server.get('ports', [])
    if len(ports) != 2 or '127.0.0.1:19081:80' not in ports:
        raise RuntimeError('Unexpected HTTP or STUN bindings.')
    others = [value for value in ports if value != '127.0.0.1:19081:80']
    if len(others) != 1:
        raise RuntimeError('Expected exactly one STUN binding.')
    old = others[0]
    if not isinstance(old, str) or not old.endswith(':3478:3478/udp'):
        raise RuntimeError('Unexpected STUN binding.')
    previous = ipaddress.ip_address(old.removesuffix(':3478:3478/udp'))
    if previous not in SUBNET:
        raise RuntimeError('Unexpected previous STUN network.')
    server['ports'] = [
        address + ':3478:3478/udp' if value == old else value
        for value in ports
    ]
    return result


def lan_address():
    interfaces = json.loads(
        call('ip', '-j', '-4', 'address', 'show', 'dev', 'wlp0s20f3'),
    )
    addresses = [
        item['local']
        for interface in interfaces
        for item in interface.get('addr_info', [])
        if item.get('scope') == 'global'
        and ipaddress.ip_address(item['local']) in SUBNET
    ]
    if len(addresses) != 1:
        raise RuntimeError(
            'Waiting for one address on the approved B Wi-Fi LAN.',
        )
    return addresses[0]


def container():
    # ps does not error when the existing managed container is absent.
    found = call(
        'docker',
        'ps',
        '-a',
        '--filter',
        'name=^/company-vpn-server-1$',
        '--format',
        '{{.ID}}',
    ).split()
    if not found:
        return None
    result = json.loads(call('docker', 'inspect', found[0]))[0]
    if (
        result['Config'].get('Labels', {}).get('com.docker.compose.project')
        != 'company-vpn'
    ):
        raise RuntimeError('Container does not belong to company-vpn.')
    return result


def data_volume(value):
    mounts = [
        m for m in value['Mounts'] if m['Destination'] == '/var/lib/netbird'
    ]
    if len(mounts) != 1 or mounts[0]['Type'] != 'volume':
        raise RuntimeError('Existing named VPN data volume is required.')
    return mounts[0]['Name']


def network_ok(value, address):
    if (
        not value
        or not value['State']['Running']
        or not value['NetworkSettings'].get('Networks')
    ):
        return False
    ports = value['NetworkSettings'].get('Ports') or {}
    return {'HostIp': '127.0.0.1', 'HostPort': '19081'} in (
        ports.get('80/tcp') or []
    ) and {'HostIp': address, 'HostPort': '3478'} in (
        ports.get('3478/udp') or []
    )


def api_ready(wait=0):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(
        'http://127.0.0.1:19081/api/instance',
        headers={'Host': 'vpn.changdar-server.mooo.com'},
    )
    deadline = time.monotonic() + wait
    while True:
        try:
            with opener.open(request, timeout=3) as response:
                body = json.load(response)
            if body.get('setup_required') is not False:
                raise RuntimeError(
                    (
                        'Unexpected setup state; refusing to initialize a new '
                        'account.'
                    ),
                )
            return True
        except (OSError, ValueError):
            if time.monotonic() >= deadline:
                return False
            time.sleep(1)


def repair_slot():
    stamp = RUNTIME / 'last-repair'
    now = time.monotonic()
    if stamp.exists() and now - float(stamp.read_text()) < 300:
        raise RuntimeError(
            (
                'Repair cooldown: will retry later; at most one recovery '
                'every five '
                'minutes.'
            ),
        )
    save(stamp, str(now))


def ensure():
    receipt = load(RECEIPT)
    address = lan_address()
    config_path = ROOT / 'compose.json'
    config = load(config_path)
    updated = config_for_address(config, address)
    value = container()
    volume = receipt['data_volume']
    # Refuse to start with a new or unrelated database volume, even if the
    # original container has been removed.
    recorded = json.loads(call('docker', 'volume', 'inspect', volume))[0]
    labels = recorded.get('Labels') or {}
    if labels.get('com.docker.compose.project') != 'company-vpn':
        raise RuntimeError('Recorded VPN volume ownership does not match.')
    if value and data_volume(value) != volume:
        raise RuntimeError(
            'Container data volume changed; manual review required.',
        )
    compose_name = (
        config.get('volumes', {})
        .get(
            'data',
            {},
        )
        .get('name', 'company-vpn_data')
    )
    if compose_name != volume:
        raise RuntimeError(
            'Compose no longer selects the recorded existing data volume.',
        )
    changed = updated != config
    if changed:
        encoded = json.dumps(config, indent=2) + '\n'
        backup = ROOT / (
            'compose.before-boot-ip-'
            + hashlib.sha256(encoded.encode()).hexdigest()[:16]
            + '.json'
        )
        if not backup.exists():
            save(backup, encoded)
        # Validate the candidate before replacing the effective configuration.
        candidate = ROOT / 'compose.boot-candidate.json'
        save(candidate, json.dumps(updated, indent=2) + '\n')
        try:
            call(
                'docker',
                'compose',
                '--project-name',
                'company-vpn',
                '--file',
                str(candidate),
                'config',
                '--quiet',
            )
            os.replace(candidate, config_path)
        finally:
            candidate.unlink(missing_ok=True)
        print(
            'Updated STUN binding to current B LAN address ' + address,
            flush=True,
        )
    if changed or not network_ok(value, address):
        repair_slot()
        call(
            *COMPOSE,
            'up',
            '-d',
            '--no-deps',
            '--force-recreate',
            '--pull',
            'never',
            'server',
        )
        print(
            'Restored VPN server network using the existing data volume.',
            flush=True,
        )
    # Start only the dashboard if it is absent/stopped; a healthy minute check
    # does not run compose up or recreate anything. Never pull images at boot.
    dashboard = call(
        'docker',
        'ps',
        '--filter',
        'label=com.docker.compose.project=company-vpn',
        '--filter',
        'label=com.docker.compose.service=dashboard',
        '--format',
        '{{.ID}}',
    ).strip()
    if not dashboard:
        call(*COMPOSE, 'up', '-d', '--no-deps', '--pull', 'never', 'dashboard')
    if not api_ready(wait=20):
        repair_slot()
        call(*COMPOSE, 'restart', '--no-deps', 'server')
        if not api_ready(wait=20):
            raise RuntimeError(
                (
                    'VPN API remains unavailable; retries are limited by the '
                    'repair cooldown.'
                ),
            )
    value = container()
    if not network_ok(value, address) or data_volume(value) != volume:
        raise RuntimeError('VPN runtime verification failed.')
    print(
        'VPN API, Docker network and persistent data volume are healthy.',
        flush=True,
    )


def managed_unit(path, contents):
    if path.exists() and not path.read_text().startswith(
        '# Managed by company-vpn/boot_guard.py',
    ):
        raise RuntimeError(
            'Refusing to overwrite an unmanaged unit: ' + str(path),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    save(path, contents, 0o644)


def install(lock):
    config = load(ROOT / 'compose.json')
    config_for_address(config, lan_address())
    value = container()
    if not value:
        raise RuntimeError(
            (
                'Install requires the existing VPN container to record its '
                'data volume.'
            ),
        )
    volume = data_volume(value)
    existing = load(RECEIPT) if RECEIPT.exists() else {'data_volume': volume}
    if existing.get('data_volume') != volume:
        raise RuntimeError('Previously recorded data volume does not match.')
    INSTALLED.parent.mkdir(parents=True, exist_ok=True)
    if (
        INSTALLED.parent.is_symlink()
        or INSTALLED.parent.stat().st_uid != 0
        or INSTALLED.parent.stat().st_mode & 0o022
    ):
        raise RuntimeError(
            'Installed helper directory must be root-owned and protected.',
        )
    save(INSTALLED, Path(__file__).read_text(), 0o644)
    save(RECEIPT, json.dumps(existing, indent=2) + '\n')
    managed_unit(
        Path(
            '/etc/systemd/system/company-vpn.service.d/20-network-health.conf',
        ),
        DROPIN,
    )
    managed_unit(
        Path('/etc/systemd/system/company-vpn-healthcheck.service'),
        CHECK_UNIT,
    )
    managed_unit(
        Path('/etc/systemd/system/company-vpn-healthcheck.timer'),
        TIMER,
    )
    call('systemctl', 'daemon-reload')
    call(
        'systemctl',
        'enable',
        'docker.service',
        'netbird.service',
        'company-vpn.service',
    )
    call('systemctl', 'enable', '--now', 'company-vpn-healthcheck.timer')
    # The service invokes this helper again. Release after installing all
    # files, before waiting for its startup check, to avoid a skipped check.
    fcntl.flock(lock, fcntl.LOCK_UN)
    call('systemctl', 'restart', 'company-vpn.service', timeout=200)
    call(
        'systemctl',
        'is-active',
        'company-vpn.service',
        'company-vpn-healthcheck.timer',
    )
    print(
        (
            'Installed root-owned startup guard and one-minute health timer. '
            'No reboot was '
            'performed.'
        ),
    )
    print(
        (
            'Reserve B LAN IP in DHCP and keep router UDP 3478 forwarding '
            'aligned with that '
            'address.'
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['install', 'ensure'])
    action = parser.parse_args().action
    if (
        os.geteuid() != 0
        or socket.gethostname() != 'changdar02-System-Product-Name'
    ):
        raise RuntimeError('Run with sudo on deployment machine B.')
    if (
        ROOT.is_symlink()
        or ROOT.stat().st_uid != 0
        or ROOT.stat().st_mode & 0o022
    ):
        raise RuntimeError('Expected the protected root-owned VPN directory.')
    RUNTIME.mkdir(mode=0o700, exist_ok=True)
    if (
        RUNTIME.is_symlink()
        or RUNTIME.stat().st_uid != 0
        or RUNTIME.stat().st_mode & 0o077
    ):
        raise RuntimeError('Expected a private root-owned runtime directory.')
    with open(RUNTIME / 'guard.lock', 'a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if action == 'install':
                raise RuntimeError(
                    'A health check is running; retry installation shortly.',
                )
            print('Another VPN check is already in progress.')
            return
        if action == 'install':
            install(lock)
        else:
            ensure()


if __name__ == '__main__':
    try:
        main()
    except (
        RuntimeError,
        ValueError,
        KeyError,
        OSError,
        subprocess.TimeoutExpired,
    ) as error:
        raise SystemExit(str(error)) from None
