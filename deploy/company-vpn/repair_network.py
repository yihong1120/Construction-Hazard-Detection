from __future__ import annotations

import copy
import json
import os
import socket
import subprocess
import tempfile
import time
import urllib.request
from datetime import datetime
from datetime import timezone
from pathlib import Path

ROOT = Path('/etc/company-vpn')
ORIGIN = 'https://vpn.changdar-server.mooo.com'
OLD_IP = '192.168.1.86'
NEW_IP = '192.168.1.88'
IMAGE = (
    'netbirdio/netbird-server@sha256:05b3d8d6d056e5062a2965d4d6e5d6d83f09f2bf9'
    '133947a8e5972e0e0e7a261'
)


def repaired_config(config):
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
    ):
        raise RuntimeError(
            (
                'Unexpected server image, network mode or mounts; nothing was '
                'changed.'
            ),
        )
    ports = server.get('ports', [])
    if (
        len(ports) != 2
        or '127.0.0.1:19081:80' not in ports
        or not any(ip + ':3478:3478/udp' in ports for ip in (OLD_IP, NEW_IP))
    ):
        raise RuntimeError('Unexpected port bindings; nothing was changed.')
    server['ports'] = [
        (
            NEW_IP + ':3478:3478/udp'
            if value == OLD_IP + ':3478:3478/udp'
            else value
        )
        for value in ports
    ]
    return result


def command(*args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=120)
    if result.returncode:
        error_path = ROOT / 'network-repair-error.log'
        fd = os.open(
            error_path,
            os.O_CREAT | os.O_TRUNC | os.O_WRONLY | os.O_NOFOLLOW,
            0o600,
        )
        with os.fdopen(fd, 'w') as stream:
            stream.write(result.stderr or result.stdout)
        raise RuntimeError(
            args[0] + ' failed; details are in ' + str(error_path),
        )
    return result.stdout


def main():
    if (
        os.geteuid() != 0
        or socket.gethostname() != 'changdar02-System-Product-Name'
    ):
        raise RuntimeError('Run with sudo on deployment machine B.')
    interfaces = json.loads(
        command('ip', '-j', '-4', 'address', 'show', 'dev', 'wlp0s20f3'),
    )
    addresses = {
        item['local']
        for interface in interfaces
        for item in interface.get('addr_info', [])
    }
    if NEW_IP not in addresses or OLD_IP in addresses:
        raise RuntimeError(
            (
                'B LAN no longer matches the confirmed address; nothing was '
                'changed.'
            ),
        )
    path = ROOT / 'compose.json'
    if ROOT.is_symlink() or path.is_symlink() or path.stat().st_uid != 0:
        raise RuntimeError(
            'Expected the root-owned company VPN configuration.',
        )
    original = path.read_bytes()
    config = json.loads(original)
    updated = repaired_config(config)
    before = json.loads(
        command('docker', 'inspect', 'company-vpn-server-1'),
    )[0]
    if (
        before['Config'].get('Labels', {}).get('com.docker.compose.project')
        != 'company-vpn'
    ):
        raise RuntimeError(
            'The existing container is not owned by this compose project.',
        )
    data_before = [
        (m.get('Type'), m.get('Name'), m.get('Source'))
        for m in before['Mounts']
        if m['Destination'] == '/var/lib/netbird'
    ]
    if len(data_before) != 1 or data_before[0][0] != 'volume':
        raise RuntimeError('Expected the existing named VPN data volume.')
    compose = [
        'docker',
        'compose',
        '--project-name',
        'company-vpn',
        '--file',
        str(path),
    ]
    fd, temporary = tempfile.mkstemp(
        prefix='compose-network-repair-',
        suffix='.json',
        dir=ROOT,
    )
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(updated, stream, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        command(
            'docker',
            'compose',
            '--project-name',
            'company-vpn',
            '--file',
            temporary,
            'config',
            '--quiet',
        )
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        backup = ROOT / ('compose.before-network-repair-' + stamp + '.json')
        fd = os.open(backup, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(original)
            stream.flush()
            os.fsync(stream.fileno())
        print('Backup:', backup, flush=True)
        print('STUN binding: ' + OLD_IP + ' -> ' + NEW_IP, flush=True)
        print(
            (
                'Recreating server only; retaining existing data volume and '
                'image.'
            ),
            flush=True,
        )
        os.replace(temporary, path)
        command(
            *compose,
            'up',
            '-d',
            '--no-deps',
            '--force-recreate',
            '--pull',
            'never',
            'server',
        )
    finally:
        Path(temporary).unlink(missing_ok=True)
    after = json.loads(command('docker', 'inspect', 'company-vpn-server-1'))[0]
    data_after = [
        (m.get('Type'), m.get('Name'), m.get('Source'))
        for m in after['Mounts']
        if m['Destination'] == '/var/lib/netbird'
    ]
    ports = after['NetworkSettings'].get('Ports') or {}
    if (
        data_before != data_after
        or not after['State']['Running']
        or not after['NetworkSettings'].get('Networks')
        or {'HostIp': '127.0.0.1', 'HostPort': '19081'}
        not in (ports.get('80/tcp') or [])
        or {'HostIp': NEW_IP, 'HostPort': '3478'}
        not in (ports.get('3478/udp') or [])
    ):
        raise RuntimeError(
            (
                'Runtime verification failed; original configuration backup '
                'was retained.'
            ),
        )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(
        'http://127.0.0.1:19081/api/instance',
        headers={'Host': 'vpn.changdar-server.mooo.com'},
    )
    for _ in range(30):
        try:
            with opener.open(request, timeout=3) as response:
                instance = json.load(response)
            if instance.get('setup_required') is not False:
                raise RuntimeError(
                    'Unexpected setup state; do not initialize a new account.',
                )
            break
        except OSError:
            time.sleep(1)
    else:
        raise RuntimeError(
            (
                'VPN API did not become ready; inspect server logs. Data '
                'volume is preserved.'
            ),
        )
    discovery = json.loads(
        command(
            'curl',
            '--noproxy',
            '*',
            '--fail',
            '--silent',
            '--show-error',
            '--max-time',
            '15',
            '--resolve',
            'vpn.changdar-server.mooo.com:443:127.0.0.1',
            ORIGIN + '/oauth2/.well-known/openid-configuration',
        ),
    )
    if discovery.get('issuer') != ORIGIN + '/oauth2':
        raise RuntimeError('Unexpected VPN issuer after recovery.')
    print(
        (
            'Verified: Docker network and both port bindings restored, '
            'existing data volume '
            'retained,'
        ),
    )
    print(
        (
            'local API ready and HTTPS OIDC discovery valid. Reconnect the '
            'employee NetBird '
            'app.'
        ),
    )
    print(
        'Router UDP 3478 forwarding must target '
        + NEW_IP
        + '; external VPN connectivity still needs testing.',
    )


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError, KeyError, OSError) as error:
        raise SystemExit(str(error)) from None
