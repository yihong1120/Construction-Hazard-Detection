# Install the NetBird control plane on B. Employee SSO/routing is a later step.
from __future__ import annotations

import argparse
import base64
import fcntl
import ipaddress
import json
import os
import secrets
import shutil
import socket
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HOST = 'vpn.changdar-server.mooo.com'
ORIGIN = 'https://' + HOST
LAN = '192.168.1.86'
VPN_RANGE = '10.203.0.0/16'
ROOT = Path('/etc/company-vpn')
SITE = Path('/etc/nginx/sites-available/company-vpn.conf')
ENABLED = Path('/etc/nginx/sites-enabled/company-vpn.conf')
SERVER = (
    'netbirdio/netbird-server@sha256:05b3d8d6d056e5062a2965d4d6e5d6d8'
    '3f09f2bf9133947a8e5972e0e0e7a261'
)
DASHBOARD = (
    'netbirdio/dashboard@sha256:70e0fc6dd3b7758148d182fb55ecd8d4f9ce0'
    '1e9bba56c9389224e0b8354009d'
)
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def save(path, value):
    data = (
        json.dumps(value, indent=2) + '\n'
        if not isinstance(value, str)
        else value
    )
    temporary = path.with_name(path.name + '.tmp')
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as stream:
        stream.write(data)
    os.replace(temporary, path)


def run(*args, timeout=90):
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode:
        save(ROOT / 'private-error.log', result.stderr or result.stdout)
        raise RuntimeError(
            args[0] + ' failed; details in /etc/company-vpn/private-error.log',
        )
    return result.stdout


def compose(*args, timeout=90):
    return run(
        'docker',
        'compose',
        '--project-name',
        'company-vpn',
        '--file',
        str(ROOT / 'compose.json'),
        *args,
        timeout=timeout,
    )


def api(path, method='GET', payload=None, token=None):
    headers = {'Content-Type': 'application/json', 'Host': HOST}
    if token:
        headers['Authorization'] = 'Token ' + token
    request = urllib.request.Request(
        'http://127.0.0.1:19081/api/' + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers=headers,
        method=method,
    )
    try:
        with OPENER.open(request, timeout=20) as response:
            body = response.read()
        return json.loads(body) if body else None
    except urllib.error.HTTPError as error:
        # Never put API bodies, tokens or user data in terminal output.
        raise RuntimeError(
            'NetBird API ' + method + ' ' + path + ': HTTP ' + str(error.code),
        ) from None


def wait_api():
    for _ in range(90):
        try:
            return api('instance')
        except (OSError, ValueError, RuntimeError):
            time.sleep(2)
    raise RuntimeError(
        'NetBird did not start; inspect company-vpn server logs locally.',
    )


def local_networks():
    networks = []
    for route in json.loads(
        run('ip', '-j', '-4', 'route', 'show', 'table', 'all'),
    ):
        if route.get('dst') and route['dst'] != 'default':
            networks.append(ipaddress.ip_network(route['dst'], strict=False))
    ids = run('docker', 'network', 'ls', '-q').split()
    if ids:
        for network in json.loads(run('docker', 'network', 'inspect', *ids)):
            for config in network.get('IPAM', {}).get('Config') or []:
                if config.get('Subnet'):
                    subnet = ipaddress.ip_network(
                        config['Subnet'],
                        strict=False,
                    )
                    if subnet.version == 4:
                        networks.append(subnet)
    return networks


def new_config(networks):
    vpn = ipaddress.ip_network(VPN_RANGE)
    if any(vpn.overlaps(n) for n in networks):
        raise RuntimeError(
            'VPN candidate overlaps an existing route or Docker network.',
        )
    for index in range(3 * 256):
        bridge = ipaddress.ip_network(
            '172.' + str(29 + index // 256) + '.' + str(index % 256) + '.0/24',
        )
        if not any(bridge.overlaps(n) for n in networks):
            break
    else:
        raise RuntimeError('No free Docker subnet found in 172.29-31.0.0/16.')
    return {
        'bridge': str(bridge),
        'gateway': str(bridge.network_address + 1),
        'auth_key': _random_key().rstrip('='),
        'cookie_key': _random_key(),
        'encryption_key': _random_key(),
    }


def server_config(config):
    # JSON is also valid YAML, which is the server's configuration format.
    return {
        'server': {
            'listenAddress': ':80',
            'exposedAddress': ORIGIN + ':443',
            'stunPorts': [3478],
            'metricsPort': 9090,
            'healthcheckAddress': ':9000',
            'logLevel': 'info',
            'logFile': 'console',
            'dataDir': '/var/lib/netbird',
            'authSecret': config['auth_key'],
            'auth': {
                'issuer': ORIGIN + '/oauth2',
                'signKeyRefreshEnabled': True,
                'sessionCookieEncryptionKey': config['cookie_key'],
                'dashboardRedirectURIs': [
                    ORIGIN + '/nb-auth',
                    ORIGIN + '/nb-silent-auth',
                ],
                'cliRedirectURIs': ['http://localhost:53000/'],
            },
            'reverseProxy': {
                'trustedHTTPProxies': [config['gateway'] + '/32'],
                'trustedPeers': [config['gateway'] + '/32'],
            },
            'store': {
                'engine': 'sqlite',
                'encryptionKey': config['encryption_key'],
            },
        },
    }


def compose_config(config, bootstrap):
    logging = {
        'driver': 'json-file',
        'options': {'max-size': '20m', 'max-file': '3'},
    }
    return {
        'services': {
            'server': {
                'image': SERVER,
                'restart': 'unless-stopped',
                'ports': ['127.0.0.1:19081:80', LAN + ':3478:3478/udp'],
                'environment': {
                    'NB_SETUP_PAT_ENABLED': str(bootstrap).lower(),
                    'NB_DISABLE_GEOLOCATION': 'true',
                },
                'volumes': [
                    'data:/var/lib/netbird',
                    './config.yaml:/etc/netbird/config.yaml:ro',
                ],
                'logging': logging,
            },
            'dashboard': {
                'image': DASHBOARD,
                'restart': 'unless-stopped',
                'ports': ['127.0.0.1:19080:80'],
                'logging': logging,
                'environment': {
                    'NETBIRD_MGMT_API_ENDPOINT': ORIGIN,
                    'NETBIRD_MGMT_GRPC_API_ENDPOINT': ORIGIN,
                    'AUTH_AUDIENCE': 'netbird-dashboard',
                    'AUTH_CLIENT_ID': 'netbird-dashboard',
                    'AUTH_CLIENT_SECRET': '',
                    'AUTH_AUTHORITY': ORIGIN + '/oauth2',
                    'USE_AUTH0': 'false',
                    'AUTH_SUPPORTED_SCOPES': 'openid profile email groups',
                    'AUTH_REDIRECT_URI': '/nb-auth',
                    'AUTH_SILENT_REDIRECT_URI': '/nb-silent-auth',
                    'NGINX_SSL_PORT': '443',
                    'LETSENCRYPT_DOMAIN': 'none',
                },
            },
        },
        'volumes': {'data': {}},
        'networks': {
            'default': {
                'ipam': {
                    'config': [
                        {
                            'subnet': config['bridge'],
                            'gateway': config['gateway'],
                        },
                    ],
                },
            },
        },
    }


def harden():
    credentials_path = ROOT / 'bootstrap.json'
    if not credentials_path.exists():
        save(
            credentials_path,
            {
                'email': 'vpn-owner@changdar-server.mooo.com',
                'name': 'VPN Recovery Owner',
                'password': secrets.token_urlsafe(36) + 'aA1!',
            },
        )
    owner = json.loads(credentials_path.read_text())
    if wait_api()['setup_required']:
        response = api(
            'setup',
            'POST',
            {
                **{key: owner[key] for key in ('email', 'name', 'password')},
                'create_pat': True,
                'pat_expire_in': 7,
            },
        )
        owner.update(response)
        save(credentials_path, owner)
    token = owner.get('personal_access_token')
    if not token:
        raise RuntimeError(
            (
                'Owner exists but bootstrap token is unavailable. '
                'Public VPN site remains unpublished.'
            ),
        )
    accounts = api('accounts', token=token)
    if len(accounts) != 1 or api('peers', token=token):
        raise RuntimeError(
            'Expected one new account with no enrolled devices.',
        )
    account = accounts[0]
    settings = account['settings']
    settings['network_range'] = VPN_RANGE
    api('accounts/' + account['id'], 'PUT', {'settings': settings}, token)
    for policy in api('policies', token=token):
        policy['enabled'] = False
        for rule in policy['rules']:
            for key in ('sources', 'destinations'):
                rule[key] = [
                    group['id'] if isinstance(group, dict) else group
                    for group in rule.get(key, [])
                ]
        api('policies/' + policy['id'], 'PUT', policy, token)
    updated = api('accounts', token=token)[0]
    if updated['settings']['network_range'] != VPN_RANGE or any(
        policy['enabled'] for policy in api('policies', token=token)
    ):
        raise RuntimeError('Initial isolation verification failed.')
    save(
        ROOT / 'isolated.json',
        {
            'account_id': account['id'],
            'network_range': VPN_RANGE,
            'employee_sso_configured': False,
            'routing_configured': False,
        },
    )


def nginx_site(tls):
    redirect = (
        'return 301 https://$host$request_uri;' if tls else 'return 503;'
    )
    http = (
        '# Managed by company-vpn/deploy.py; only this host is '
        'changed.\nserver {\n    listen 80;\n    listen [::]:80;\n    '
        'server_name '
        f'{HOST}'
        ';\n    location ^~ /.well-known/acme-challenge/ { root '
        '/var/www/html; }\n    location / { '
        f'{redirect}'
        ' }\n}\n'
    )
    if not tls:
        return http
    return http + (
        '\nserver {\n    listen 443 ssl;\n    listen [::]:443 ssl;\n '
        '   http2 on;\n    server_name '
        f'{HOST}'
        ';\n    ssl_certificate /etc/letsencrypt/live/'
        f'{HOST}'
        '/fullchain.pem;\n    ssl_certificate_key '
        '/etc/letsencrypt/live/'
        f'{HOST}'
        '/privkey.pem;\n    ssl_protocols TLSv1.2 TLSv1.3;\n    '
        'client_header_timeout 1d;\n    client_body_timeout 1d;\n  '
        '  proxy_set_header Host $host;\n    proxy_set_header '
        'X-Real-IP $remote_addr;\n    proxy_set_header '
        'X-Forwarded-For $remote_addr;\n    proxy_set_header '
        'X-Forwarded-Proto https;\n    proxy_set_header '
        'X-Forwarded-Host $host;\n    grpc_set_header Host $host;\n'
        '    grpc_set_header X-Forwarded-For $remote_addr;\n    '
        'grpc_set_header X-Forwarded-Proto https;\n    location = '
        '/api/setup { return 403; }\n    location ~ '
        '^/(relay|ws-proxy/) {\n        proxy_pass '
        'http://127.0.0.1:19081;\n        proxy_http_version 1.1;\n'
        '        proxy_set_header Host $host;\n        '
        'proxy_set_header X-Real-IP $remote_addr;\n        '
        'proxy_set_header X-Forwarded-For $remote_addr;\n        '
        'proxy_set_header X-Forwarded-Proto https;\n        '
        'proxy_set_header X-Forwarded-Host $host;\n        '
        'proxy_set_header Upgrade $http_upgrade;\n        '
        'proxy_set_header Connection "Upgrade";\n        '
        'proxy_read_timeout 1d;\n    }\n    location ~ '
        '^/(signalexchange\\.SignalExchange|management\\.Management'
        'Service)/ {\n        grpc_pass grpc://127.0.0.1:19081;\n  '
        '      grpc_read_timeout 1d;\n        grpc_send_timeout '
        '1d;\n        grpc_socket_keepalive on;\n    }\n    '
        'location ~ ^/(api|oauth2)/ {\n        proxy_pass '
        'http://127.0.0.1:19081;\n        proxy_read_timeout '
        '120s;\n    }\n    location / { proxy_pass '
        'http://127.0.0.1:19080; }\n}\n'
    )


def install_site(tls):
    previous = SITE.read_bytes() if SITE.exists() else None
    linked = ENABLED.is_symlink()
    if ENABLED.exists() and not linked:
        raise RuntimeError(
            'Refusing to replace an unmanaged enabled Nginx file.',
        )
    if linked and ENABLED.resolve() != SITE:
        raise RuntimeError('Refusing to replace an unmanaged Nginx link.')
    try:
        save(SITE, nginx_site(tls))
        SITE.chmod(0o644)
        if not linked:
            ENABLED.symlink_to(SITE)
        run('nginx', '-t')
        run('systemctl', 'reload', 'nginx')
    except Exception:
        if previous is not None:
            SITE.write_bytes(previous)
        else:
            SITE.unlink(missing_ok=True)
        if not linked:
            ENABLED.unlink(missing_ok=True)
        run('nginx', '-t')
        run('systemctl', 'reload', 'nginx')
        raise


def https_check(host, path='/', local=False):
    args = [
        '/usr/bin/curl',
        '-q',
        '--noproxy',
        '*',
        '--fail',
        '--silent',
        '--show-error',
        '--max-time',
        '20',
    ]
    if local:
        args += ['--resolve', host + ':443:127.0.0.1']
    return run(*args, 'https://' + host + path)


def check_stun(address, port=3478):
    transaction = secrets.token_bytes(12)
    packet = struct.pack('!HHI', 1, 0, 0x2112A442) + transaction
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client:
        client.settimeout(5)
        client.sendto(packet, (address, port))
        response, _ = client.recvfrom(4096)
    if (
        len(response) < 20
        or response[:2] != b'\x01\x01'
        or response[4:20] != packet[4:20]
    ):
        raise RuntimeError('Local STUN response verification failed.')


def verify():
    if wait_api()['setup_required']:
        raise RuntimeError('Owner setup is incomplete.')
    for attempt in range(15):
        try:
            status = json.loads(https_check(HOST, '/api/instance', local=True))
            discovery = json.loads(
                https_check(
                    HOST,
                    '/oauth2/.well-known/openid-configuration',
                    local=True,
                ),
            )
            if (
                status['setup_required']
                or discovery['issuer'] != ORIGIN + '/oauth2'
            ):
                raise RuntimeError('Unexpected instance or issuer.')
            https_check(HOST, local=True)
            break
        except (RuntimeError, ValueError):
            if attempt == 14:
                raise
            time.sleep(2)
    for host in (
        'changdar-server.mooo.com',
        'ai.changdar-server.mooo.com',
        'wiki.changdar-server.mooo.com',
    ):
        https_check(host)
    config = json.loads((ROOT / 'compose.json').read_text())
    if (
        config['services']['server']['environment']['NB_SETUP_PAT_ENABLED']
        != 'false'
    ):
        raise RuntimeError('Setup-time token creation must be disabled.')
    ids = compose('ps', '-q').split()
    if len(ids) != 2:
        raise RuntimeError('Expected two running NetBird containers.')
    for container in json.loads(run('docker', 'inspect', *ids)):
        if (
            not container['State']['Running']
            or container['HostConfig']['RestartPolicy']['Name']
            != 'unless-stopped'
        ):
            raise RuntimeError(
                'NetBird runtime or autostart configuration differs.',
            )
        environment = container['Config'].get('Env', [])
        if any(
            item.startswith('NB_SETUP_PAT_ENABLED=')
            and item != 'NB_SETUP_PAT_ENABLED=false'
            for item in environment
        ):
            raise RuntimeError(
                'Runtime setup-time token creation is still enabled.',
            )
    run('systemctl', 'is-enabled', 'docker')
    run('systemctl', 'is-enabled', 'company-vpn.service')
    run('systemctl', 'is-active', 'company-vpn.service')
    run('systemctl', 'is-active', 'nginx')
    check_stun(LAN)
    print(
        (
            'Verified: VPN HTTPS, owner setup, OIDC discovery, local '
            'STUN, container autostart and existing three public '
            'websites.'
        ),
    )
    print(
        (
            'Router UDP 3478 forwarding and external employee '
            'connectivity still require testing.'
        ),
    )
    if (ROOT / 'sso-ready.json').exists():
        print(
            (
                'Keycloak connector is configured; employee approval,'
                ' office routing and external connectivity are '
                'separate checks.'
            ),
        )
    else:
        print(
            (
                'Control plane only: employee Keycloak SSO, device '
                'access policies and office routing are not '
                'configured.'
            ),
        )


def install_boot_unit():
    unit = Path('/etc/systemd/system/company-vpn.service')
    text = (
        '# Managed by company-vpn/deploy.py\n[Unit]\n'
        'Description=Company NetBird control plane\n'
        'Wants=network-online.target\nAfter=network-online.target '
        'docker.service\nRequires=docker.service\n'
        'StartLimitIntervalSec=0\n\n[Service]\nType=oneshot\n'
        'RemainAfterExit=yes\nExecStart=/usr/bin/docker compose '
        '--project-name company-vpn --file '
        '/etc/company-vpn/compose.json up -d\nRestart=on-failure\n'
        'RestartSec=10s\nTimeoutStartSec=180s\n\n[Install]\n'
        'WantedBy=multi-user.target\n'
    )
    if unit.exists() and not unit.read_text().startswith(
        '# Managed by company-vpn/deploy.py',
    ):
        raise RuntimeError(
            'Refusing to replace an unmanaged company-vpn.service.',
        )
    save(unit, text)
    unit.chmod(0o644)
    run('systemctl', 'daemon-reload')
    run('systemctl', 'enable', '--now', 'company-vpn.service', timeout=180)


def install():
    if (ROOT / 'installed.json').exists():
        verify()
        return
    interfaces = json.loads(run('ip', '-j', '-4', 'address', 'show'))
    if LAN not in {
        a['local']
        for interface in interfaces
        for a in interface.get('addr_info', [])
    }:
        raise RuntimeError('Run on B with LAN address ' + LAN + '.')
    main_ips = {
        r[4][0]
        for r in socket.getaddrinfo(
            'changdar-server.mooo.com',
            443,
            socket.AF_INET,
        )
    }
    vpn_ips = {r[4][0] for r in socket.getaddrinfo(HOST, 443, socket.AF_INET)}
    if main_ips != vpn_ips:
        raise RuntimeError('VPN DNS does not match the main website.')
    for executable in ('docker', 'nginx', 'certbot'):
        if not shutil.which(executable):
            raise RuntimeError('Required command is missing: ' + executable)
    config_path = ROOT / 'private-config.json'
    if not config_path.exists():
        for kind, address, port in (
            (socket.SOCK_STREAM, '127.0.0.1', 19080),
            (socket.SOCK_STREAM, '127.0.0.1', 19081),
            (socket.SOCK_DGRAM, LAN, 3478),
        ):
            with socket.socket(socket.AF_INET, kind) as listener:
                listener.bind((address, port))
        # Check before adding any public configuration.
        existing = run('nginx', '-T')
        if HOST in existing:
            raise RuntimeError(
                (
                    'An existing Nginx configuration already '
                    'mentions the VPN domain.'
                ),
            )
        for host in (
            'changdar-server.mooo.com',
            'ai.changdar-server.mooo.com',
            'wiki.changdar-server.mooo.com',
        ):
            https_check(host)
        config = new_config(local_networks())
        backup = ROOT / ('nginx-backup-' + str(time.time_ns()))
        backup.mkdir(mode=0o700)
        shutil.copytree('/etc/nginx', backup / 'nginx', symlinks=True)
        save(config_path, config)
    config = json.loads(config_path.read_text())
    save(ROOT / 'config.yaml', server_config(config))
    bootstrap = not (ROOT / 'isolated.json').exists()
    save(ROOT / 'compose.json', compose_config(config, bootstrap))
    print(
        'Pulling pinned NetBird server 0.80.0 and dashboard 2.94.0 images.',
        flush=True,
    )
    compose('pull', timeout=1800)
    compose('up', '-d', timeout=180)
    if bootstrap:
        harden()
        save(ROOT / 'compose.json', compose_config(config, False))
        compose('up', '-d', timeout=180)
        wait_api()
    print(
        (
            'Private owner and VPN isolation verified; enabling the '
            'dedicated HTTPS host.'
        ),
        flush=True,
    )
    cert = Path('/etc/letsencrypt/live') / HOST / 'fullchain.pem'
    if not cert.exists():
        install_site(False)
        run(
            'certbot',
            'certonly',
            '--webroot',
            '-w',
            '/var/www/html',
            '-d',
            HOST,
            '--non-interactive',
            '--agree-tos',
            '--register-unsafely-without-email',
            '--deploy-hook',
            '/usr/sbin/nginx -t && /usr/bin/systemctl reload nginx',
            timeout=600,
        )
    install_site(True)
    install_boot_unit()
    verify()
    save(
        ROOT / 'installed.json',
        {'host': HOST, 'installed_at': time.time(), 'phase': 'control-plane'},
    )
    print(
        (
            'NetBird control plane installed. Administrator '
            'credentials: sudo python3 ~/company-vpn/deploy.py '
            'credentials'
        ),
    )


def show_credentials_path():
    credentials = ROOT / 'bootstrap.json'
    if not credentials.is_file():
        raise RuntimeError('Private recovery credentials are missing.')
    print('Private recovery credentials file:', credentials)
    print(
        'Read this root-only file locally when needed; '
        'do not copy its contents into logs or issue reports.',
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'action',
        nargs='?',
        choices=('install', 'verify', 'credentials'),
        default='install',
    )
    action = parser.parse_args().action
    if os.geteuid() != 0:
        raise SystemExit('Run with sudo on B.')
    os.umask(0o077)
    ROOT.mkdir(mode=0o700, exist_ok=True)
    if ROOT.is_symlink() or ROOT.stat().st_uid != 0:
        raise SystemExit(
            (
                'Configuration directory must be owned by root and '
                'must not be a symlink.'
            ),
        )
    ROOT.chmod(0o700)
    with (ROOT / 'deploy.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if action == 'credentials':
            show_credentials_path()
        elif action == 'install':
            install()
        else:
            verify()


def _random_key():
    return base64.b64encode(secrets.token_bytes(32)).decode()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('Stopped:', str(error), file=sys.stderr)
        raise SystemExit(1)
