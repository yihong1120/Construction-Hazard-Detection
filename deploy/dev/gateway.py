from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import os
import shutil
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

HOST = 'dev.changdar-server.mooo.com'
ORIGIN = 'https://' + HOST
MARKER = '# Managed by construction-dev-v1'
DEST = Path('/etc/nginx/sites-available/construction-dev.conf')
LINK = Path('/etc/nginx/sites-enabled/construction-dev.conf')
CERT = Path('/etc/letsencrypt/live') / HOST


def run(*args):
    subprocess.run(args, check=True)


def configuration(tls):
    text = f"""{MARKER}
server {{
    listen 80;
    listen [::]:80;
    server_name {HOST};
    location /.well-known/acme-challenge/ {{ root /var/www/html; }}
    location / {{ return 308 https://{HOST}$request_uri; }}
}}
"""
    if tls:
        text += f"""
server {{
    listen 443 ssl;
    listen [::]:443 ssl;
    http2 on;
    server_name {HOST};
    ssl_certificate {CERT}/fullchain.pem;
    ssl_certificate_key {CERT}/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    add_header X-Construction-Gateway dev-b always;
    add_header X-Content-Type-Options nosniff always;
    client_max_body_size 100M;
    location / {{
        proxy_pass http://100.96.17.117:18090;
        proxy_http_version 1.1;
        proxy_set_header Host {HOST};
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $remote_addr;
        proxy_set_header X-Forwarded-Proto https;
        proxy_set_header X-Forwarded-Host {HOST};
        proxy_set_header X-Forwarded-Port 443;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_redirect off;
        proxy_buffering off;
        proxy_request_buffering off;
        proxy_cache off;
        proxy_connect_timeout 10s;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }}
}}
"""
    return text


def install(text):
    temporary = DEST.with_suffix('.tmp')
    temporary.write_text(text)
    temporary.chmod(0o644)
    temporary.replace(DEST)
    if not LINK.is_symlink():
        LINK.symlink_to(DEST)
    run('/usr/sbin/nginx', '-t')
    run('/usr/bin/systemctl', 'reload', 'nginx')


def wait_gateway(timeout=45):
    deadline = time.monotonic() + timeout
    while True:
        result = subprocess.run(
            [
                '/usr/bin/curl',
                '--noproxy',
                '*',
                '--fail',
                '--silent',
                '--show-error',
                '--connect-timeout',
                '3',
                '--max-time',
                '5',
                '--dump-header',
                '-',
                '--resolve',
                HOST + ':443:127.0.0.1',
                ORIGIN + '/health',
            ],
            capture_output=True,
            text=True,
        )
        headers, _, body = result.stdout.partition('\n\n')
        try:
            healthy = json.loads(body) == {
                'status': True,
                'environment': 'dev',
            }
        except ValueError:
            healthy = False
        markers = {line.strip().lower() for line in headers.splitlines()}
        if (
            result.returncode == 0
            and healthy
            and {
                'x-construction-gateway: dev-b',
                'x-construction-environment: dev',
            }
            <= markers
        ):
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(
                'Dev HTTPS not ready: '
                + (result.stderr.strip() or 'Wrong app or gateway marker.'),
            )
        time.sleep(1)


def private_checks():
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    artifact = None
    index_digest = None
    for path in ('/health', '/', '/dev-build.json'):
        request = urllib.request.Request(
            'http://100.96.17.117:18090' + path,
            headers={'Host': HOST},
        )
        with opener.open(request, timeout=20) as response:
            body = response.read()
            if (
                response.status != 200
                or response.headers.get('X-Construction-Environment') != 'dev'
            ):
                raise RuntimeError('A dev preflight failed: ' + path)
            if path == '/health' and json.loads(body) != {
                'status': True,
                'environment': 'dev',
            }:
                raise RuntimeError('Private dev health failed.')
            if path == '/':
                index_digest = hashlib.sha256(body).hexdigest()
            if (
                path == '/dev-build.json'
                and json.loads(body).get('environment') != 'dev'
            ):
                raise RuntimeError('Missing versioned dev artifact.')
            if path == '/dev-build.json':
                artifact = json.loads(body)
    request = urllib.request.Request(
        'http://100.96.17.117:18090/main.dart.js',
        headers={'Host': HOST},
    )
    with opener.open(request, timeout=30) as response:
        digest = hashlib.file_digest(response, 'sha256').hexdigest()
    if digest != artifact.get('main_dart_sha256'):
        raise RuntimeError(
            'Dev frontend checksum differs from its build manifest.',
        )
    if index_digest != artifact.get('index_sha256'):
        raise RuntimeError(
            'Dev index checksum differs from its artifact manifest.',
        )
    print(
        (
            'A independent dev backend, database/Redis health and Flutter '
            'artifact verified.'
        ),
        flush=True,
    )


def verify():
    wait_gateway()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for url in (
        ORIGIN + '/health',
        ORIGIN + '/',
        ORIGIN + '/dev-build.json',
        'https://ai-dev.changdar-server.mooo.com/health',
        'https://changdar-server.mooo.com/',
        'https://ai.changdar-server.mooo.com/health',
        'https://wiki.changdar-server.mooo.com/',
    ):
        with opener.open(url, timeout=30) as response:
            if response.status != 200:
                raise RuntimeError('Public check failed: ' + url)
            if (
                url.startswith(ORIGIN)
                and response.headers.get('X-Construction-Gateway') != 'dev-b'
            ):
                raise RuntimeError(
                    'Public DNS/router is not reaching the dev gateway.',
                )
    run('/usr/bin/systemctl', 'is-enabled', 'nginx')
    run('/usr/bin/systemctl', 'is-active', 'certbot.timer')
    print(
        (
            'Passed: dev HTTPS, isolated frontend/backend and all existing '
            'public websites are '
            'healthy.'
        ),
    )
    print(
        'Open '
        + ORIGIN
        + ' and sign in with the existing ChangDar Keycloak account.',
    )


def main():
    if (
        os.geteuid() != 0
        or socket.gethostname() != 'changdar02-System-Product-Name'
    ):
        raise RuntimeError('Run with sudo on B (changdar02).')
    if DEST.exists() and not DEST.read_text().startswith(MARKER):
        raise RuntimeError('Refusing to overwrite an unowned site.')
    if (LINK.exists() or LINK.is_symlink()) and (
        not LINK.is_symlink() or LINK.resolve() != DEST
    ):
        raise RuntimeError('Unexpected enabled-site path.')
    private_checks()
    backup = Path('/etc/nginx/construction-dev-backups') / str(time.time_ns())
    backup.mkdir(parents=True, mode=0o700)
    previous = DEST.read_bytes() if DEST.exists() else None
    linked = LINK.is_symlink()
    if previous is not None:
        shutil.copy2(DEST, backup / DEST.name)
    (backup / 'receipt.json').write_text(
        json.dumps(
            {'site_existed': previous is not None, 'linked': linked},
        ),
    )
    (Path('/var/www/html/.well-known/acme-challenge')).mkdir(
        parents=True,
        exist_ok=True,
    )
    try:
        install(configuration(CERT.joinpath('fullchain.pem').exists()))
        renewal = configparser.ConfigParser()
        renewal.read_string(
            '[certificate]\n'
            + Path(
                '/etc/letsencrypt/renewal/ai.changdar-server.mooo.com.conf',
            ).read_text(),
        )
        print(
            (
                'Obtaining a dedicated certificate for dev; existing '
                'certificates are '
                'retained.'
            ),
            flush=True,
        )
        run(
            '/usr/bin/certbot',
            'certonly',
            '--webroot',
            '-w',
            '/var/www/html',
            '-d',
            HOST,
            '--cert-name',
            HOST,
            '--non-interactive',
            '--agree-tos',
            '--keep-until-expiring',
            '--account',
            renewal.get('renewalparams', 'account'),
            '--server',
            renewal.get(
                'renewalparams',
                'server',
            ),
        )
        install(configuration(True))
        verify()
    except BaseException:
        if previous is None:
            DEST.unlink(missing_ok=True)
        else:
            DEST.write_bytes(previous)
        if not linked:
            LINK.unlink(missing_ok=True)
        run('/usr/sbin/nginx', '-t')
        run('/usr/bin/systemctl', 'reload', 'nginx')
        print('Dev gateway rolled back; backup: ' + str(backup), flush=True)
        raise
    hook = Path('/etc/letsencrypt/renewal-hooks/deploy/construction-dev-nginx')
    hook.parent.mkdir(parents=True, exist_ok=True)
    if hook.exists() and MARKER not in hook.read_text():
        raise RuntimeError('Existing unowned renewal hook.')
    hook.write_text(
        '#!/bin/sh\n'
        + MARKER
        + '\ncase " $RENEWED_DOMAINS " in\n'
        + (
            '  *" '
            f"{HOST}"
            ' "*) /usr/sbin/nginx -t && /usr/bin/systemctl '
            'reload nginx '
            ';;\nesac\n'
        ),
    )
    hook.chmod(0o755)
    run('/bin/sh', '-n', str(hook))
    print(
        'Separate certificate renewal hook installed. Backup: ' + str(backup),
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.verify_only:
        verify()
    elif args.preflight:
        private_checks()
    else:
        main()
