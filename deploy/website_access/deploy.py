from __future__ import annotations

import hashlib
import json
import os
import pwd
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent
OWNER = 'changdar02'
HOME = Path('/home') / OWNER
APP = HOME / 'service-migration-phase5/app'
RUNTIME = HOME / 'service-migration-phase5/cpu-runtime/bin/python'
ENV = HOME / 'service-migration-phase5/production-bff-config.json'
INSTALL = Path('/opt/website-access')
OUTPUT = Path('/etc/nginx/conf.d/website-access.conf')
CONFIG = Path('/etc/website-access.json')
HOSTS = {
    'main': 'changdar-server.mooo.com',
    'ai': 'ai.changdar-server.mooo.com',
    'wiki': 'wiki.changdar-server.mooo.com',
}
NEW_FILES = (
    'examples/db_management/website_access_models.py',
    'examples/db_management/schemas/website_access.py',
    'examples/db_management/services/website_access.py',
    'examples/db_management/routers/website_access.py',
)
PATCHES = {
    'examples/db_management/app.py': [
        (
            'from examples.deployment_registry.router import '
            'router as registry_router',
            'from examples.db_management.routers.website_access import '
            'router as website_access_router\n'
            'from examples.deployment_registry.router import '
            'router as registry_router',
        ),
        (
            'app.include_router(stream_cfg_router)',
            'app.include_router(stream_cfg_router)\n'
            'app.include_router(website_access_router)',
        ),
    ],
    'examples/db_management/schemas/user.py': [
        (
            '    can_assign_group_admins: bool',
            '    can_assign_group_admins: bool\n'
            '    can_manage_website_access: bool = False',
        ),
    ],
    'examples/db_management/routers/users.py': [
        (
            '        can_assign_group_admins=super_admin,',
            '        can_assign_group_admins=super_admin,\n'
            '        can_manage_website_access=super_admin,',
        ),
    ],
}


def run(args):
    """Suppress errors which can contain private configuration values."""
    result = subprocess.run(
        [str(a) for a in args], capture_output=True, text=True, timeout=90,
    )
    if result.returncode:
        raise RuntimeError('Command failed: ' + str(args[0]))
    return result.stdout


def patched_code(content, replacements):
    """Preserve production changes and refuse ambiguous patch anchors."""
    for old, new in replacements:
        if new in content:
            continue
        if content.count(old) != 1:
            raise ValueError('Runtime patch anchor differs')
        content = content.replace(old, new, 1)
    return content


def gateway_content(content, site):
    """Add a server rewrite guard before SSL configuration, once per host."""
    host = HOSTS[site]
    if not re.search(r'\bserver_name\s+' + re.escape(host) + r'\s*;', content):
        raise ValueError('Unexpected gateway host')
    include = f'    include /etc/nginx/snippets/website-access-{site}.conf;\n'
    if include.strip() in content:
        return content
    match = re.search(r'^\s*ssl_certificate\s', content, re.MULTILINE)
    if match is None:
        raise ValueError('HTTPS server not found')
    offset = match.start()
    return content[:offset] + '\n' + include + content[offset:]


def main():
    check_only = sys.argv[1:] == ['--check']
    if sys.argv[1:] and not check_only:
        raise SystemExit('Usage: deploy.py [--check]')
    if not check_only and os.geteuid() != 0:
        raise SystemExit(
            'Run with sudo on B to install the gateway integration.',
        )
    if not APP.is_dir() or not RUNTIME.is_file() or not ENV.is_file():
        raise RuntimeError('Expected B production runtime is missing')
    if CONFIG.exists() or OUTPUT.exists():
        raise RuntimeError(
            'Already installed; preserve policies and use the existing agent.',
        )
    for name in NEW_FILES:
        if not (BASE / 'payload' / name).is_file():
            raise RuntimeError('Incomplete runtime bundle')
    for name in ('agent.py', 'snapshot.py', 'migrate.py', 'migration.sql'):
        if not (BASE / name).is_file():
            raise RuntimeError('Incomplete deployment bundle')
    checksums = json.loads((BASE / 'checksums.json').read_text())
    required = {'payload/' + name for name in NEW_FILES}
    required.update({
        'deploy.py', 'agent.py', 'snapshot.py', 'migrate.py',
        'migration.sql', 'README.md',
    })
    if set(checksums) != required:
        raise RuntimeError('Unexpected bundle manifest')
    for name, expected in checksums.items():
        if hashlib.sha256((BASE / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Bundle checksum mismatch: ' + name)
    candidate = {
        APP / name: patched_code((APP / name).read_text(), patches)
        for name, patches in PATCHES.items()
    }
    for site in HOSTS:
        path = Path('/etc/nginx/sites-enabled/phase6-' + site + '.conf')
        candidate[path] = gateway_content(path.read_text(), site)
    if check_only:
        for path, content in candidate.items():
            if str(path).startswith(str(APP)):
                compile(content, str(path), 'exec')
        for name in NEW_FILES:
            compile((BASE / 'payload' / name).read_text(), name, 'exec')
        print(
            'Bundle checksums, runtime patches and three gateway hosts '
            'verified; no production files changed.',
        )
        return
    backup = (
        HOME
        / 'website-access-management'
        / ('production-backup-' + str(time.time_ns()))
    )
    backup.mkdir(parents=True, mode=0o700)
    # Save precisely every destination before changing files. This includes
    # absent files so rollback can distinguish new from pre-existing paths.
    destinations = [
        *candidate,
        *(APP / name for name in NEW_FILES),
        OUTPUT,
        CONFIG,
        Path('/etc/systemd/system/website-access-agent.service'),
    ]
    destinations += [
        Path('/etc/nginx/snippets/website-access-' + site + '.conf')
        for site in HOSTS
    ]
    saved = {}
    for index, path in enumerate(destinations):
        original = backup / str(index)
        if path.exists():
            shutil.copy2(path, original)
            saved[str(path)] = str(original)
        else:
            saved[str(path)] = None
    (backup / 'files.json').write_text(json.dumps(saved, indent=2))
    account = pwd.getpwnam(OWNER)
    try:
        for name in NEW_FILES:
            destination = APP / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(BASE / 'payload' / name, destination)
            os.chown(destination, account.pw_uid, account.pw_gid)
        for path, content in candidate.items():
            if str(path).startswith(str(APP)):
                path.write_text(content)
        reader = [
            '/usr/sbin/runuser',
            '-u',
            OWNER,
            '--',
            str(RUNTIME),
            str(BASE / 'snapshot.py'),
            '--config',
            str(ENV),
            '--app-root',
            str(APP),
        ]
        run(
            [
                '/usr/sbin/runuser',
                '-u',
                OWNER,
                '--',
                RUNTIME,
                BASE / 'migrate.py',
                '--config',
                ENV,
                '--app-root',
                APP,
                '--sql',
                BASE / 'migration.sql',
            ],
        )
        snapshot = json.loads(run(reader))
        if snapshot['hosts'] != HOSTS or any(
            p['restricted'] for p in snapshot['policies']
        ):
            raise RuntimeError(
                'Initial gateway installation requires '
                'all website restrictions disabled',
            )
        sys.path.insert(0, str(BASE))
        from agent import atomic_write, render

        rendered, _versions = render(snapshot)
        INSTALL.mkdir(mode=0o755, exist_ok=True)
        for name in ('agent.py', 'snapshot.py'):
            shutil.copyfile(BASE / name, INSTALL / name)
            os.chmod(INSTALL / name, 0o644)
        atomic_write(OUTPUT, rendered)
        atomic_write(INSTALL / 'last-good.conf', rendered)
        for site in HOSTS:
            atomic_write(
                Path('/etc/nginx/snippets/website-access-' + site + '.conf'),
                f'if ($vn_website_{site}_allowed = 0) {{ return 403; }}\n',
            )
        for path, content in candidate.items():
            if str(path).startswith('/etc/nginx/'):
                path.write_text(content)
        reader[5] = str(INSTALL / 'snapshot.py')
        atomic_write(
            CONFIG,
            json.dumps(
                {
                    'reader': reader,
                    'hosts': HOSTS,
                    'output': str(OUTPUT),
                    'last_good': str(INSTALL / 'last-good.conf'),
                    'status': '/run/website-access/status.json',
                },
                indent=2,
            ),
        )
        atomic_write(
            Path('/etc/systemd/system/website-access-agent.service'),
            '''[Unit]
Description=Synchronize company website IP policies to Nginx
After=network-online.target nginx.service postgresql.service
Wants=network-online.target nginx.service
[Service]
Type=simple
ExecStart=/usr/bin/python3 /opt/website-access/agent.py \\
    --config /etc/website-access.json
Restart=on-failure
RestartSec=5
RuntimeDirectory=website-access
RuntimeDirectoryMode=0755
UMask=0022
NoNewPrivileges=true
[Install]
WantedBy=multi-user.target
''',
        )
        run(['/usr/sbin/nginx', '-t'])
        run(['/usr/bin/systemctl', 'restart', 'phase5-bff-production.service'])
        for attempt in range(40):
            try:
                with urllib.request.urlopen(
                    'http://127.0.0.1:8006/openapi.json', timeout=2,
                ) as response:
                    paths = json.load(response)['paths']
                    if '/admin/website-access' not in paths:
                        raise RuntimeError('Management API was not installed')
                break
            except (OSError, ValueError):
                if attempt == 39:
                    raise RuntimeError(
                        'Management API did not start',
                    ) from None
                time.sleep(1)
        run(['/usr/bin/systemctl', 'reload', 'nginx'])
        run(['/usr/bin/systemctl', 'daemon-reload'])
        run(
            [
                '/usr/bin/systemctl',
                'enable',
                '--now',
                'website-access-agent.service',
            ],
        )
        for _ in range(10):
            if Path('/run/website-access/status.json').exists():
                break
            time.sleep(1)
        else:
            raise RuntimeError('Gateway synchronization did not become ready')
        for host in HOSTS.values():
            run(
                [
                    '/usr/bin/curl',
                    '--noproxy',
                    '*',
                    '--silent',
                    '--show-error',
                    '--fail',
                    '--resolve',
                    host + ':443:127.0.0.1',
                    '--max-time',
                    '15',
                    '--output',
                    '/dev/null',
                    'https://' + host + '/',
                ],
            )
    except Exception:
        subprocess.run(
            [
                '/usr/bin/systemctl',
                'disable',
                '--now',
                'website-access-agent.service',
            ],
            capture_output=True,
        )
        for name, original in saved.items():
            path = Path(name)
            if original:
                shutil.copy2(original, path)
            else:
                path.unlink(missing_ok=True)
        subprocess.run(
            ['/usr/bin/systemctl', 'daemon-reload'], capture_output=True,
        )
        subprocess.run(
            ['/usr/bin/systemctl', 'restart', 'phase5-bff-production.service'],
            capture_output=True,
        )
        subprocess.run(
            ['/usr/bin/systemctl', 'reload', 'nginx'], capture_output=True,
        )
        # Additive database tables remain for inspection; no DROP or data loss.
        raise
    print(
        'Website access management installed; '
        'all three websites remain unrestricted.',
    )
    print(
        'Gateway synchronization and autostart verified. Backup: '
        + str(backup),
    )


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(
            'Deployment stopped: ' + type(exc).__name__ + ': ' + str(exc),
            file=sys.stderr,
        )
        raise SystemExit(1) from None
