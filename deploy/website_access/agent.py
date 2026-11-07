from __future__ import annotations

import argparse
import ipaddress
import json
import os
import subprocess
import tempfile
import time
from datetime import datetime
from datetime import timezone
from operator import itemgetter
from pathlib import Path

SITES = ('main', 'ai', 'wiki')


def render(snapshot: dict) -> tuple[str, dict[str, int]]:
    """Validate the entire catalog before generating any privileged config."""
    rows = snapshot['policies']
    if len(rows) != 3 or {r['site_key'] for r in rows} != set(SITES):
        raise ValueError('Incomplete website catalog')
    lines = ['# Managed by website-access-agent; numeric IP networks only.']
    versions = {}
    for row in sorted(rows, key=itemgetter('site_key')):
        site = row['site_key']
        if (
            type(row['restricted']) is not bool
            or type(row['version']) is not int
            or row['version'] < 0
        ):
            raise ValueError('Invalid policy state')
        versions[site] = row['version']
        rules = row['rules']
        if len(rules) > 256:
            raise ValueError('Too many networks')
        networks = []
        for rule in rules:
            if type(rule['enabled']) is not bool:
                raise ValueError('Invalid rule state')
            network = ipaddress.ip_network(rule['network'], strict=True)
            if network.prefixlen == 0 or '%' in rule['network']:
                raise ValueError('Invalid network')
            if network.version == 6 and network.overlaps(
                ipaddress.ip_network('::ffff:0:0/96'),
            ):
                raise ValueError('Mapped IPv6 network')
            if rule['enabled']:
                networks.append(str(network))
        if row['restricted'] and not networks:
            raise ValueError('Empty active allowlist')
        lines.extend(
            [
                f'geo $vn_website_{site}_allowed {{',
                '    default ' + ('0;' if row['restricted'] else '1;'),
            ],
        )
        if row['restricted']:
            lines.extend(
                '    ' + network + ' 1;' for network in sorted(set(networks))
            )
        lines.append('}')
    return '\n'.join(lines) + '\n', versions


def atomic_write(path: Path, content: str):
    """Replace a single generated file without publishing a partial policy."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(name, 0o644)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def apply_once(config: dict):
    """Keep the previous working config on reader, syntax or reload failure."""
    result = subprocess.run(
        config['reader'],
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    data = json.loads(result.stdout)
    if data['hosts'] != config['hosts']:
        raise ValueError('Policy export belongs to another deployment')
    rendered, versions = render(data)
    target = Path(config['output'])
    previous = Path(config['last_good']).read_text()
    # Include versions even for label-only changes so the applied receipt can
    # acknowledge the exact saved revision without always reloading Nginx.
    if rendered != previous or target.read_text() != rendered:
        atomic_write(target, rendered)
        try:
            subprocess.run(
                ['/usr/sbin/nginx', '-t'],
                check=True,
                capture_output=True,
                timeout=10,
            )
            subprocess.run(
                ['/usr/bin/systemctl', 'reload', 'nginx'],
                check=True,
                capture_output=True,
                timeout=15,
            )
        except Exception:
            atomic_write(target, previous)
            # A reload may have been accepted just before the process timed
            # out. Explicitly restore the last known good runtime config too.
            subprocess.run(
                ['/usr/bin/systemctl', 'reload', 'nginx'],
                check=True,
                capture_output=True,
                timeout=15,
            )
            raise
        atomic_write(Path(config['last_good']), rendered)
    atomic_write(
        Path(config['status']),
        json.dumps(
            {
                'checked_at': datetime.now(timezone.utc).isoformat(),
                'versions': versions,
                'hosts': config['hosts'],
            },
        )
        + '\n',
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    while True:
        try:
            apply_once(config)
        except Exception as exc:
            print(
                'Website access sync failed: ' + type(exc).__name__,
                flush=True,
            )
            if args.once:
                raise SystemExit(1) from None
        if args.once:
            return
        time.sleep(5)


if __name__ == '__main__':
    main()
