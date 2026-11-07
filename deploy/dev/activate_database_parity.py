from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import psycopg
from psycopg import sql

from deploy.dev.database_parity import compare
from deploy.dev.database_parity import dev_options
from deploy.dev.database_parity import production_options
from deploy.dev.database_parity import snapshot
from deploy.dev.run import PORTS
from deploy.dev.run import ROOT
from deploy.dev.run import STATE

COMPOSE = ROOT / 'deploy/dev/compose.yaml'
COMMAND = ['docker', 'compose', '-f', str(COMPOSE)]
UNITS = [f"construction-dev-{name}.service" for name in PORTS]


def run(args, **kwargs):
    subprocess.run(args, check=True, **kwargs)


def fingerprint(options):
    with psycopg.connect(
        **options,
        options='-c default_transaction_read_only=on',
    ) as db:
        db.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ')
        tables = db.execute(
            (
                'SELECT schemaname,tablename FROM pg_tables WHERE '
                "schemaname='public' ORDER BY "
                '2'
            ),
        ).fetchall()
        result = {}
        for schema, name in tables:
            hashes = db.execute(
                sql.SQL(
                    'SELECT md5(to_jsonb(t)::text) FROM {} t ORDER BY 1',
                ).format(sql.Identifier(schema, name)),
            ).fetchall()
            result[name] = {
                'count': len(hashes),
                'sha256': hashlib.sha256(
                    json.dumps(hashes).encode(),
                ).hexdigest(),
            }
        return result


def health():
    deadline = time.monotonic() + 60
    with httpx.Client(trust_env=False, timeout=3) as client:
        while time.monotonic() < deadline:
            try:
                if client.get(
                    'http://127.0.0.1:19005/health',
                ).json() == {'status': True, 'environment': 'dev'} and all(
                    client.get(
                        f"http://127.0.0.1:{port}/openapi.json",
                    ).status_code
                    == 200
                    for port in PORTS.values()
                ):
                    return
            except (httpx.HTTPError, ValueError):
                pass
            time.sleep(1)
    raise RuntimeError(
        'Dev APIs did not become healthy after database activation.',
    )


def main():
    state_path = STATE / 'parity/current.json'
    state = json.loads(state_path.read_text())
    if state.get('active'):
        result = compare(
            snapshot(production_options()),
            snapshot(dev_options()),
        )
        if not result['matched']:
            raise RuntimeError(
                'Active dev schema differs; inspect database_parity first.',
            )
        health()
        print(
            (
                'Verified dev baseline is already active; no volume switch '
                'performed.'
            ),
        )
        return
    result = compare(
        snapshot(production_options()),
        snapshot(dev_options(15434)),
    )
    if not result['matched']:
        raise RuntimeError(
            'Candidate schema/authorization differs from current production.',
        )
    current = json.loads(
        subprocess.check_output(
            ['docker', 'inspect', 'construction-dev-postgres-1'],
            text=True,
        ),
    )[0]
    candidate = json.loads(
        subprocess.check_output(
            ['docker', 'inspect', 'construction-dev-postgres-parity'],
            text=True,
        ),
    )[0]
    candidate_volumes = {x.get('Name') for x in candidate['Mounts']}
    if 'construction-dev_postgres-parity-data' not in candidate_volumes:
        raise RuntimeError('Unexpected candidate volume.')
    image = (
        'pgvector/pgvector@sha256:2fd905ba95f99a51be207d0ff0b8d5b8538cbce7e908'
        '3b3ccfa65b93bb28b938'
    )
    expected_image = subprocess.check_output(
        ['docker', 'image', 'inspect', image, '--format', '{{.Id}}'],
        text=True,
    ).strip()
    if candidate['Image'] != expected_image:
        raise RuntimeError('Unexpected candidate image.')
    text = COMPOSE.read_text()
    if (
        image not in text
        or 'volumes: ["postgres-parity-data:/var/lib/postgresql"]' not in text
    ):
        raise RuntimeError(
            'Compose is not configured for the verified candidate.',
        )
    backup = Path(state['directory'])
    rollback = text.replace(image, current['Config']['Image']).replace(
        'volumes: ["postgres-parity-data:/var/lib/postgresql"]',
        'volumes: ["postgres-data:/var/lib/postgresql"]',
    )
    (backup / 'rollback-compose.yaml').write_text(rollback)
    (backup / 'rollback-compose.yaml').chmod(0o600)
    run(['systemctl', '--user', 'stop', *UNITS])
    changed = False
    try:
        if fingerprint(dev_options()) != state['source_fingerprints']:
            raise RuntimeError(
                (
                    'Dev data changed since candidate preparation; rebuild '
                    'candidate before '
                    'switching.'
                ),
            )
        options = dev_options()
        dump = subprocess.run(
            [
                '/usr/lib/postgresql/18/bin/pg_dump',
                '--host',
                options['host'],
                '--port',
                str(
                    options['port'],
                ),
                '--username',
                options['user'],
                '--dbname',
                options['dbname'],
                '--no-owner',
                '--no-acl',
                '--format=custom',
                '--file',
                str(backup / 'dev-frozen.dump'),
            ],
            env={
                **os.environ,
                'PGPASSWORD': options['password'],
                'PGOPTIONS': '-c default_transaction_read_only=on',
            },
            capture_output=True,
        )
        if dump.returncode:
            raise RuntimeError('Frozen dev backup failed.')
        (backup / 'dev-frozen.dump').chmod(0o600)
        run(
            ['docker', 'stop', 'construction-dev-postgres-parity'],
            stdout=subprocess.DEVNULL,
        )
        changed = True
        run(
            [
                *COMMAND,
                'up',
                '-d',
                '--wait',
                '--wait-timeout',
                '90',
                'postgres',
            ],
        )
        run(['systemctl', '--user', 'start', *UNITS])
        health()
        result = compare(
            snapshot(production_options()),
            snapshot(dev_options()),
        )
        if not result['matched']:
            raise RuntimeError(
                'Dev application startup changed the canonical schema: '
                + ','.join(result['differences']),
            )
        run([sys.executable, '-m', 'deploy.dev.smoke'], cwd=ROOT)
        state.update(
            active=True,
            previous_volume='construction-dev_postgres-data',
            active_result=result,
            schema_archive_sha256=hashlib.sha256(
                (backup / 'production-schema.dump').read_bytes(),
            ).hexdigest(),
        )
        state_path.write_text(json.dumps(state, indent=2))
        state_path.chmod(0o600)
        print(
            (
                'Dev canonical schema, retained test records and matching '
                'feature grants are active. Old dev volume '
                'retained.'
            ),
        )
    except BaseException:
        if changed:
            COMPOSE.write_text(rollback)
            run(
                [
                    *COMMAND,
                    'up',
                    '-d',
                    '--wait',
                    '--wait-timeout',
                    '90',
                    'postgres',
                ],
            )
        run(['systemctl', '--user', 'start', *UNITS])
        raise


if __name__ == '__main__':
    main()
