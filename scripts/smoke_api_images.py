"""Boot API images with isolated databases and no application-source mounts."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from uuid import uuid4


def docker(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ['docker', *args], check=check, capture_output=True, text=True,
        timeout=120,
    )


def wait_ready(container: str, command: list[str], seconds: int = 60) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if docker('exec', container, *command, check=False).returncode == 0:
            return
        running = docker('inspect', '--format={{.State.Running}}', container)
        if running.stdout.strip() != 'true':
            raise RuntimeError(f'{container} exited during startup')
        time.sleep(0.5)
    raise RuntimeError(f'{container} did not become ready within {seconds}s')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--management-image', required=True)
    parser.add_argument('--violation-image', required=True)
    parser.add_argument('--postgres-image', default='postgres:17.11-bookworm')
    parser.add_argument('--redis-image', default='redis:7.4.10-alpine3.21')
    args = parser.parse_args()
    prefix = f'construction-smoke-{uuid4().hex[:10]}'
    network = prefix
    containers: list[str] = []
    network_created = False
    try:
        docker('network', 'create', network)
        network_created = True
        postgres = f'{prefix}-postgres'
        containers.append(postgres)
        docker(
            'run',
            '-d',
            '--name',
            postgres,
            '--network',
            network,
            '-e',
            'POSTGRES_DB=smoke',
            '-e',
            'POSTGRES_USER=smoke',
            '-e',
            'POSTGRES_PASSWORD=isolated-smoke-only',
            args.postgres_image,
        )
        redis = f'{prefix}-redis'
        containers.append(redis)
        docker(
            'run',
            '-d',
            '--name',
            redis,
            '--network',
            network,
            args.redis_image,
        )
        # PostgreSQL's entrypoint uses a temporary Unix-socket server while
        # creating the database. Wait for the final TCP listener instead.
        wait_ready(
            postgres, [
                'pg_isready', '-h', '127.0.0.1', '-U', 'smoke', '-d', 'smoke',
            ],
        )
        wait_ready(redis, ['redis-cli', 'ping'])
        docker(
            'exec', postgres, 'psql', '-X', '-v', 'ON_ERROR_STOP=1',
            '-U', 'smoke', '-d', 'smoke',
            '-c', 'CREATE EXTENSION IF NOT EXISTS pg_trgm;',
        )
        for label, image, port in (
            ('management', args.management_image, 8005),
            ('violation', args.violation_image, 8081),
        ):
            container = f'{prefix}-{label}'
            containers.append(container)
            docker(
                'run',
                '-d',
                '--name',
                container,
                '--network',
                network,
                '-e',
                'DATABASE_URL=postgresql+asyncpg://smoke:'
                f'isolated-smoke-only@{postgres}/smoke',
                '-e',
                f'REDIS_HOST={redis}',
                '-e',
                'REDIS_PORT=6379',
                '-e',
                'REDIS_PASSWORD=',
                '-e',
                'AUTO_CREATE_SCHEMA=true',
                '-e',
                'OIDC_ENABLED=true',
                '-e',
                'OIDC_ISSUER_URL=https://oidc.smoke.invalid/realms/smoke',
                '-e',
                'OIDC_JWKS_URL=https://oidc.smoke.invalid/realms/smoke/certs',
                '-e',
                'OIDC_AUDIENCE=smoke-api',
                '-e',
                'OIDC_ACCOUNT_URL='
                'https://oidc.smoke.invalid/realms/smoke/account',
                '-e',
                'BFF_TOKEN_ENCRYPTION_KEY=isolated-smoke-only',
                image,
            )
            probe = (
                'import json, urllib.request; '
                'r=urllib.request.urlopen('
                f'"http://127.0.0.1:{port}/openapi.json", timeout=2); '
                'assert r.status == 200; assert json.load(r)["paths"]'
            )
            wait_ready(container, ['python', '-c', probe])
            mounts = json.loads(
                docker(
                    'inspect',
                    '--format={{json .Mounts}}',
                    container,
                ).stdout,
            )
            if mounts:
                raise RuntimeError(
                    'API smoke containers must not use source mounts',
                )
            check_http = '''import asyncio
from src.http_client_pool import HttpClientPool
async def check():
    pool = HttpClientPool()
    await pool.get('smoke', timeout=2.0)
    await pool.close()
asyncio.run(check())
'''
            docker('exec', container, 'python', '-c', check_http)
            print(
                f'{label}: ASGI startup, OpenAPI and HTTP/2 client OK; '
                'no mounts',
                flush=True,
            )
            # Avoid concurrent fixture schema creation by the next service.
            docker('stop', '--time=10', container)
    except (RuntimeError, subprocess.SubprocessError):
        for container in containers:
            logs = docker('logs', '--tail=30', container, check=False)
            if logs.returncode == 0:
                print(f'{container}:\n{logs.stdout}{logs.stderr}')
        raise
    finally:
        for container in reversed(containers):
            docker('rm', '-f', '-v', container, check=False)
        if network_created:
            docker('network', 'rm', network, check=False)


if __name__ == '__main__':
    main()
