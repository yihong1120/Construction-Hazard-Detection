from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
from pathlib import Path
from urllib.parse import urlsplit

import asyncpg

from deploy.migrations.backup import BackupError
from deploy.migrations.backup import DatabaseBackup

ROOT = Path(__file__).resolve().parent
LOCK_NAME = 'construction.evidence_identity.20261030'


class MigrationError(RuntimeError):
    pass


class MigrationRunner:

    def __init__(
        self,
        connection: asyncpg.Connection,
        expected_database: str,
        manifest: Path = ROOT / 'manifest.json',
    ) -> None:
        self.db, self.expected_database = (connection, expected_database)
        self.directory = manifest.parent
        self.steps = json.loads(manifest.read_text())
        self.max_batches = 10000

    async def _target(self) -> None:
        if (
            await self.db.fetchval('SELECT current_database()')
            != self.expected_database
        ):
            raise MigrationError(
                'Connected database differs from --expected-database',
            )
        if not all(
            [
                await self.db.fetchval(
                    'SELECT to_regclass($1) IS NOT NULL',
                    name,
                )
                for name in ('sites', 'violations', 'notifications')
            ],
        ):
            raise MigrationError(
                (
                    'Application tables are missing from the '
                    'configured search_path'
                ),
            )

    def _source(self, step: dict) -> tuple[str, str, str]:
        path = (self.directory / step['file']).resolve()
        if path.parent != self.directory.resolve() or path.suffix != '.sql':
            raise MigrationError('Unexpected migration path')
        source = path.read_text()
        return (path.stem, hashlib.sha256(source.encode()).hexdigest(), source)

    async def plan(self) -> list[dict]:
        await self._target()
        versions = {}
        if await self.db.fetchval(
            "SELECT to_regclass('schema_migrations') IS NOT NULL",
        ):
            versions = {
                row['version']: row['checksum'].strip()
                for row in await self.db.fetch(
                    'SELECT version,checksum FROM schema_migrations',
                )
            }
        result = []
        for step in self.steps:
            version, checksum, _ = self._source(step)
            applied = version in versions
            if applied and versions[version] != checksum:
                raise MigrationError(
                    'Applied migration checksum differs: ' + version,
                )
            if applied and (not await self.db.fetchval(step['verify'])):
                raise MigrationError(
                    'Applied migration postcondition differs: ' + version,
                )
            result.append(
                {'version': version, 'checksum': checksum, 'applied': applied},
            )
        return result

    @staticmethod
    def _body(source: str) -> str:
        source, count = re.subn(
            '^BEGIN;\\s*$',
            '',
            source,
            count=1,
            flags=re.M,
        )
        if count != 1:
            raise MigrationError(
                'Migration requires one explicit outer transaction',
            )
        source, count = re.subn(
            '^COMMIT;\\s*$',
            '',
            source,
            count=1,
            flags=re.M,
        )
        if count != 1:
            raise MigrationError(
                'Migration requires one explicit outer commit',
            )
        return source

    async def apply(self, backup: DatabaseBackup | None = None) -> list[dict]:
        await self._target()
        schema = await self.db.fetchval('SELECT current_schema()')
        lock = LOCK_NAME if schema == 'public' else LOCK_NAME + ':' + schema
        if not await self.db.fetchval(
            'SELECT pg_try_advisory_lock(hashtextextended($1,0))',
            lock,
        ):
            raise MigrationError('Another evidence migration is running')
        try:
            plan = await self.plan()
            if backup is not None:
                await backup.create(self.db)
            await self.db.execute(
                (
                    'CREATE TABLE IF NOT EXISTS schema_migrations ( '
                    'version VARCHAR(255) PRIMARY KEY, checksum CHAR(64) '
                    'NOT NULL, applied_at TIMESTAMPTZ NOT NULL DEFAULT '
                    'now() )'
                ),
            )
            for step, state in zip(self.steps, plan, strict=True):
                if state['applied']:
                    continue
                version, checksum, source = self._source(step)
                previous = (
                    await self.db.fetchval(step['remaining'])
                    if 'remaining' in step
                    else None
                )
                for _ in range(self.max_batches):
                    async with self.db.transaction():
                        await self.db.execute(self._body(source))
                        complete = bool(await self.db.fetchval(step['verify']))
                        if not complete and 'remaining' not in step:
                            raise MigrationError(
                                'Postcondition failed: ' + version,
                            )
                        if complete:
                            await self.db.execute(
                                (
                                    'INSERT INTO '
                                    'schema_migrations(version,checksum) '
                                    'VALUES($1,$2)'
                                ),
                                version,
                                checksum,
                            )
                    if complete:
                        break
                    remaining = await self.db.fetchval(step['remaining'])
                    if previous is not None and remaining >= previous:
                        raise MigrationError(
                            (
                                'Backfill made no progress; '
                                'inspect unmatched or '
                                'locked evidence'
                            ),
                        )
                    previous = remaining
                    await asyncio.sleep(0)
                else:
                    raise MigrationError(
                        'Backfill batch limit reached; rerun to continue',
                    )
            return await self.plan()
        finally:
            await self.db.execute(
                'SELECT pg_advisory_unlock(hashtextextended($1,0))',
                lock,
            )


async def run(args: argparse.Namespace) -> None:
    os.umask(63)
    values = os.environ
    if args.config:
        config = json.loads(args.config.read_text())
        values = config.get('environment', config)
    url = values.get('DATABASE_URL', '').replace(
        'postgresql+asyncpg://',
        'postgresql://',
        1,
    )
    if urlsplit(url).scheme not in ('postgresql', 'postgres'):
        raise MigrationError('Configure a PostgreSQL DATABASE_URL')
    db = await asyncpg.connect(
        url,
        timeout=10,
        server_settings={
            'application_name': 'evidence-migration-runner',
            'statement_timeout': '120000',
            'lock_timeout': '5000',
        },
    )
    try:
        backup = None
        if args.backup:
            if not args.apply or args.backup_verifier_config is None:
                raise MigrationError(
                    '--backup requires --apply and --backup-verifier-config',
                )
            verifier = json.loads(args.backup_verifier_config.read_text())
            verifier_values = verifier.get('environment', verifier)
            verifier_url = verifier_values.get('DATABASE_URL', '').replace(
                'postgresql+asyncpg://', 'postgresql://', 1,
            )
            if urlsplit(verifier_url).scheme not in ('postgresql', 'postgres'):
                raise MigrationError(
                    'Configure the isolated verifier PostgreSQL DATABASE_URL',
                )
            backup = DatabaseBackup(url, args.backup, verifier_url)
        runner = MigrationRunner(db, args.expected_database)
        if args.schema:
            if any(
                not re.fullmatch('[A-Za-z_][A-Za-z0-9_]*', s)
                for s in args.schema
            ):
                raise MigrationError('Invalid schema name')
            await db.execute(
                'SET search_path TO '
                + ','.join('"' + s + '"' for s in args.schema),
            )
        await runner._target()
        result = (
            await runner.apply(backup) if args.apply else await runner.plan()
        )
        print(
            json.dumps(
                {'database': args.expected_database, 'migrations': result},
            ),
        )
        if args.check and any(not s['applied'] for s in result):
            raise SystemExit(1)
    finally:
        await db.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Plan, verify, or apply the evidence identity migrations',
    )
    parser.add_argument('--expected-database', required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--schema', action='append')
    parser.add_argument('--backup', type=Path)
    parser.add_argument('--backup-verifier-config', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--apply', action='store_true')
    mode.add_argument('--check', action='store_true')
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except (MigrationError, BackupError, asyncpg.PostgresError) as error:
        parser.exit(
            1,
            (
                'Migration failed: ' + str(error)
                if isinstance(
                    error,
                    (MigrationError, BackupError),
                )
                else 'Migration failed: ' + type(error).__name__ + '\n'
            ),
        )


if __name__ == '__main__':
    main()
