from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from urllib.parse import parse_qsl
from urllib.parse import unquote
from urllib.parse import urlencode
from urllib.parse import urlsplit
from urllib.parse import urlunsplit
from uuid import uuid4

import asyncpg


class BackupError(RuntimeError):
    pass


def connection_arguments(url: str) -> tuple[str, dict[str, str]]:
    parsed = urlsplit(url)
    parameters = dict(parse_qsl(parsed.query, keep_blank_values=True))
    password = parameters.pop('password', unquote(parsed.password or ''))
    ssl_password = parameters.pop('sslpassword', None)
    authority = parsed.netloc.rsplit('@', 1)[-1]
    if parsed.username:
        authority = parsed.username + '@' + authority
    safe = urlunsplit(
        (parsed.scheme, authority, parsed.path, urlencode(parameters), ''),
    )
    env = {**os.environ, 'PGPASSWORD': password}
    if ssl_password is not None:
        env['PGSSLPASSWORD'] = ssl_password
    return safe, env


def _database_url(url: str, database: str) -> str:
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.pop('dbname', None)
    return urlunsplit(
        (parts.scheme, parts.netloc, '/' + database, urlencode(query), ''),
    )


def _identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


async def _row_counts(db: asyncpg.Connection) -> dict[str, int]:
    tables = await db.fetch("""
        SELECT schemaname, tablename FROM pg_tables
        WHERE schemaname NOT LIKE 'pg_%'
          AND schemaname <> 'information_schema'
        ORDER BY schemaname, tablename
    """)
    counts = {}
    for schema, table in tables:
        qualified = _identifier(schema) + '.' + _identifier(table)
        counts[qualified] = await db.fetchval(
            'SELECT count(*) FROM ' + qualified,
        )
    return counts


@dataclass(frozen=True)
class DatabaseBackup:
    source_url: str = field(repr=False)
    directory: Path
    verifier_url: str = field(repr=False)

    async def _command(
        self,
        name: str,
        arguments: list[str],
        env: dict[str, str],
    ) -> None:
        executable = shutil.which(arguments[0])
        if executable is None:
            raise BackupError('Install matching PostgreSQL client tools')
        process = await asyncio.create_subprocess_exec(
            executable,
            *arguments[1:],
            env=env,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            async with asyncio.timeout(900):
                _, error = await process.communicate()
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        if process.returncode:
            path = self.directory / (name + '.log')
            path.write_bytes(error)
            path.chmod(0o600)
            raise BackupError(name + ' failed; inspect its private log')

    async def create(self, db: asyncpg.Connection) -> dict:
        self.directory.mkdir(parents=True, mode=0o700, exist_ok=False)
        archive = self.directory / 'database.dump'
        source, env = connection_arguments(self.source_url)
        env['PGOPTIONS'] = '-c default_transaction_read_only=on'
        async with db.transaction(isolation='repeatable_read', readonly=True):
            snapshot = await db.fetchval('SELECT pg_export_snapshot()')
            counts = await _row_counts(db)
            await self._command(
                'dump',
                [
                    'pg_dump',
                    '--dbname',
                    source,
                    '--no-password',
                    '--format=custom',
                    '--no-owner',
                    '--no-privileges',
                    '--snapshot',
                    snapshot,
                    '--lock-wait-timeout=5s',
                    '--file',
                    str(archive),
                ],
                env,
            )
        archive.chmod(0o600)
        await self._verify(archive, counts)
        with archive.open('rb') as stream:
            checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
        receipt = {
            'file': archive.name,
            'bytes': archive.stat().st_size,
            'sha256': checksum,
            'tables': counts,
            'restore_verified': True,
        }
        path = self.directory / 'backup.json'
        path.write_text(json.dumps(receipt, indent=2))
        path.chmod(0o600)
        return receipt

    async def _verify(self, archive: Path, expected: dict[str, int]) -> None:
        # Only a newly created random database is ever restored or removed.
        name = 'chd_restore_' + uuid4().hex
        control = await asyncpg.connect(self.verifier_url, timeout=10)
        created = False
        restored = None
        try:
            create = asyncio.create_task(
                control.execute(
                    'CREATE DATABASE '
                    + _identifier(name)
                    + ' TEMPLATE template0',
                ),
            )
            try:
                await asyncio.shield(create)
            except asyncio.CancelledError:
                await create
                created = True
                raise
            created = True
            target = _database_url(self.verifier_url, name)
            safe, env = connection_arguments(target)
            await self._command(
                'restore',
                [
                    'pg_restore',
                    '--dbname',
                    safe,
                    '--no-password',
                    '--no-owner',
                    '--no-privileges',
                    '--exit-on-error',
                    str(archive),
                ],
                env,
            )
            restored = await asyncpg.connect(target, timeout=10)
            if await _row_counts(restored) != expected:
                raise BackupError(
                    'Restored table counts differ from the exported snapshot',
                )
            invalid = await restored.fetchval("""
                SELECT count(*) FROM pg_index i
                JOIN pg_class c ON c.oid=i.indexrelid
                JOIN pg_namespace n ON n.oid=c.relnamespace
                WHERE n.nspname NOT LIKE 'pg_%' AND NOT i.indisvalid
            """)
            if invalid:
                raise BackupError('Restored database contains invalid indexes')
        finally:
            if restored is not None:
                await restored.close()
            try:
                if created:
                    await control.execute(
                        'DROP DATABASE ' + _identifier(name) + ' WITH (FORCE)',
                    )
            finally:
                await control.close()
