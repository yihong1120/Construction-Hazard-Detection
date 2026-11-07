from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from deploy.migrations import backup as module
from deploy.migrations.backup import BackupError
from deploy.migrations.backup import connection_arguments
from deploy.migrations.backup import DatabaseBackup
from deploy.migrations.run import MigrationError
from deploy.migrations.run import MigrationRunner
from tests.postgres_fixtures_test import DATABASE
from tests.postgres_fixtures_test import IsolatedPostgresDatabase


async def _actual_restore_uses_one_snapshot_inspect(
    counts,
    observed,
    /,
    db,
):
    result = await counts(db)
    if (await db.fetchval('SELECT current_database()')).startswith(
        'chd_restore_',
    ):
        observed.extend(
            await db.fetch('SELECT body FROM violations ORDER BY id'),
        )
    return result


async def _actual_restore_uses_one_snapshot_run(
    command,
    testcase,
    /,
    name,
    arguments,
    env,
):
    if name == 'dump':
        await testcase.control.execute(
            'SELECT 1',
        )
        import asyncpg

        writer = await asyncpg.connect(testcase.url)
        try:
            await writer.execute(
                'INSERT INTO violations(site_id,body) '
                "VALUES(1,'later evidence')",
            )
        finally:
            await writer.close()
    await command(name, arguments, env)


def _failed_restore_blocks_migration_and_record_name(
    created_names,
    database_url,
    /,
    url,
    name,
):
    created_names.append(name)
    return database_url(url, name)


async def _failed_restore_blocks_migration_and_wrong_counts(
    verify,
    /,
    db,
):
    actual = await verify(db)
    if (await db.fetchval('SELECT current_database()')).startswith(
        'chd_restore_',
    ):
        return {}
    return actual


async def _migration_lock_is_held_through_backup(
    finish,
    started,
    /,
    db,
):
    started.set()
    await finish.wait()


class _CancellationDuringDatabaseCreationStillCleansItCallbacks:
    operations: Any
    release: Any
    started: Any

    async def _call_execute(self, sql):
        self.operations.append(sql)
        if sql.startswith('CREATE DATABASE'):
            self.started.set()
            await self.release.wait()


class BackupCredentialsTest(unittest.TestCase):
    def test_passwords_are_only_in_the_subprocess_environment(self):
        safe, env = connection_arguments(
            'postgresql://operator:authority-secret@localhost/test'
            '?password=query-secret&sslpassword=key-secret'
            '&host=%2Ftmp%2Fpgsocket',
        )
        self.assertNotIn('secret', safe)
        self.assertEqual(env['PGPASSWORD'], 'query-secret')
        self.assertEqual(env['PGSSLPASSWORD'], 'key-secret')
        self.assertIn('host=%2Ftmp%2Fpgsocket', safe)


class DatabaseBackupTest(IsolatedPostgresDatabase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        await self.db.execute('''
            CREATE TABLE sites(id SERIAL PRIMARY KEY, name TEXT UNIQUE);
            CREATE TABLE violations(id SERIAL PRIMARY KEY,
                site_id INTEGER REFERENCES sites(id), body TEXT NOT NULL);
            CREATE TABLE notifications(id SERIAL PRIMARY KEY,
                violation_id INTEGER REFERENCES violations(id));
            CREATE INDEX evidence_body_idx ON violations(body);
            INSERT INTO sites(name) VALUES ('original');
            INSERT INTO violations(site_id, body)
                VALUES (1,'original evidence');
            INSERT INTO notifications(violation_id) VALUES (1);
        ''')
        self.backup = DatabaseBackup(self.url, self.path / 'backup', DATABASE)

    async def test_actual_restore_uses_one_snapshot_and_preserves_evidence(
        self,
    ):
        counts = module._row_counts
        observed = []

        inspect = partial(
            _actual_restore_uses_one_snapshot_inspect,
            counts,
            observed,
        )

        command = self.backup._command

        run = partial(_actual_restore_uses_one_snapshot_run, command, self)

        with patch.object(module, '_row_counts', inspect), patch.object(
            DatabaseBackup,
            '_command',
            side_effect=run,
        ):
            receipt = await self.backup.create(self.db)
        self.assertTrue(receipt['restore_verified'])
        self.assertEqual(receipt['tables']['"public"."violations"'], 1)
        self.assertEqual(
            [row['body'] for row in observed],
            ['original evidence'],
        )
        self.assertEqual(
            await self.db.fetchval('SELECT count(*) FROM violations'),
            2,
        )
        self.assertEqual(self.backup.directory.stat().st_mode & 0o777, 0o700)
        self.assertEqual(
            (self.backup.directory / 'database.dump').stat().st_mode & 0o777,
            0o600,
        )
        self.assertTrue(
            json.loads((self.backup.directory / 'backup.json').read_text())[
                'restore_verified'
            ],
        )

    def runner(self):
        source = self.path / 'migration.sql'
        source.write_text(
            'BEGIN;\n'
            'ALTER TABLE violations ADD COLUMN applied BOOLEAN;\n'
            'COMMIT;\n',
        )
        manifest = self.path / 'manifest.json'
        manifest.write_text(
            json.dumps(
                [
                    {
                        'file': source.name,
                        'verify': 'SELECT EXISTS(SELECT 1 FROM pg_attribute '
                        "WHERE attrelid='violations'::regclass "
                        "AND attname='applied')",
                    },
                ],
            ),
        )
        return MigrationRunner(self.db, self.database, manifest)

    async def test_failed_restore_blocks_migration_and_cleans_scratch(
        self,
    ):
        runner = self.runner()
        created_names = []
        database_url = module._database_url

        record_name = partial(
            _failed_restore_blocks_migration_and_record_name,
            created_names,
            database_url,
        )

        verify = module._row_counts

        wrong_counts = partial(
            _failed_restore_blocks_migration_and_wrong_counts,
            verify,
        )

        with patch.object(module, '_row_counts', wrong_counts), patch.object(
            module,
            '_database_url',
            record_name,
        ):
            with self.assertRaisesRegex(BackupError, 'counts differ'):
                await runner.apply(self.backup)
        self.assertIsNone(
            await self.db.fetchval("SELECT to_regclass('schema_migrations')"),
        )
        self.assertEqual(
            await self.db.fetchval(
                'SELECT count(*) FROM pg_attribute '
                "WHERE attrelid='violations'::regclass "
                "AND attname='applied'",
            ),
            0,
        )
        self.assertEqual(len(created_names), 1)
        for name in created_names:
            self.assertEqual(
                await self.control.fetchval(
                    'SELECT count(*) FROM pg_database WHERE datname=$1',
                    name,
                ),
                0,
            )
        self.assertFalse((self.backup.directory / 'backup.json').exists())

    async def test_migration_lock_is_held_through_backup(self):
        import asyncpg

        runner = self.runner()
        started, finish = asyncio.Event(), asyncio.Event()

        backup = partial(
            _migration_lock_is_held_through_backup,
            finish,
            started,
        )

        other = await asyncpg.connect(self.url)
        try:
            with patch.object(
                DatabaseBackup,
                'create',
                AsyncMock(side_effect=backup),
            ):
                task = asyncio.create_task(runner.apply(self.backup))
                await started.wait()
                try:
                    with self.assertRaisesRegex(
                        MigrationError,
                        'Another evidence migration',
                    ):
                        await MigrationRunner(
                            other,
                            self.database,
                            self.path / 'manifest.json',
                        ).apply()
                finally:
                    finish.set()
                    await task
        finally:
            await other.close()


class BackupLifetimeTest(unittest.IsolatedAsyncioTestCase):
    async def test_cancellation_during_database_creation_still_cleans_it(self):
        _callbacks = (
            _CancellationDuringDatabaseCreationStillCleansItCallbacks()
        )
        control = MagicMock()
        _callbacks.started, _callbacks.release = (
            asyncio.Event(),
            asyncio.Event(),
        )
        _callbacks.operations = []

        execute = _callbacks._call_execute

        control.execute = AsyncMock(side_effect=execute)
        control.close = AsyncMock()
        backup = DatabaseBackup(
            'postgresql://source/db',
            Path('/unused'),
            'postgresql://verifier/db',
        )
        with patch.object(
            module.asyncpg,
            'connect',
            AsyncMock(return_value=control),
        ):
            task = asyncio.create_task(
                backup._verify(Path('/unused/archive'), {}),
            )
            await _callbacks.started.wait()
            task.cancel()
            await asyncio.sleep(0)
            _callbacks.release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(len(_callbacks.operations), 2)
        created = _callbacks.operations[0].split('"')[1]
        self.assertTrue(created.startswith('chd_restore_'))
        self.assertIn(
            'DROP DATABASE "' + created + '"',
            _callbacks.operations[1],
        )
        control.close.assert_awaited_once()
