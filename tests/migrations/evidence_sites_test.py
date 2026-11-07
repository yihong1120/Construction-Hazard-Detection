from __future__ import annotations

import os
import unittest
from pathlib import Path
from uuid import uuid4

import asyncpg
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import dialect

from examples.auth.models import Violation
from examples.violation_records.site_scope import violation_site_scope

ROOT = Path(__file__).resolve().parents[2]
DATABASE = os.getenv('EVIDENCE_TEST_DATABASE_URL')


@unittest.skipUnless(DATABASE, 'Requires the isolated migration test database')
class EvidenceMigrationDatabase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.db = await asyncpg.connect(DATABASE)
        self.assertEqual(
            await self.db.fetchval('SELECT current_database()'),
            'chd_refactor_test',
        )
        self.extra_schemas: list[str] = []
        self.schema = 'migration_' + uuid4().hex
        await self.db.execute(f'CREATE SCHEMA {self.schema}')
        await self.db.execute(f'SET search_path TO {self.schema}')
        await self.db.execute('''
            CREATE TABLE sites(
                id INTEGER PRIMARY KEY, name VARCHAR(80) UNIQUE NOT NULL
            );
            CREATE TABLE violations(
                id INTEGER PRIMARY KEY,
                site VARCHAR(80) NOT NULL REFERENCES sites(name)
                    ON DELETE CASCADE,
                detection_time TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            CREATE TABLE notifications(
                id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL
            );
            INSERT INTO sites VALUES (1, 'original');
            INSERT INTO violations(id, site)
                SELECT id, 'original' FROM generate_series(1, 5001) id;
            INSERT INTO notifications VALUES (1, 1), (2, 1);
        ''')

    async def asyncTearDown(self) -> None:
        await self.db.execute('ROLLBACK')
        await self.db.execute('SET search_path TO public')
        for schema in self.extra_schemas:
            await self.db.execute(f'DROP SCHEMA {schema} CASCADE')
        await self.db.execute(f'DROP SCHEMA {self.schema} CASCADE')
        await self.db.close()

    async def apply(self, suffix: str) -> None:
        path = next(
            (ROOT / 'deploy/migrations').glob('20261030_' + suffix + '*.sql'),
        )
        await self.db.execute(path.read_text())


@unittest.skipUnless(DATABASE, 'Requires the isolated migration test database')
class EvidenceSiteMigrationTest(EvidenceMigrationDatabase):
    async def test_repeatable_migration_preserves_snapshot(
        self,
    ) -> None:
        await self.apply('01')
        await self.apply('01')
        with self.assertRaisesRegex(asyncpg.RaiseError, 'Backfill incomplete'):
            await self.apply('03')
        await self.db.execute('ROLLBACK')
        await self.apply('02')
        self.assertEqual(
            await self.db.fetchval(
                'SELECT count(*) FROM violations WHERE site_id IS NULL',
            ),
            1,
        )
        await self.apply('02')
        await self.apply('02')
        await self.apply('03')
        await self.apply('03')
        await self.db.execute("UPDATE sites SET name='renamed' WHERE id=1")
        await self.db.execute("INSERT INTO sites VALUES (2, 'original')")
        self.assertEqual(
            await self.db.fetchval('SELECT site FROM violations LIMIT 1'),
            'original',
        )
        for name, expected in [('renamed', 5001), ('original', 0)]:
            statement = select(Violation.id).where(
                violation_site_scope([name]),
            )
            sql = str(
                statement.compile(
                    dialect=dialect(),
                    compile_kwargs={'literal_binds': True},
                ),
            )
            self.assertEqual(len(await self.db.fetch(sql)), expected)
        await self.db.execute('DELETE FROM sites WHERE id=1')
        self.assertEqual(
            await self.db.fetchval('SELECT count(*) FROM violations'),
            0,
        )

    async def test_event_indexes_allow_legacy_nulls_and_reject_duplicates(
        self,
    ) -> None:
        await self.apply('01')
        await self.db.execute(
            "UPDATE violations SET event_id='event-1' WHERE id=1",
        )
        with self.assertRaises(asyncpg.UniqueViolationError):
            await self.db.execute(
                "UPDATE violations SET event_id='event-1' WHERE id=2",
            )
        await self.db.execute(
            "UPDATE notifications SET event_key='event-1' WHERE id=1",
        )
        with self.assertRaises(asyncpg.UniqueViolationError):
            await self.db.execute(
                "UPDATE notifications SET event_key='event-1' WHERE id=2",
            )

    async def test_rolling_bridge_handles_split_schemas_and_old_writers(
        self,
    ) -> None:
        evidence_schema = 'evidence_' + uuid4().hex
        self.extra_schemas.append(evidence_schema)
        await self.db.execute(f'CREATE SCHEMA {evidence_schema}')
        for table in ('sites', 'violations'):
            await self.db.execute(
                f'ALTER TABLE {table} SET SCHEMA {evidence_schema}',
            )
        await self.db.execute(
            f'SET search_path TO {self.schema}, {evidence_schema}',
        )
        await self.apply('01')
        bridge = (
            ROOT
            / 'deploy/migrations'
            / ('20261031_01_legacy_evidence_site_dual_write.sql')
        )
        await self.db.execute(bridge.read_text())
        await self.db.execute(bridge.read_text())
        await self.db.execute(
            "INSERT INTO violations(id, site) VALUES (6000, 'original')",
        )
        self.assertEqual(
            await self.db.fetchval(
                'SELECT site_id FROM violations WHERE id=6000',
            ),
            1,
        )
        await self.apply('02')
        await self.apply('02')
        await self.apply('03')
        await self.db.execute("UPDATE sites SET name='renamed' WHERE id=1")
        await self.db.execute(
            "INSERT INTO violations(id, site) VALUES (6001, 'renamed')",
        )
        await self.db.execute(
            'INSERT INTO violations(id, site, site_id) '
            "VALUES (6002, 'historical snapshot', 1)",
        )
        self.assertEqual(
            await self.db.fetchval(
                'SELECT count(*) FROM violations WHERE site_id IS NULL',
            ),
            0,
        )
        with self.assertRaises(asyncpg.ForeignKeyViolationError):
            await self.db.execute(
                "INSERT INTO violations(id, site) VALUES (6003, 'unknown')",
            )
        self.assertEqual(
            await self.db.fetchval('SELECT site FROM violations WHERE id=1'),
            'original',
        )
