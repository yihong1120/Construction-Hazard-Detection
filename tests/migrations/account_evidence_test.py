from __future__ import annotations

from pathlib import Path

import asyncpg

from tests.postgres_fixtures_test import IsolatedPostgresDatabase


class AccountEvidenceMigrationTest(IsolatedPostgresDatabase):
    async def test_repeatable_upgrade_preserves_users_and_guards_deletion(
        self,
    ):
        await self.db.execute(
            '''CREATE TABLE users(id integer PRIMARY KEY,status text NOT NULL);
            CREATE TABLE violations(id integer PRIMARY KEY);
            INSERT INTO users VALUES(1,'active');
            INSERT INTO violations VALUES(1);''',
        )
        script = Path(
            'deploy/migrations/20261102_02_account_evidence_lifecycle.sql',
        ).read_text()
        for _ in range(2):
            await self.db.execute(script)
        self.assertFalse(
            await self.db.fetchval(
                'SELECT deletion_pending FROM users WHERE id=1',
            ),
        )
        self.assertIsNone(
            await self.db.fetchval(
                'SELECT event_fingerprint FROM violations WHERE id=1',
            ),
        )
        with self.assertRaises(asyncpg.CheckViolationError):
            await self.db.execute(
                'UPDATE users SET deletion_pending=true WHERE id=1',
            )
        await self.db.execute(
            "UPDATE users SET status='suspended',deletion_pending=true "
            'WHERE id=1',
        )
        await self.db.execute(
            """INSERT INTO account_operations(
                id,kind,phase,user_id,tenant_id,username)
            VALUES('first','delete','pending',1,
                '00000000-0000-0000-0000-000000000001','alice')""",
        )
        with self.assertRaises(asyncpg.UniqueViolationError):
            await self.db.execute(
                """INSERT INTO account_operations(
                id,kind,phase,user_id,tenant_id,username)
                VALUES('second','delete','pending',1,
                    '00000000-0000-0000-0000-000000000001','alice')""",
            )
        await self.db.execute('DELETE FROM users WHERE id=1')
        self.assertEqual(
            await self.db.fetchval('SELECT count(*) FROM account_operations'),
            1,
        )

    async def test_fresh_initialization_contains_the_same_recovery_schema(
        self,
    ):
        await self.db.execute(Path('scripts/init.postgres.sql').read_text())
        self.assertEqual(
            await self.db.fetchval(
                'SELECT count(*) FROM information_schema.columns WHERE '
                "table_schema='public' AND ((table_name='users' AND "
                "column_name='deletion_pending') OR "
                "(table_name='violations' AND "
                "column_name='event_fingerprint'))",
            ),
            2,
        )
        for table in ('account_operations', 'evidence_write_operations'):
            self.assertIsNotNone(
                await self.db.fetchval('SELECT to_regclass($1)', table),
            )
