from __future__ import annotations

from pathlib import Path

from tests.postgres_fixtures_test import IsolatedPostgresDatabase


class AccountStateRecoveryMigrationTest(IsolatedPostgresDatabase):
    async def test_upgrade_is_repeatable_and_preserves_pending_work(self):
        await self.db.execute(
            '''CREATE TABLE users(id integer PRIMARY KEY,
                status text NOT NULL);
            CREATE TABLE violations(id integer PRIMARY KEY);
            INSERT INTO users VALUES(1,'active');''',
        )
        await self.db.execute(
            Path(
                'deploy/migrations/20261102_02_account_evidence_lifecycle'
                '.sql',
            ).read_text(),
        )
        await self.db.execute(
            'INSERT INTO evidence_write_operations(id,path) '
            "VALUES('pending','date/image.png')",
        )
        script = Path(
            'deploy/migrations/20261103_01_account_state_evidence_recover'
            'y.sql',
        ).read_text()
        for _ in range(2):
            await self.db.execute(script)
        self.assertEqual(
            await self.db.fetchval(
                'SELECT state_revision FROM users WHERE id=1',
            ),
            1,
        )
        row = await self.db.fetchrow(
            'SELECT attempts,last_error,next_attempt FROM '
            "evidence_write_operations WHERE id='pending'",
        )
        self.assertEqual(row['attempts'], 0)
        self.assertIsNone(row['last_error'])
        self.assertIsNotNone(row['next_attempt'])
        index = await self.db.fetchval(
            'SELECT indexdef FROM pg_indexes WHERE '
            "indexname='evidence_write_due'",
        )
        self.assertIn('(next_attempt, created_at)', index)
