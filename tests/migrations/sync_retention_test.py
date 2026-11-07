from __future__ import annotations

from pathlib import Path

from tests.postgres_fixtures_test import IsolatedPostgresDatabase


class SyncRetentionMigrationTest(IsolatedPostgresDatabase):
    async def test_upgrade_backfills_receipts_and_completion_times_once(self):
        await self.db.execute(
            'CREATE TABLE users(id integer PRIMARY KEY,status text NOT '
            'NULL); CREATE TABLE violations(id integer PRIMARY KEY)',
        )
        for name in (
            '20261102_02_account_evidence_lifecycle.sql',
            '20261103_01_account_state_evidence_recovery.sql',
        ):
            await self.db.execute(
                (Path('deploy/migrations') / name).read_text(),
            )
        await self.db.execute(
            """INSERT INTO account_operations(
                id,kind,phase,user_id,tenant_id,username)
            VALUES('deleted','delete','completed',7,
                '00000000-0000-0000-0000-000000000001','alice')""",
        )
        script = Path(
            'deploy/migrations/20261103_02_sync_progress_retention.sql',
        ).read_text()
        await self.db.execute(script)
        first = await self.db.fetchrow(
            'SELECT completed_at,expires_at FROM '
            'account_deletion_receipts WHERE user_id=7',
        )
        await self.db.execute(script)
        second = await self.db.fetchrow(
            'SELECT completed_at,expires_at FROM '
            'account_deletion_receipts WHERE user_id=7',
        )
        self.assertEqual(first, second)
        self.assertIsNotNone(
            await self.db.fetchval(
                'SELECT completed_at FROM account_operations WHERE '
                "id='deleted'",
            ),
        )
        self.assertEqual(
            await self.db.fetchval(
                'SELECT sync_parts::text FROM account_operations WHERE '
                "id='deleted'",
            ),
            '{}',
        )
        index = await self.db.fetchval(
            'SELECT indexdef FROM pg_indexes WHERE '
            "indexname='account_operation_due'",
        )
        self.assertIn('WHERE', index)
        self.assertIn('phase', index)
