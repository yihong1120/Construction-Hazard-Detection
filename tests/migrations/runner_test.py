from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import asyncpg

from deploy.migrations.run import LOCK_NAME
from deploy.migrations.run import MigrationError
from deploy.migrations.run import MigrationRunner
from deploy.migrations.run import ROOT
from tests.migrations.evidence_sites_test import EvidenceMigrationDatabase

DATABASE = os.getenv('EVIDENCE_TEST_DATABASE_URL')


@unittest.skipUnless(DATABASE, 'Requires isolated migration database')
class MigrationRunnerTest(EvidenceMigrationDatabase):

    async def asyncSetUp(self):
        await super().asyncSetUp()
        await self.db.execute(
            """
            CREATE TABLE detection_model_catalog(
                model_key varchar(120) PRIMARY KEY, definition json NOT NULL,
                display_order integer NOT NULL DEFAULT 0,
                is_default boolean NOT NULL DEFAULT false
            );
            CREATE TABLE users(
                id integer PRIMARY KEY, status text, role text,
                group_id integer
            );
            CREATE TABLE site_notification_preferences(
                user_id integer, site_id integer, is_enabled boolean
            );
            CREATE TABLE user_sites(user_id integer, site_id integer);
            CREATE TABLE site_groups(group_id integer, site_id integer);
        """,
        )

    def runner(self):
        return MigrationRunner(self.db, 'chd_refactor_test')

    async def test_plan_has_no_mutations_and_apply_is_repeatable(self) -> None:
        plan = await self.runner().plan()
        self.assertEqual(len(plan), 10)
        self.assertTrue(all(not step['applied'] for step in plan))
        self.assertIsNone(
            await self.db.fetchval("SELECT to_regclass('schema_migrations')"),
        )
        applied = await self.runner().apply()
        self.assertTrue(all(step['applied'] for step in applied))
        self.assertEqual(await self.runner().apply(), applied)
        self.assertEqual(
            await self.db.fetchval('SELECT count(*) FROM schema_migrations'),
            10,
        )

    async def test_wrong_database_is_rejected_before_ddl(self) -> None:
        with self.assertRaisesRegex(MigrationError, 'database differs'):
            await MigrationRunner(self.db, 'wrong_database').apply()
        self.assertIsNone(
            await self.db.fetchval("SELECT to_regclass('schema_migrations')"),
        )

    async def test_applied_checksum_and_schema_drift_are_detected(
        self,
    ) -> None:
        await self.runner().apply()
        await self.db.execute(
            'ALTER TABLE violations DROP CONSTRAINT fk_violations_site_id',
        )
        with self.assertRaisesRegex(MigrationError, 'postcondition differs'):
            await self.runner().plan()
        await self.db.execute(
            (
                "UPDATE schema_migrations SET checksum=repeat('0',64) "
                "WHERE version='20261030_01_evidence_identity'"
            ),
        )
        with self.assertRaisesRegex(MigrationError, 'checksum differs'):
            await self.runner().plan()

    async def test_batch_limit_can_resume_without_replaying_completed_stages(
        self,
    ) -> None:
        runner = self.runner()
        runner.max_batches = 1
        with self.assertRaisesRegex(MigrationError, 'batch limit'):
            await runner.apply()
        self.assertEqual(
            await self.db.fetchval(
                'SELECT count(*) FROM violations WHERE site_id IS NULL',
            ),
            1,
        )
        self.assertEqual(
            await self.db.fetchval('SELECT count(*) FROM schema_migrations'),
            2,
        )
        self.assertTrue(
            all(step['applied'] for step in await self.runner().apply()),
        )

    async def test_unmatched_site_stops_backfill_without_guessing(
        self,
    ) -> None:
        await self.db.execute(
            'ALTER TABLE violations DROP CONSTRAINT violations_site_fkey',
        )
        await self.db.execute(
            "INSERT INTO violations(id,site) VALUES(6000,'unknown')",
        )
        with self.assertRaisesRegex(MigrationError, 'no progress'):
            await self.runner().apply()
        self.assertIsNone(
            await self.db.fetchval(
                'SELECT site_id FROM violations WHERE id=6000',
            ),
        )
        self.assertEqual(
            await self.db.fetchval(
                (
                    'SELECT count(*) FROM schema_migrations WHERE '
                    "version='20261030_03_contract_evidence_sites'"
                ),
            ),
            0,
        )

    async def test_concurrent_runner_lock_is_rejected_and_released(
        self,
    ) -> None:
        assert DATABASE is not None
        other = await asyncpg.connect(DATABASE)
        try:
            lock = LOCK_NAME + ':' + self.schema
            await other.execute(
                'SELECT pg_advisory_lock(hashtextextended($1,0))',
                lock,
            )
            with self.assertRaisesRegex(
                MigrationError,
                'Another evidence migration',
            ):
                await self.runner().apply()
            await other.execute(
                'SELECT pg_advisory_unlock(hashtextextended($1,0))',
                lock,
            )
            self.assertTrue(
                all(step['applied'] for step in await self.runner().apply()),
            )
        finally:
            await other.close()

    async def test_failed_contract_postcondition_rolls_back_its_ddl(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            for path in ROOT.glob('*.sql'):
                shutil.copy(path, directory / path.name)
            steps = json.loads((ROOT / 'manifest.json').read_text())
            contract = next(
                step for step in steps
                if step['file'] == '20261030_03_contract_evidence_sites.sql'
            )
            contract['verify'] = 'SELECT false'
            manifest = directory / 'manifest.json'
            manifest.write_text(json.dumps(steps))
            with self.assertRaisesRegex(
                MigrationError,
                'Postcondition failed',
            ):
                await MigrationRunner(
                    self.db,
                    'chd_refactor_test',
                    manifest,
                ).apply()
        self.assertTrue(
            await self.db.fetchval(
                (
                    'SELECT EXISTS(SELECT 1 FROM pg_constraint WHERE '
                    "conrelid='violations'::regclass AND "
                    "conname='violations_site_fkey')"
                ),
            ),
        )
        self.assertEqual(
            await self.db.fetchval('SELECT count(*) FROM schema_migrations'),
            3,
        )
