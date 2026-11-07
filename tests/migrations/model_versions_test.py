from __future__ import annotations

import json
import unittest
from pathlib import Path

import asyncpg

from tests.migrations.evidence_sites_test import DATABASE
from tests.migrations.evidence_sites_test import EvidenceMigrationDatabase
from tests.postgres_fixtures_test import IsolatedPostgresDatabase


@unittest.skipUnless(DATABASE, 'Requires isolated migration database')
class ModelVersionMigrationTest(EvidenceMigrationDatabase):
    async def test_backfill_preserves_current_and_journal_history(
        self,
    ):
        await self.db.execute(
            '''CREATE TABLE detection_model_catalog(
            model_key varchar(120) PRIMARY KEY, definition json NOT NULL);
            CREATE TABLE model_publications(
                id text PRIMARY KEY, snapshot json NOT NULL)''',
        )
        old = {
            'version': 'v1',
            'sha256': 'a' * 64,
            'classes': [{'id': 7, 'code': 'helmet', 'display_name': 'Helmet'}],
        }
        new = {**old, 'version': 'v2', 'sha256': 'b' * 64}
        await self.db.execute(
            'INSERT INTO detection_model_catalog VALUES($1,$2::json)',
            'detector',
            json.dumps(new),
        )
        snapshot = {
            'previous': {'detector': {'definition': old}},
            'prepared': [[{'id': 'detector', **new}, 0, True]],
        }
        await self.db.execute(
            'INSERT INTO model_publications VALUES($1,$2::json)',
            'publication',
            json.dumps(snapshot),
        )
        script = Path(
            'deploy/migrations/20261102_01_model_versions.sql',
        ).read_text()
        await self.db.execute(script)
        await self.db.execute(script)
        versions = await self.db.fetch(
            'SELECT version FROM detection_model_versions ORDER BY version',
        )
        self.assertEqual([r['version'] for r in versions], ['v1', 'v2'])
        with self.assertRaises(asyncpg.CheckViolationError):
            await self.db.execute(
                'UPDATE detection_model_catalog SET definition='
                "jsonb_set(definition::jsonb,'{sha256}',"
                "to_jsonb(repeat('c',64)))",
            )
        with self.assertRaises(asyncpg.CheckViolationError):
            await self.db.execute('DELETE FROM detection_model_versions')


class ModelVersionInitializationTest(IsolatedPostgresDatabase):
    async def test_fresh_initialization_installs_history_and_archiving(self):
        await self.db.execute(Path('scripts/init.postgres.sql').read_text())
        definition = {'version': 'v1', 'sha256': 'a' * 64, 'classes': []}
        await self.db.execute(
            'INSERT INTO detection_model_catalog(model_key,definition) '
            'VALUES($1,$2::json)',
            'detector',
            json.dumps(definition),
        )
        self.assertEqual(
            await self.db.fetchval(
                'SELECT version FROM detection_model_versions '
                'WHERE model_key=$1',
                'detector',
            ),
            'v1',
        )
