from __future__ import annotations

import os
import tempfile
import unittest
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import patch

from sqlalchemy.engine import make_url

from deploy.dev import schema
from deploy.dev.run import STATE
from deploy.migrations.run import MigrationError


async def _restore_then_migrate_uses_only_restore(steps, /, _):
    steps.append('baseline')


async def _restore_then_migrate_uses_only_plan(steps, /):
    steps.append('plan')
    return [{'applied': False}]


async def _restore_then_migrate_uses_only_apply(
    testcase,
    steps,
    /,
    backup,
):
    steps.append('verified-backup-and-migrate')
    testcase.assertEqual(backup.source_url, backup.verifier_url)
    testcase.assertIn(
        '127.0.0.1:15433/construction_dev',
        backup.source_url,
    )
    testcase.assertFalse(backup.directory.exists())
    return [{'applied': True}]


def dev_values():
    values = {
        'DATABASE_URL': (
            'postgresql+asyncpg://construction_dev:test@127.0.0.1:15433/'
            'construction_dev'
        ),
        'REDIS_HOST': '127.0.0.1',
        'REDIS_PORT': '16380',
        'STORAGE_ROOT': str(STATE / 'documents'),
        'SIGNATURE_ROOT': str(STATE / 'documents/signatures'),
        'WIN_UPLOAD_ROOT': str(STATE / 'documents'),
        'BFF_SESSION_COOKIE_NAME': '__Host-vn_dev_session',
        'OIDC_AUDIENCE': 'visionnaire-dev-api',
    }
    for name, port in {
        'MANAGEMENT': 19005,
        'FILES': 19004,
        'FCM': 19003,
        'VIOLATIONS': 19002,
        'STREAMING': 19800,
        'CHAT': 19009,
        'DETECTION': 19000,
        'AI_USAGE': 19009,
        'AI_ACCESS': 18083,
    }.items():
        values['BFF_' + name + '_API_URL'] = f'http://127.0.0.1:{port}'
    values['STREAMING_PLAYBACK_API_URL'] = 'http://127.0.0.1:19800'
    return values


class DevSchemaTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.values = dev_values()
        self.env = patch.dict(os.environ, self.values, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.db = SimpleNamespace(close=AsyncMock())
        self.engine = SimpleNamespace(
            url=make_url(self.values['DATABASE_URL']),
        )

    async def test_non_dev_target_rejected_before_baseline_or_connection(self):
        os.environ['DATABASE_URL'] = (
            'postgresql+asyncpg://user:pass@shared/production'
        )
        with patch(
            'deploy.dev.database_parity.restore_empty_baseline',
            AsyncMock(),
        ) as baseline, patch.object(
            schema,
            'connect',
            AsyncMock(),
        ) as connection:
            with self.assertRaises(RuntimeError):
                await schema.prepare_schema(self.engine)
            baseline.assert_not_awaited()
            connection.assert_not_awaited()

    async def test_mismatched_engine_refused_before_baseline_write(self):
        self.engine.url = make_url('postgresql+asyncpg://shared/production')
        with patch(
            'deploy.dev.database_parity.restore_empty_baseline',
            AsyncMock(),
        ) as baseline:
            with self.assertRaises(MigrationError):
                await schema.prepare_schema(self.engine)
            baseline.assert_not_awaited()

    async def test_restore_then_migrate_uses_only_a_new_verifier_database(
        self,
    ):
        steps = []

        restore = partial(_restore_then_migrate_uses_only_restore, steps)

        plan = partial(_restore_then_migrate_uses_only_plan, steps)

        apply = partial(_restore_then_migrate_uses_only_apply, self, steps)

        runner = SimpleNamespace(plan=plan, apply=apply)
        with tempfile.TemporaryDirectory() as directory, patch.object(
            schema,
            'STATE',
            Path(directory),
        ), patch(
            'deploy.dev.database_parity.restore_empty_baseline',
            restore,
        ), patch.object(
            schema,
            'connect',
            AsyncMock(return_value=self.db),
        ), patch.object(
            schema,
            'MigrationRunner',
            return_value=runner,
        ):
            await schema.prepare_schema(self.engine)
            receipts = list(
                Path(directory).glob('migrations/setup-*/migration.json'),
            )
            self.assertEqual(len(receipts), 1)
            self.assertEqual(receipts[0].stat().st_mode & 0o777, 0o600)
        self.assertEqual(
            steps,
            ['baseline', 'plan', 'verified-backup-and-migrate'],
        )
        self.db.close.assert_awaited_once()

    async def test_existing_current_schema_does_not_backup_or_apply(self):
        runner = SimpleNamespace(
            plan=AsyncMock(return_value=[{'applied': True}]),
            apply=AsyncMock(),
        )
        with patch(
            'deploy.dev.database_parity.restore_empty_baseline',
            AsyncMock(),
        ), patch.object(
            schema,
            'connect',
            AsyncMock(return_value=self.db),
        ), patch.object(
            schema,
            'MigrationRunner',
            return_value=runner,
        ):
            await schema.prepare_schema(self.engine)
        runner.apply.assert_not_awaited()
        self.db.close.assert_awaited_once()

    async def test_pending_or_drifted_schema_prevents_service_startup(self):
        for result in [
            [{'applied': False}],
            MigrationError('checksum differs'),
        ]:
            runner = SimpleNamespace(plan=AsyncMock())
            if isinstance(result, Exception):
                runner.plan.side_effect = result
            else:
                runner.plan.return_value = result
            with patch.object(
                schema,
                'connect',
                AsyncMock(return_value=self.db),
            ) as connection, patch.object(
                schema,
                'MigrationRunner',
                return_value=runner,
            ):
                with self.assertRaises(MigrationError):
                    await schema.require_current_schema()
                connection.assert_awaited_once_with(readonly=True)
        self.assertEqual(self.db.close.await_count, 2)

    async def test_failed_migration_prevents_seed_queries(
        self,
    ):
        from deploy.dev import seed

        runner = SimpleNamespace(
            plan=AsyncMock(return_value=[{'applied': False}]),
            apply=AsyncMock(side_effect=MigrationError('postcondition')),
        )
        with tempfile.TemporaryDirectory() as directory, patch.object(
            schema,
            'STATE',
            Path(directory),
        ), patch(
            'deploy.dev.database_parity.restore_empty_baseline',
            AsyncMock(),
        ), patch.object(
            schema,
            'connect',
            AsyncMock(return_value=self.db),
        ), patch.object(
            schema,
            'MigrationRunner',
            return_value=runner,
        ):
            with self.assertRaises(MigrationError):
                await schema.prepare_schema(self.engine)
        self.db.close.assert_awaited_once()
        with patch.object(seed, 'load_environment'), patch.object(
            schema,
            'prepare_schema',
            AsyncMock(side_effect=MigrationError('pending')),
        ), patch('examples.auth.database.AsyncSessionLocal') as sessions:
            with self.assertRaises(MigrationError):
                await seed.seed()
            sessions.assert_not_called()

    async def test_connections_are_bounded_and_startup_checks_are_read_only(
        self,
    ):
        with patch.object(
            schema.asyncpg,
            'connect',
            AsyncMock(return_value=self.db),
        ) as connect:
            await schema.connect(readonly=True)
            options = connect.await_args.kwargs
            self.assertEqual(options['timeout'], 10)
            self.assertEqual(
                options['server_settings']['default_transaction_read_only'],
                'on',
            )
