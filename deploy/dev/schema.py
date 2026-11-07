from __future__ import annotations

import json
import os
from uuid import uuid4

import asyncpg

from deploy.dev.run import STATE
from deploy.dev.run import validate
from deploy.migrations.backup import DatabaseBackup
from deploy.migrations.run import MigrationError
from deploy.migrations.run import MigrationRunner


def dev_database_url() -> str:
    validate(dict(os.environ))
    return os.environ['DATABASE_URL'].replace(
        'postgresql+asyncpg://',
        'postgresql://',
        1,
    )


async def connect(*, readonly=False):
    return await asyncpg.connect(
        dev_database_url(),
        timeout=10,
        server_settings={
            'application_name': 'construction-dev-schema',
            'search_path': 'public,construction_hazard_detection',
            'statement_timeout': '120000',
            'lock_timeout': '5000',
            'default_transaction_read_only': 'on' if readonly else 'off',
        },
    )


async def require_current_schema():
    db = await connect(readonly=True)
    try:
        plan = await MigrationRunner(db, 'construction_dev').plan()
        if any(not step['applied'] for step in plan):
            raise MigrationError(
                'Dev migrations pending; run deploy/dev/setup.py before '
                'startup',
            )
    finally:
        await db.close()


async def prepare_schema(engine):
    """Restore an empty baseline, migrate with verified backup, then seed."""
    from deploy.dev.database_parity import restore_empty_baseline

    # Refuse shared resources before the baseline restoration can write.
    url = dev_database_url()
    from sqlalchemy.engine import make_url

    if engine.url != make_url(os.environ['DATABASE_URL']):
        raise MigrationError('Dev baseline engine differs from configuration')
    await restore_empty_baseline(engine)
    db = await connect()
    try:
        runner = MigrationRunner(db, 'construction_dev')
        plan = await runner.plan()
        if not any(not step['applied'] for step in plan):
            return
        operation = STATE / 'migrations' / ('setup-' + uuid4().hex)
        operation.mkdir(parents=True, mode=0o700)
        # Verify in a new random database on this isolated dev server.
        # Never restore over construction_dev or a shared database.
        result = await runner.apply(
            DatabaseBackup(url, operation / 'backup', url),
        )
        receipt = operation / 'migration.json'
        receipt.write_text(
            json.dumps(
                {
                    'database': 'construction_dev',
                    'migrations': result,
                },
                indent=2,
            ),
        )
        receipt.chmod(0o600)
        if any(not step['applied'] for step in result):
            raise MigrationError('Dev schema preparation did not complete')
    finally:
        await db.close()
