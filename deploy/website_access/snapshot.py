from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path


async def export():
    from examples.auth.database import AsyncSessionLocal, engine
    from examples.db_management.services.website_access import (
        policies,
        snapshot,
        website_hosts,
    )

    try:
        async with AsyncSessionLocal() as db:
            # Policies and rules must come from the same PostgreSQL snapshot.
            from sqlalchemy import text

            await db.execute(
                text(
                    (
                        'SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ '
                        'ONLY'
                    ),
                ),
            )
            rows = await policies(db)
            return {
                'hosts': website_hosts(),
                'policies': [snapshot(p) for p in rows],
            }
    finally:
        await engine.dispose()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--app-root', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    os.environ.update(config.get('environment', config))
    os.environ['PYTHON_DOTENV_DISABLED'] = '1'
    os.chdir(args.app_root)
    sys.path.insert(0, str(args.app_root))
    try:
        print(json.dumps(asyncio.run(export())))
    except Exception as exc:
        # Database exception text can contain private credentials or SQL data.
        print('Policy export failed: ' + type(exc).__name__, file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
