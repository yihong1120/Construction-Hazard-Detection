from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path


async def migrate(sql_file):
    import asyncpg
    from examples.auth.config import Settings

    url = Settings().sqlalchemy_database_uri.replace(
        'postgresql+asyncpg://', 'postgresql://', 1,
    )
    connection = await asyncpg.connect(url, timeout=10)
    try:
        await connection.execute(sql_file.read_text())
    finally:
        await connection.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--app-root', type=Path, required=True)
    parser.add_argument('--sql', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    os.environ.update(config.get('environment', config))
    sql_file = args.sql.resolve()
    os.environ['PYTHON_DOTENV_DISABLED'] = '1'
    os.chdir(args.app_root)
    sys.path.insert(0, str(args.app_root))
    try:
        asyncio.run(migrate(sql_file))
    except Exception as exc:
        print(
            'Website migration failed: ' + type(exc).__name__, file=sys.stderr,
        )
        raise SystemExit(1) from None
    print('Website access schema ready; existing website modes retained.')


if __name__ == '__main__':
    main()
