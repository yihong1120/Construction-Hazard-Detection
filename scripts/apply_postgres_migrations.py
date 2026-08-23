from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import asyncpg


MIGRATION_DIRECTORY = Path(__file__).with_name('migrations')
MIGRATION_PATTERN = re.compile(r'^(?P<date>\d{8})_.+_postgres\.sql$')
LEDGER_TABLE = 'schema_migrations'
DOLLAR_QUOTE_PATTERN = re.compile(r'\$(?:[A-Za-z_][A-Za-z0-9_]*)?\$')
CONCURRENT_INDEX_PATTERN = re.compile(
    r'\b(?:CREATE|DROP)\s+INDEX\s+CONCURRENTLY\b',
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Migration:
    """Describe one immutable PostgreSQL migration file.

    Attributes:
        version: Immutable migration identifier derived from the filename.
        date: Date prefix used for chronological baseline selection.
        path: SQL file path.
        checksum: SHA-256 checksum of the SQL content.
    """

    version: str
    date: str
    path: Path
    checksum: str


def normalise_database_url(database_url: str) -> str:
    """Convert a SQLAlchemy async URL to the asyncpg connection form.

    Args:
        database_url: PostgreSQL connection URL from the environment.

    Returns:
        An asyncpg-compatible PostgreSQL URL.

    Raises:
        ValueError: If the URL does not select PostgreSQL.
    """
    if database_url.startswith('postgresql+asyncpg://'):
        return database_url.replace('postgresql+asyncpg://', 'postgresql://', 1)
    if database_url.startswith('postgresql://'):
        return database_url
    raise ValueError('DATABASE_URL must use postgresql or postgresql+asyncpg.')


def discover_migrations(directory: Path) -> list[Migration]:
    """Discover canonical PostgreSQL migrations in execution order.

    Args:
        directory: Directory holding versioned migration files.

    Returns:
        Sorted immutable migration descriptions.

    Raises:
        ValueError: If two files share a version or the directory is invalid.
    """
    if not directory.is_dir():
        raise ValueError(f'Migration directory does not exist: {directory}')

    migrations: list[Migration] = []
    versions: set[str] = set()
    for path in sorted(directory.glob('*_postgres.sql')):
        match = MIGRATION_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        date = match.group('date')
        version = path.name.removesuffix('_postgres.sql')
        if version in versions:
            raise ValueError(f'Duplicate migration version: {version}')
        versions.add(version)
        migrations.append(
            Migration(
                version=version,
                date=date,
                path=path,
                checksum=hashlib.sha256(path.read_bytes()).hexdigest(),
            ),
        )
    return migrations


def split_sql_statements(source: str) -> list[str]:
    """Split PostgreSQL SQL without breaking quoted or dollar-quoted blocks.

    Args:
        source: Complete text of one SQL migration.

    Returns:
        Executable SQL statements in their source order.

    Raises:
        ValueError: If the SQL source ends inside a quoted or comment block.
    """
    statements: list[str] = []
    start = 0
    index = 0
    length = len(source)
    in_single_quote = False
    in_double_quote = False
    in_line_comment = False
    in_block_comment = False
    dollar_delimiter: str | None = None

    while index < length:
        if in_line_comment:
            if source[index] == '\n':
                in_line_comment = False
            index += 1
            continue

        if in_block_comment:
            if source.startswith('*/', index):
                in_block_comment = False
                index += 2
            else:
                index += 1
            continue

        if dollar_delimiter is not None:
            if source.startswith(dollar_delimiter, index):
                index += len(dollar_delimiter)
                dollar_delimiter = None
            else:
                index += 1
            continue

        character = source[index]
        if in_single_quote:
            if character == "'":
                if index + 1 < length and source[index + 1] == "'":
                    index += 2
                    continue
                in_single_quote = False
            index += 1
            continue

        if in_double_quote:
            if character == '"':
                if index + 1 < length and source[index + 1] == '"':
                    index += 2
                    continue
                in_double_quote = False
            index += 1
            continue

        if source.startswith('--', index):
            in_line_comment = True
            index += 2
            continue
        if source.startswith('/*', index):
            in_block_comment = True
            index += 2
            continue
        if character == "'":
            in_single_quote = True
            index += 1
            continue
        if character == '"':
            in_double_quote = True
            index += 1
            continue
        if character == '$':
            match = DOLLAR_QUOTE_PATTERN.match(source, index)
            if match is not None:
                dollar_delimiter = match.group(0)
                index = match.end()
                continue
        if character == ';':
            statement = source[start:index + 1].strip()
            if statement:
                statements.append(statement)
            start = index + 1
        index += 1

    if (
        in_single_quote
        or in_double_quote
        or in_block_comment
        or dollar_delimiter is not None
    ):
        raise ValueError('SQL source ends inside a quoted or comment block.')

    remainder = source[start:].strip()
    if remainder:
        statements.append(remainder)
    return statements


def statements_for_execution(source: str) -> list[str]:
    """Return execution units that preserve PostgreSQL transaction rules.

    Args:
        source: Complete text of one SQL migration.

    Returns:
        One source unit normally, or individual statements for concurrent
        index operations which PostgreSQL forbids inside a transaction.
    """
    if CONCURRENT_INDEX_PATTERN.search(source) is None:
        return [source]
    return split_sql_statements(source)


async def ensure_ledger(connection: asyncpg.Connection) -> None:
    """Create the migration ledger when it has not been initialised.

    Args:
        connection: Open PostgreSQL connection.
    """
    await connection.execute(
        f'''
        CREATE TABLE IF NOT EXISTS {LEDGER_TABLE} (
            version VARCHAR(255) PRIMARY KEY,
            checksum CHAR(64) NOT NULL,
            applied_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        ''',
    )


async def read_applied(
    connection: asyncpg.Connection,
) -> dict[str, str]:
    """Return migration checksums recorded by the database.

    Args:
        connection: Open PostgreSQL connection.

    Returns:
        Mapping of migration version to immutable checksum.
    """
    rows = await connection.fetch(
        f'SELECT version, checksum FROM {LEDGER_TABLE} ORDER BY version',
    )
    return {str(row['version']): str(row['checksum']) for row in rows}


def validate_ledger(
    migrations: Iterable[Migration],
    applied: dict[str, str],
) -> None:
    """Reject edited or unknown migrations before running new SQL.

    Args:
        migrations: Canonical migration sequence.
        applied: Checksums read from the migration ledger.

    Raises:
        ValueError: If the ledger references missing or changed migrations.
    """
    expected = {migration.version: migration.checksum for migration in migrations}
    unknown = sorted(set(applied) - set(expected))
    if unknown:
        raise ValueError(
            'Ledger contains migrations not present in this checkout: '
            f'{", ".join(unknown)}',
        )
    changed = sorted(
        version
        for version, checksum in applied.items()
        if expected[version] != checksum
    )
    if changed:
        raise ValueError(
            'Applied migration content has changed: '
            f'{", ".join(changed)}',
        )


async def record_baseline(
    connection: asyncpg.Connection,
    migrations: Iterable[Migration],
    through_version: str,
) -> int:
    """Record verified legacy migrations without rerunning their SQL.

    Args:
        connection: Open PostgreSQL connection.
        migrations: Canonical migration sequence.
        through_version: Inclusive date version already present in the schema.

    Returns:
        Number of newly recorded migration rows.
    """
    recorded = 0
    for migration in migrations:
        if migration.date > through_version:
            continue
        result = await connection.execute(
            f'''
            INSERT INTO {LEDGER_TABLE} (version, checksum)
            VALUES ($1, $2)
            ON CONFLICT (version) DO NOTHING
            ''',
            migration.version,
            migration.checksum,
        )
        if result == 'INSERT 0 1':
            recorded += 1
    return recorded


async def apply_pending(
    connection: asyncpg.Connection,
    migrations: Iterable[Migration],
    applied: dict[str, str],
    dry_run: bool,
) -> int:
    """Execute and record each unapplied migration without an outer transaction.

    Args:
        connection: Open PostgreSQL connection.
        migrations: Canonical migration sequence.
        applied: Checksums already present in the migration ledger.
        dry_run: Whether to print planned migrations without executing them.

    Returns:
        Number of migrations executed or planned.
    """
    pending = [
        migration for migration in migrations
        if migration.version not in applied
    ]
    for migration in pending:
        print(f'{"Would apply" if dry_run else "Applying"} {migration.path.name}')
        if dry_run:
            continue
        # asyncpg wraps a multi-statement query in a transaction.  Only split
        # migrations containing concurrent index work: ordinary migrations
        # retain their all-or-nothing database execution semantics.
        source = migration.path.read_text(encoding='utf-8')
        for statement in statements_for_execution(source):
            await connection.execute(statement)
        await connection.execute(
            f'INSERT INTO {LEDGER_TABLE} (version, checksum) VALUES ($1, $2)',
            migration.version,
            migration.checksum,
        )
    return len(pending)


async def run(
    database_url: str,
    migration_directory: Path,
    baseline: str | None,
    dry_run: bool,
) -> int:
    """Run the PostgreSQL migration workflow for one database.

    Args:
        database_url: PostgreSQL database URL.
        migration_directory: Canonical migration directory.
        baseline: Optional inclusive version to record without executing.
        dry_run: Whether to print planned work only.

    Returns:
        Number of migrations recorded, executed or planned.
    """
    migrations = discover_migrations(migration_directory)
    connection = await asyncpg.connect(normalise_database_url(database_url))
    try:
        await ensure_ledger(connection)
        applied = await read_applied(connection)
        validate_ledger(migrations, applied)
        if baseline is not None:
            if dry_run:
                count = sum(
                    migration.date <= baseline for migration in migrations
                    if migration.version not in applied
                )
                print(f'Would record {count} migrations through {baseline}.')
                return count
            return await record_baseline(connection, migrations, baseline)
        return await apply_pending(connection, migrations, applied, dry_run)
    finally:
        await connection.close()


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for the PostgreSQL migration runner.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(
        description='Apply immutable PostgreSQL schema migrations.',
    )
    parser.add_argument(
        '--database-url',
        default=os.getenv('DATABASE_URL'),
        help='PostgreSQL URL; defaults to DATABASE_URL.',
    )
    parser.add_argument(
        '--migrations',
        type=Path,
        default=MIGRATION_DIRECTORY,
        help='Directory containing canonical *_postgres.sql migrations.',
    )
    parser.add_argument(
        '--baseline',
        metavar='YYYYMMDD',
        help='Record existing migrations through this date without running them.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print migration work without changing the database.',
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the PostgreSQL migration command-line entry point.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Process exit status.
    """
    arguments = parse_arguments(argv)
    if not arguments.database_url:
        raise SystemExit('DATABASE_URL or --database-url is required.')
    if arguments.baseline is not None and not re.fullmatch(
        r'\d{8}',
        arguments.baseline,
    ):
        raise SystemExit('--baseline must use YYYYMMDD.')
    count = asyncio.run(
        run(
            arguments.database_url,
            arguments.migrations,
            arguments.baseline,
            arguments.dry_run,
        ),
    )
    print(f'Completed {count} migration operation(s).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
