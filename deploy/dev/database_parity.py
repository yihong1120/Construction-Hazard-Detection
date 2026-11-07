from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from urllib.parse import unquote
from urllib.parse import urlsplit

from deploy.dev.run import CONFIG
from deploy.dev.run import STATE
from deploy.dev.run import validate

QUERIES = {
    ('columns'): (
        'SELECT n.nspname AS schema,c.relname AS relation,a.attname AS '
        'name,\n        a.attnum,format_type(a.atttypid,a.atttypmod) AS '
        'type,a.attnotnull,\n        a.attidentity,a.attgenerated,pg_get_expr('
        'd.adbin,d.adrelid) AS default\n        FROM pg_attribute a JOIN '
        'pg_class c ON c.oid=a.attrelid\n        JOIN pg_namespace n ON '
        'n.oid=c.relnamespace\n        LEFT JOIN pg_attrdef d ON '
        'd.adrelid=a.attrelid AND d.adnum=a.attnum\n        WHERE c.relkind '
        "IN ('r','p') AND a.attnum>0 AND NOT a.attisdropped\n        AND "
        "n.nspname IN ('public','construction_hazard_detection') ORDER BY "
        '1,2,4'
    ),
    ('constraints'): (
        'SELECT n.nspname AS schema,r.relname AS relation,c.conname AS '
        'name,\n        c.contype,pg_get_constraintdef(c.oid,true) AS '
        'definition,c.convalidated\n        FROM pg_constraint c JOIN '
        'pg_namespace n ON n.oid=c.connamespace\n        JOIN pg_class r ON '
        'r.oid=c.conrelid\n        WHERE n.nspname IN '
        "('public','construction_hazard_detection') ORDER BY "
        '1,2,3'
    ),
    ('indexes'): (
        'SELECT schemaname AS schema,tablename AS relation,indexname AS '
        'name,indexdef AS definition\n        FROM pg_indexes WHERE '
        "schemaname IN ('public','construction_hazard_detection') ORDER BY "
        '1,2,3'
    ),
    ('triggers'): (
        'SELECT n.nspname AS schema,c.relname AS relation,t.tgname AS '
        'name,t.tgenabled,\n        pg_get_triggerdef(t.oid,true) AS '
        'definition FROM pg_trigger t\n        JOIN pg_class c ON '
        'c.oid=t.tgrelid JOIN pg_namespace n ON n.oid=c.relnamespace\n        '
        "WHERE n.nspname IN ('public','construction_hazard_detection') AND "
        'NOT t.tgisinternal ORDER BY '
        '1,2,3'
    ),
    'functions': (
        'SELECT n.nspname AS schema,p.proname AS name,\n        '
        'pg_get_function_identity_arguments(p.oid) AS '
        'args,pg_get_functiondef(p.oid) AS definition\n        FROM pg_proc p '
        'JOIN pg_namespace n ON n.oid=p.pronamespace\n        WHERE n.nspname '
        "IN ('public','construction_hazard_detection') AND p.prokind='f'\n    "
        '    AND NOT EXISTS (SELECT 1 FROM pg_depend d WHERE '
        "d.classid='pg_proc'::regclass\n          AND d.objid=p.oid AND "
        "d.deptype='e') ORDER BY "
        '1,2,3'
    ),
    ('enums'): (
        'SELECT n.nspname AS schema,t.typname AS '
        'name,e.enumlabel,e.enumsortorder\n        FROM pg_type t JOIN '
        'pg_namespace n ON n.oid=t.typnamespace JOIN pg_enum e ON '
        'e.enumtypid=t.oid\n        WHERE n.nspname IN '
        "('public','construction_hazard_detection') ORDER BY "
        '1,2,4'
    ),
    ('sequences'): (
        'SELECT schemaname,sequencename,data_type::text,start_value,min_value,'
        'max_value,\n        increment_by,cycle,cache_size FROM '
        'pg_sequences\n        WHERE schemaname IN '
        "('public','construction_hazard_detection') ORDER BY "
        '1,2'
    ),
    'extensions': 'SELECT extname,extversion FROM pg_extension ORDER BY 1',
    ('policies'): (
        'SELECT * FROM pg_policies WHERE schemaname IN '
        "('public','construction_hazard_detection')\n        ORDER BY "
        'schemaname,tablename,policyname'
    ),
}


def dev_options(port=15433):
    env = json.loads((CONFIG / 'environment.json').read_text())
    validate(env)
    u = urlsplit(env['DATABASE_URL'])
    if port not in {15433, 15434}:
        raise RuntimeError('Only isolated dev/candidate ports are allowed.')
    return dict(
        host='127.0.0.1',
        port=port,
        user=unquote(u.username),
        password=unquote(u.password),
        dbname='construction_dev',
    )


def production_options():
    credentials = json.loads(
        (
            Path.home() / 'service-migration-phase5/source-db-credentials.json'
        ).read_text(),
    )
    return dict(
        host='100.93.178.78',
        port=15432,
        user=credentials['user'],
        password=credentials['password'],
        dbname='construction_hazard_detection',
    )


async def restore_empty_baseline(engine):
    (
        'Restore the private schema snapshot only into an entirely empty dev '
        'DB.'
    )
    options = dev_options()
    from sqlalchemy import text

    async with engine.begin() as db:
        rows = (
            await db.execute(
                text(
                    (
                        'SELECT n.nspname,c.relname '
                        'FROM pg_class c\n            '
                        'JOIN pg_namespace n ON '
                        'n.oid=c.relnamespace WHERE '
                        "c.relkind IN ('r','p')\n      "
                        '      AND n.nspname IN '
                        "('public','construction_hazard"
                        "_detection')"
                    ),
                ),
            )
        ).all()
        if rows:
            if ('construction_hazard_detection', 'users') not in rows or (
                'public',
                'schema_migrations',
            ) not in rows:
                raise RuntimeError(
                    (
                        'Partial dev schema detected; restore a verified '
                        'candidate instead of using ORM '
                        'create_all.'
                    ),
                )
            return
        state = json.loads((STATE / 'parity/current.json').read_text())
        archive = Path(state['directory']) / 'production-schema.dump'
        if not archive.resolve().is_relative_to((STATE / 'parity').resolve()):
            raise RuntimeError('Unexpected schema baseline path.')
        if (
            hashlib.sha256(archive.read_bytes()).hexdigest()
            != state['schema_archive_sha256']
        ):
            raise RuntimeError('Schema baseline checksum mismatch.')
        await db.execute(
            text(
                (
                    'ALTER ROLE construction_dev SET '
                    'search_path=public,construction_hazard_detection'
                ),
            ),
        )
    await engine.dispose()
    result = subprocess.run(
        [
            '/usr/lib/postgresql/18/bin/pg_restore',
            '--host',
            options['host'],
            '--port',
            str(
                options['port'],
            ),
            '--username',
            options['user'],
            '--dbname',
            options['dbname'],
            '--no-owner',
            '--no-acl',
            '--exit-on-error',
            '--single-transaction',
            str(archive),
        ],
        env={**os.environ, 'PGPASSWORD': options['password']},
        capture_output=True,
    )
    if result.returncode:
        raise RuntimeError('Empty dev schema baseline restore failed.')


def snapshot(options):
    import psycopg
    from psycopg.rows import dict_row

    with psycopg.connect(
        **options,
        connect_timeout=8,
        row_factory=dict_row,
        options=(
            '-c default_transaction_read_only=on -c ' 'statement_timeout=15000'
        ),
    ) as db:
        db.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ')
        db.execute(
            'SET LOCAL search_path=public,construction_hazard_detection',
        )
        schema = {
            name: db.execute(query).fetchall()
            for name, query in QUERIES.items()
        }
        catalog = db.execute(
            (
                'SELECT feature_name,description FROM features ORDER BY '
                'feature_name'
            ),
        ).fetchall()
        operator = db.execute(
            (
                'SELECT u.role,u.status,g.max_allowed_streams,\n            '
                'array_agg(f.feature_name ORDER BY f.feature_name) AS '
                'features\n            FROM users u JOIN group_info g ON '
                'g.id=u.group_id\n            LEFT JOIN group_features gf ON '
                'gf.group_id=g.id LEFT JOIN features f ON '
                'f.id=gf.feature_id\n            WHERE '
                "lower(u.username)='changdar' GROUP BY "
                'u.role,u.status,g.max_allowed_streams'
            ),
        ).fetchone()
        migrations = db.execute(
            (
                'SELECT version,checksum FROM public.schema_migrations ORDER '
                'BY version'
            ),
        ).fetchall()
    return {
        'schema': schema,
        'catalog': catalog,
        'operator': operator,
        'migrations': migrations,
    }


def compare(production, dev):
    differences = [
        kind
        for kind in QUERIES
        if production['schema'][kind] != dev['schema'][kind]
    ]
    differences += [
        kind
        for kind in (
            'catalog',
            'operator',
            'migrations',
        )
        if production[kind] != dev[kind]
    ]
    counts = {kind: len(rows) for kind, rows in dev['schema'].items()}
    counts['tables'] = len(
        {(r['schema'], r['relation']) for r in dev['schema']['columns']},
    )
    digest = hashlib.sha256(
        json.dumps(
            dev['schema'],
            sort_keys=True,
            default=str,
        ).encode(),
    ).hexdigest()
    return {
        'matched': not differences,
        'differences': differences,
        'dev_counts': counts,
        'schema_sha256': digest,
        'features': [x['feature_name'] for x in dev['catalog']],
        'operator': dev['operator'],
        'migrations': len(dev['migrations']),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', action='store_true')
    args = parser.parse_args()
    production = snapshot(production_options())
    dev = snapshot(dev_options(15434 if args.candidate else 15433))
    result = compare(production, dev)
    destination = (
        STATE
        / 'parity'
        / ('candidate-audit.json' if args.candidate else 'active-audit.json')
    )
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination.write_text(
        json.dumps(
            {'result': result, 'production': production, 'dev': dev},
            indent=2,
            default=str,
        ),
    )
    destination.chmod(0o600)
    print(json.dumps(result, ensure_ascii=False, default=str))
    if not result['matched']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
