from __future__ import annotations

import argparse
import asyncio
import json

from sqlalchemy import select

from examples.auth.database import engine
from examples.auth.models import AccountOperation
from examples.auth.operation_database import close_operation_database
from examples.db_management.services.account_lifecycle import AccountLifecycle


async def run(args):
    lifecycle = AccountLifecycle()
    if args.command == 'list':
        async with lifecycle.sessions() as db:
            rows = await db.scalars(
                select(AccountOperation)
                .where(
                    AccountOperation.phase != 'completed',
                )
                .order_by(AccountOperation.created_at)
                .limit(args.limit),
            )
            for record in rows:
                print(
                    json.dumps(
                        {
                            key: getattr(record, key)
                            for key in (
                                'id',
                                'kind',
                                'phase',
                                'subject',
                                'user_id',
                                'attempts',
                                'last_error',
                                'target_revision',
                                'sync_parts',
                            )
                        },
                    ),
                )
        return 0
    async with lifecycle.ownership(args.operation) as acquired:
        if not acquired:
            raise RuntimeError(
                'Operation is active; retry after its owner exits',
            )
        async with lifecycle.sessions() as db:
            record = await db.get(AccountOperation, args.operation)
            if record is None:
                raise ValueError('Operation not found')
            if args.subject:
                if (
                    record.kind != 'provision'
                    or record.phase != 'uncertain'
                    or record.subject
                ):
                    raise ValueError(
                        'Only an uncertain provision without a subject '
                        'can be repaired',
                    )
                # Verify Keycloak creation logs and the journal operation ID.
                # Matching by a mutable username is insufficient.
                record.subject = args.subject
                record.phase = 'remote_created'
                await db.commit()
        return 0 if await lifecycle.reconcile_owned(args.operation) else 1


def main():
    parser = argparse.ArgumentParser(
        description='Inspect and resume durable account operations',
    )
    commands = parser.add_subparsers(dest='command', required=True)
    listing = commands.add_parser('list')
    listing.add_argument(
        '--limit',
        type=int,
        default=100,
        choices=range(1, 1001),
    )
    retry = commands.add_parser('retry')
    retry.add_argument('operation')
    retry.add_argument(
        '--subject',
        help='Explicitly verified Keycloak subject for an uncertain creation',
    )

    return asyncio.run(_execute(parser))


async def _execute(parser):
    try:
        return await run(parser.parse_args())
    finally:
        await close_operation_database(engine)
        await engine.dispose()


if __name__ == '__main__':
    raise SystemExit(main())
