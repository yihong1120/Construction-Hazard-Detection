from __future__ import annotations

import asyncio

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from examples.auth.operation_database import close_operation_database
from examples.auth.operation_database import operation_database
from tests.database_fixtures_test import OrmDatabase


class OperationDatabaseTest(OrmDatabase):
    async def test_pool_preserves_custom_connection_options_and_is_reused(
        self,
    ):
        await self.db.execute('CREATE SCHEMA journal_scope')
        source = create_async_engine(
            self.engine.url,
            connect_args={
                'server_settings': {'search_path': 'journal_scope,public'},
            },
            pool_size=1,
            max_overflow=0,
            pool_timeout=0.2,
        )
        try:
            resources = operation_database(source)
            self.assertIs(resources, operation_database(source))
            async with source.begin() as occupied:
                self.assertEqual(
                    await occupied.scalar(text('SELECT current_schema()')),
                    'journal_scope',
                )
                async with asyncio.timeout(2):
                    async with resources.locks.begin() as lock:
                        async with resources.sessions() as journal:
                            self.assertEqual(
                                await lock.scalar(
                                    text('SELECT current_schema()'),
                                ),
                                'journal_scope',
                            )
                            self.assertEqual(
                                await journal.scalar(
                                    text('SELECT current_schema()'),
                                ),
                                'journal_scope',
                            )
            await close_operation_database(source)
            self.assertIsNot(operation_database(source), resources)
        finally:
            await close_operation_database(source)
            await source.dispose()
