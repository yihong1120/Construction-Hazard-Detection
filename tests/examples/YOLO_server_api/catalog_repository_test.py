from __future__ import annotations

import asyncio
import os
import unittest
from unittest.mock import patch
from uuid import uuid4

import asyncpg
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine

from examples.YOLO_server_api.catalog_repository import CatalogRepository
from examples.YOLO_server_api.catalog_repository import (
    SynchronousCatalogReader,
)

DATABASE = os.getenv('EVIDENCE_TEST_DATABASE_URL')


@unittest.skipUnless(DATABASE, 'Requires the isolated catalog test database')
class CatalogRepositoryTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        assert DATABASE is not None
        self.connection = await asyncpg.connect(DATABASE)
        self.assertEqual(
            await self.connection.fetchval('SELECT current_database()'),
            'chd_refactor_test',
        )
        await self.connection.execute(
            '''CREATE TABLE IF NOT EXISTS detection_model_catalog (
            model_key VARCHAR(120) PRIMARY KEY, definition JSON NOT NULL,
            is_default BOOLEAN NOT NULL DEFAULT false,
            display_order INTEGER NOT NULL DEFAULT 0
        )''',
        )
        self.key = 'test-' + uuid4().hex
        await self.connection.execute(
            'INSERT INTO detection_model_catalog(model_key, definition) '
            'VALUES ($1, $2)',
            self.key,
            '{"enabled": true}',
        )
        self.uri = DATABASE.replace(
            'postgresql://',
            'postgresql+asyncpg://',
            1,
        )
        self.engine = create_async_engine(
            self.uri,
            pool_size=1,
            max_overflow=0,
        )
        self.repository = CatalogRepository(async_sessionmaker(self.engine))

    async def asyncTearDown(self) -> None:
        await self.engine.dispose()
        await self.connection.execute(
            'DELETE FROM detection_model_catalog WHERE model_key=$1',
            self.key,
        )
        await self.connection.close()

    async def test_async_reader_reuses_pool_and_observes_disable_immediately(
        self,
    ) -> None:
        first = await self.repository.rows()
        self.assertTrue(
            next(row for row in first if row['model_key'] == self.key)[
                'definition'
            ]['enabled'],
        )
        await self.connection.execute(
            'UPDATE detection_model_catalog SET definition=$1 '
            'WHERE model_key=$2',
            '{"enabled": false}',
            self.key,
        )
        second = await self.repository.rows()
        self.assertFalse(
            next(row for row in second if row['model_key'] == self.key)[
                'definition'
            ]['enabled'],
        )
        self.assertEqual(self.engine.pool.checkedout(), 0)
        self.assertEqual(self.engine.pool.size(), 1)

    async def test_sync_workers_share_one_owned_loop_and_close_its_pool(
        self,
    ) -> None:
        reader = SynchronousCatalogReader()
        try:
            with patch.dict(os.environ, {'DATABASE_URL': self.uri}):
                results = await asyncio.gather(
                    *(asyncio.to_thread(reader.rows) for _ in range(8)),
                )
            self.assertTrue(
                all(
                    any(row['model_key'] == self.key for row in result)
                    for result in results
                ),
            )
            engine = reader.engine
            thread = reader.thread
            assert engine is not None and thread is not None
            self.assertEqual(engine.pool.size(), 1)
            await self.connection.execute(
                'UPDATE detection_model_catalog SET definition=$1 '
                'WHERE model_key=$2',
                '{"enabled": false}',
                self.key,
            )
            result = await asyncio.to_thread(reader.rows)
            self.assertFalse(
                next(row for row in result if row['model_key'] == self.key)[
                    'definition'
                ]['enabled'],
            )
            self.assertIs(reader.engine, engine)
            self.assertIs(reader.thread, thread)
        finally:
            await asyncio.to_thread(reader.close)
        self.assertIsNone(reader.thread)
        self.assertFalse(thread.is_alive())
