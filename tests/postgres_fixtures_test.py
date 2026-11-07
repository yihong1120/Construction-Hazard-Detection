from __future__ import annotations

import os
import unittest
from uuid import uuid4

import asyncpg

from deploy.migrations.backup import _database_url

DATABASE = os.getenv('EVIDENCE_TEST_DATABASE_URL')


@unittest.skipUnless(DATABASE, 'Requires the isolated PostgreSQL test server')
class IsolatedPostgresDatabase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.control = await asyncpg.connect(DATABASE)
        self.database = 'chd_test_' + uuid4().hex
        self.url = _database_url(DATABASE, self.database)
        await self.control.execute(
            'CREATE DATABASE "' + self.database + '" TEMPLATE template0',
        )
        self.db = await asyncpg.connect(self.url)

    async def asyncTearDown(self):
        await self.db.close()
        try:
            await self.control.execute(
                'DROP DATABASE "' + self.database + '" WITH (FORCE)',
            )
        finally:
            await self.control.close()
