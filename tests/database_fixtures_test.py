from __future__ import annotations

from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine

from examples.auth.models import Base
from examples.auth.operation_database import close_operation_database
from tests.postgres_fixtures_test import IsolatedPostgresDatabase


class OrmDatabase(IsolatedPostgresDatabase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        await self.db.execute('CREATE EXTENSION IF NOT EXISTS pg_trgm')
        self.engine = create_async_engine(
            self.url.replace('postgresql://', 'postgresql+asyncpg://', 1),
        )
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def asyncTearDown(self):
        await close_operation_database(self.engine)
        await self.engine.dispose()
        await super().asyncTearDown()
