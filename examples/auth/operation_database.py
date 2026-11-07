from __future__ import annotations

from weakref import WeakKeyDictionary

from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.ext.asyncio import AsyncConnection
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio import create_async_engine

from src.async_tasks import finish_on_cancel


class OperationDatabase:
    """Own independent journal and ownership pools with the source connector.

    Recreating the source pool preserves connect_args, async_creator and pool
    connection listeners. Locks have a separate pool so lock holders cannot
    exhaust the connections needed to commit their durable progress.
    """

    def __init__(self, source: AsyncEngine):

        self.engine = _clone_engine(source)
        self.locks = _clone_engine(source)
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)

    async def close(self):

        await finish_on_cancel(self._dispose())

    async def _dispose(self):
        try:
            await self.locks.dispose()
        finally:
            await self.engine.dispose()


_databases: WeakKeyDictionary[AsyncEngine, OperationDatabase] = (
    WeakKeyDictionary()
)


def operation_database(
    database: AsyncEngine | AsyncConnection,
) -> OperationDatabase:
    source = (
        database.engine if isinstance(database, AsyncConnection) else database
    )
    if source not in _databases:
        _databases[source] = OperationDatabase(source)
    return _databases[source]


async def close_operation_database(database: AsyncEngine):
    resources = _databases.pop(database, None)
    if resources is not None:
        await resources.close()


def _clone_engine(source: AsyncEngine) -> AsyncEngine:
    return create_async_engine(
        source.url,
        pool=source.sync_engine.pool.recreate(),
        execution_options=source.sync_engine.get_execution_options(),
    )
