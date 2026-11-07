from __future__ import annotations

import asyncio
import atexit
import os
import threading
from collections.abc import Mapping
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine

from examples.auth.config import Settings
from examples.auth.database import AsyncSessionLocal
from examples.auth.models import DetectionModelCatalog
from examples.auth.models import DetectionModelVersion


class CatalogRepository:
    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession] = AsyncSessionLocal,
    ) -> None:
        self.sessions = sessions

    async def rows(self) -> list[Mapping[str, Any]]:
        async with asyncio.timeout(5), self.sessions() as db:
            result = await db.execute(
                select(
                    DetectionModelCatalog.model_key,
                    DetectionModelCatalog.definition,
                    DetectionModelCatalog.is_default,
                ).order_by(
                    DetectionModelCatalog.display_order,
                    DetectionModelCatalog.model_key,
                ),
            )
            return list(result.mappings())

    async def version(self, model_key: str, version: str):
        async with asyncio.timeout(5), self.sessions() as db:
            return await db.get(DetectionModelVersion, (model_key, version))


catalog_repository = CatalogRepository()


class SynchronousCatalogReader:
    """Own one event loop and a bounded DB pool for synchronous ML workers."""

    def __init__(self) -> None:
        self._reset()

    def _reset(self) -> None:
        self.pid = os.getpid()
        self.lock = threading.Lock()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.thread: threading.Thread | None = None
        self.repository: CatalogRepository | None = None
        self.engine: AsyncEngine | None = None
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None

    def _serve(self) -> None:
        loop = asyncio.new_event_loop()
        self.loop = loop
        asyncio.set_event_loop(loop)
        try:
            self.engine = create_async_engine(
                Settings().sqlalchemy_database_uri,
                pool_size=1,
                max_overflow=0,
                pool_pre_ping=True,
                pool_timeout=5,
                pool_recycle=1800,
            )
            self.repository = CatalogRepository(
                async_sessionmaker(self.engine, expire_on_commit=False),
            )
        except BaseException as exc:
            self._startup_error = exc
            self._ready.set()
            loop.close()
            return
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            loop.close()

    def rows(self) -> list[Mapping[str, Any]]:
        if self.pid != os.getpid():
            self._reset()
        with self.lock:
            if self.thread is None:
                self.thread = threading.Thread(
                    target=self._serve,
                    name='model-catalog',
                    daemon=True,
                )
                self.thread.start()
            if not self._ready.wait(timeout=10):
                raise TimeoutError('Model catalog did not start')
            if self._startup_error is not None:
                raise RuntimeError(
                    'Model catalog could not start',
                ) from self._startup_error
            assert self.loop is not None and self.repository is not None
            future = asyncio.run_coroutine_threadsafe(
                self.repository.rows(),
                self.loop,
            )
        try:
            return future.result(timeout=10)
        except BaseException:
            future.cancel()
            raise

    def close(self) -> None:
        if self.pid != os.getpid():
            return
        with self.lock:
            if self.loop is not None and self.loop.is_running():
                try:
                    if self.engine is not None:
                        asyncio.run_coroutine_threadsafe(
                            self.engine.dispose(),
                            self.loop,
                        ).result(timeout=10)
                finally:
                    self.loop.call_soon_threadsafe(self.loop.stop)
                    assert self.thread is not None
                    self.thread.join(timeout=10)
            self.thread = None
            self.loop = None
            self.repository = None
            self.engine = None
            self._ready.clear()
            self._startup_error = None


synchronous_catalog = SynchronousCatalogReader()
atexit.register(synchronous_catalog.close)
