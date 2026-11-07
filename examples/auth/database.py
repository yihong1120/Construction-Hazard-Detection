from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import DeclarativeBase

from examples.auth.config import Settings

settings: Settings = Settings()
DB_POOL_SIZE = settings.db_pool_size
DB_MAX_OVERFLOW = settings.db_max_overflow
DB_POOL_TIMEOUT_SECONDS = settings.db_pool_timeout_seconds
DB_POOL_RECYCLE_SECONDS = settings.db_pool_recycle_seconds

# Create an asynchronous SQLAlchemy engine
# using the database URI from settings.
sqlalchemy_database_uri = settings.sqlalchemy_database_uri

engine = create_async_engine(
    sqlalchemy_database_uri,
    pool_size=DB_POOL_SIZE,
    max_overflow=DB_MAX_OVERFLOW,
    pool_timeout=DB_POOL_TIMEOUT_SECONDS,
    pool_pre_ping=True,
    pool_recycle=DB_POOL_RECYCLE_SECONDS,
)

# Generate an asynchronous session factory using the configured engine.
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Base class for all ORM models in this application."""


async def get_db() -> AsyncGenerator[AsyncSession]:
    """Provide a SQLAlchemy asynchronous session for database operations.

    Yields:
        AsyncSession: A SQLAlchemy AsyncSession connected to the configured
            asynchronous engine.
    """

    async with AsyncSessionLocal() as session:
        yield session
