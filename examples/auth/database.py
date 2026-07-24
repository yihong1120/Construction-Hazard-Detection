from __future__ import annotations

import os
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import DeclarativeBase

from examples.auth.config import Settings


def _nonnegative_int_env(name: str, default: int) -> int:
    """Read a non-negative integer environment setting with a safe default."""
    try:
        return max(0, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _positive_float_env(name: str, default: float) -> float:
    """Read a positive float environment setting with a safe default."""
    try:
        return max(0.1, float(os.getenv(name, str(default))))
    except ValueError:
        return default


DB_POOL_SIZE = max(1, _nonnegative_int_env('DB_POOL_SIZE', 2))
DB_MAX_OVERFLOW = _nonnegative_int_env('DB_MAX_OVERFLOW', 1)
DB_POOL_TIMEOUT_SECONDS = _positive_float_env(
    'DB_POOL_TIMEOUT_SECONDS',
    10.0,
)
DB_POOL_RECYCLE_SECONDS = max(
    1,
    _nonnegative_int_env('DB_POOL_RECYCLE_SECONDS', 1800),
)

# Instantiate the Settings object to retrieve environment-based configurations
settings: Settings = Settings()

# Create an asynchronous SQLAlchemy engine
# using the database URI from settings.
sqlalchemy_database_uri = settings.sqlalchemy_database_uri
if sqlalchemy_database_uri.startswith('postgres://'):
    sqlalchemy_database_uri = sqlalchemy_database_uri.replace(
        'postgres://', 'postgresql+asyncpg://', 1,
    )
elif sqlalchemy_database_uri.startswith('postgresql://'):
    sqlalchemy_database_uri = sqlalchemy_database_uri.replace(
        'postgresql://', 'postgresql+asyncpg://', 1,
    )

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
    """
    Base class for all ORM models in this application.
    """


async def get_db() -> AsyncGenerator[AsyncSession]:
    """
    Provide a SQLAlchemy asynchronous session for database operations.

    Yields:
        AsyncSession: A SQLAlchemy AsyncSession connected to the configured
            asynchronous engine.
    """
    async with AsyncSessionLocal() as session:
        yield session
