from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import DeclarativeBase

from examples.auth.config import Settings

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
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=3600,
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
