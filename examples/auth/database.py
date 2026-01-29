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
engine = create_async_engine(
    settings.sqlalchemy_database_uri.replace('mysql://', 'mysql+aiomysql://'),
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
