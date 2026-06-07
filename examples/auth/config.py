from __future__ import annotations

import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables from .env file
load_dotenv()


def _postgres_async_url(database_url: str) -> str:
    """Normalize PostgreSQL URLs to the asyncpg driver."""
    replacements = (
        ('postgres://', 'postgresql+asyncpg://'),
        ('postgresql://', 'postgresql+asyncpg://'),
    )
    for old, new in replacements:
        if database_url.startswith(old):
            return database_url.replace(old, new, 1)
    return database_url


class Settings(BaseSettings):
    """
    Configuration settings for the application.

    Attributes:
        authjwt_secret_key (str): The secret key for signing JWT tokens.
            Defaults to the value of the JWT_SECRET_KEY environment variable.
        sqlalchemy_database_uri (str): The database connection URI (async).
            Defaults to the value of the DATABASE_URL environment variable
            or 'postgresql+asyncpg://user:password@localhost/dbname' if
            not set.
        sqlalchemy_track_modifications (bool): Indicates whether SQLAlchemy
            should track modifications. Defaults to False.
    """

    authjwt_secret_key: str = os.getenv('JWT_SECRET_KEY', '')
    sqlalchemy_database_uri: str = _postgres_async_url(
        os.getenv(
            'DATABASE_URL',
            'postgresql+asyncpg://user:password@localhost/dbname',
        ),
    )
    sqlalchemy_track_modifications: bool = False

    ALGORITHM: str = 'HS256'

    def __init__(self) -> None:
        """
        Construct the Settings object.
        """
        super().__init__()
        if not self.authjwt_secret_key:
            raise RuntimeError('JWT_SECRET_KEY is required')
