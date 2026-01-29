from __future__ import annotations

import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """
    Configuration settings for the application.

    Attributes:
        authjwt_secret_key (str): The secret key for signing JWT tokens.
            Defaults to the value of the JWT_SECRET_KEY environment variable
            or 'your_fallback_secret_key' if not set.
        sqlalchemy_database_uri (str): The database connection URI (async).
            Defaults to the value of the DATABASE_URL environment variable
            or 'mysql+aiomysql://user:password@localhost/dbname' if not set.
        sqlalchemy_track_modifications (bool): Indicates whether SQLAlchemy
            should track modifications. Defaults to False.
    """

    authjwt_secret_key: str = os.getenv(
        'JWT_SECRET_KEY', 'your_fallback_secret_key',
    )
    sqlalchemy_database_uri: str = os.getenv(
        'DATABASE_URL', 'mysql+aiomysql://user:password@localhost/dbname',
    )
    sqlalchemy_track_modifications: bool = False

    ALGORITHM: str = 'HS256'

    def __init__(self) -> None:
        """
        Construct the Settings object.
        """
        super().__init__()
