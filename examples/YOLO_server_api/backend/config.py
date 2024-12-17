from __future__ import annotations

import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables from a .env file
load_dotenv()


class Settings(BaseSettings):
    """
    A class to represent the application settings.

    Attributes:
        authjwt_secret_key (str): The secret key for JWT authentication.
        sqlalchemy_database_uri (str): The URI for the SQLAlchemy database
            connection.
        sqlalchemy_track_modifications (bool): Flag to track modifications in
            SQLAlchemy.
    """

    authjwt_secret_key: str = os.getenv(
        'JWT_SECRET_KEY',
        'your_fallback_secret_key',
    )
    sqlalchemy_database_uri: str = os.getenv(
        'DATABASE_URL',
        'mysql+asyncmy://user:password@localhost/dbname',
    )
    sqlalchemy_track_modifications: bool = False

    def __init__(self) -> None:
        """
        Initialise the Settings instance with environment variables.

        If the environment variables are not set, fallback values will be used.
        """
        super().__init__()  # Ensure the BaseSettings initialisation is called
