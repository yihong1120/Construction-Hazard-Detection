from __future__ import annotations

import os

from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()


class Config:
    """
    Configuration class for setting and retrieving environment variables.

    Attributes:
        JWT_SECRET_KEY (str): The secret key used for JWT authentication.
            Defaults to a fallback key if not set.
        SQLALCHEMY_DATABASE_URI (str): The URI for the SQL database connection.
        SQLALCHEMY_TRACK_MODIFICATIONS (bool): Flag to disable or enable
            track modifications feature of SQLAlchemy.
    """

    # Fetch the JWT secret key from environment or use a fallback
    JWT_SECRET_KEY: str = os.getenv(
        'JWT_SECRET_KEY',
        'your_fallback_secret_key',
    )

    # Get database URL from environment or use fallback connection string
    SQLALCHEMY_DATABASE_URI: str = os.getenv(
        'DATABASE_URL',
        'mysql://user:password@localhost/dbname',
    )

    # Set SQLAlchemy to not track modifications for performance benefits
    SQLALCHEMY_TRACK_MODIFICATIONS: bool = False
