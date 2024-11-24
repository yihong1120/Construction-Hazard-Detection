from __future__ import annotations

from datetime import datetime
from datetime import timezone

from sqlalchemy import Boolean
from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy.orm import declarative_base
from werkzeug.security import check_password_hash
from werkzeug.security import generate_password_hash

Base = declarative_base()


class User(Base):  # type: ignore
    """
    Represents a user entity in the database with password hashing for
    security.
    """

    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(80), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    # Roles: admin, model_manage, user, guest
    role = Column(String(20), default='user', nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(
        DateTime, default=datetime.now(
            timezone.utc,
        ), nullable=False,
    )
    updated_at = Column(
        DateTime,
        default=datetime.now(timezone.utc),
        onupdate=datetime.now(timezone.utc),
        nullable=False,
    )

    def set_password(self, password: str) -> None:
        """
        Generates a password hash from a plain text password and stores it.

        Args:
            password (str): The plain text password to hash and store.
        """
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        """
        Checks if the provided password matches the stored hashed password.

        Args:
            password (str): The plain text password to check against the hash.

        Returns:
            bool: True if the password matches the hash, else False.
        """
        return check_password_hash(self.password_hash, password)
