from __future__ import annotations

from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash
from werkzeug.security import generate_password_hash

db = SQLAlchemy()


class User(db.Model):  # type: ignore
    """
    User model that represents an authenticated user in the database.

    Attributes:
        id (int): The primary key for the user.
        username (str): The username, must be unique and not nullable.
        password_hash (str): The hashed password, not stored as plain text.
    """
    id: int = db.Column(db.Integer, primary_key=True)
    username: str = db.Column(db.String(50), unique=True, nullable=False)
    password_hash: str = db.Column(db.String(255), nullable=False)

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
