from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from typing import Any

db = SQLAlchemy()

class User(db.Model):
    """
    User model that represents an authenticated user in the database.
    
    Attributes:
        id (int): The primary key for the user.
        username (str): The username of the user, must be unique and is not nullable.
        password_hash (str): The hashed password for the user, not stored as plain text for security reasons.
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
            password (str): The plain text password to check against the stored hash.

        Returns:
            bool: True if the password matches the stored hash, otherwise False.
        """
        return check_password_hash(self.password_hash, password)
