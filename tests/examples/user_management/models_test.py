from __future__ import annotations

import unittest

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from examples.user_management.models import User, db


class UserModelTestCase(unittest.TestCase):
    """
    Unit tests for the User model.
    """

    def setUp(self) -> None:
        """
        Set up the test environment before each test.
        """
        # Create a Flask application for testing
        self.app = Flask(__name__)
        self.app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        self.app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

        # Initialize the SQLAlchemy database
        db.init_app(self.app)

        # Create the database and the User table
        with self.app.app_context():
            db.create_all()

        # Create a test client
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        """
        Clean up the test environment after each test.
        """
        # Drop all tables from the database
        with self.app.app_context():
            db.session.remove()
            db.drop_all()

    def test_set_password(self) -> None:
        """
        Test that the password is hashed correctly when set.
        """
        with self.app.app_context():
            user = User(username="testuser")
            user.set_password("password123")

            # Ensure the password_hash is not equal to the plain text password
            self.assertNotEqual(user.password_hash, "password123")

            # Ensure the password_hash is not empty
            self.assertTrue(user.password_hash)

    def test_check_password(self) -> None:
        """
        Test that the correct password validates against the hash.
        """
        with self.app.app_context():
            user = User(username="testuser")
            user.set_password("password123")

            # Ensure the password check returns True for correct password
            self.assertTrue(user.check_password("password123"))

            # Ensure the password check returns False for incorrect password
            self.assertFalse(user.check_password("wrongpassword"))

    def test_unique_username(self) -> None:
        """
        Test that usernames must be unique in the database.
        """
        with self.app.app_context():
            # Create a user with a unique username
            user1 = User(username="uniqueuser")
            user1.set_password("password123")
            db.session.add(user1)
            db.session.commit()

            # Attempt to create another user with the same username
            user2 = User(username="uniqueuser")
            user2.set_password("password456")
            db.session.add(user2)

            # Ensure an integrity error is raised due to the unique constraint
            with self.assertRaises(Exception):
                db.session.commit()


if __name__ == "__main__":
    unittest.main()
