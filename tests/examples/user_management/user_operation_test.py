from __future__ import annotations

import unittest

from flask import Flask

from examples.user_management.models import db
from examples.user_management.models import User
from examples.user_management.user_operation import add_user
from examples.user_management.user_operation import delete_user
from examples.user_management.user_operation import update_password
from examples.user_management.user_operation import update_username


class UserOperationTestCase(unittest.TestCase):
    """
    Unit tests for user operations
    (add, delete, update username, update password).
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

    def test_add_user_success(self) -> None:
        """
        Test adding a user successfully.
        """
        with self.app.app_context():
            response = add_user('testuser', 'password123')
            self.assertEqual(response, 'User testuser added successfully.')

            # Verify user was added
            user = User.query.filter_by(username='testuser').first()
            self.assertIsNotNone(user)
            self.assertTrue(user.check_password('password123'))

    def test_add_user_duplicate_username(self) -> None:
        """
        Test adding a user with a duplicate username.
        """
        with self.app.app_context():
            add_user('testuser', 'password123')
            response = add_user('testuser', 'password456')

            self.assertIn('Error adding user', response)

    def test_delete_user_success(self) -> None:
        """
        Test deleting a user successfully.
        """
        with self.app.app_context():
            add_user('testuser', 'password123')
            response = delete_user('testuser')
            self.assertEqual(response, 'User testuser deleted successfully.')

            # Verify user was deleted
            user = User.query.filter_by(username='testuser').first()
            self.assertIsNone(user)

    def test_delete_user_not_found(self) -> None:
        """
        Test deleting a non-existent user.
        """
        with self.app.app_context():
            response = delete_user('nonexistentuser')
            self.assertEqual(response, 'User nonexistentuser not found.')

    def test_update_username_success(self) -> None:
        """
        Test updating a user's username successfully.
        """
        with self.app.app_context():
            add_user('oldusername', 'password123')
            response = update_username('oldusername', 'newusername')
            self.assertEqual(
                response, 'Username updated successfully to newusername.',
            )

            # Verify the username was updated
            user = User.query.filter_by(username='newusername').first()
            self.assertIsNotNone(user)
            self.assertTrue(user.check_password('password123'))

    def test_update_username_not_found(self) -> None:
        """
        Test updating the username of a non-existent user.
        """
        with self.app.app_context():
            response = update_username('nonexistentuser', 'newusername')
            self.assertEqual(response, 'User nonexistentuser not found.')

    def test_update_password_success(self) -> None:
        """
        Test updating a user's password successfully.
        """
        with self.app.app_context():
            add_user('testuser', 'password123')
            response = update_password('testuser', 'newpassword123')
            self.assertEqual(
                response, 'Password updated successfully for user testuser.',
            )

            # Verify the password was updated
            user = User.query.filter_by(username='testuser').first()
            self.assertIsNotNone(user)
            self.assertTrue(user.check_password('newpassword123'))

    def test_update_password_user_not_found(self) -> None:
        """
        Test updating the password of a non-existent user.
        """
        with self.app.app_context():
            response = update_password('nonexistentuser', 'newpassword123')
            self.assertEqual(response, 'User nonexistentuser not found.')


if __name__ == '__main__':
    unittest.main()
