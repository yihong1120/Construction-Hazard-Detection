from __future__ import annotations

import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

from examples.user_management.app import app
from examples.user_management.app import db


class TestUserManagement(unittest.TestCase):
    """
    Test suite for user management functionalities in the Flask app.
    """

    def setUp(self) -> None:
        """
        Set up the test environment before each test.
        """
        self.app = app
        self.client = self.app.test_client()
        self.app.testing = True

        # Set up an in-memory database for testing
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
        with app.app_context():
            db.create_all()

    def tearDown(self) -> None:
        """
        Clean up the test environment after each test.
        """
        with app.app_context():
            db.session.remove()
            db.drop_all()

    @patch('examples.user_management.app.add_user', return_value=True)
    def test_add_user_success(self, mock_add_user: MagicMock) -> None:
        """
        Test successful addition of a user.

        Args:
            mock_add_user: Mocked version of the add_user function.

        Asserts that the user is added successfully
        and the correct message is returned.
        """
        response = self.client.post(
            '/add_user',
            data={'username': 'testuser', 'password': 'password123'},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'User added successfully', response.data)
        mock_add_user.assert_called_once_with('testuser', 'password123')

    @patch('examples.user_management.app.add_user', return_value=False)
    def test_add_user_failure(self, mock_add_user: MagicMock) -> None:
        """
        Test failed addition of a user.

        Args:
            mock_add_user: Mocked version of the add_user function.

        Asserts that the user addition fails
        and the correct error message is returned.
        """
        response = self.client.post(
            '/add_user',
            data={'username': 'testuser', 'password': 'password123'},
        )
        self.assertEqual(response.status_code, 500)
        self.assertIn(b'Failed to add user', response.data)

    def test_add_user_missing_data(self) -> None:
        """
        Test addition of a user with missing data.

        Asserts that the request fails due to missing data
        and the appropriate error message is returned.
        """
        response = self.client.post('/add_user', data={'username': 'testuser'})
        self.assertEqual(response.status_code, 400)
        self.assertIn(b'Invalid input', response.data)

    @patch('examples.user_management.app.delete_user', return_value=True)
    def test_delete_user_success(self, mock_delete_user: MagicMock) -> None:
        """
        Test successful deletion of a user.

        Args:
            mock_delete_user: Mocked version
            of the delete_user function.

        Asserts that the user is deleted successfully
        and the correct message is returned.
        """
        response = self.client.delete('/delete_user/testuser')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'User deleted successfully', response.data)
        mock_delete_user.assert_called_once_with('testuser')

    @patch('examples.user_management.app.delete_user', return_value=False)
    def test_delete_user_failure(self, mock_delete_user: MagicMock) -> None:
        """
        Test failed deletion of a user.

        Args:
            mock_delete_user: Mocked version
            of the delete_user function.

        Asserts that the user deletion fails
        and the correct error message is returned.
        """
        response = self.client.delete('/delete_user/testuser')
        self.assertEqual(response.status_code, 500)
        self.assertIn(b'Failed to delete user', response.data)

    def test_delete_user_missing_username(self) -> None:
        """
        Test deletion of a user with a missing username.

        Asserts that the request fails due to a missing username
        and the appropriate error message is returned.
        """
        response = self.client.delete('/delete_user/')
        self.assertEqual(response.status_code, 404)

    @patch('examples.user_management.app.update_username', return_value=True)
    def test_update_username_success(
        self,
        mock_update_username: MagicMock,
    ) -> None:
        """
        Test successful username update.

        Args:
            mock_update_username: Mocked version
            of the update_username function.

        Asserts that the username is updated successfully
        and the correct message is returned.
        """
        response = self.client.put(
            '/update_username',
            data={'old_username': 'oldname', 'new_username': 'newname'},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Username updated successfully', response.data)
        mock_update_username.assert_called_once_with('oldname', 'newname')

    @patch(
        'examples.user_management.app.update_username',
        return_value=False,
    )
    def test_update_username_failure(
        self,
        mock_update_username: MagicMock,
    ) -> None:
        """
        Test failed username update.

        Args:
            mock_update_username: Mocked version
            of the update_username function.

        Asserts that the username update fails
        and the correct error message is returned.
        """
        response = self.client.put(
            '/update_username',
            data={'old_username': 'oldname', 'new_username': 'newname'},
        )
        self.assertEqual(response.status_code, 500)
        self.assertIn(b'Failed to update username', response.data)

    def test_update_username_missing_data(self) -> None:
        """
        Test username update with missing data.

        Asserts that the request fails due to missing data
        and the appropriate error message is returned.
        """
        response = self.client.put(
            '/update_username', data={'old_username': 'oldname'},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn(b'Invalid input', response.data)

    @patch(
        'examples.user_management.app.update_password',
        return_value=True,
    )
    def test_update_password_success(
        self, mock_update_password: MagicMock,
    ) -> None:
        """
        Test successful password update.

        Args:
            mock_update_password: Mocked version
            of the update_password function.

        Asserts that the password is updated successfully
        and the correct message is returned.
        """
        response = self.client.put(
            '/update_password',
            data={'username': 'testuser', 'new_password': 'newpassword123'},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Password updated successfully', response.data)
        mock_update_password.assert_called_once_with(
            'testuser', 'newpassword123',
        )

    @patch('examples.user_management.app.update_password', return_value=False)
    def test_update_password_failure(
        self,
        mock_update_password: MagicMock,
    ) -> None:
        """
        Test failed password update.

        Args:
            mock_update_password: Mocked version
            of the update_password function.

        Asserts that the password update fails
        and the correct error message is returned.
        """
        response = self.client.put(
            '/update_password',
            data={'username': 'testuser', 'new_password': 'newpassword123'},
        )
        self.assertEqual(response.status_code, 500)
        self.assertIn(b'Failed to update password', response.data)

    def test_update_password_missing_data(self) -> None:
        """
        Test password update with missing data.

        Asserts that the request fails due to missing data
        and the appropriate error message is returned.
        """
        response = self.client.put(
            '/update_password', data={'username': 'testuser'},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn(b'Invalid input', response.data)

    @patch(
        'examples.user_management.app.add_user',
        side_effect=Exception('Test exception'),
    )
    def test_add_user_internal_error(
        self,
        mock_add_user: MagicMock,
    ) -> None:
        """
        Test handling of an internal error during user addition.

        Args:
            mock_add_user: Mocked version of the add_user function.

        Asserts that an internal error during user addition is
        handled correctly.
        """
        response = self.client.post(
            '/add_user',
            data={'username': 'testuser', 'password': 'password123'},
        )
        self.assertEqual(response.status_code, 500)
        self.assertIn(b'An internal error occurred', response.data)

    @patch(
        'examples.user_management.app.delete_user',
        side_effect=Exception('Test exception'),
    )
    def test_delete_user_internal_error(
        self,
        mock_delete_user: MagicMock,
    ) -> None:
        """
        Test handling of an internal error during user deletion.

        Args:
            mock_delete_user: Mocked version
            of the delete_user function.

        Asserts that an internal error during user deletion is
        handled correctly.
        """
        response = self.client.delete('/delete_user/testuser')
        self.assertEqual(response.status_code, 500)
        self.assertIn(b'An internal error occurred', response.data)

    @patch(
        'examples.user_management.app.update_username',
        side_effect=Exception('Test exception'),
    )
    def test_update_username_internal_error(
        self,
        mock_update_username: MagicMock,
    ) -> None:
        """
        Test handling of an internal error during username update.

        Args:
            mock_update_username: Mocked version
            of the update_username function.

        Asserts that an internal error during username update is
        handled correctly.
        """
        response = self.client.put(
            '/update_username',
            data={'old_username': 'oldname', 'new_username': 'newname'},
        )
        self.assertEqual(response.status_code, 500)
        self.assertIn(b'An internal error occurred', response.data)

    @patch(
        'examples.user_management.app.update_password',
        side_effect=Exception('Test exception'),
    )
    def test_update_password_internal_error(
        self,
        mock_update_password: MagicMock,
    ) -> None:
        """
        Test handling of an internal error during password update.

        Args:
            mock_update_password: Mocked version
            of the update_password function.

        Asserts that an internal error during password update is
        handled correctly.
        """
        response = self.client.put(
            '/update_password',
            data={'username': 'testuser', 'new_password': 'newpassword123'},
        )
        self.assertEqual(response.status_code, 500)
        self.assertIn(b'An internal error occurred', response.data)


if __name__ == '__main__':
    unittest.main()
