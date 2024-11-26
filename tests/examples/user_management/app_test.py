from __future__ import annotations

import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi.testclient import TestClient

from examples.user_management.app import app


class TestUserManagement(unittest.TestCase):
    """
    Test suite for user management functionalities in the FastAPI app.
    """

    def setUp(self) -> None:
        """
        Set up the test environment before each test.
        """
        self.client = TestClient(app)

    @patch(
        'examples.user_management.app.add_user',
        return_value={
            'success': True,
            'message': "User 'testuser' added successfully.",
        },
    )
    def test_add_user_success(self, mock_add_user: MagicMock) -> None:
        """
        Test successful addition of a user.

        Args:
            mock_add_user: Mocked version of the add_user function.
        """
        response = self.client.post(
            '/add_user',
            json={
                'username': 'testuser',
                'password': 'password123', 'role': 'user',
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn('User added successfully.', response.json()['message'])
        mock_add_user.assert_called_once_with(
            'testuser', 'password123', 'user', unittest.mock.ANY,
        )

    @patch(
        'examples.user_management.app.add_user',
        return_value={
            'success': False, 'error': 'IntegrityError',
            'message': "Username 'testuser' already exists.",
        },
    )
    def test_add_user_failure(self, mock_add_user: MagicMock) -> None:
        """
        Test failed addition of a user.

        Args:
            mock_add_user: Mocked version of the add_user function.
        """
        response = self.client.post(
            '/add_user',
            json={
                'username': 'testuser',
                'password': 'password123', 'role': 'user',
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn('Failed to add user.', response.json()['detail'])

    def test_add_user_missing_data(self) -> None:
        """
        Test addition of a user with missing data.
        """
        response = self.client.post('/add_user', json={'username': 'testuser'})
        self.assertEqual(response.status_code, 422)  # Validation error

    @patch(
        'examples.user_management.app.delete_user',
        return_value={
            'success': True,
            'message': "User 'testuser' deleted successfully.",
        },
    )
    def test_delete_user_success(self, mock_delete_user: MagicMock) -> None:
        """
        Test successful deletion of a user.

        Args:
            mock_delete_user: Mocked version of the delete_user function.
        """
        response = self.client.delete('/delete_user/testuser')
        self.assertEqual(response.status_code, 200)
        self.assertIn('User deleted successfully.', response.json()['message'])
        mock_delete_user.assert_called_once_with('testuser', unittest.mock.ANY)

    @patch(
        'examples.user_management.app.delete_user',
        return_value={
            'success': False, 'error': 'NotFound',
            'message': "User 'testuser' not found.",
        },
    )
    def test_delete_user_failure(self, mock_delete_user: MagicMock) -> None:
        """
        Test failed deletion of a user.

        Args:
            mock_delete_user: Mocked version of the delete_user function.
        """
        response = self.client.delete('/delete_user/testuser')
        self.assertEqual(response.status_code, 404)
        self.assertIn('Failed to delete user.', response.json()['detail'])

    @patch(
        'examples.user_management.app.update_username',
        return_value={
            'success': True,
            'message': "Username updated from 'olduser' to 'newuser'.",
        },
    )
    def test_update_username_success(
        self, mock_update_username: MagicMock,
    ) -> None:
        """
        Test successful update of a user's username.

        Args:
            mock_update_username: Mocked version of the
                update_username function.
        """
        response = self.client.put(
            '/update_username',
            json={'old_username': 'olduser', 'new_username': 'newuser'},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(
            'Username updated successfully.',
            response.json()['message'],
        )
        mock_update_username.assert_called_once_with(
            'olduser', 'newuser', unittest.mock.ANY,
        )

    @patch(
        'examples.user_management.app.update_username',
        return_value={
            'success': False, 'error': 'NotFound',
            'message': "User 'olduser' not found.",
        },
    )
    def test_update_username_failure(
        self, mock_update_username: MagicMock,
    ) -> None:
        """
        Test failed update of a user's username.

        Args:
            mock_update_username: Mocked version of the
                update_username function.
        """
        response = self.client.put(
            '/update_username',
            json={'old_username': 'olduser', 'new_username': 'newuser'},
        )
        self.assertEqual(response.status_code, 404)
        self.assertIn('Failed to update username.', response.json()['detail'])

    @patch(
        'examples.user_management.app.update_password',
        return_value={
            'success': True,
            'message': "Password updated successfully for user 'testuser'.",
        },
    )
    def test_update_password_success(
        self, mock_update_password: MagicMock,
    ) -> None:
        """
        Test successful update of a user's password.

        Args:
            mock_update_password: Mocked version of the
                update_password function.
        """
        response = self.client.put(
            '/update_password',
            json={'username': 'testuser', 'new_password': 'newpassword123'},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(
            'Password updated successfully.',
            response.json()['message'],
        )
        mock_update_password.assert_called_once_with(
            'testuser', 'newpassword123', unittest.mock.ANY,
        )

    @patch(
        'examples.user_management.app.update_password',
        return_value={
            'success': False, 'error': 'NotFound',
            'message': "User 'testuser' not found.",
        },
    )
    def test_update_password_failure(
        self, mock_update_password: MagicMock,
    ) -> None:
        """
        Test failed update of a user's password.

        Args:
            mock_update_password: Mocked version of the
                update_password function.
        """
        response = self.client.put(
            '/update_password',
            json={'username': 'testuser', 'new_password': 'newpassword123'},
        )
        self.assertEqual(response.status_code, 404)
        self.assertIn('Failed to update password.', response.json()['detail'])


if __name__ == '__main__':
    unittest.main()
