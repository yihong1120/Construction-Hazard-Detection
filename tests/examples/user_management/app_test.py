from __future__ import annotations

import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

from examples.user_management.app import app
from examples.user_management.app import db


class TestUserManagement(unittest.TestCase):

    def setUp(self):
        # 每个测试前调用，创建测试客户端
        self.app = app
        self.client = self.app.test_client()
        self.app.testing = True

        # 使用SQLite内存数据库进行测试
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
        with app.app_context():
            db.create_all()

    @patch('examples.user_management.app.add_user', return_value=True)
    def test_add_user_success(self, mock_add_user):
        response = self.client.post(
            '/add_user', data={'username': 'testuser', 'password': 'password123'},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'User added successfully', response.data)
        mock_add_user.assert_called_once_with('testuser', 'password123')

    @patch('examples.user_management.app.add_user', return_value=False)
    def test_add_user_failure(self, mock_add_user):
        response = self.client.post(
            '/add_user', data={'username': 'testuser', 'password': 'password123'},
        )
        self.assertEqual(response.status_code, 500)
        self.assertIn(b'Failed to add user', response.data)

    def test_add_user_missing_data(self):
        response = self.client.post('/add_user', data={'username': 'testuser'})
        self.assertEqual(response.status_code, 400)
        self.assertIn(b'Invalid input', response.data)

    @patch('examples.user_management.app.delete_user', return_value=True)
    def test_delete_user_success(self, mock_delete_user):
        response = self.client.delete('/delete_user/testuser')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'User deleted successfully', response.data)
        mock_delete_user.assert_called_once_with('testuser')

    @patch('examples.user_management.app.delete_user', return_value=False)
    def test_delete_user_failure(self, mock_delete_user):
        response = self.client.delete('/delete_user/testuser')
        self.assertEqual(response.status_code, 500)
        self.assertIn(b'Failed to delete user', response.data)

    def test_delete_user_missing_username(self):
        response = self.client.delete('/delete_user/')
        self.assertEqual(response.status_code, 404)

    @patch('examples.user_management.app.update_username', return_value=True)
    def test_update_username_success(self, mock_update_username):
        response = self.client.put(
            '/update_username', data={'old_username': 'oldname', 'new_username': 'newname'},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Username updated successfully', response.data)
        mock_update_username.assert_called_once_with('oldname', 'newname')

    @patch('examples.user_management.app.update_username', return_value=False)
    def test_update_username_failure(self, mock_update_username):
        response = self.client.put(
            '/update_username', data={'old_username': 'oldname', 'new_username': 'newname'},
        )
        self.assertEqual(response.status_code, 500)
        self.assertIn(b'Failed to update username', response.data)

    def test_update_username_missing_data(self):
        response = self.client.put(
            '/update_username', data={'old_username': 'oldname'},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn(b'Invalid input', response.data)

    @patch('examples.user_management.app.update_password', return_value=True)
    def test_update_password_success(self, mock_update_password):
        response = self.client.put(
            '/update_password', data={'username': 'testuser', 'new_password': 'newpassword123'},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Password updated successfully', response.data)
        mock_update_password.assert_called_once_with(
            'testuser', 'newpassword123',
        )

    @patch('examples.user_management.app.update_password', return_value=False)
    def test_update_password_failure(self, mock_update_password):
        response = self.client.put(
            '/update_password', data={'username': 'testuser', 'new_password': 'newpassword123'},
        )
        self.assertEqual(response.status_code, 500)
        self.assertIn(b'Failed to update password', response.data)

    def test_update_password_missing_data(self):
        response = self.client.put(
            '/update_password', data={'username': 'testuser'},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn(b'Invalid input', response.data)

    @patch('examples.user_management.app.add_user', side_effect=Exception('Test exception'))
    def test_add_user_internal_error(self, mock_add_user):
        response = self.client.post(
            '/add_user', data={'username': 'testuser', 'password': 'password123'},
        )
        self.assertEqual(response.status_code, 500)
        self.assertIn(b'An internal error occurred', response.data)

    @patch('examples.user_management.app.delete_user', side_effect=Exception('Test exception'))
    def test_delete_user_internal_error(self, mock_delete_user):
        response = self.client.delete('/delete_user/testuser')
        self.assertEqual(response.status_code, 500)
        self.assertIn(b'An internal error occurred', response.data)

    @patch('examples.user_management.app.update_username', side_effect=Exception('Test exception'))
    def test_update_username_internal_error(self, mock_update_username):
        response = self.client.put(
            '/update_username', data={'old_username': 'oldname', 'new_username': 'newname'},
        )
        self.assertEqual(response.status_code, 500)
        self.assertIn(b'An internal error occurred', response.data)

    @patch('examples.user_management.app.update_password', side_effect=Exception('Test exception'))
    def test_update_password_internal_error(self, mock_update_password):
        response = self.client.put(
            '/update_password', data={'username': 'testuser', 'new_password': 'newpassword123'},
        )
        self.assertEqual(response.status_code, 500)
        self.assertIn(b'An internal error occurred', response.data)


if __name__ == '__main__':
    unittest.main()
