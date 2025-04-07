from __future__ import annotations

import unittest

from pydantic import ValidationError

from examples.auth.schemas import DeleteUser
from examples.auth.schemas import LogoutRequest
from examples.auth.schemas import RefreshRequest
from examples.auth.schemas import SetUserActiveStatus
from examples.auth.schemas import UpdatePassword
from examples.auth.schemas import UpdateUsername
from examples.auth.schemas import UserCreate
from examples.auth.schemas import UserLogin


class TestUserLoginSchema(unittest.TestCase):
    """
    Tests the UserLogin schema which requires `username` and `password`.
    """

    def test_valid_userlogin(self) -> None:
        """
        Verify that a valid UserLogin with both fields provided
        is parsed correctly without errors.
        """
        user_login: UserLogin = UserLogin(
            username='testuser',
            password='secret123',
        )
        self.assertEqual(user_login.username, 'testuser')
        self.assertEqual(user_login.password, 'secret123')

    def test_missing_username(self) -> None:
        """
        Omit `username`; expect a ValidationError for the missing field.
        """
        with self.assertRaises(ValidationError):
            UserLogin(password='secret123')  # username missing

    def test_missing_password(self) -> None:
        """
        Omit `password`; expect a ValidationError for the missing field.
        """
        with self.assertRaises(ValidationError):
            UserLogin(username='testuser')  # password missing


class TestLogoutRequestSchema(unittest.TestCase):
    """
    Tests the LogoutRequest schema which requires `refresh_token`.
    """

    def test_valid_logout_request(self) -> None:
        """
        Provide a refresh token; expect instantiation to succeed.
        """
        request: LogoutRequest = LogoutRequest(
            refresh_token='some_refresh_token',
        )
        self.assertEqual(request.refresh_token, 'some_refresh_token')

    def test_missing_refresh_token(self) -> None:
        """
        Omit `refresh_token`; expect a ValidationError.
        """
        with self.assertRaises(ValidationError):
            LogoutRequest()  # no refresh_token


class TestRefreshRequestSchema(unittest.TestCase):
    """
    Tests the RefreshRequest schema which requires `refresh_token`.
    """

    def test_valid_refresh_request(self) -> None:
        """
        Provide a refresh token; expect normal instantiation.
        """
        refresh_req: RefreshRequest = RefreshRequest(
            refresh_token='valid_refresh_token',
        )
        self.assertEqual(refresh_req.refresh_token, 'valid_refresh_token')

    def test_missing_refresh_token(self) -> None:
        """
        Omit `refresh_token`; expect a ValidationError.
        """
        with self.assertRaises(ValidationError):
            RefreshRequest()  # no refresh_token


class TestUserCreateSchema(unittest.TestCase):
    """
    Tests the UserCreate schema which requires `username` and `password`,
    with an optional `role` defaulting to 'user'.
    """

    def test_valid_user_create(self) -> None:
        """
        Ensure `username` and `password` are set correctly,
        with `role` defaulting to 'user'.
        """
        user_create: UserCreate = UserCreate(
            username='u1',
            password='p1',
        )
        self.assertEqual(user_create.username, 'u1')
        self.assertEqual(user_create.password, 'p1')
        self.assertEqual(user_create.role, 'user')  # default

    def test_custom_role(self) -> None:
        """
        Provide a custom role 'admin' to verify it overrides the default.
        """
        user_create: UserCreate = UserCreate(
            username='u2',
            password='p2',
            role='admin',
        )
        self.assertEqual(user_create.role, 'admin')

    def test_missing_username(self) -> None:
        """
        Omit `username`; expect a ValidationError.
        """
        with self.assertRaises(ValidationError):
            UserCreate(password='p1')

    def test_missing_password(self) -> None:
        """
        Omit `password`; expect a ValidationError.
        """
        with self.assertRaises(ValidationError):
            UserCreate(username='u1')


class TestDeleteUserSchema(unittest.TestCase):
    """
    Tests the DeleteUser schema which requires `username`.
    """

    def test_valid_delete_user(self) -> None:
        """
        Provide a username; expect normal instantiation.
        """
        delete_req: DeleteUser = DeleteUser(username='target_user')
        self.assertEqual(delete_req.username, 'target_user')

    def test_missing_username(self) -> None:
        """
        Omit `username`; expect a ValidationError.
        """
        with self.assertRaises(ValidationError):
            DeleteUser()  # username missing


class TestUpdateUsernameSchema(unittest.TestCase):
    """
    Tests the UpdateUsername schema which requires `old_username`
    and `new_username`.
    """

    def test_valid_update_username(self) -> None:
        """
        Provide `old_username` and `new_username`; expect valid instantiation.
        """
        update_req: UpdateUsername = UpdateUsername(
            old_username='old',
            new_username='new',
        )
        self.assertEqual(update_req.old_username, 'old')
        self.assertEqual(update_req.new_username, 'new')

    def test_missing_fields(self) -> None:
        """
        Omit either `old_username` or `new_username`; expect ValidationError.
        """
        with self.assertRaises(ValidationError):
            UpdateUsername(old_username='only_old')
        with self.assertRaises(ValidationError):
            UpdateUsername(new_username='only_new')


class TestUpdatePasswordSchema(unittest.TestCase):
    """
    Tests the UpdatePassword schema which requires `username` and
    `new_password`, with an optional `role` defaulting to 'user'.
    """

    def test_valid_update_password(self) -> None:
        """
        Provide `username` and `new_password` with default role='user'.
        """
        update_req: UpdatePassword = UpdatePassword(
            username='u1',
            new_password='p1',
        )
        self.assertEqual(update_req.username, 'u1')
        self.assertEqual(update_req.new_password, 'p1')
        self.assertEqual(update_req.role, 'user')

    def test_custom_role(self) -> None:
        """
        Override the default role to 'admin'.
        """
        update_req: UpdatePassword = UpdatePassword(
            username='u2',
            new_password='p2',
            role='admin',
        )
        self.assertEqual(update_req.role, 'admin')

    def test_missing_username(self) -> None:
        """
        Omit `username`; expect ValidationError.
        """
        with self.assertRaises(ValidationError):
            UpdatePassword(new_password='p1')

    def test_missing_password(self) -> None:
        """
        Omit `new_password`; expect ValidationError.
        """
        with self.assertRaises(ValidationError):
            UpdatePassword(username='u1')


class TestSetUserActiveStatusSchema(unittest.TestCase):
    """
    Tests the SetUserActiveStatus schema which requires both `username`
    and `is_active`.
    """

    def test_valid_active_status(self) -> None:
        """
        Provide both fields; check correct instantiation and types.
        """
        status_req: SetUserActiveStatus = SetUserActiveStatus(
            username='testuser',
            is_active=True,
        )
        self.assertEqual(status_req.username, 'testuser')
        self.assertTrue(status_req.is_active)

    def test_missing_username(self) -> None:
        """
        Omit `username`; expect ValidationError.
        """
        with self.assertRaises(ValidationError):
            SetUserActiveStatus(is_active=False)

    def test_missing_is_active(self) -> None:
        """
        Omit `is_active`; expect ValidationError.
        """
        with self.assertRaises(ValidationError):
            SetUserActiveStatus(username='u1')

    def test_is_active_must_be_bool(self) -> None:
        """
        Provide a non-boolean for `is_active`; expect ValidationError.
        """
        with self.assertRaises(ValidationError):
            SetUserActiveStatus(username='u1', is_active='not_boolean')


if __name__ == '__main__':
    unittest.main()

'''
pytest\
    --cov=examples.auth.schemas \
    --cov-report=term-missing tests/examples/auth/schemas_test.py
'''
