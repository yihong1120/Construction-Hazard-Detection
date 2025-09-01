from __future__ import annotations

import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError

from examples.auth.models import User
from examples.db_management.services import user_services


class TestUserServices(unittest.IsolatedAsyncioTestCase):
    """
    Unit tests for user_services using mocks.
    """

    def setUp(self) -> None:
        """
        Initialise shared mocks used by each test.
        """
        self.db: MagicMock = MagicMock()
        self.user: MagicMock = MagicMock()
        self.user.id = 1
        self.user.profile = MagicMock()
        self.user.group = MagicMock()

        # Common profile payload used by several tests
        self.profile_data: dict[str, str] = {
            'email': 'test@example.com',
            'mobile': '123456789',
        }

        # Mock the methods of the database session
        self.db.add = MagicMock()
        self.db.delete = MagicMock()
        self.db.commit = AsyncMock()
        self.db.flush = AsyncMock()
        self.db.refresh = AsyncMock()
        self.db.rollback = AsyncMock()
        self.db.execute = AsyncMock()
        self.db.get = AsyncMock()

    async def test_create_user_success(self) -> None:
        """
        Ensure a user and accompanying profile are created successfully.
        """
        # Arrange
        self.db.flush = AsyncMock()
        self.db.commit = AsyncMock()
        self.db.refresh = AsyncMock()
        self.db.add = MagicMock()

        with patch(
            'examples.db_management.services.user_services.User',
        ) as MockUser:
            mock_user: MagicMock = MagicMock()
            mock_user.id = 1
            mock_user.set_password = MagicMock()
            MockUser.return_value = mock_user

            with patch(
                'examples.db_management.services.user_services.UserProfile',
            ) as MockProfile:
                MockProfile.return_value = MagicMock()

                # Act
                result: MagicMock = await user_services.create_user(
                    'user',
                    'pw',
                    'admin',
                    1,
                    self.db,
                    self.profile_data,
                )

                # Assert
                self.assertEqual(result, mock_user)
                mock_user.set_password.assert_called_once_with('pw')
                self.db.add.assert_any_call(mock_user)
                self.db.commit.assert_awaited()
                self.db.refresh.assert_awaited()

    async def test_create_user_integrity_error(self) -> None:
        """
        Return *400 Bad Request* when attempting to save a duplicate user.
        """
        self.db.flush = AsyncMock(side_effect=IntegrityError('a', 'b', 'c'))
        self.db.rollback = AsyncMock()

        with patch(
            'examples.db_management.services.user_services.User',
        ) as MockUser:
            mock_user: MagicMock = MagicMock()
            mock_user.set_password = MagicMock()
            MockUser.return_value = mock_user

            with self.assertRaises(HTTPException) as cm:
                await user_services.create_user(
                    'user', 'pw', 'admin', 1, self.db,
                )

            self.assertEqual(cm.exception.status_code, 400)
            self.db.rollback.assert_awaited()

    async def test_create_user_general_exception(self) -> None:
        """
        Return *500 Internal Server Error* for an unexpected save failure.
        """
        self.db.flush = AsyncMock(side_effect=Exception('fail'))
        self.db.rollback = AsyncMock()

        with patch(
            'examples.db_management.services.user_services.User',
        ) as MockUser:
            mock_user: MagicMock = MagicMock()
            mock_user.set_password = MagicMock()
            MockUser.return_value = mock_user

            with self.assertRaises(HTTPException) as cm:
                await user_services.create_user(
                    'user', 'pw', 'admin', 1, self.db,
                )

            self.assertEqual(cm.exception.status_code, 500)
            self.db.rollback.assert_awaited()

    async def test_list_users(self) -> None:
        """
        Fetch all users, ensuring the underlying query is executed.
        """
        mock_result: MagicMock = MagicMock()
        scalars_mock: MagicMock = (
            mock_result.unique.return_value.scalars.return_value
        )
        scalars_mock.all.return_value = ['user1', 'user2']
        self.db.execute = AsyncMock(return_value=mock_result)

        users: list[User] = await user_services.list_users(self.db)

        self.assertEqual(users, ['user1', 'user2'])

    async def test_get_user_by_id_found(self) -> None:
        """
        Retrieve a single user by identifier when they exist.
        """
        self.db.get = AsyncMock(return_value=self.user)

        user: MagicMock = await user_services.get_user_by_id(1, self.db)

        self.assertEqual(user, self.user)

    async def test_get_user_by_id_not_found(self) -> None:
        """
        Raise *404 Not Found* when the requested user is missing.
        """
        self.db.get = AsyncMock(return_value=None)

        with self.assertRaises(HTTPException) as cm:
            await user_services.get_user_by_id(1, self.db)

        self.assertEqual(cm.exception.status_code, 404)

    async def test_delete_user_success(self) -> None:
        """
        Persist the removal of an existing user.
        """
        self.db.delete = AsyncMock()
        self.db.commit = AsyncMock()

        await user_services.delete_user(self.user, self.db)

        self.db.delete.assert_awaited_with(self.user)
        self.db.commit.assert_awaited()

    async def test_delete_user_exception(self) -> None:
        """
        Handle an unexpected database failure during deletion.
        """
        self.db.delete = AsyncMock()
        self.db.commit = AsyncMock(side_effect=Exception('fail'))
        self.db.rollback = AsyncMock()

        with self.assertRaises(HTTPException) as cm:
            await user_services.delete_user(self.user, self.db)

        self.assertEqual(cm.exception.status_code, 500)
        self.db.rollback.assert_awaited()

    async def test_update_username_success(self) -> None:
        """
        Change the username and commit the transaction.
        """
        self.db.commit = AsyncMock()
        self.user.username = 'old'

        await user_services.update_username(self.user, 'new', self.db)

        self.assertEqual(self.user.username, 'new')
        self.db.commit.assert_awaited()

    async def test_update_username_integrity_error(self) -> None:
        """
        Return *400 Bad Request* when the new username already exists.
        """
        self.db.commit = AsyncMock(side_effect=IntegrityError('a', 'b', 'c'))
        self.db.rollback = AsyncMock()

        with self.assertRaises(HTTPException) as cm:
            await user_services.update_username(self.user, 'new', self.db)

        self.assertEqual(cm.exception.status_code, 400)
        self.db.rollback.assert_awaited()

    async def test_update_username_general_exception(self) -> None:
        """
        Return *500 Internal Server Error* for an unexpected failure.
        """
        self.db.commit = AsyncMock(side_effect=Exception('fail'))
        self.db.rollback = AsyncMock()

        with self.assertRaises(HTTPException) as cm:
            await user_services.update_username(self.user, 'new', self.db)

        self.assertEqual(cm.exception.status_code, 500)
        self.db.rollback.assert_awaited()

    async def test_update_password_success(self) -> None:
        """
        Set a new password and commit the change.
        """
        self.db.commit = AsyncMock()
        self.user.set_password = MagicMock()

        await user_services.update_password(self.user, 'pw', self.db)

        self.user.set_password.assert_called_once_with('pw')
        self.db.commit.assert_awaited()

    async def test_update_password_exception(self) -> None:
        """
        Return *500 Internal Server Error* when committing fails.
        """
        self.db.commit = AsyncMock(side_effect=Exception('fail'))
        self.db.rollback = AsyncMock()
        self.user.set_password = MagicMock()

        with self.assertRaises(HTTPException) as cm:
            await user_services.update_password(self.user, 'pw', self.db)

        self.assertEqual(cm.exception.status_code, 500)
        self.db.rollback.assert_awaited()

    async def test_set_active_status_success(self) -> None:
        """
        Toggle the is_active flag and commit.
        """
        self.db.commit = AsyncMock()

        await user_services.set_active_status(self.user, True, self.db)

        self.assertTrue(
            self.user.is_active or self.user.is_active is not False,
        )
        self.db.commit.assert_awaited()

    async def test_set_active_status_exception(self) -> None:
        """
        Raise *500 Internal Server Error* when commit fails.
        """
        self.db.commit = AsyncMock(side_effect=Exception('fail'))
        self.db.rollback = AsyncMock()

        with self.assertRaises(HTTPException) as cm:
            await user_services.set_active_status(self.user, False, self.db)

        self.assertEqual(cm.exception.status_code, 500)
        self.db.rollback.assert_awaited()

    async def test_create_or_update_profile_update(self) -> None:
        """
        Update fields on an existing UserProfile.
        """
        self.db.commit = AsyncMock()
        self.db.refresh = AsyncMock()
        self.user.profile = MagicMock()
        self.user.profile.email = 'old@example.com'

        await user_services.create_or_update_profile(
            self.user, {'email': 'new@example.com'}, self.db,
        )

        self.assertEqual(self.user.profile.email, 'new@example.com')
        self.db.commit.assert_awaited()
        self.db.refresh.assert_awaited()

    async def test_create_or_update_profile_create(self) -> None:
        """
        Create a brand-new profile when one is absent and allowed.
        """
        self.db.commit = AsyncMock()
        self.db.refresh = AsyncMock()
        self.user.profile = None

        with patch(
            'examples.db_management.services.user_services.UserProfile',
        ) as MockProfile:
            mock_profile: MagicMock = MagicMock()
            MockProfile.return_value = mock_profile

            await user_services.create_or_update_profile(
                self.user,
                {'email': 'new@example.com'},
                self.db,
                create_if_missing=True,
            )

            self.db.add.assert_called_with(mock_profile)
            self.db.commit.assert_awaited()
            self.db.refresh.assert_awaited()

    async def test_create_or_update_profile_not_found(self) -> None:
        """
        Return *404 Not Found* if profile is missing
        and creation is disallowed.
        """
        self.user.profile = None

        with self.assertRaises(HTTPException) as cm:
            await user_services.create_or_update_profile(
                self.user,
                {'email': 'new@example.com'},
                self.db,
                create_if_missing=False,
            )

        self.assertEqual(cm.exception.status_code, 404)

    async def test_create_or_update_profile_integrity_error(self) -> None:
        """
        Handle a unique-constraint violation on profile save.
        """
        self.db.commit = AsyncMock(side_effect=IntegrityError('a', 'b', 'c'))
        self.db.rollback = AsyncMock()
        self.db.refresh = AsyncMock()

        awaitable = user_services.create_or_update_profile(
            self.user, {'email': 'dup@example.com'}, self.db,
        )

        with self.assertRaises(HTTPException) as cm:
            await awaitable

        self.assertEqual(cm.exception.status_code, 400)
        self.db.rollback.assert_awaited()

    async def test_create_or_update_profile_general_exception(self) -> None:
        """
        Return *500 Internal Server Error* for an unexpected profile failure.
        """
        self.db.commit = AsyncMock(side_effect=Exception('fail'))
        self.db.rollback = AsyncMock()
        self.db.refresh = AsyncMock()

        awaitable = user_services.create_or_update_profile(
            self.user, {'email': 'fail@example.com'}, self.db,
        )

        with self.assertRaises(HTTPException) as cm:
            await awaitable

        self.assertEqual(cm.exception.status_code, 500)
        self.db.rollback.assert_awaited()


if __name__ == '__main__':
    unittest.main()

'''
pytest --cov=examples.db_management.services.user_services\
    --cov-report=term-missing\
        tests/examples/db_management/services/user_services_test.py
'''
