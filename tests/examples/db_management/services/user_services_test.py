from __future__ import annotations

import unittest
from typing import Any
from typing import cast
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
        self.user = cast(User, MagicMock())
        self.user.id = 1
        self.user.profile = MagicMock()
        self.user.group = MagicMock()

        # Common profile payload used by several tests
        self.profile_data: dict[str, str] = {
            'email': 'test@example.com',
            'mobile_number': '123456789',
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
                result = await user_services.create_user(
                    'user',
                    'password',
                    'admin',
                    1,
                    self.db,
                    self.profile_data,
                )

                # Assert
                self.assertEqual(result, mock_user)
                mock_user.set_password.assert_called_once_with('password')
                self.db.add.assert_any_call(mock_user)
                self.db.commit.assert_awaited()
                self.db.refresh.assert_awaited()

    async def test_create_user_rejects_short_password(self) -> None:
        """Reject user creation before any database writes for short password."""
        with self.assertRaises(HTTPException) as cm:
            await user_services.create_user(
                'user', 'short', 'admin', 1, self.db,
            )

        self.assertEqual(cm.exception.status_code, 400)
        self.assertEqual(
            cm.exception.detail,
            {'code': 'password_too_short', 'min_length': 8},
        )
        self.db.add.assert_not_called()

    async def test_create_user_inactive_success(self) -> None:
        """
        Allow creating a pending account with pending status.
        """
        self.db.flush = AsyncMock()
        self.db.commit = AsyncMock()
        self.db.refresh = AsyncMock()
        self.db.add = MagicMock()

        with patch(
            'examples.db_management.services.user_services.User',
        ) as MockUser:
            mock_user: MagicMock = MagicMock()
            mock_user.id = 2
            mock_user.set_password = MagicMock()
            MockUser.return_value = mock_user

            await user_services.create_user(
                'pending',
                'password',
                'user',
                None,
                self.db,
                self.profile_data,
                status='pending',
            )

            self.assertEqual(MockUser.call_args.kwargs['status'], 'pending')

    async def test_create_user_integrity_error(self) -> None:
        """
        Return *400 Bad Request* when attempting to save a duplicate user.
        """
        self.db.flush = AsyncMock(
            side_effect=IntegrityError('a', 'b', Exception('c')),
        )
        self.db.rollback = AsyncMock()

        with patch(
            'examples.db_management.services.user_services.User',
        ) as MockUser:
            mock_user: MagicMock = MagicMock()
            mock_user.set_password = MagicMock()
            MockUser.return_value = mock_user

            with self.assertRaises(HTTPException) as cm:
                await user_services.create_user(
                    'user', 'password', 'admin', 1, self.db,
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
                    'user', 'password', 'admin', 1, self.db,
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
        result = MagicMock()
        result.unique.return_value.scalar_one_or_none.return_value = self.user
        self.db.execute = AsyncMock(return_value=result)

        user = await user_services.get_user_by_id(1, self.db)

        self.assertEqual(user, self.user)

    async def test_get_user_by_id_not_found(self) -> None:
        """
        Raise *404 Not Found* when the requested user is missing.
        """
        result = MagicMock()
        result.unique.return_value.scalar_one_or_none.return_value = None
        self.db.execute = AsyncMock(return_value=result)

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
        self.db.commit = AsyncMock(
            side_effect=IntegrityError('a', 'b', Exception('c')),
        )
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
        db = cast(Any, self.db)
        user = cast(Any, self.user)
        db.commit = AsyncMock()
        user.set_password = MagicMock()

        await user_services.update_password(self.user, 'password', self.db)

        user.set_password.assert_called_once_with('password')
        db.commit.assert_awaited()

    async def test_update_password_rejects_short_password(self) -> None:
        """Reject short replacement passwords without committing."""
        db = cast(Any, self.db)
        user = cast(Any, self.user)
        user.set_password = MagicMock()

        with self.assertRaises(HTTPException) as cm:
            await user_services.update_password(self.user, 'short', self.db)

        self.assertEqual(cm.exception.status_code, 400)
        self.assertEqual(
            cm.exception.detail,
            {'code': 'password_too_short', 'min_length': 8},
        )
        user.set_password.assert_not_called()
        db.commit.assert_not_awaited()

    async def test_update_password_exception(self) -> None:
        """
        Return *500 Internal Server Error* when committing fails.
        """
        db = cast(Any, self.db)
        user = cast(Any, self.user)
        db.commit = AsyncMock(side_effect=Exception('fail'))
        db.rollback = AsyncMock()
        user.set_password = MagicMock()

        with self.assertRaises(HTTPException) as cm:
            await user_services.update_password(self.user, 'password', self.db)

        self.assertEqual(cm.exception.status_code, 500)
        db.rollback.assert_awaited()

    async def test_set_user_status_success(self) -> None:
        """
        Update the status field and commit.
        """
        self.db.commit = AsyncMock()

        await user_services.set_user_status(self.user, 'active', self.db)

        self.assertEqual(self.user.status, 'active')
        self.db.commit.assert_awaited()

    async def test_set_user_status_exception(self) -> None:
        """
        Raise *500 Internal Server Error* when commit fails.
        """
        self.db.commit = AsyncMock(side_effect=Exception('fail'))
        self.db.rollback = AsyncMock()

        with self.assertRaises(HTTPException) as cm:
            await user_services.set_user_status(self.user, 'suspended', self.db)

        self.assertEqual(cm.exception.status_code, 500)
        self.db.rollback.assert_awaited()

    async def test_set_user_status_invalid(self) -> None:
        """Reject unknown account statuses."""
        with self.assertRaises(HTTPException) as cm:
            await user_services.set_user_status(self.user, 'unknown', self.db)

        self.assertEqual(cm.exception.status_code, 400)

    async def test_create_or_update_profile_update(self) -> None:
        """
        Update fields on an existing UserProfile.
        """
        db = cast(Any, self.db)
        user = cast(Any, self.user)
        db.commit = AsyncMock()
        db.refresh = AsyncMock()
        user.profile = MagicMock()
        user.profile.email = 'old@example.com'
        user.profile.family_name = 'Old'

        await user_services.create_or_update_profile(
            self.user,
            {'email': 'new@example.com', 'family_name': 'New'},
            self.db,
        )

        self.assertEqual(user.profile.email, 'new@example.com')
        self.assertEqual(user.profile.family_name, 'New')
        db.commit.assert_awaited()
        db.refresh.assert_awaited()

    async def test_create_or_update_profile_create(self) -> None:
        """
        Create a brand-new profile when one is absent and allowed.
        """
        db = cast(Any, self.db)
        user = cast(Any, self.user)
        db.commit = AsyncMock()
        db.refresh = AsyncMock()
        user.profile = None

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

            db.add.assert_called_with(mock_profile)
            db.commit.assert_awaited()
            db.refresh.assert_awaited()

    async def test_create_or_update_profile_not_found(self) -> None:
        """
        Return *404 Not Found* if profile is missing
        and creation is disallowed.
        """
        user = cast(Any, self.user)
        user.profile = None

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
        self.db.commit = AsyncMock(
            side_effect=IntegrityError('a', 'b', Exception('c')),
        )
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
