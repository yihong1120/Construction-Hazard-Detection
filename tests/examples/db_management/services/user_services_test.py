from __future__ import annotations

import unittest
from types import SimpleNamespace
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
    """Unit tests for user_services using mocks."""

    def setUp(self) -> None:
        """Initialise shared mocks used by each test."""
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

    async def test_list_users(self) -> None:
        """Fetch all users, ensuring the underlying query is executed."""
        mock_result: MagicMock = MagicMock()
        scalars_mock: MagicMock = (
            mock_result.unique.return_value.scalars.return_value
        )
        scalars_mock.all.return_value = ['user1', 'user2']
        self.db.execute = AsyncMock(return_value=mock_result)

        users, next_cursor = await user_services.list_users(self.db)

        self.assertEqual(users, ['user1', 'user2'])
        self.assertIsNone(next_cursor)

    async def test_list_users_filters_by_group(self) -> None:
        """An administrator list can be restricted to its own group."""
        mock_result: MagicMock = MagicMock()
        mock_result_rows = mock_result.unique.return_value.scalars.return_value
        mock_result_rows.all.return_value = [
            'group-user',
        ]
        self.db.execute = AsyncMock(return_value=mock_result)

        users, next_cursor = await user_services.list_users(
            self.db,
            group_id=9,
        )

        self.assertEqual(users, ['group-user'])
        self.assertIsNone(next_cursor)
        query = self.db.execute.await_args.args[0]
        self.assertIn('users.group_id', str(query))

    async def test_get_user_by_id_found(self) -> None:
        """Retrieve a single user by identifier when they exist."""
        result = MagicMock()
        result.unique.return_value.scalar_one_or_none.return_value = self.user
        self.db.execute = AsyncMock(return_value=result)

        user = await user_services.get_user_by_id(1, self.db)

        self.assertEqual(user, self.user)

    async def test_get_user_by_id_not_found(self) -> None:
        """Raise *404 Not Found* when the requested user is missing."""
        result = MagicMock()
        result.unique.return_value.scalar_one_or_none.return_value = None
        self.db.execute = AsyncMock(return_value=result)

        with self.assertRaises(HTTPException) as cm:
            await user_services.get_user_by_id(1, self.db)

        self.assertEqual(cm.exception.status_code, 404)

    async def test_delete_user_success(self) -> None:
        """Persist the removal of an existing user."""
        self.db.delete = AsyncMock()
        self.db.commit = AsyncMock()

        await user_services.delete_user(self.user, self.db)

        self.db.delete.assert_awaited_with(self.user)
        self.db.commit.assert_awaited()

    async def test_delete_user_exception(self) -> None:
        """Handle an unexpected database failure during deletion."""
        self.db.delete = AsyncMock()
        self.db.commit = AsyncMock(side_effect=Exception('fail'))
        self.db.rollback = AsyncMock()

        with self.assertRaises(HTTPException) as cm:
            await user_services.delete_user(self.user, self.db)

        self.assertEqual(cm.exception.status_code, 500)
        self.db.rollback.assert_awaited()

    async def test_update_username_success(self) -> None:
        """Change the username and commit the transaction."""
        self.db.commit = AsyncMock()
        self.user.username = 'old'

        await user_services.update_username(self.user, 'new', self.db)

        self.assertEqual(self.user.username, 'new')
        self.db.commit.assert_awaited()

    async def test_update_username_integrity_error(self) -> None:
        """Return *400 Bad Request* when the new username already exists."""
        self.db.commit = AsyncMock(
            side_effect=IntegrityError('a', 'b', Exception('c')),
        )
        self.db.rollback = AsyncMock()

        with self.assertRaises(HTTPException) as cm:
            await user_services.update_username(self.user, 'new', self.db)

        self.assertEqual(cm.exception.status_code, 400)
        self.db.rollback.assert_awaited()

    async def test_update_username_general_exception(self) -> None:
        """Return *500 Internal Server Error* for an unexpected failure."""
        self.db.commit = AsyncMock(side_effect=Exception('fail'))
        self.db.rollback = AsyncMock()

        with self.assertRaises(HTTPException) as cm:
            await user_services.update_username(self.user, 'new', self.db)

        self.assertEqual(cm.exception.status_code, 500)
        self.db.rollback.assert_awaited()

    async def test_set_user_status_success(self) -> None:
        """Update the status field and commit."""
        self.db.commit = AsyncMock()

        await user_services.set_user_status(self.user, 'active', self.db)

        self.assertEqual(self.user.status, 'active')
        self.db.commit.assert_awaited()

    async def test_set_user_status_exception(self) -> None:
        """Raise *500 Internal Server Error* when commit fails."""
        self.db.commit = AsyncMock(side_effect=Exception('fail'))
        self.db.rollback = AsyncMock()

        with self.assertRaises(HTTPException) as cm:
            await user_services.set_user_status(
                self.user,
                'suspended',
                self.db,
            )

        self.assertEqual(cm.exception.status_code, 500)
        self.db.rollback.assert_awaited()

    async def test_set_user_status_invalid(self) -> None:
        """Reject unknown account statuses."""
        with self.assertRaises(HTTPException) as cm:
            await user_services.set_user_status(self.user, 'unknown', self.db)

        self.assertEqual(cm.exception.status_code, 400)

    async def test_create_or_update_profile_update(self) -> None:
        """Update fields on an existing UserProfile."""
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
        """Create a brand-new profile when one is absent and allowed."""
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
        """Return *404 Not Found* if profile is missing and creation is
        disallowed."""
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
        """Handle a unique-constraint violation on profile save."""
        self.db.commit = AsyncMock(
            side_effect=IntegrityError('a', 'b', Exception('c')),
        )
        self.db.rollback = AsyncMock()
        self.db.refresh = AsyncMock()

        awaitable = user_services.create_or_update_profile(
            self.user,
            {'email': 'dup@example.com'},
            self.db,
        )

        with self.assertRaises(HTTPException) as cm:
            await awaitable

        self.assertEqual(cm.exception.status_code, 400)
        self.db.rollback.assert_awaited()

    async def test_create_or_update_profile_general_exception(self) -> None:
        """Return *500 Internal Server Error* for an unexpected profile
        failure."""
        self.db.commit = AsyncMock(side_effect=Exception('fail'))
        self.db.rollback = AsyncMock()
        self.db.refresh = AsyncMock()

        awaitable = user_services.create_or_update_profile(
            self.user,
            {'email': 'fail@example.com'},
            self.db,
        )

        with self.assertRaises(HTTPException) as cm:
            await awaitable

        self.assertEqual(cm.exception.status_code, 500)
        self.db.rollback.assert_awaited()


if __name__ == '__main__':
    unittest.main()


class TestOidcAccountStorage(unittest.IsolatedAsyncioTestCase):
    async def test_subject_must_already_be_linked(self):
        db = AsyncMock()
        user = SimpleNamespace(id=1, username='alice')
        with patch.object(
            user_services,
            'keycloak_subject_for_user',
            AsyncMock(return_value=None),
        ):
            with self.assertRaises(HTTPException) as raised:
                await user_services.ensure_keycloak_subject_for_user(user, db)
            self.assertEqual(
                raised.exception.detail,
                'keycloak_identity_not_linked',
            )
        with patch.object(
            user_services,
            'keycloak_subject_for_user',
            AsyncMock(return_value='subject'),
        ):
            self.assertEqual(
                await user_services.ensure_keycloak_subject_for_user(user, db),
                'subject',
            )
        self.assertNotIn('password_hash', User.__table__.columns)
