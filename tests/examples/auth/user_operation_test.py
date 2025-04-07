from __future__ import annotations

import unittest
from unittest.mock import AsyncMock
from unittest.mock import patch

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine

from examples.auth.models import Base
from examples.auth.models import User
from examples.auth.user_operation import add_user
from examples.auth.user_operation import delete_user
from examples.auth.user_operation import set_user_active_status
from examples.auth.user_operation import update_password
from examples.auth.user_operation import update_username


class UserOperationTestCase(unittest.IsolatedAsyncioTestCase):
    """
    Test suite for user operations in the database.
    """

    async def asyncSetUp(self) -> None:
        """
        Prepare the test environment before each test.
        """
        self.TEST_DATABASE_URL: str = 'sqlite+aiosqlite:///:memory:'
        self.test_engine = create_async_engine(
            self.TEST_DATABASE_URL,
            echo=False,  # Set to True to debug SQL statements
        )
        self.test_sessionmaker = async_sessionmaker(
            bind=self.test_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        self.session: AsyncSession = self.test_sessionmaker()

        # Create the database tables
        async with self.test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def asyncTearDown(self) -> None:
        """
        Clean up the test environment after each test.
        """
        async with self.test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

        await self.session.close()
        await self.test_engine.dispose()

    async def test_add_user_success(self) -> None:
        """
        Verify that `add_user` can successfully add a new user to the database.
        """
        result = await add_user(
            'testuser', 'password123', 'user', self.session,
        )
        self.assertTrue(result['success'])
        self.assertEqual(
            result['message'],
            "User 'testuser' added successfully.",
        )

        stmt = select(User).where(User.username == 'testuser')
        execution_result = await self.session.execute(stmt)
        user: User | None = execution_result.scalars().first()
        self.assertIsNotNone(
            user, 'Expected user to be in the DB after add_user.',
        )
        if user:
            self.assertTrue(
                await user.check_password('password123'),
                "The stored password hash should match 'password123'.",
            )

    async def test_add_user_duplicate_username(self) -> None:
        """
        Verify that attempting to add a user with a duplicate username fails
        and returns an IntegrityError.
        """
        await add_user('testuser', 'password123', 'user', self.session)
        result = await add_user(
            'testuser',
            'password456',
            'user',
            self.session,
        )
        self.assertFalse(result['success'])
        self.assertEqual(result['error'], 'IntegrityError')
        self.assertEqual(
            result['message'],
            "Username 'testuser' already exists.",
        )

    async def test_delete_user_success(self) -> None:
        """
        Verify that `delete_user` can remove
        an existing user from the database.
        """
        await add_user('testuser', 'password123', 'user', self.session)
        result = await delete_user('testuser', self.session)
        self.assertTrue(result['success'])
        self.assertEqual(
            result['message'],
            "User 'testuser' deleted successfully.",
        )

        stmt = select(User).where(User.username == 'testuser')
        execution_result = await self.session.execute(stmt)
        user: User | None = execution_result.scalars().first()
        self.assertIsNone(
            user, 'User should no longer exist after delete_user.',
        )

    async def test_delete_user_not_found(self) -> None:
        """
        Verify that trying to delete a non-existent user returns
        a NotFound error.
        """
        result = await delete_user('nonexistentuser', self.session)
        self.assertFalse(result['success'])
        self.assertEqual(result['error'], 'NotFound')
        self.assertEqual(
            result['message'],
            "User 'nonexistentuser' not found.",
        )

    async def test_update_username_success(self) -> None:
        """
        Verify that `update_username` can successfully change
        a user's username.
        """
        await add_user('oldusername', 'password123', 'user', self.session)
        result = await update_username(
            'oldusername', 'newusername', self.session,
        )
        self.assertTrue(result['success'])
        self.assertEqual(
            result['message'],
            "Username updated from 'oldusername' to 'newusername'.",
        )

        stmt = select(User).where(User.username == 'newusername')
        execution_result = await self.session.execute(stmt)
        user: User | None = execution_result.scalars().first()
        self.assertIsNotNone(
            user, 'User with new username should exist in the DB.',
        )

    async def test_update_username_not_found(self) -> None:
        """
        Verify that updating username for a non-existent user returns NotFound.
        """
        result = await update_username(
            'nonexistentuser', 'newusername', self.session,
        )
        self.assertFalse(result['success'])
        self.assertEqual(result['error'], 'NotFound')
        self.assertEqual(
            result['message'],
            "User 'nonexistentuser' not found.",
        )

    async def test_update_password_success(self) -> None:
        """
        Verify that `update_password` can successfully change
        a user's password.
        """
        await add_user('testuser', 'password123', 'user', self.session)
        result = await update_password(
            'testuser', 'newpassword123', self.session,
        )
        self.assertTrue(result['success'])
        self.assertEqual(
            result['message'],
            "Password updated successfully for user 'testuser'.",
        )

        stmt = select(User).where(User.username == 'testuser')
        execution_result = await self.session.execute(stmt)
        user: User | None = execution_result.scalars().first()
        if user:
            self.assertTrue(
                await user.check_password('newpassword123'),
                "The updated password hash should match 'newpassword123'.",
            )

    async def test_update_password_user_not_found(self) -> None:
        """
        Verify that attempting to update the password of a non-existent user
        returns a NotFound error.
        """
        result = await update_password(
            'nonexistentuser', 'newpassword123', self.session,
        )
        self.assertFalse(result['success'])
        self.assertEqual(result['error'], 'NotFound')
        self.assertEqual(
            result['message'],
            "User 'nonexistentuser' not found.",
        )

    async def test_set_user_active_status_success(self) -> None:
        """
        Verify that `set_user_active_status` can successfully change a user's
        active status.
        """
        await add_user('testuser', 'password123', 'user', self.session)
        result = await set_user_active_status(
            'testuser', is_active=False, db=self.session,
        )
        self.assertTrue(result['success'])
        self.assertEqual(
            result['message'],
            "User 'testuser' is now inactive.",
        )

        stmt = select(User).where(User.username == 'testuser')
        execution_result = await self.session.execute(stmt)
        user: User | None = execution_result.scalars().first()
        if user:
            self.assertFalse(
                user.is_active, "User's is_active should be False.",
            )

    async def test_set_user_active_status_user_not_found(self) -> None:
        """
        Verify that attempting to set the active status of a non-existent user
        returns a NotFound error.
        """
        result = await set_user_active_status(
            'nonexistentuser', is_active=True, db=self.session,
        )
        self.assertFalse(result['success'])
        self.assertEqual(result['error'], 'NotFound')
        self.assertEqual(
            result['message'],
            "User 'nonexistentuser' not found.",
        )

    async def test_add_user_unknown_error(self) -> None:
        """
        Verify that an unknown exception in `add_user` returns 'UnknownError'.
        """
        with patch.object(
            self.session,
            'commit',
            new_callable=AsyncMock,
        ) as mock_commit:
            mock_commit.side_effect = Exception('Unknown error')
            result = await add_user(
                'testuser',
                'password123',
                'user',
                self.session,
            )
            self.assertFalse(result['success'])
            self.assertEqual(result['error'], 'UnknownError')
            self.assertIn('Failed to add user', result['message'])

    async def test_delete_user_unknown_error(self) -> None:
        """
        Verify that an unknown exception in `delete_user` returns
        'UnknownError'.
        """
        await add_user('testuser', 'password123', 'user', self.session)
        with patch.object(
            self.session,
            'commit',
            new_callable=AsyncMock,
        ) as mock_commit:
            mock_commit.side_effect = Exception('Unknown error')
            result = await delete_user('testuser', self.session)
            self.assertFalse(result['success'])
            self.assertEqual(result['error'], 'UnknownError')
            self.assertIn('Failed to delete user', result['message'])

    async def test_update_username_unknown_error(self) -> None:
        """
        Verify that an unknown exception in `update_username` returns
        'UnknownError'.
        """
        await add_user('testuser', 'password123', 'user', self.session)
        with patch.object(
            self.session,
            'commit',
            new_callable=AsyncMock,
        ) as mock_commit:
            mock_commit.side_effect = Exception('Unknown error')
            result = await update_username(
                'testuser',
                'newusername',
                self.session,
            )
            self.assertFalse(result['success'])
            self.assertEqual(result['error'], 'UnknownError')
            self.assertIn('Failed to update username', result['message'])

    async def test_update_password_unknown_error(self) -> None:
        """
        Verify that an unknown exception in `update_password` returns
        'UnknownError'.
        """
        await add_user('testuser', 'password123', 'user', self.session)
        with patch.object(
            self.session,
            'commit',
            new_callable=AsyncMock,
        ) as mock_commit:
            mock_commit.side_effect = Exception('Unknown error')
            result = await update_password(
                'testuser',
                'newpassword123',
                self.session,
            )
            self.assertFalse(result['success'])
            self.assertEqual(result['error'], 'UnknownError')
            self.assertIn('Failed to update password', result['message'])

    async def test_set_user_active_status_unknown_error(self) -> None:
        """
        Verify that an unknown exception in `set_user_active_status` returns
        'UnknownError'.
        """
        await add_user('testuser', 'password123', 'user', self.session)
        with patch.object(
            self.session,
            'commit',
            new_callable=AsyncMock,
        ) as mock_commit:
            mock_commit.side_effect = Exception('Unknown error')
            result = await set_user_active_status(
                'testuser',
                is_active=True,
                db=self.session,
            )
            self.assertFalse(result['success'])
            self.assertEqual(result['error'], 'UnknownError')
            self.assertIn('Failed to update active status', result['message'])

    async def test_update_username_integrity_error(self) -> None:
        """
        Verify that updating a username to one that already exists causes an
        IntegrityError to be returned.
        """
        # Add two distinct users
        await add_user('user1', 'password123', 'user', self.session)
        await add_user('user2', 'password123', 'user', self.session)

        # Attempt to rename 'user1' to 'user2'
        result = await update_username('user1', 'user2', self.session)
        self.assertFalse(result['success'])
        self.assertEqual(result['error'], 'IntegrityError')
        self.assertEqual(result['message'], "Username 'user2' already exists.")


if __name__ == '__main__':
    unittest.main()

'''
pytest \
    --cov=examples.auth.user_operation \
    --cov-report=term-missing tests/examples/auth/user_operation_test.py
'''
