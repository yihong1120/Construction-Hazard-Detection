from __future__ import annotations

import unittest
from unittest.mock import AsyncMock
from unittest.mock import patch

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine

from examples.YOLO_server_api.backend.models import Base
from examples.YOLO_server_api.backend.models import User
from examples.YOLO_server_api.backend.user_operation import add_user
from examples.YOLO_server_api.backend.user_operation import delete_user
from examples.YOLO_server_api.backend.user_operation import (
    set_user_active_status,
)
from examples.YOLO_server_api.backend.user_operation import update_password
from examples.YOLO_server_api.backend.user_operation import update_username


class UserOperationTestCase(unittest.IsolatedAsyncioTestCase):
    """
    Unit tests for user operations, including adding, deleting,
    updating usernames, updating passwords, and setting user active status.

    Attributes:
        TEST_DATABASE_URL (str): The URL for the test database.
        test_engine (AsyncEngine): The test database engine.
        test_sessionmaker (sessionmaker): Session maker for
            creating database sessions.
        session (AsyncSession): The current database session.
    """

    async def asyncSetUp(self) -> None:
        """
        Set up the test environment before each test by configuring an
        in-memory SQLite database and creating the required tables.
        """
        self.TEST_DATABASE_URL: str = 'sqlite+aiosqlite:///:memory:'
        self.test_engine = create_async_engine(
            self.TEST_DATABASE_URL, echo=False,
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
        Clean up the test environment after each test by dropping all tables
        and closing the session and engine.
        """
        async with self.test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await self.session.close()
        await self.test_engine.dispose()

    async def test_add_user_success(self) -> None:
        """
        Test that a user can be added successfully.
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
        user = execution_result.scalars().first()
        self.assertIsNotNone(user)
        if user:
            self.assertTrue(await user.check_password('password123'))

    async def test_add_user_duplicate_username(self) -> None:
        """
        Test that adding a user with a duplicate username returns an error.
        """
        await add_user('testuser', 'password123', 'user', self.session)
        result = await add_user(
            'testuser', 'password456', 'user', self.session,
        )
        self.assertFalse(result['success'])
        self.assertEqual(result['error'], 'IntegrityError')
        self.assertEqual(
            result['message'],
            "Username 'testuser' already exists.",
        )

    async def test_delete_user_success(self) -> None:
        """
        Test that a user can be deleted successfully.
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
        user = execution_result.scalars().first()
        self.assertIsNone(user)

    async def test_delete_user_not_found(self) -> None:
        """
        Test that deleting a non-existent user returns an error.
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
        Test that a user's username can be updated successfully.
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
        user = execution_result.scalars().first()
        self.assertIsNotNone(user)

    async def test_update_username_not_found(self) -> None:
        """
        Test that updating a non-existent user's username returns an error.
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
        Test that a user's password can be updated successfully.
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
        user = execution_result.scalars().first()
        if user:
            self.assertTrue(await user.check_password('newpassword123'))

    async def test_update_password_user_not_found(self) -> None:
        """
        Test that updating the password of a non-existent user
        returns an error.
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
        Test that a user's active status can be updated successfully.
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
        user = execution_result.scalars().first()
        if user:
            self.assertFalse(user.is_active)

    async def test_set_user_active_status_user_not_found(self) -> None:
        """
        Test that updating the active status of a non-existent user
        returns an error.
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
        Test that add_user returns the correct error message
        when an unknown exception occurs.
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
        Test that delete_user returns the correct error message
        when an unknown exception occurs.
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
        Test that update_username returns the correct error message
        when an unknown exception occurs。
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
        Test that update_password returns the correct error message
        when an unknown exception occurs.
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
        Test that set_user_active_status returns the correct error message
        when an unknown exception occurs.
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
        Test to return an IntegrityError
        when updating username to an existing username.
        """
        # Add two users
        await add_user('user1', 'password123', 'user', self.session)
        await add_user('user2', 'password123', 'user', self.session)

        # Attempt to update username 'user1' to 'user2'
        result = await update_username('user1', 'user2', self.session)
        self.assertFalse(result['success'])
        self.assertEqual(result['error'], 'IntegrityError')
        self.assertEqual(result['message'], "Username 'user2' already exists.")


if __name__ == '__main__':
    unittest.main()
