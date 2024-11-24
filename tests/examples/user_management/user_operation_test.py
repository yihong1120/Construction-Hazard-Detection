from __future__ import annotations

import unittest

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import sessionmaker

from examples.user_management.models import Base
from examples.user_management.models import User
from examples.user_management.user_operation import add_user
from examples.user_management.user_operation import delete_user
from examples.user_management.user_operation import update_password
from examples.user_management.user_operation import update_username


class UserOperationTestCase(unittest.IsolatedAsyncioTestCase):
    """
    Unit tests for user operations, including adding, deleting,
    updating usernames, and updating passwords for users.

    Attributes:
        TEST_DATABASE_URL (str): The URL for the test database.
        test_engine (AsyncEngine): The test database engine.
        test_sessionmaker (sessionmaker): Session maker
            for creating database sessions.
        session (AsyncSession): The current database session.
    """

    async def asyncSetUp(self) -> None:
        """
        Set up the test environment before each test by configuring an
        in-memory SQLite database and creating the required tables.
        """
        # Configure the in-memory SQLite database for testing
        self.TEST_DATABASE_URL: str = 'sqlite+aiosqlite:///:memory:'
        self.test_engine = create_async_engine(
            self.TEST_DATABASE_URL, echo=False,
        )
        self.test_sessionmaker = sessionmaker(
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
        # Drop all database tables
        async with self.test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await self.session.close()
        await self.test_engine.dispose()

    async def test_add_user_success(self) -> None:
        """
        Test that a user can be added successfully, and their details
        are correctly stored in the database.
        """
        add_user_result: str = await add_user(
            'testuser', 'password123', 'user', self.session,
        )
        self.assertEqual(
            add_user_result,
            'User testuser with role user added successfully.',
        )

        # Verify if the user was successfully added
        stmt = select(User).where(User.username == 'testuser')
        execution_result = await self.session.execute(stmt)
        user: User = execution_result.scalars().first()
        self.assertIsNotNone(user)
        self.assertTrue(user.check_password('password123'))

    async def test_add_user_duplicate_username(self) -> None:
        """
        Test that adding a user with a duplicate username returns the
        appropriate error message.
        """
        await add_user('testuser', 'password123', 'user', self.session)
        add_user_result: str = await add_user(
            'testuser', 'password456', 'user', self.session,
        )
        self.assertIn(
            'Error: Username testuser already exists.',
            add_user_result,
        )

    async def test_delete_user_success(self) -> None:
        """
        Test that a user can be deleted successfully and is no longer present
        in the database.
        """
        await add_user('testuser', 'password123', 'user', self.session)
        delete_user_result: str = await delete_user('testuser', self.session)
        self.assertEqual(
            delete_user_result,
            'User testuser deleted successfully.',
        )

        # Verify if the user was deleted
        stmt = select(User).where(User.username == 'testuser')
        execution_result = await self.session.execute(stmt)
        user: User = execution_result.scalars().first()
        self.assertIsNone(user)

    async def test_delete_user_not_found(self) -> None:
        """
        Test that deleting a non-existent user returns the appropriate
        error message.
        """
        result: str = await delete_user('nonexistentuser', self.session)
        self.assertEqual(result, 'User nonexistentuser not found.')

    async def test_update_username_success(self) -> None:
        """
        Test that a user's username can be updated successfully and the
        changes are reflected in the database.
        """
        await add_user('oldusername', 'password123', 'user', self.session)
        update_username_result: str = await update_username(
            'oldusername', 'newusername', self.session,
        )
        self.assertEqual(
            update_username_result,
            'Username updated successfully to newusername.',
        )

        # Verify if the username was updated
        stmt = select(User).where(User.username == 'newusername')
        execution_result = await self.session.execute(stmt)
        user: User = execution_result.scalars().first()
        self.assertIsNotNone(user)
        self.assertTrue(user.check_password('password123'))

    async def test_update_username_not_found(self) -> None:
        """
        Test that updating the username of a non-existent user returns the
        appropriate error message.
        """
        result: str = await update_username(
            'nonexistentuser', 'newusername', self.session,
        )
        self.assertEqual(result, 'User nonexistentuser not found.')

    async def test_update_password_success(self) -> None:
        """
        Test that a user's password can be updated successfully and the
        changes are reflected in the database.
        """
        await add_user('testuser', 'password123', 'user', self.session)
        update_password_result: str = await update_password(
            'testuser', 'newpassword123', self.session,
        )
        self.assertEqual(
            update_password_result,
            'Password updated successfully for user testuser.',
        )

        # Verify if the password was updated
        stmt = select(User).where(User.username == 'testuser')
        execution_result = await self.session.execute(stmt)
        user: User = execution_result.scalars().first()
        self.assertTrue(user.check_password('newpassword123'))

    async def test_update_password_user_not_found(self) -> None:
        """
        Test that updating the password of a non-existent user returns the
        appropriate error message.
        """
        result: str = await update_password(
            'nonexistentuser', 'newpassword123', self.session,
        )
        self.assertEqual(result, 'User nonexistentuser not found.')


if __name__ == '__main__':
    unittest.main()
