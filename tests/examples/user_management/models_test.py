from __future__ import annotations

import unittest

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import sessionmaker

from examples.user_management.models import Base
from examples.user_management.models import User


class UserModelTestCase(unittest.IsolatedAsyncioTestCase):
    """
    Unit tests for the User model using an in-memory SQLite database.
    """

    async def asyncSetUp(self) -> None:
        """
        Set up the test environment before each test.
        """
        # Use an in-memory SQLite database for testing
        self.TEST_DATABASE_URL = 'sqlite+aiosqlite:///:memory:'
        self.test_engine = create_async_engine(
            self.TEST_DATABASE_URL, echo=False,
        )
        self.test_sessionmaker = sessionmaker(
            bind=self.test_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        self.session = self.test_sessionmaker()

        # Create database schema
        async with self.test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def asyncTearDown(self) -> None:
        """
        Clean up the test environment after each test.
        """
        # Drop all tables
        async with self.test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

        # Close the session
        await self.session.close()
        await self.test_engine.dispose()

    async def test_set_password(self) -> None:
        """
        Test that the password is hashed correctly when set.
        """
        user = User(username='testuser')
        user.set_password('securepassword123')

        # Ensure the password hash is not the same as the plain text password
        self.assertNotEqual(user.password_hash, 'securepassword123')

        # Ensure the password hash is not empty
        self.assertTrue(user.password_hash)

    async def test_check_password(self) -> None:
        """
        Test that the correct password matches the hash
        and an incorrect one does not.
        """
        user = User(username='testuser')
        user.set_password('securepassword123')

        # Ensure the correct password matches the hash
        self.assertTrue(user.check_password('securepassword123'))

        # Ensure an incorrect password does not match the hash
        self.assertFalse(user.check_password('wrongpassword'))

    async def test_user_creation(self) -> None:
        """
        Test that a user can be created and stored in the database.
        """
        user = User(username='testuser', role='user')
        user.set_password('securepassword123')
        self.session.add(user)
        await self.session.commit()

        # Retrieve the user from the database
        retrieved_user = await self.session.get(User, user.id)

        self.assertIsNotNone(retrieved_user)
        self.assertEqual(retrieved_user.username, 'testuser')
        self.assertEqual(retrieved_user.role, 'user')
        self.assertTrue(retrieved_user.is_active)

    async def test_unique_username_constraint(self) -> None:
        """
        Test that the username must be unique in the database.
        """
        user1 = User(username='uniqueuser')
        user1.set_password('password123')
        self.session.add(user1)
        await self.session.commit()

        # Attempt to add another user with the same username
        user2 = User(username='uniqueuser')
        user2.set_password('password456')
        self.session.add(user2)

        with self.assertRaises(Exception):
            await self.session.commit()

    async def test_default_values(self) -> None:
        """
        Test that default values are correctly assigned when a user is created.
        """
        user = User(username='defaultuser')
        user.set_password('password123')
        self.session.add(user)
        await self.session.commit()

        # Retrieve the user from the database
        retrieved_user = await self.session.get(User, user.id)

        # Check default values
        self.assertEqual(retrieved_user.role, 'user')
        self.assertTrue(retrieved_user.is_active)
        self.assertIsNotNone(retrieved_user.created_at)
        self.assertIsNotNone(retrieved_user.updated_at)


if __name__ == '__main__':
    unittest.main()
