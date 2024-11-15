from __future__ import annotations

import unittest
from unittest.mock import _patch
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from examples.YOLO_server_api.auth import auth_router
from examples.YOLO_server_api.auth import UserLogin
from examples.YOLO_server_api.models import User

# Initialise FastAPI app and test client
app = FastAPI()
app.include_router(auth_router)
client = TestClient(app)


class TestAuth(unittest.TestCase):
    """
    Unit tests for the FastAPI authentication routes.
    """

    def setUp(self) -> None:
        """
        Set up required mocks and patchers before each test.
        """
        self.username: str = 'testuser'
        self.password: str = 'testpassword'
        self.user_login: UserLogin = UserLogin(
            username=self.username, password=self.password,
        )

        # Create a mock user instance and define its properties
        self.mock_user: MagicMock = MagicMock(spec=User)
        self.mock_user.username = self.username
        self.mock_user.role = 'user'
        self.mock_user.is_active = True
        self.mock_user.check_password.return_value = True

        # Patch user cache and JWT access token generator
        self.user_cache_patcher: _patch = patch(
            'examples.YOLO_server_api.auth.user_cache', {},
        )
        self.jwt_access_patcher: _patch = patch(
            'examples.YOLO_server_api.auth.jwt_access',
        )

        # Start patchers and assign mock objects to instance variables
        self.mock_user_cache: dict = self.user_cache_patcher.start()
        self.mock_jwt_access: MagicMock = self.jwt_access_patcher.start()

        # Mock the creation of a JWT access token
        self.mock_jwt_access.create_access_token.return_value = 'mocked_token'

    def tearDown(self) -> None:
        """
        Stop all active patchers after each test.
        """
        self.user_cache_patcher.stop()
        self.jwt_access_patcher.stop()

    @patch('examples.YOLO_server_api.auth.get_db')
    async def test_create_token_successful_login(
        self, mock_get_db: MagicMock,
    ) -> None:
        """
        Test successful token creation with valid user credentials.

        Args:
            mock_get_db (MagicMock): Mocked database session dependency.
        """
        # Mock the AsyncSession for database interactions
        mock_db: MagicMock = MagicMock(spec=AsyncSession)
        mock_get_db.return_value = mock_db

        # Add mock user to cache to simulate cached authentication
        self.mock_user_cache[self.username] = self.mock_user

        # Send a POST request to retrieve a token
        response = client.post(
            '/token', json={
                'username': self.username,
                'password': self.password,
            },
        )

        # Assert successful response with token in response JSON
        self.assertEqual(response.status_code, 200)
        self.assertIn('access_token', response.json())
        self.assertEqual(response.json()['access_token'], 'mocked_token')

    @patch('examples.YOLO_server_api.auth.get_db')
    async def test_create_token_incorrect_password(
        self, mock_get_db: MagicMock,
    ) -> None:
        """
        Test token creation failure due to incorrect password.

        Args:
            mock_get_db (MagicMock): Mocked database session dependency.
        """
        # Set mock user to return False for password check
        self.mock_user.check_password.return_value = False
        mock_db: MagicMock = MagicMock(spec=AsyncSession)
        mock_get_db.return_value = mock_db

        # Send a POST request with incorrect password
        response = client.post(
            '/token', json={
                'username': self.username,
                'password': 'wrongpassword',
            },
        )

        # Assert the response indicates failed authentication
        self.assertEqual(response.status_code, 401)
        self.assertEqual(
            response.json()['detail'], 'Wrong username or password',
        )

    @patch('examples.YOLO_server_api.auth.get_db')
    async def test_create_token_user_not_found(
        self, mock_get_db: MagicMock,
    ) -> None:
        """
        Test token creation failure when the user is not found.

        Args:
            mock_get_db (MagicMock): Mocked database session dependency.
        """
        # Mock scenario where user is not found in the database
        mock_db: MagicMock = MagicMock(spec=AsyncSession)
        mock_get_db.return_value = mock_db

        # Send a POST request for a non-existent user
        response = client.post(
            '/token', json={'username': 'nonexistent', 'password': 'password'},
        )

        # Assert the response indicates user not found
        self.assertEqual(response.status_code, 401)
        self.assertEqual(
            response.json()['detail'], 'Wrong username or password',
        )

    @patch('examples.YOLO_server_api.auth.get_db')
    async def test_create_token_user_in_cache(
        self, mock_get_db: MagicMock,
    ) -> None:
        """
        Test token creation using a cached user without querying the database.

        Args:
            mock_get_db (MagicMock): Mocked database session dependency.
        """
        # Add the mock user to cache to simulate cached authentication
        self.mock_user_cache[self.username] = self.mock_user

        # Send a POST request to retrieve a token for cached user
        response = client.post(
            '/token', json={
                'username': self.username,
                'password': self.password,
            },
        )

        # Assert the response contains the token and database was not queried
        self.assertEqual(response.status_code, 200)
        self.assertIn('access_token', response.json())
        self.assertEqual(response.json()['access_token'], 'mocked_token')
        mock_get_db.assert_not_called()  # Ensure database query was bypassed


if __name__ == '__main__':
    unittest.main()
