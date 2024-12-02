from __future__ import annotations

import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import FastAPI
from httpx import ASGITransport
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from examples.YOLO_server_api.auth import auth_router
from examples.YOLO_server_api.auth import UserLogin
from examples.YOLO_server_api.models import get_db
from examples.YOLO_server_api.models import User

# Initialise FastAPI app
app = FastAPI()
app.include_router(auth_router)


class TestAuth(unittest.IsolatedAsyncioTestCase):
    """
    Unit tests for the FastAPI authentication routes.
    """

    # Add type annotations for all instance attributes
    user_cache_patcher: unittest.mock._patch
    jwt_access_patcher: unittest.mock._patch
    mock_user_cache: dict
    mock_jwt_access: MagicMock
    aclient: AsyncClient
    mock_db: MagicMock

    async def asyncSetUp(self) -> None:
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
        self.user_cache_patcher = patch(
            'examples.YOLO_server_api.auth.user_cache', {},
        )
        self.jwt_access_patcher = patch(
            'examples.YOLO_server_api.auth.jwt_access',
        )

        # Start patchers and assign mock objects to instance variables
        self.mock_user_cache = self.user_cache_patcher.start()
        self.mock_jwt_access = self.jwt_access_patcher.start()

        # Mock the creation of a JWT access token
        self.mock_jwt_access.create_access_token.return_value = 'mocked_token'

        # Initialize AsyncClient with ASGITransport
        self.aclient = AsyncClient(
            transport=ASGITransport(app=app),
            base_url='http://test',
        )

        # Mock the database session
        self.mock_db = MagicMock(spec=AsyncSession)

        # Override get_db dependency
        async def override_get_db():
            yield self.mock_db

        app.dependency_overrides[get_db] = override_get_db

    async def asyncTearDown(self) -> None:
        """
        Stop all active patchers after each test.
        """
        self.user_cache_patcher.stop()
        self.jwt_access_patcher.stop()
        await self.aclient.aclose()

        # Clear dependency overrides
        app.dependency_overrides.clear()

    async def test_create_token_user_not_found(self) -> None:
        """
        Test token creation failure when the user is not found in
        cache or database.
        """
        # Ensure the user is not in cache
        if self.username in self.mock_user_cache:
            del self.mock_user_cache[self.username]

        # Mock the database execute result to return None (user not found)
        mock_result = MagicMock()
        mock_result.scalar.return_value = None
        self.mock_db.execute.return_value = mock_result

        # Send a POST request with the username that does not exist
        response = await self.aclient.post(
            '/token', json={
                'username': self.username,
                'password': self.password,
            },
        )

        # Assert the response indicates failed authentication
        self.assertEqual(response.status_code, 401)
        self.assertEqual(
            response.json()['detail'], 'Wrong username or password',
        )

    async def test_create_token_successful_login(self) -> None:
        """
        Test successful token creation with valid user credentials.
        """
        # Mock the database execute result to return the user
        mock_result = MagicMock()
        mock_result.scalar.return_value = self.mock_user
        self.mock_db.execute.return_value = mock_result

        # Send a POST request to retrieve a token
        response = await self.aclient.post(
            '/token', json={
                'username': self.username,
                'password': self.password,
            },
        )

        # Assert successful response with token in response JSON
        self.assertEqual(response.status_code, 200)
        self.assertIn('access_token', response.json())
        self.assertEqual(response.json()['access_token'], 'mocked_token')

    async def test_create_token_incorrect_password(self) -> None:
        """
        Test token creation failure due to incorrect password.
        """
        # Set mock user to return False for password check
        self.mock_user.check_password.return_value = False

        # Mock the database execute result to return the user
        mock_result = MagicMock()
        mock_result.scalar.return_value = self.mock_user
        self.mock_db.execute.return_value = mock_result

        # Send a POST request with incorrect password
        response = await self.aclient.post(
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

    async def test_create_token_user_in_cache(self) -> None:
        """
        Test token creation using a cached user without querying the database.
        """
        # Add the mock user to cache to simulate cached authentication
        self.mock_user_cache[self.username] = self.mock_user

        # Send a POST request to retrieve a token for cached user
        response = await self.aclient.post(
            '/token', json={
                'username': self.username,
                'password': self.password,
            },
        )

        # Assert the response contains the token and database was not queried
        self.assertEqual(response.status_code, 200)
        self.assertIn('access_token', response.json())
        self.assertEqual(response.json()['access_token'], 'mocked_token')
        self.mock_db.execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()
