from __future__ import annotations

import unittest
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.testclient import TestClient

from examples.auth.database import get_db
from examples.auth.redis_pool import get_redis_pool
from examples.db_management.routers import auth
from examples.db_management.schemas.auth import LogoutRequest
from examples.db_management.schemas.auth import RefreshRequest
from examples.db_management.schemas.auth import UserLogin


class TestAuthRouter(unittest.IsolatedAsyncioTestCase):
    """
    Unit tests for the authentication router endpoints
    using FastAPI and unittest.
    """

    def setUp(self) -> None:
        """
        Set up the FastAPI app, TestClient,
        and dependency overrides for each test.
        """
        # Create FastAPI app and mount the auth router
        self.app: FastAPI = FastAPI()
        self.app.include_router(auth.router, prefix='/auth')
        # Mock Redis client in app state
        self.app.state.redis_client = MagicMock()
        self.client: TestClient = TestClient(self.app)

        async def override_get_db() -> AsyncGenerator[MagicMock]:
            """Override for get_db dependency, yields a mock DB session."""
            db_mock: MagicMock = MagicMock()
            # Avoid await db.scalar errors
            db_mock.scalar = AsyncMock(return_value=None)
            yield db_mock

        async def override_get_redis_pool() -> MagicMock:
            """
            Override for get_redis_pool dependency, returns a mock Redis pool.
            """
            redis_mock: MagicMock = MagicMock()
            # Avoid await redis.get errors
            redis_mock.get = AsyncMock(return_value=None)
            return redis_mock

        # Apply dependency overrides
        self.app.dependency_overrides[get_db] = override_get_db
        self.app.dependency_overrides[get_redis_pool] = override_get_redis_pool

    @patch(
        'examples.db_management.routers.auth.login_user',
        new_callable=AsyncMock,
    )
    async def test_login_success(self, mock_login_user: AsyncMock) -> None:
        """
        Test successful login returns correct token pair and user info.

        Args:
            mock_login_user (AsyncMock): Mocked login_user service function.
        """
        mock_login_user.return_value = {
            'access_token': 'access123',
            'refresh_token': 'refresh123',
            'username': 'testuser',
            'role': 'user',
            'user_id': 1,
            'group_id': 2,
            'feature_names': ['f1', 'f2'],
        }
        payload: UserLogin = UserLogin(username='testuser', password='pw')
        response = self.client.post('/auth/login', json=payload.model_dump())
        self.assertEqual(response.status_code, 200)
        data: dict = response.json()
        self.assertIn('access_token', data)
        self.assertIn('refresh_token', data)
        self.assertEqual(data['username'], 'testuser')

    @patch(
        'examples.db_management.routers.auth.logout_user',
        new_callable=AsyncMock,
    )
    async def test_logout_success(self, mock_logout_user: AsyncMock) -> None:
        """
        Test successful logout returns a confirmation message.

        Args:
            mock_logout_user (AsyncMock): Mocked logout_user service function.
        """
        payload: LogoutRequest = LogoutRequest(refresh_token='refresh123')
        headers: dict[str, str] = {'Authorization': 'Bearer access123'}
        response = self.client.post(
            '/auth/logout',
            json=payload.model_dump(),
            headers=headers,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()['message'],
            'Logged out successfully.',
        )

    @patch(
        'examples.db_management.routers.auth.refresh_tokens',
        new_callable=AsyncMock,
    )
    async def test_refresh_success(
        self, mock_refresh_tokens: AsyncMock,
    ) -> None:
        """
        Test successful token refresh returns new access and refresh tokens.

        Args:
            mock_refresh_tokens (AsyncMock):
                Mocked refresh_tokens service function.
        """
        mock_refresh_tokens.return_value = {
            'access_token': 'new_access',
            'refresh_token': 'new_refresh',
            'feature_names': ['f1'],
        }
        payload: RefreshRequest = RefreshRequest(refresh_token='refresh123')
        response = self.client.post('/auth/refresh', json=payload.model_dump())
        self.assertEqual(response.status_code, 200)
        data: dict = response.json()
        self.assertIn('access_token', data)
        self.assertIn('refresh_token', data)

    @patch(
        'examples.db_management.routers.auth.login_user',
        new_callable=AsyncMock,
    )
    async def test_login_fail(self, mock_login_user: AsyncMock) -> None:
        """
        Test failed login returns 401 and error detail.

        Args:
            mock_login_user (AsyncMock): Mocked login_user service function.
        """
        mock_login_user.side_effect = HTTPException(
            status_code=401,
            detail='fail',
        )
        payload: UserLogin = UserLogin(username='bad', password='bad')
        response = self.client.post('/auth/login', json=payload.model_dump())
        self.assertEqual(response.status_code, 401)
        self.assertIn('detail', response.json())

    @patch(
        'examples.db_management.routers.auth.logout_user',
        new_callable=AsyncMock,
    )
    async def test_logout_fail(self, mock_logout_user: AsyncMock) -> None:
        """
        Test failed logout returns 401 and error detail.

        Args:
            mock_logout_user (AsyncMock): Mocked logout_user service function.
        """
        mock_logout_user.side_effect = HTTPException(
            status_code=401,
            detail='fail',
        )
        payload: LogoutRequest = LogoutRequest(refresh_token='bad')
        headers: dict[str, str] = {
            'Authorization': 'Bearer access123',
        }  # Ensure mock is called
        response = self.client.post(
            '/auth/logout',
            json=payload.model_dump(),
            headers=headers,
        )
        self.assertEqual(response.status_code, 401)
        self.assertIn('detail', response.json())

    @patch(
        'examples.db_management.routers.auth.refresh_tokens',
        new_callable=AsyncMock,
    )
    async def test_refresh_fail(self, mock_refresh_tokens: AsyncMock) -> None:
        """
        Test failed token refresh returns 401 and error detail.

        Args:
            mock_refresh_tokens (AsyncMock):
                Mocked refresh_tokens service function.
        """
        mock_refresh_tokens.side_effect = HTTPException(
            status_code=401,
            detail='fail',
        )
        payload: RefreshRequest = RefreshRequest(refresh_token='bad')
        response = self.client.post(
            '/auth/refresh',
            json=payload.model_dump(),
        )
        self.assertEqual(response.status_code, 401)
        self.assertIn('detail', response.json())


if __name__ == '__main__':
    unittest.main()

'''
pytest --cov=examples.db_management.routers.auth\
    --cov-report=term-missing\
        tests/examples/db_management/routers/auth_test.py
'''
