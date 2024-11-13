from __future__ import annotations

import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI
from sqlalchemy.orm import Session
from examples.YOLO_server.auth import auth_router, UserLogin
from examples.YOLO_server.models import User
from examples.YOLO_server.cache import user_cache

app = FastAPI()
app.include_router(auth_router)
client = TestClient(app)


class TestAuth(unittest.TestCase):
    def setUp(self):
        """Set up required mocks and patchers."""
        self.username = "testuser"
        self.password = "testpassword"
        self.user_login = UserLogin(username=self.username, password=self.password)

        # Create a mock user and mock its methods
        self.mock_user = MagicMock(spec=User)
        self.mock_user.username = self.username
        self.mock_user.role = 'user'  # 確保角色有效
        self.mock_user.is_active = True  # 確保帳戶是活躍的
        self.mock_user.check_password.return_value = True

        # Patch user_cache and JWT access
        self.user_cache_patcher = patch("examples.YOLO_server.auth.user_cache", {})
        self.jwt_access_patcher = patch("examples.YOLO_server.auth.jwt_access")

        # Start the patchers and set them to the test instance
        self.mock_user_cache = self.user_cache_patcher.start()
        self.mock_jwt_access = self.jwt_access_patcher.start()

    def tearDown(self):
        """Stop all patches."""
        self.user_cache_patcher.stop()
        self.jwt_access_patcher.stop()

    @patch("examples.YOLO_server.auth.get_db")
    def test_create_token_successful_login(self, mock_get_db):
        """Test successful token creation with valid credentials."""
        
        # Mock the database query to return our mock user
        mock_db = MagicMock(spec=Session)
        mock_db.query.return_value.filter.return_value.first.return_value = self.mock_user
        mock_get_db.return_value = mock_db

        # Mock JWT access token creation
        self.mock_jwt_access.create_access_token.return_value = "mocked_token"

        # Add mock user to cache to simulate caching logic
        self.mock_user_cache[self.username] = self.mock_user

        # Make the token request
        response = client.post("/token", json={"username": self.username, "password": self.password})

        # Assertions
        self.assertEqual(response.status_code, 200)
        self.assertIn("access_token", response.json())
        self.assertEqual(response.json()["access_token"], "mocked_token")

    @patch("examples.YOLO_server.auth.get_db")
    def test_create_token_incorrect_password(self, mock_get_db):
        """Test token creation failure with an incorrect password."""
        
        # Simulate a user with an incorrect password
        self.mock_user.check_password.return_value = False
        mock_db = MagicMock(spec=Session)
        mock_db.query.return_value.filter.return_value.first.return_value = self.mock_user
        mock_get_db.return_value = mock_db

        response = client.post("/token", json={"username": self.username, "password": "wrongpassword"})

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "Wrong username or password")

    @patch("examples.YOLO_server.auth.get_db")
    def test_create_token_user_not_found(self, mock_get_db):
        """Test token creation failure when the user is not found."""
        
        # Simulate no user found in the database
        mock_db = MagicMock(spec=Session)
        mock_db.query.return_value.filter.return_value.first.return_value = None
        mock_get_db.return_value = mock_db

        response = client.post("/token", json={"username": "nonexistent", "password": "password"})

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "Wrong username or password")

    @patch("examples.YOLO_server.auth.get_db")
    def test_create_token_user_in_cache(self, mock_get_db):
        """Test token creation when the user is already cached."""

        # Add the mock user to cache
        self.mock_user_cache[self.username] = self.mock_user

        # Mock JWT access token creation
        self.mock_jwt_access.create_access_token.return_value = "mocked_token"

        response = client.post("/token", json={"username": self.username, "password": self.password})

        self.assertEqual(response.status_code, 200)
        self.assertIn("access_token", response.json())
        self.assertEqual(response.json()["access_token"], "mocked_token")

        # Ensure database was not queried
        mock_get_db.assert_not_called()


if __name__ == "__main__":
    unittest.main()
