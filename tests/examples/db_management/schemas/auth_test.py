from __future__ import annotations

import unittest

from pydantic import ValidationError

from examples.db_management.schemas import auth


class TestUserLogin(unittest.TestCase):
    """Tests for UserLogin schema."""

    def test_valid(self) -> None:
        """Test UserLogin with valid inputs."""
        data: auth.UserLogin = auth.UserLogin(
            identifier='user',
            password='pw',
            hcaptcha_token='captcha-token',
        )
        self.assertEqual(data.identifier, 'user')
        self.assertEqual(data.password, 'pw')
        self.assertEqual(data.hcaptcha_token, 'captcha-token')

    def test_hcaptcha_token_defaults_to_none(self) -> None:
        """Test UserLogin keeps hCaptcha validation in service layer."""
        data: auth.UserLogin = auth.UserLogin(
            identifier='user',
            password='pw',
        )
        self.assertIsNone(data.hcaptcha_token)

    def test_missing_field(self) -> None:
        """Test UserLogin raises ValidationError when required fields are
        missing."""
        with self.assertRaises(ValidationError):
            auth.UserLogin(identifier='user')

        with self.assertRaises(ValidationError):
            auth.UserLogin(password='pw')


class TestLogoutRequest(unittest.TestCase):
    """Tests for LogoutRequest schema."""

    def test_valid(self) -> None:
        """Test LogoutRequest with valid inputs."""
        data: auth.LogoutRequest = auth.LogoutRequest(refresh_token='token')
        self.assertEqual(data.refresh_token, 'token')

    def test_missing(self) -> None:
        """LogoutRequest can omit refresh_token for Web cookie logout."""
        data = auth.LogoutRequest()
        self.assertIsNone(data.refresh_token)


class TestRefreshRequest(unittest.TestCase):
    """Tests for RefreshRequest schema."""

    def test_valid(self) -> None:
        """Test RefreshRequest with valid inputs."""
        data: auth.RefreshRequest = auth.RefreshRequest(refresh_token='token')
        self.assertEqual(data.refresh_token, 'token')

    def test_missing(self) -> None:
        """RefreshRequest can omit refresh_token for Web cookie refresh."""
        data = auth.RefreshRequest()
        self.assertIsNone(data.refresh_token)


class TestTokenPair(unittest.TestCase):
    """Tests for TokenPair schema."""

    def test_valid(self) -> None:
        """Test TokenPair with all valid inputs provided."""
        data: auth.TokenPair = auth.TokenPair(
            access_token='a',
            refresh_token='r',
            username='u',
            role='admin',
            user_id=1,
            group_id=2,
            feature_names=['f1', 'f2'],
        )
        self.assertEqual(data.access_token, 'a')
        self.assertEqual(data.refresh_token, 'r')
        self.assertEqual(data.username, 'u')
        self.assertEqual(data.role, 'admin')
        self.assertEqual(data.user_id, 1)
        self.assertEqual(data.group_id, 2)
        self.assertEqual(data.feature_names, ['f1', 'f2'])

    def test_optional_and_default(self) -> None:
        """Test TokenPair with optional fields omitted, using default
        values."""
        data: auth.TokenPair = auth.TokenPair(
            access_token='a',
        )
        self.assertIsNone(data.username)
        self.assertIsNone(data.refresh_token)
        self.assertIsNone(data.role)
        self.assertIsNone(data.user_id)
        self.assertIsNone(data.group_id)
        self.assertEqual(data.feature_names, [])

    def test_missing_required(self) -> None:
        """Test TokenPair raises ValidationError when required fields are
        missing."""
        with self.assertRaises(ValidationError):
            auth.TokenPair(refresh_token='r')

        data = auth.TokenPair(access_token='a')
        self.assertIsNone(data.refresh_token)


if __name__ == '__main__':
    unittest.main()
