from __future__ import annotations

import unittest

from pydantic import ValidationError

from examples.db_management.schemas.password_reset import (
    ForgotPasswordRequest,
)
from examples.db_management.schemas.password_reset import (
    PasswordMessageResponse,
)
from examples.db_management.schemas.password_reset import ResetPasswordRequest


class TestPasswordResetSchemas(unittest.TestCase):
    """Tests for password reset request and response schemas."""

    def test_forgot_password_request_accepts_email(self) -> None:
        """Test forgot password request accepts email."""
        payload = ForgotPasswordRequest(email='user@example.com')

        self.assertEqual(str(payload.email), 'user@example.com')

    def test_forgot_password_request_rejects_invalid_email(self) -> None:
        """Test forgot password request rejects invalid email."""
        with self.assertRaises(ValidationError):
            ForgotPasswordRequest(email='not-an-email')

    def test_reset_password_request_requires_token_and_password(self) -> None:
        """Test reset password request requires token and password."""
        payload = ResetPasswordRequest(
            token='raw-token',
            new_password='NewPass123',
        )

        self.assertEqual(payload.token, 'raw-token')
        self.assertEqual(payload.new_password, 'NewPass123')

    def test_reset_password_request_allows_service_layer_validation(
        self,
    ) -> None:
        """Test reset password request allows service layer validation."""
        payload = ResetPasswordRequest()

        self.assertIsNone(payload.token)
        self.assertIsNone(payload.new_password)

    def test_password_message_response(self) -> None:
        """Test password message response."""
        payload = PasswordMessageResponse(message='ok')

        self.assertEqual(payload.message, 'ok')
