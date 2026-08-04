from __future__ import annotations

import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

from firebase_admin import messaging

from examples.local_notification_server.fcm_service import (
    _is_invalid_registration_token_error,
)
from examples.local_notification_server.fcm_service import _token_log_id
from examples.local_notification_server.fcm_service import FcmSendResult
from examples.local_notification_server.fcm_service import init_firebase_app
from examples.local_notification_server.fcm_service import (
    send_fcm_notification_service,
)
from examples.local_notification_server.fcm_service import WEBPUSH_CFG

patch(
    'firebase_admin.credentials.Certificate',
    return_value=MagicMock(),
).start()


class TestInitFirebaseApp(unittest.TestCase):
    """Test suite for initialising the Firebase application."""

    @patch('firebase_admin.credentials.Certificate', return_value=MagicMock())
    @patch('firebase_admin.initialize_app')
    @patch('firebase_admin._apps', new=[])
    def test_init_firebase_app_when_not_inited(
        self,
        mock_init_app: MagicMock,
        mock_cred: MagicMock,
    ) -> None:
        """Test that init_firebase_app() calls initialise_app when no apps are
        registered.

        Args:
            mock_init_app (MagicMock): Mocked initialise_app function.
            mock_cred (MagicMock): Mocked Certificate constructor.
        """
        init_firebase_app('dummy/path.json', 'dummy-project')
        mock_init_app.assert_called_once()
        mock_cred.assert_called_once()

    @patch('firebase_admin.credentials.Certificate', return_value=MagicMock())
    @patch('firebase_admin.initialize_app')
    @patch('firebase_admin._apps', new=['already_inited'])
    def test_init_firebase_app_when_already_inited(
        self,
        mock_init_app: MagicMock,
        mock_cred: MagicMock,
    ) -> None:
        """Test that init_firebase_app() does not re-initialise if an app is
        already registered.

        Args:
            mock_init_app (MagicMock): Mocked initialise_app function.
            mock_cred (MagicMock): Mocked Certificate constructor.
        """
        init_firebase_app('dummy/path.json', 'dummy-project')
        mock_init_app.assert_not_called()
        mock_cred.assert_not_called()

    def test_init_firebase_app_empty_cred_path(self) -> None:
        """Test that init_firebase_app raises ValueError if cred_path is
        empty."""
        with self.assertRaises(ValueError) as ctx:
            init_firebase_app('', 'dummy-project')
        self.assertIn('cred_path', str(ctx.exception))

    def test_init_firebase_app_empty_project_id(self) -> None:
        """Test that init_firebase_app raises ValueError if project_id is
        empty."""
        with self.assertRaises(ValueError) as ctx:
            init_firebase_app('dummy/path.json', '')
        self.assertIn('project_id', str(ctx.exception))


class TestFcmTokenHelpers(unittest.TestCase):
    """Tests for safe token logging and Firebase error compatibility."""

    def test_token_log_id_never_returns_the_raw_token(self) -> None:
        """Token logs expose a bounded hash and handle absent values."""
        self.assertEqual(_token_log_id(None), '<missing-token>')
        self.assertNotEqual(
            _token_log_id(
                'raw-device-token',
            ),
            'raw-device-token',
        )
        self.assertEqual(len(_token_log_id('raw-device-token')), 12)

    def test_invalid_token_errors_support_exception_code_and_messages(
        self,
    ) -> None:
        """Invalid registration tokens are recognized across SDK error
        shapes."""
        self.assertTrue(
            _is_invalid_registration_token_error(
                messaging.SenderIdMismatchError('mismatch'),
            ),
        )
        self.assertTrue(
            _is_invalid_registration_token_error(
                MagicMock(code='not-found'),
            ),
        )
        self.assertTrue(
            _is_invalid_registration_token_error(
                RuntimeError('Registration token is not registered'),
            ),
        )
        self.assertFalse(
            _is_invalid_registration_token_error(
                RuntimeError('timeout'),
            ),
        )


@patch(
    'examples.local_notification_server.fcm_service.init_firebase_app',
    lambda: None,
)
class TestSendFCMNotificationService(unittest.IsolatedAsyncioTestCase):
    """Test suite for sending FCM notifications using the
    send_fcm_notification_service function."""

    async def test_no_tokens(self) -> None:
        """Test that sending notifications with an empty token list returns
        False."""
        result = await send_fcm_notification_service([], 'Title', 'Body')
        self.assertFalse(result)
        self.assertIsInstance(result, FcmSendResult)
        self.assertEqual(result.success_count, 0)
        self.assertEqual(result.failure_count, 0)

    @patch('firebase_admin.messaging.send_each')
    async def test_all_success(self, mock_send_each: MagicMock) -> None:
        """Test that the service returns True when all tokens send
        successfully.

        Args:
            mock_send_each (MagicMock): Mocked send_each function.
        """
        mock_response = MagicMock()
        mock_response.failure_count = 0
        mock_response.responses = [MagicMock(success=True)]
        mock_send_each.return_value = mock_response

        tokens = ['valid_token']
        result = await send_fcm_notification_service(tokens, 'Title', 'Body')
        self.assertTrue(result)
        self.assertEqual(result.success_count, 1)
        self.assertEqual(result.failure_count, 0)
        self.assertEqual(result.invalid_tokens, ())
        message = mock_send_each.call_args.args[0][0]
        self.assertEqual(message.webpush, WEBPUSH_CFG)

    @patch('firebase_admin.messaging.send_each')
    async def test_partial_fail(self, mock_send_each: MagicMock) -> None:
        """Test that the service returns False when at least one token fails.

        Args:
            mock_send_each (MagicMock): Mocked send_each function.
        """
        mock_response = MagicMock()
        mock_response.failure_count = 1
        mock_response.responses = [
            MagicMock(success=True),
            MagicMock(
                success=False,
                exception=messaging.UnregisteredError(
                    'Requested entity was not found.',
                ),
            ),
        ]
        mock_send_each.return_value = mock_response

        tokens = ['valid_token', 'invalid_token']
        result = await send_fcm_notification_service(tokens, 'Title', 'Body')
        self.assertFalse(result)
        self.assertEqual(result.success_count, 1)
        self.assertEqual(result.failure_count, 1)
        self.assertEqual(result.invalid_tokens, ('invalid_token',))

    @patch('firebase_admin.messaging.send_each')
    async def test_exception(self, mock_send_each: MagicMock) -> None:
        """Test that the service returns False when an exception occurs.

        Args:
            mock_send_each (MagicMock): Mocked send_each function.
        """
        mock_send_each.side_effect = Exception('FCM send error')

        tokens = ['token']
        result = await send_fcm_notification_service(tokens, 'Title', 'Body')
        self.assertFalse(result)
        self.assertEqual(result.success_count, 0)
        self.assertEqual(result.failure_count, 1)
        self.assertEqual(result.invalid_tokens, ())


if __name__ == '__main__':
    unittest.main()
\
"""Pytest \

--cov=examples.local_notification_server.fcm_service \
--cov-report=term-missing \
tests/examples/local_notification_server/fcm_service_test.py
"""
