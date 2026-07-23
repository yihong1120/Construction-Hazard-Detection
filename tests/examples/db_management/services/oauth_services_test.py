from __future__ import annotations

import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import HTTPException

from examples.auth.models import USER_STATUS_ACTIVE
from examples.auth.models import USER_STATUS_INACTIVE
from examples.auth.models import USER_STATUS_PENDING
from examples.auth.models import UserIdentity
from examples.db_management.services import oauth_services as svc


class TestOAuthServices(unittest.IsolatedAsyncioTestCase):
    """Tests for Google/Apple provider account resolution."""

    def setUp(self) -> None:
        self.db: MagicMock = MagicMock()
        self.db.scalar = AsyncMock()
        self.db.get = AsyncMock()
        self.db.commit = AsyncMock()
        self.redis: AsyncMock = AsyncMock()
        self.consent_payload: MagicMock = MagicMock()
        self.consent_payload.accepted_terms = True
        self.consent_payload.terms_version = '2026-06-27'
        self.consent_payload.privacy_version = '2026-06-27'
        self.consent_payload.notification_consent = True
        self.consent_payload.ai_terms_accepted = True
        self.consent_payload.ai_terms_version = '2026-06-27'

    @patch.object(svc, 'issue_token_pair_for_user', new_callable=AsyncMock)
    async def test_existing_identity_active_user_gets_tokens(
        self,
        mock_issue_tokens: AsyncMock,
    ) -> None:
        user = MagicMock(
            id=10,
            username='user',
            role='user',
            group_id=1,
            status=USER_STATUS_ACTIVE,
        )
        identity = MagicMock(user_id=10)
        self.db.scalar.return_value = identity
        self.db.get.return_value = user
        mock_issue_tokens.return_value = {
            'access_token': 'access',
            'refresh_token': 'refresh',
        }

        result = await svc.authenticate_provider_user(
            'google',
            {
                'sub': 'google-sub',
                'email': 'user@example.com',
                'email_verified': True,
            },
            self.db,
            self.redis,
        )

        self.assertEqual(result['access_token'], 'access')
        mock_issue_tokens.assert_awaited_once_with(
            user,
            self.db,
            self.redis,
            hash_refresh_token=False,
        )

    async def test_existing_identity_pending_user_is_rejected(self) -> None:
        user = MagicMock(status=USER_STATUS_PENDING)
        self.db.scalar.return_value = MagicMock(user_id=10)
        self.db.get.return_value = user

        with self.assertRaises(HTTPException) as ctx:
            await svc.authenticate_provider_user(
                'google',
                {'sub': 'google-sub'},
                self.db,
                self.redis,
            )

        self.assertEqual(ctx.exception.status_code, 403)
        self.assertEqual(
            ctx.exception.detail,
            {
                'code': 'pending_admin_approval',
                'status': 'pending_admin_approval',
            },
        )

    async def test_existing_identity_inactive_user_is_rejected(self) -> None:
        user = MagicMock(status=USER_STATUS_INACTIVE)
        self.db.scalar.return_value = MagicMock(user_id=10)
        self.db.get.return_value = user

        with self.assertRaises(HTTPException) as ctx:
            await svc.authenticate_provider_user(
                'apple',
                {'sub': 'apple-sub'},
                self.db,
                self.redis,
            )

        self.assertEqual(ctx.exception.status_code, 403)
        self.assertEqual(
            ctx.exception.detail,
            {'code': 'account_suspended', 'status': 'suspended'},
        )

    async def test_verified_email_existing_user_requires_binding(self) -> None:
        user = MagicMock(status=USER_STATUS_ACTIVE)
        self.db.scalar.side_effect = [None, user]

        with self.assertRaises(HTTPException) as ctx:
            await svc.authenticate_provider_user(
                'google',
                {
                    'sub': 'google-sub',
                    'email': 'existing@example.com',
                    'email_verified': True,
                },
                self.db,
                self.redis,
            )

        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.detail['code'], 'account_link_required')

    async def test_unverified_email_existing_user_requires_binding(self) -> None:
        self.db.scalar.side_effect = [None, MagicMock()]

        with self.assertRaises(HTTPException) as ctx:
            await svc.authenticate_provider_user(
                'google',
                {
                    'sub': 'google-sub',
                    'email': 'existing@example.com',
                    'email_verified': False,
                },
                self.db,
                self.redis,
            )

        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.detail['code'], 'account_link_required')

    @patch.object(
        svc,
        '_create_pending_user_with_identity',
        new_callable=AsyncMock,
    )
    @patch.object(svc, 'record_user_consent', new_callable=AsyncMock)
    @patch.object(svc, 'validate_signup_consents', new_callable=AsyncMock)
    async def test_new_provider_account_creates_pending_user(
        self,
        mock_validate_consents: AsyncMock,
        mock_record_consent: AsyncMock,
        mock_create_pending: AsyncMock,
    ) -> None:
        self.db.scalar.side_effect = [None, None]
        user = MagicMock(id=99, status=USER_STATUS_PENDING)
        mock_create_pending.return_value = user

        with self.assertRaises(HTTPException) as ctx:
            await svc.authenticate_provider_user(
                'apple',
                {
                    'sub': 'apple-sub',
                    'email': 'new@example.com',
                    'email_verified': True,
                },
                self.db,
                self.redis,
                consent_payload=self.consent_payload,
            )

        self.assertEqual(ctx.exception.status_code, 403)
        self.assertEqual(
            ctx.exception.detail['code'],
            'pending_admin_approval',
        )
        mock_validate_consents.assert_awaited_once_with(
            self.consent_payload,
            self.db,
        )
        mock_record_consent.assert_awaited_once_with(
            99,
            self.consent_payload,
            self.db,
            request=None,
        )
        mock_create_pending.assert_awaited_once()

    async def test_new_provider_account_requires_legal_consent(self) -> None:
        """New social accounts must submit legal consent versions."""
        self.db.scalar.side_effect = [None, None]

        with self.assertRaises(HTTPException) as ctx:
            await svc.authenticate_provider_user(
                'apple',
                {
                    'sub': 'apple-sub',
                    'email': 'new@example.com',
                    'email_verified': True,
                },
                self.db,
                self.redis,
            )

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(
            ctx.exception.detail['code'],
            'legal_consent_required',
        )

    async def test_link_provider_rejects_identity_bound_to_another_user(
        self,
    ) -> None:
        user = MagicMock(id=1)
        self.db.scalar.return_value = MagicMock(user_id=2)

        with self.assertRaises(HTTPException) as ctx:
            await svc.link_provider_identity(
                user,
                'google',
                {
                    'sub': 'google-sub',
                    'email': 'user@example.com',
                    'email_verified': True,
                },
                self.db,
            )

        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(
            ctx.exception.detail['code'],
            'identity_already_linked_to_another_user',
        )

    async def test_link_provider_creates_identity_for_current_user(self) -> None:
        user = MagicMock(id=1)
        self.db.scalar.side_effect = [None, None]
        identity = MagicMock(
            id=12,
            provider='google',
            email='user@example.com',
            display_name='User',
            linked_at=svc.datetime(2026, 6, 21, tzinfo=svc.timezone.utc),
        )
        self.db.refresh = AsyncMock(side_effect=lambda obj: None)

        def add_side_effect(obj: object) -> None:
            self.assertIsInstance(obj, UserIdentity)
            obj.id = identity.id
            obj.provider = identity.provider
            obj.email = identity.email
            obj.display_name = identity.display_name
            obj.linked_at = identity.linked_at

        self.db.add.side_effect = add_side_effect

        result = await svc.link_provider_identity(
            user,
            'google',
            {
                'sub': 'google-sub',
                'email': 'user@example.com',
                'email_verified': True,
                'name': 'User',
            },
            self.db,
        )

        self.assertEqual(result.id, 12)
        self.db.commit.assert_awaited_once()

    async def test_link_provider_rejects_second_identity_for_same_provider(
        self,
    ) -> None:
        user = MagicMock(id=1)
        self.db.scalar.side_effect = [
            None,
            MagicMock(user_id=1, provider_user_id='other-google-sub'),
        ]

        with self.assertRaises(HTTPException) as ctx:
            await svc.link_provider_identity(
                user,
                'google',
                {
                    'sub': 'google-sub',
                    'email': 'user@example.com',
                    'email_verified': True,
                },
                self.db,
            )

        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(
            ctx.exception.detail['code'], 'provider_already_linked',
        )

    async def test_unlink_rejects_last_login_method(self) -> None:
        user = MagicMock(id=1, password_hash=svc.OAUTH_DISABLED_PASSWORD_HASH)
        self.db.get.return_value = MagicMock(user_id=1)
        self.db.scalar.return_value = 1

        with self.assertRaises(HTTPException) as ctx:
            await svc.unlink_identity(user, 12, self.db)

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.detail['code'], 'last_login_method')

    @patch.object(svc, '_exchange_apple_authorization_code', new_callable=AsyncMock)
    @patch.object(svc, '_verify_jwt_with_jwks')
    async def test_apple_nonce_mismatch_is_rejected(
        self,
        mock_verify_jwt: MagicMock,
        mock_exchange_code: AsyncMock,
    ) -> None:
        mock_verify_jwt.return_value = {
            'sub': 'apple-sub',
            'aud': 'com.changdar.visionnaire',
            'nonce': 'actual-nonce',
        }
        mock_exchange_code.return_value = {}

        with self.assertRaises(HTTPException) as ctx:
            await svc.verify_apple_identity_token(
                'identity-token',
                'code',
                expected_nonce='expected-nonce',
            )

        self.assertEqual(ctx.exception.status_code, 401)


if __name__ == '__main__':
    unittest.main()
