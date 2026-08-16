from __future__ import annotations

import unittest
from datetime import datetime
from datetime import timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import mock_open
from unittest.mock import patch

import jwt
from fastapi import HTTPException

from examples.auth.models import USER_STATUS_ACTIVE
from examples.auth.models import USER_STATUS_EMAIL_UNVERIFIED
from examples.auth.models import USER_STATUS_PENDING_ADMIN_APPROVAL
from examples.auth.models import USER_STATUS_REJECTED
from examples.auth.models import USER_STATUS_SUSPENDED
from examples.auth.models import UserIdentity
from examples.db_management.schemas.auth import ProviderClaims
from examples.db_management.services import oauth_services as svc


def _claims(**values: object) -> ProviderClaims:
    return ProviderClaims.model_validate(
        {'sub': 'provider-user', **values},
    )


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
            _claims(
                sub='google-sub',
                email='user@example.com',
                email_verified=True,
            ),
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
        user = MagicMock(status=USER_STATUS_PENDING_ADMIN_APPROVAL)
        self.db.scalar.return_value = MagicMock(user_id=10)
        self.db.get.return_value = user

        with self.assertRaises(HTTPException) as ctx:
            await svc.authenticate_provider_user(
                'google',
                _claims(sub='google-sub'),
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
        user = MagicMock(status=USER_STATUS_SUSPENDED)
        self.db.scalar.return_value = MagicMock(user_id=10)
        self.db.get.return_value = user

        with self.assertRaises(HTTPException) as ctx:
            await svc.authenticate_provider_user(
                'apple',
                _claims(sub='apple-sub'),
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
                _claims(
                    sub='google-sub',
                    email='existing@example.com',
                    email_verified=True,
                ),
                self.db,
                self.redis,
            )

        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.detail['code'], 'account_link_required')

    async def test_unverified_email_existing_user_requires_binding(
        self,
    ) -> None:
        self.db.scalar.side_effect = [None, MagicMock()]

        with self.assertRaises(HTTPException) as ctx:
            await svc.authenticate_provider_user(
                'google',
                _claims(
                    sub='google-sub',
                    email='existing@example.com',
                    email_verified=False,
                ),
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
        user = MagicMock(id=99, status=USER_STATUS_PENDING_ADMIN_APPROVAL)
        mock_create_pending.return_value = user

        with self.assertRaises(HTTPException) as ctx:
            await svc.authenticate_provider_user(
                'apple',
                _claims(
                    sub='apple-sub',
                    email='new@example.com',
                    email_verified=True,
                ),
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
                _claims(
                    sub='apple-sub',
                    email='new@example.com',
                    email_verified=True,
                ),
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
                _claims(
                    sub='google-sub',
                    email='user@example.com',
                    email_verified=True,
                ),
                self.db,
            )

        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(
            ctx.exception.detail['code'],
            'identity_already_linked_to_another_user',
        )

    async def test_link_provider_creates_identity_for_current_user(
        self,
    ) -> None:
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
            assert isinstance(obj, UserIdentity)
            obj.id = identity.id
            obj.provider = identity.provider
            obj.email = identity.email
            obj.display_name = identity.display_name
            obj.linked_at = identity.linked_at

        self.db.add.side_effect = add_side_effect

        result = await svc.link_provider_identity(
            user,
            'google',
            _claims(
                sub='google-sub',
                email='user@example.com',
                email_verified=True,
                name='User',
            ),
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
                _claims(
                    sub='google-sub',
                    email='user@example.com',
                    email_verified=True,
                ),
                self.db,
            )

        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(
            ctx.exception.detail['code'],
            'provider_already_linked',
        )

    async def test_unlink_rejects_last_login_method(self) -> None:
        user = MagicMock(id=1, password_hash=svc.OAUTH_DISABLED_PASSWORD_HASH)
        self.db.get.return_value = MagicMock(user_id=1)
        self.db.scalar.return_value = 1

        with self.assertRaises(HTTPException) as ctx:
            await svc.unlink_identity(user, 12, self.db)

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.detail['code'], 'last_login_method')

    @patch.object(
        svc, '_exchange_apple_authorization_code', new_callable=AsyncMock,
    )
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


def _settings(**values: object) -> SimpleNamespace:
    defaults: dict[str, object] = {
        'google_client_ids': '',
        'apple_client_ids': '',
        'apple_private_key': '',
        'apple_private_key_path': '',
        'apple_team_id': '',
        'apple_key_id': '',
        'apple_service_id': '',
        'apple_bundle_id': '',
        'apple_redirect_uri': '',
    }
    defaults.update(values)
    return SimpleNamespace(**defaults)


class TestOAuthHelperCoverage(unittest.TestCase):
    def test_configured_client_ids_and_claim_normalisation(self) -> None:
        settings = _settings(
            google_client_ids=' web-client, ,mobile-client ',
            apple_client_ids=' ios-client, service-client ',
        )
        with patch.object(svc, 'settings', settings):
            self.assertEqual(
                svc._configured_google_client_ids(),
                ['web-client', 'mobile-client'],
            )
            self.assertEqual(
                svc._configured_apple_client_ids(),
                ['ios-client', 'service-client'],
            )

        self.assertEqual(
            svc._normalise_email(
                ' User@Example.COM ',
            ),
            'user@example.com',
        )
        self.assertIsNone(svc._normalise_email('  '))
        self.assertIsNone(svc._normalise_email(None))

    def test_verify_jwt_rejects_unconfigured_invalid_and_missing_subject(
        self,
    ) -> None:
        with self.assertRaises(HTTPException) as missing_audience:
            svc._verify_jwt_with_jwks('token', 'jwks', [], ['issuer'])
        self.assertEqual(missing_audience.exception.status_code, 500)

        with patch.object(
            svc.jwt,
            'PyJWKClient',
            side_effect=jwt.PyJWTError('invalid'),
        ):
            with self.assertRaises(HTTPException) as invalid_token:
                svc._verify_jwt_with_jwks(
                    'token',
                    'jwks',
                    ['client'],
                    ['issuer'],
                )
        self.assertEqual(invalid_token.exception.status_code, 401)

        signing_key = MagicMock(key='public-key')
        jwk_client = MagicMock()
        jwk_client.get_signing_key_from_jwt.return_value = signing_key
        with patch.object(svc.jwt, 'PyJWKClient', return_value=jwk_client):
            with patch.object(
                svc.jwt, 'decode', return_value={'aud': 'client'},
            ):
                with self.assertRaises(HTTPException) as missing_subject:
                    svc._verify_jwt_with_jwks(
                        'token',
                        'jwks',
                        ['client'],
                        ['issuer'],
                    )
        self.assertEqual(missing_subject.exception.status_code, 401)

    def test_verify_jwt_returns_verified_claims(self) -> None:
        signing_key = MagicMock(key='public-key')
        jwk_client = MagicMock()
        jwk_client.get_signing_key_from_jwt.return_value = signing_key
        claims = {'sub': 'provider-user', 'email': 'user@example.com'}
        with patch.object(svc.jwt, 'PyJWKClient', return_value=jwk_client):
            with patch.object(
                svc.jwt, 'decode', return_value=claims,
            ) as decode:
                result = svc._verify_jwt_with_jwks(
                    'token',
                    'jwks',
                    ['client'],
                    ['issuer'],
                )

        self.assertEqual(result.sub, claims['sub'])
        self.assertEqual(result.email, claims['email'])
        decode.assert_called_once_with(
            'token',
            'public-key',
            algorithms=['RS256'],
            audience=['client'],
            issuer=['issuer'],
        )

    def test_load_apple_private_key_from_value_file_and_missing_config(
        self,
    ) -> None:
        with patch.object(
            svc,
            'settings',
            _settings(apple_private_key='line-one\\nline-two'),
        ):
            self.assertEqual(
                svc._load_apple_private_key(),
                'line-one\nline-two',
            )

        with patch.object(
            svc,
            'settings',
            _settings(apple_private_key_path='/tmp/apple.key'),
        ):
            with patch('builtins.open', mock_open(read_data='file-key')):
                self.assertEqual(svc._load_apple_private_key(), 'file-key')

        with patch.object(svc, 'settings', _settings()):
            with self.assertRaises(HTTPException) as missing_config:
                svc._load_apple_private_key()
        self.assertEqual(missing_config.exception.status_code, 500)

    def test_build_apple_client_secret_checks_config_and_signs(self) -> None:
        with patch.object(svc, 'settings', _settings()):
            with self.assertRaises(HTTPException) as missing_config:
                svc._build_apple_client_secret('client')
        self.assertEqual(missing_config.exception.status_code, 500)

        settings = _settings(
            apple_team_id='team-id',
            apple_key_id='key-id',
            apple_private_key='private-key',
        )
        with patch.object(svc, 'settings', settings):
            with patch.object(
                svc.jwt, 'encode', return_value='client-secret',
            ) as encode:
                self.assertEqual(
                    svc._build_apple_client_secret('service-id'),
                    'client-secret',
                )

        payload = encode.call_args.args[0]
        self.assertEqual(payload['iss'], 'team-id')
        self.assertEqual(payload['aud'], svc.APPLE_ISSUER)
        self.assertEqual(payload['sub'], 'service-id')
        self.assertEqual(encode.call_args.kwargs['algorithm'], 'ES256')
        self.assertEqual(encode.call_args.kwargs['headers'], {'kid': 'key-id'})

    def test_apple_candidate_and_user_profile_helpers(self) -> None:
        settings = _settings(
            apple_service_id='service',
            apple_bundle_id='bundle',
            apple_client_ids='service, bundle, native',
        )
        with patch.object(svc, 'settings', settings):
            self.assertEqual(
                svc._apple_exchange_client_id_candidates(),
                ['service', 'bundle', 'native'],
            )

        self.assertEqual(
            svc._username_from_claims(
                'google', _claims(email='A B@example.com'),
            ),
            'a_b',
        )
        self.assertEqual(
            svc._username_from_claims('apple', _claims(sub='subject')),
            'apple_subject',
        )
        self.assertEqual(
            svc._username_from_claims(
                'apple', _claims(email='---@example.com'),
            ),
            'apple_user',
        )
        self.assertEqual(
            svc._profile_names(
                'google',
                _claims(given_name='Given', family_name='Family'),
            ),
            ('Family', 'Given'),
        )
        self.assertEqual(
            svc._profile_names('google', _claims(name='First Last')),
            ('First', 'Last'),
        )
        self.assertEqual(
            svc._profile_names('apple', _claims(name='Solo')),
            ('Apple', 'Solo'),
        )
        self.assertEqual(
            svc._profile_names('google', _claims(email='name@example.com')),
            ('Google', 'name'),
        )
        self.assertEqual(
            svc._profile_names('apple', _claims()),
            ('Apple', 'User'),
        )
        self.assertEqual(
            svc._display_name_from_claims(
                _claims(given_name='Given', family_name='Family'),
            ),
            'Given Family',
        )
        self.assertIsNone(svc._display_name_from_claims(_claims()))

    def test_status_identity_and_password_helpers(self) -> None:
        self.assertEqual(
            svc._status_error(USER_STATUS_EMAIL_UNVERIFIED).detail['code'],
            'email_unverified',
        )
        self.assertEqual(
            svc._status_error(USER_STATUS_SUSPENDED).detail['code'],
            'account_suspended',
        )
        self.assertEqual(
            svc._status_error(USER_STATUS_REJECTED).detail['code'],
            'account_rejected',
        )
        self.assertEqual(
            svc._status_error(
                'other',
            ).detail['code'],
            'user_not_active',
        )

        with self.assertRaises(HTTPException):
            svc._ensure_active_user(MagicMock(status=USER_STATUS_REJECTED))
        svc._ensure_active_user(MagicMock(status=USER_STATUS_ACTIVE))
        self.assertTrue(
            svc._user_has_password(
                MagicMock(password_hash='hashed'),
            ),
        )
        self.assertFalse(
            svc._user_has_password(
                MagicMock(password_hash=svc.OAUTH_DISABLED_PASSWORD_HASH),
            ),
        )

        identity = MagicMock(
            id=5,
            provider='google',
            email='user@example.com',
            display_name='User',
            linked_at=datetime(2026, 7, 24, tzinfo=timezone.utc),
        )
        self.assertEqual(
            svc._identity_read(
                identity,
            ).linked_at,
            '2026-07-24T00:00:00Z',
        )

    def test_new_identity_and_claim_update(self) -> None:
        user = MagicMock()
        with patch.object(
            svc,
            'UserIdentity',
            side_effect=lambda **kwargs: kwargs,
        ):
            identity = svc._new_identity(
                user,
                'apple',
                _claims(
                    sub='apple-user',
                    email='relay@privaterelay.appleid.com',
                    email_verified=True,
                    given_name='Given',
                    family_name='Family',
                ),
            )
        self.assertEqual(identity['display_name'], 'Given Family')
        self.assertTrue(identity['raw_email_is_private'])

        stored = MagicMock(
            email='old@example.com',
            email_verified=False,
            display_name='Old',
        )
        svc._update_identity_from_claims(
            stored,
            _claims(
                email=' NEW@EXAMPLE.COM ',
                email_verified=True,
                name='New Name',
                is_private_email=False,
            ),
        )
        self.assertEqual(stored.email, 'new@example.com')
        self.assertTrue(stored.email_verified)
        self.assertEqual(stored.display_name, 'New Name')
        self.assertFalse(stored.raw_email_is_private)


class TestOAuthAsyncCoverage(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.db = MagicMock()
        self.db.scalar = AsyncMock()
        self.db.get = AsyncMock()
        self.db.flush = AsyncMock()
        self.db.commit = AsyncMock()
        self.db.refresh = AsyncMock()
        self.db.delete = AsyncMock()

    async def test_google_token_verification_validates_email(self) -> None:
        with patch.object(
            svc.asyncio,
            'to_thread',
            new=AsyncMock(
                return_value={
                    'sub': 'google-user',
                    'email_verified': True,
                    'email': 'user@example.com',
                },
            ),
        ):
            result = await svc.verify_google_id_token('id-token')
        self.assertEqual(result.sub, 'google-user')

        for claims in [
            {
                'sub': 'google-user',
                'email_verified': False,
                'email': 'user@example.com',
            },
            {'sub': 'google-user', 'email_verified': True},
        ]:
            with patch.object(
                svc.asyncio, 'to_thread', new=AsyncMock(return_value=claims),
            ):
                with self.assertRaises(HTTPException) as invalid_claims:
                    await svc.verify_google_id_token('id-token')
            self.assertEqual(invalid_claims.exception.status_code, 401)

    async def test_verify_apple_identity_token_all_exchange_paths(
        self,
    ) -> None:
        settings = _settings(
            apple_client_ids='web-client, native-client',
            apple_service_id='web-client',
            apple_bundle_id='native-client',
        )
        with patch.object(svc, 'settings', settings):
            with patch.object(
                svc.asyncio,
                'to_thread',
                new=AsyncMock(
                    side_effect=[
                        {
                            'sub': 'apple-user',
                            'aud': 'web-client',
                            'nonce': 'nonce',
                        },
                        {'sub': 'apple-user', 'aud': 'web-client'},
                    ],
                ),
            ):
                with patch.object(
                    svc,
                    '_exchange_apple_authorization_code',
                    new=AsyncMock(return_value={'id_token': 'exchanged'}),
                ) as exchange:
                    result = await svc.verify_apple_identity_token(
                        'identity',
                        'code',
                        expected_nonce='nonce',
                    )
        self.assertEqual(result.sub, 'apple-user')
        exchange.assert_awaited_once_with('code', ['web-client'])

        with patch.object(svc, 'settings', settings):
            with patch.object(
                svc.asyncio,
                'to_thread',
                new=AsyncMock(
                    return_value={'sub': 'apple-user', 'aud': 'web-client'},
                ),
            ):
                with patch.object(
                    svc,
                    '_exchange_apple_authorization_code',
                    new=AsyncMock(return_value={'id_token': 'exchanged'}),
                ) as exchange:
                    result = await svc.verify_apple_identity_token(
                        None, 'code',
                    )
        self.assertEqual(result.sub, 'apple-user')
        exchange.assert_awaited_once_with(
            'code',
            ['web-client', 'native-client'],
        )

        with patch.object(svc, 'settings', settings):
            with patch.object(
                svc.asyncio,
                'to_thread',
                new=AsyncMock(
                    return_value={'sub': 'apple-user', 'aud': 'wrong'},
                ),
            ):
                with self.assertRaises(HTTPException) as invalid_audience:
                    await svc.verify_apple_identity_token('identity', 'code')
        self.assertEqual(invalid_audience.exception.status_code, 401)

        with patch.object(svc, 'settings', settings):
            with patch.object(
                svc.asyncio,
                'to_thread',
                new=AsyncMock(
                    side_effect=[
                        {'sub': 'one', 'aud': 'web-client'},
                        {'sub': 'two', 'aud': 'web-client'},
                    ],
                ),
            ):
                with patch.object(
                    svc,
                    '_exchange_apple_authorization_code',
                    new=AsyncMock(return_value={'id_token': 'exchanged'}),
                ):
                    with self.assertRaises(
                        HTTPException,
                    ) as mismatched_subject:
                        await svc.verify_apple_identity_token(
                            'identity', 'code',
                        )
        self.assertEqual(mismatched_subject.exception.status_code, 401)

        with patch.object(svc, 'settings', settings):
            with patch.object(
                svc,
                '_exchange_apple_authorization_code',
                new=AsyncMock(return_value={}),
            ):
                with self.assertRaises(HTTPException) as missing_identity:
                    await svc.verify_apple_identity_token(None, 'code')
        self.assertEqual(missing_identity.exception.status_code, 401)

    async def test_apple_code_exchange_retries_and_validates_response(
        self,
    ) -> None:
        error = HTTPException(status_code=401, detail='Invalid provider token')
        with patch.object(
            svc,
            '_exchange_apple_authorization_code_once',
            new=AsyncMock(side_effect=[error, {'id_token': 'id'}]),
        ):
            self.assertEqual(
                (
                    await svc._exchange_apple_authorization_code(
                        'code', ['one', 'two'],
                    )
                ).id_token,
                'id',
            )
        with patch.object(
            svc,
            '_exchange_apple_authorization_code_once',
            new=AsyncMock(side_effect=error),
        ):
            with self.assertRaises(HTTPException) as final_error:
                await svc._exchange_apple_authorization_code('code', ['one'])
        self.assertEqual(final_error.exception.status_code, 401)
        with self.assertRaises(HTTPException) as no_clients:
            await svc._exchange_apple_authorization_code('code', [])
        self.assertEqual(no_clients.exception.status_code, 500)

        response = MagicMock(status_code=200)
        response.json.return_value = {'id_token': 'id'}
        client = MagicMock()
        client.post = AsyncMock(return_value=response)
        context = MagicMock()
        context.__aenter__ = AsyncMock(return_value=client)
        context.__aexit__ = AsyncMock(return_value=None)
        settings = _settings(
            apple_service_id='web-client',
            apple_redirect_uri='https://app/callback',
        )
        with patch.object(svc, 'settings', settings):
            with patch.object(
                svc, '_build_apple_client_secret', return_value='secret',
            ):
                with patch.object(
                    svc.httpx, 'AsyncClient', return_value=context,
                ):
                    result = await svc._exchange_apple_authorization_code_once(
                        'code', 'web-client',
                    )
        self.assertEqual(result.id_token, 'id')
        self.assertEqual(
            client.post.call_args.kwargs['data']['redirect_uri'],
            'https://app/callback',
        )

        bad_response = MagicMock(status_code=400)
        bad_client = MagicMock()
        bad_client.post = AsyncMock(return_value=bad_response)
        bad_context = MagicMock()
        bad_context.__aenter__ = AsyncMock(return_value=bad_client)
        bad_context.__aexit__ = AsyncMock(return_value=None)
        with patch.object(
            svc, '_build_apple_client_secret', return_value='secret',
        ):
            with patch.object(
                svc.httpx, 'AsyncClient', return_value=bad_context,
            ):
                with self.assertRaises(HTTPException) as rejected:
                    await svc._exchange_apple_authorization_code_once(
                        'code', 'native',
                    )
        self.assertEqual(rejected.exception.status_code, 401)

        invalid_json_response = MagicMock(status_code=200)
        invalid_json_response.json.side_effect = ValueError('invalid json')
        invalid_json_client = MagicMock()
        invalid_json_client.post = AsyncMock(
            return_value=invalid_json_response,
        )
        invalid_json_context = MagicMock()
        invalid_json_context.__aenter__ = AsyncMock(
            return_value=invalid_json_client,
        )
        invalid_json_context.__aexit__ = AsyncMock(return_value=None)
        with patch.object(
            svc, '_build_apple_client_secret', return_value='secret',
        ):
            with patch.object(
                svc.httpx, 'AsyncClient', return_value=invalid_json_context,
            ):
                with self.assertRaises(HTTPException) as invalid_json:
                    await svc._exchange_apple_authorization_code_once(
                        'code', 'native',
                    )
        self.assertEqual(invalid_json.exception.status_code, 401)

    async def test_database_lookup_and_new_pending_user(self) -> None:
        with self.assertRaises(HTTPException) as missing_provider_subject:
            svc._provider_claims({})
        self.assertEqual(missing_provider_subject.exception.status_code, 401)

        self.db.scalar.return_value = MagicMock(user_id=9)
        self.db.get.return_value = 'user'
        self.assertEqual(
            await svc._find_identity_user(self.db, 'google', 'provider-user'),
            'user',
        )
        self.db.scalar.return_value = None
        self.assertIsNone(
            await svc._find_identity_user(self.db, 'google', 'provider-user'),
        )
        self.db.scalar.return_value = 'email-user'
        self.assertEqual(
            await svc._find_user_by_email(self.db, 'USER@example.com'),
            'email-user',
        )
        self.db.scalar.side_effect = [True, True, None]
        self.assertEqual(
            await svc._unique_username(
                self.db,
                'google',
                _claims(email='name@example.com'),
            ),
            'name_3',
        )

        with self.assertRaises(HTTPException) as missing_email:
            await svc._create_pending_user_with_identity(
                self.db,
                'google',
                _claims(),
            )
        self.assertEqual(missing_email.exception.status_code, 400)

        user = MagicMock(id=77)
        profile = MagicMock()
        identity = MagicMock()
        with patch.object(svc, 'User', return_value=user) as make_user:
            with patch.object(svc, 'UserProfile', return_value=profile):
                with patch.object(svc, '_new_identity', return_value=identity):
                    with patch.object(
                        svc,
                        '_unique_username',
                        new=AsyncMock(return_value='unique'),
                    ):
                        result = await svc._create_pending_user_with_identity(
                            self.db,
                            'google',
                            _claims(
                                sub='provider-user',
                                email='user@example.com',
                            ),
                        )
        self.assertIs(result, user)
        self.assertEqual(make_user.call_args.kwargs['username'], 'unique')
        self.db.flush.assert_awaited()
        self.db.commit.assert_awaited()
        self.db.refresh.assert_awaited_once_with(
            user,
            attribute_names=['profile', 'group'],
        )

    async def test_login_wrappers_and_identity_list(self) -> None:
        google_claims = _claims(sub='google-user')
        with patch.object(
            svc,
            'verify_google_id_token',
            new=AsyncMock(return_value=google_claims),
        ):
            with patch.object(
                svc,
                'authenticate_provider_user',
                new=AsyncMock(return_value={'access_token': 'token'}),
            ) as authenticate:
                result = await svc.login_with_google(
                    'id-token',
                    self.db,
                    MagicMock(),
                    display_name='User',
                    device_lang='zh-TW',
                )
        self.assertEqual(result['access_token'], 'token')
        self.assertEqual(google_claims.name, 'User')
        self.assertEqual(google_claims.device_lang, 'zh-TW')
        self.assertEqual(
            authenticate.call_args.args[:2],
            ('google', google_claims),
        )

        apple_claims = _claims(sub='apple-user')
        with patch.object(
            svc,
            'verify_apple_identity_token',
            new=AsyncMock(return_value=apple_claims),
        ):
            with patch.object(
                svc,
                'authenticate_provider_user',
                new=AsyncMock(return_value={'access_token': 'token'}),
            ) as authenticate:
                await svc.login_with_apple(
                    None,
                    'code',
                    self.db,
                    MagicMock(),
                    email='user@example.com',
                    given_name='Given',
                    family_name='Family',
                    nonce='nonce',
                    device_lang='en',
                    hash_refresh_token=True,
                )
        self.assertEqual(apple_claims.email, 'user@example.com')
        self.assertEqual(apple_claims.given_name, 'Given')
        self.assertEqual(apple_claims.family_name, 'Family')
        self.assertEqual(apple_claims.device_lang, 'en')
        self.assertTrue(authenticate.call_args.kwargs['hash_refresh_token'])

        identity = MagicMock(
            id=1,
            provider='google',
            email='user@example.com',
            display_name='User',
            linked_at=datetime(2026, 7, 24, tzinfo=timezone.utc),
        )
        execute_result = MagicMock()
        execute_result.scalars.return_value.all.return_value = [identity]
        self.db.execute = AsyncMock(return_value=execute_result)
        listed = await svc.list_user_identities(
            MagicMock(id=1, password_hash=svc.OAUTH_DISABLED_PASSWORD_HASH),
            self.db,
        )
        self.assertFalse(listed.has_password)
        self.assertEqual(listed.identities[0].provider, 'google')

    async def test_link_wrappers_update_and_unlink(self) -> None:
        user = MagicMock(id=1, password_hash='hashed')
        existing = MagicMock(
            id=9,
            user_id=1,
            provider='google',
            email='old@example.com',
            email_verified=False,
            display_name='Old',
            linked_at=datetime(2026, 7, 24, tzinfo=timezone.utc),
        )
        self.db.scalar.return_value = existing
        updated = await svc.link_provider_identity(
            user,
            'google',
            _claims(
                sub='google-user',
                email='new@example.com',
                name='New',
            ),
            self.db,
        )
        self.assertEqual(updated.email, 'new@example.com')
        self.db.commit.assert_awaited()

        with self.assertRaises(HTTPException) as missing_subject:
            svc._provider_claims({})
        self.assertEqual(missing_subject.exception.status_code, 401)

        with patch.object(
            svc,
            'verify_google_id_token',
            new=AsyncMock(return_value=_claims(sub='google-user')),
        ):
            with patch.object(
                svc,
                'link_provider_identity',
                new=AsyncMock(return_value='google-linked'),
            ):
                self.assertEqual(
                    await svc.link_google_identity(user, 'id-token', self.db),
                    'google-linked',
                )
        with patch.object(
            svc,
            'verify_apple_identity_token',
            new=AsyncMock(return_value=_claims(sub='apple-user')),
        ):
            with patch.object(
                svc,
                'link_provider_identity',
                new=AsyncMock(return_value='apple-linked'),
            ):
                self.assertEqual(
                    await svc.link_apple_identity(
                        user, None, 'code', self.db, 'nonce',
                    ),
                    'apple-linked',
                )

        self.db.get.return_value = None
        with self.assertRaises(HTTPException) as missing_identity:
            await svc.unlink_identity(user, 7, self.db)
        self.assertEqual(missing_identity.exception.status_code, 404)

        identity = MagicMock(user_id=1)
        self.db.get.return_value = identity
        self.db.scalar.return_value = 2
        self.assertEqual(
            await svc.unlink_identity(user, 7, self.db),
            {'message': 'Identity unlinked successfully.'},
        )
        self.db.delete.assert_awaited_once_with(identity)


if __name__ == '__main__':
    unittest.main()
