from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import mock_open
from unittest.mock import patch

import jwt
from fastapi import HTTPException

from examples.db_management.schemas.auth import ProviderClaims
from examples.db_management.services import oauth_services as svc


def _claims(**values: object) -> ProviderClaims:
    """Perform claims.

    Args:
        **values: Value used by this callable.

    Returns:
        The callable result.
    """
    return ProviderClaims.model_validate(
        {'sub': 'provider-user', **values},
    )


class TestOAuthServices(unittest.IsolatedAsyncioTestCase):
    """Tests for Google/Apple provider account resolution."""

    def setUp(self) -> None:
        """Perform setUp."""
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

    @patch.object(
        svc,
        '_exchange_apple_authorization_code',
        new_callable=AsyncMock,
    )
    @patch.object(svc, '_verify_jwt_with_jwks')
    async def test_apple_nonce_mismatch_is_rejected(
        self,
        mock_verify_jwt: MagicMock,
        mock_exchange_code: AsyncMock,
    ) -> None:
        """Test apple nonce mismatch is rejected.

        Args:
            mock_verify_jwt: Value used by this callable.
            mock_exchange_code: Value used by this callable.
        """
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
    """Perform settings.

    Args:
        **values: Value used by this callable.

    Returns:
        The callable result.
    """
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
    """Provide TestOAuthHelperCoverage."""

    def test_configured_client_ids_and_claim_normalisation(self) -> None:
        """Test configured client ids and claim normalisation."""
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
        """Test verify jwt rejects unconfigured invalid and missing subject."""
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
                svc.jwt,
                'decode',
                return_value={'aud': 'client'},
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
        """Test verify jwt returns verified claims."""
        signing_key = MagicMock(key='public-key')
        jwk_client = MagicMock()
        jwk_client.get_signing_key_from_jwt.return_value = signing_key
        claims = {'sub': 'provider-user', 'email': 'user@example.com'}
        with patch.object(svc.jwt, 'PyJWKClient', return_value=jwk_client):
            with patch.object(
                svc.jwt,
                'decode',
                return_value=claims,
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
        """Test load apple private key from value file and missing config."""
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
        """Test build apple client secret checks config and signs."""
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
                svc.jwt,
                'encode',
                return_value='client-secret',
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
        """Test apple candidate and user profile helpers."""
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


class TestOAuthAsyncCoverage(unittest.IsolatedAsyncioTestCase):
    """Provide TestOAuthAsyncCoverage."""

    def setUp(self) -> None:
        """Perform setUp."""
        self.db = MagicMock()
        self.db.scalar = AsyncMock()
        self.db.get = AsyncMock()
        self.db.flush = AsyncMock()
        self.db.commit = AsyncMock()
        self.db.refresh = AsyncMock()
        self.db.delete = AsyncMock()

    async def test_google_token_verification_validates_email(self) -> None:
        """Test google token verification validates email."""
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
                svc.asyncio,
                'to_thread',
                new=AsyncMock(return_value=claims),
            ):
                with self.assertRaises(HTTPException) as invalid_claims:
                    await svc.verify_google_id_token('id-token')
            self.assertEqual(invalid_claims.exception.status_code, 401)

    async def test_verify_apple_identity_token_all_exchange_paths(
        self,
    ) -> None:
        """Test verify apple identity token all exchange paths."""
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
                        None,
                        'code',
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
                            'identity',
                            'code',
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
        """Test apple code exchange retries and validates response."""
        error = HTTPException(status_code=401, detail='Invalid provider token')
        with patch.object(
            svc,
            '_exchange_apple_authorization_code_once',
            new=AsyncMock(side_effect=[error, {'id_token': 'id'}]),
        ):
            self.assertEqual(
                (
                    await svc._exchange_apple_authorization_code(
                        'code',
                        ['one', 'two'],
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
                svc,
                '_build_apple_client_secret',
                return_value='secret',
            ):
                with patch.object(
                    svc.httpx,
                    'AsyncClient',
                    return_value=context,
                ):
                    result = await svc._exchange_apple_authorization_code_once(
                        'code',
                        'web-client',
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
            svc,
            '_build_apple_client_secret',
            return_value='secret',
        ):
            with patch.object(
                svc.httpx,
                'AsyncClient',
                return_value=bad_context,
            ):
                with self.assertRaises(HTTPException) as rejected:
                    await svc._exchange_apple_authorization_code_once(
                        'code',
                        'native',
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
            svc,
            '_build_apple_client_secret',
            return_value='secret',
        ):
            with patch.object(
                svc.httpx,
                'AsyncClient',
                return_value=invalid_json_context,
            ):
                with self.assertRaises(HTTPException) as invalid_json:
                    await svc._exchange_apple_authorization_code_once(
                        'code',
                        'native',
                    )
        self.assertEqual(invalid_json.exception.status_code, 401)


if __name__ == '__main__':
    unittest.main()
