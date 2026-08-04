from __future__ import annotations

import asyncio
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import patch
from urllib.parse import parse_qs
from urllib.parse import urlsplit

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.testclient import TestClient

from examples.auth.database import get_db
from examples.auth.jwt_config import jwt_access
from examples.auth.jwt_config import jwt_refresh
from examples.auth.redis_pool import get_redis_pool
from examples.auth.session_store import create_auth_session
from examples.bff import security as bff
from examples.db_management.routers import oauth
from tests.examples.auth.session_store_test import FakeRedis


class OAuthRouterTest(unittest.TestCase):
    verifier = 'dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk'
    challenge = 'E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM'
    redirect_uri = 'com.changdar.visionnaire:/oauth2redirect'

    def setUp(self) -> None:
        app = FastAPI()
        app.include_router(oauth.router)
        self.redis = FakeRedis()
        self.db = AsyncMock()
        self.db.scalar.return_value = SimpleNamespace(
            id=1,
            username='alice',
            role='user',
            group_id=1,
            status='active',
        )
        app.dependency_overrides[get_redis_pool] = lambda: self.redis
        app.dependency_overrides[get_db] = lambda: self.db
        self.client = TestClient(app)

        access = jwt_access.create_access_token({'username': 'alice'})
        refresh = jwt_refresh.create_access_token({'username': 'alice'})
        self.session_id, _ = asyncio.run(
            create_auth_session(
                self.redis,  # type: ignore[arg-type]
                {'access_token': access, 'refresh_token': refresh},
                {'id': 1, 'username': 'alice'},
            ),
        )

    @patch(
        'examples.db_management.routers.oauth.issue_token_pair_for_user',
        new_callable=AsyncMock,
    )
    def test_pkce_code_is_bound_and_single_use(
        self,
        issue_token_pair: AsyncMock,
    ) -> None:
        issued_access = jwt_access.create_access_token({'username': 'alice'})
        issued_refresh = jwt_refresh.create_access_token(
            {'username': 'alice'},
        )
        issue_token_pair.return_value = {
            'access_token': issued_access,
            'refresh_token': issued_refresh,
        }
        authorize = self.client.get(
            '/oauth/authorize',
            follow_redirects=False,
            cookies={bff.SESSION_COOKIE: self.session_id},
            params={
                'response_type': 'code',
                'client_id': 'visionnaire-ios',
                'redirect_uri': self.redirect_uri,
                'code_challenge': self.challenge,
                'code_challenge_method': 'S256',
                'state': 'client-state',
            },
        )
        self.assertEqual(authorize.status_code, 302)
        query = parse_qs(urlsplit(authorize.headers['location']).query)
        self.assertEqual(query['state'], ['client-state'])
        code = query['code'][0]

        form = {
            'grant_type': 'authorization_code',
            'client_id': 'visionnaire-ios',
            'redirect_uri': self.redirect_uri,
            'code': code,
            'code_verifier': self.verifier,
        }
        exchanged = self.client.post('/oauth/token', data=form)
        self.assertEqual(exchanged.status_code, 200)
        self.assertEqual(exchanged.json()['expires_in'], 900)
        self.assertEqual(exchanged.json()['token_type'], 'Bearer')

        reused = self.client.post('/oauth/token', data=form)
        self.assertEqual(reused.status_code, 400)
        self.assertEqual(reused.json()['detail'], 'invalid_grant')

    def test_authorize_rejects_unregistered_redirect(self) -> None:
        response = self.client.get(
            '/oauth/authorize',
            follow_redirects=False,
            cookies={bff.SESSION_COOKIE: self.session_id},
            params={
                'response_type': 'code',
                'client_id': 'visionnaire-ios',
                'redirect_uri': 'https://attacker.example/callback',
                'code_challenge': self.challenge,
                'code_challenge_method': 'S256',
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()['detail'], 'invalid_oauth_client')


if __name__ == '__main__':
    unittest.main()


class _Request:
    def __init__(
        self,
        data: object | None = None,
        *,
        content_type: str = 'application/json',
        cookies: dict[str, str] | None = None,
        authorization: str | None = None,
    ) -> None:
        self._data = data if data is not None else {}
        self.cookies = cookies or {}
        self.headers = {'content-type': content_type}
        if authorization:
            self.headers['authorization'] = authorization

    async def json(self) -> object:
        return self._data

    async def form(self) -> object:
        return self._data


class TestOAuthRouterCoverage(unittest.IsolatedAsyncioTestCase):
    client_id = 'visionnaire-ios'
    redirect_uri = 'com.changdar.visionnaire:/oauth2redirect'
    verifier = 'dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk'
    challenge = oauth._pkce_challenge(verifier)

    async def test_helpers_parse_json_and_reject_invalid_native_config(
        self,
    ) -> None:
        self.assertEqual(
            await oauth._request_data(
                _Request({'grant_type': 'refresh_token'}),
            ),
            {'grant_type': 'refresh_token'},
        )
        self.assertEqual(await oauth._request_data(_Request([])), {})
        self.assertEqual(
            await oauth._request_data(
                _Request(
                    {'answer': 42},
                    content_type='application/x-www-form-urlencoded',
                ),
            ),
            {'answer': '42'},
        )
        with patch.dict(os.environ, {'OAUTH_NATIVE_CLIENTS_JSON': '{'}):
            with self.assertRaisesRegex(
                RuntimeError, 'Invalid OAUTH_NATIVE_CLIENTS_JSON',
            ):
                oauth._native_clients()

    async def test_authorize_rejects_invalid_pkce_and_missing_login(
        self,
    ) -> None:
        redis = AsyncMock()
        request = _Request(cookies={oauth.SESSION_COOKIE: 'session'})
        cases = [
            (
                {'response_type': 'token', 'code_challenge_method': 'S256'},
                'pkce_s256_required',
            ),
            (
                {'response_type': 'code', 'code_challenge_method': 'plain'},
                'pkce_s256_required',
            ),
            (
                {
                    'response_type': 'code',
                    'code_challenge_method': 'S256',
                    'code_challenge': 'invalid',
                },
                'invalid_code_challenge',
            ),
        ]
        for values, detail in cases:
            with self.assertRaises(HTTPException) as raised:
                await oauth.authorize(
                    request,
                    client_id=self.client_id,
                    redirect_uri=self.redirect_uri,
                    code_challenge=values.get(
                        'code_challenge',
                        self.challenge,
                    ),
                    redis=redis,
                    **{
                        'response_type': values['response_type'],
                        'code_challenge_method': values[
                            'code_challenge_method'
                        ],
                    },
                )
            self.assertEqual(raised.exception.detail, detail)

        with patch.object(
            oauth, 'get_auth_session', AsyncMock(return_value=None),
        ):
            with self.assertRaises(HTTPException) as raised:
                await oauth.authorize(
                    request,
                    response_type='code',
                    client_id=self.client_id,
                    redirect_uri=self.redirect_uri,
                    code_challenge=self.challenge,
                    code_challenge_method='S256',
                    redis=redis,
                )
        self.assertEqual(raised.exception.detail, 'login_required')

        with patch.object(
            oauth, 'get_auth_session', AsyncMock(return_value={'user': {}}),
        ):
            with self.assertRaises(HTTPException) as raised:
                await oauth.authorize(
                    request,
                    response_type='code',
                    client_id=self.client_id,
                    redirect_uri=self.redirect_uri,
                    code_challenge=self.challenge,
                    code_challenge_method='S256',
                    redis=redis,
                )
        self.assertEqual(raised.exception.detail, 'login_required')

    async def test_token_rejects_invalid_authorization_codes(self) -> None:
        db = AsyncMock()
        redis = AsyncMock()
        base = {
            'grant_type': 'authorization_code',
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri,
        }
        for values in [
            {**base, 'code': '', 'code_verifier': self.verifier},
            {**base, 'code': 'code', 'code_verifier': 'short'},
        ]:
            with self.assertRaises(HTTPException) as raised:
                await oauth.token(_Request(values), db=db, redis=redis)
            self.assertEqual(raised.exception.detail, 'invalid_grant')

        redis.getdel.return_value = None
        with self.assertRaises(HTTPException) as raised:
            await oauth.token(
                _Request(
                    {
                        **base,
                        'code': 'missing',
                        'code_verifier': self.verifier,
                    },
                ),
                db=db,
                redis=redis,
            )
        self.assertEqual(raised.exception.detail, 'invalid_grant')

        redis.getdel.return_value = 'not-json'
        with self.assertRaises(HTTPException) as raised:
            await oauth.token(
                _Request(
                    {**base, 'code': 'broken', 'code_verifier': self.verifier},
                ),
                db=db,
                redis=redis,
            )
        self.assertEqual(raised.exception.detail, 'invalid_grant')

        redis.getdel.return_value = b'not-json'
        with self.assertRaises(HTTPException) as raised:
            await oauth.token(
                _Request(
                    {**base, 'code': 'bytes', 'code_verifier': self.verifier},
                ),
                db=db,
                redis=redis,
            )
        self.assertEqual(raised.exception.detail, 'invalid_grant')

        redis.getdel.return_value = json.dumps(
            {
                'client_id': self.client_id,
                'redirect_uri': self.redirect_uri,
                'code_challenge': 'wrong',
                'user_id': 1,
            },
        )
        with self.assertRaises(HTTPException) as raised:
            await oauth.token(
                _Request(
                    {
                        **base,
                        'code': 'mismatch',
                        'code_verifier': self.verifier,
                    },
                ),
                db=db,
                redis=redis,
            )
        self.assertEqual(raised.exception.detail, 'invalid_grant')

        redis.getdel.return_value = json.dumps(
            {
                'client_id': self.client_id,
                'redirect_uri': self.redirect_uri,
                'code_challenge': self.challenge,
                'user_id': 1,
            },
        )
        db.scalar.return_value = SimpleNamespace(status='disabled')
        with self.assertRaises(HTTPException) as raised:
            await oauth.token(
                _Request(
                    {
                        **base,
                        'code': 'inactive',
                        'code_verifier': self.verifier,
                    },
                ),
                db=db,
                redis=redis,
            )
        self.assertEqual(raised.exception.detail, 'invalid_grant')

    async def test_token_refresh_and_unsupported_grants(self) -> None:
        redis = AsyncMock()
        with patch.object(
            oauth,
            'refresh_tokens',
            AsyncMock(
                return_value={
                    'access_token': 'access',
                    'refresh_token': 'refresh',
                },
            ),
        ) as refresh:
            response = await oauth.token(
                _Request(
                    {
                        'grant_type': 'refresh_token',
                        'client_id': self.client_id,
                        'refresh_token': 'source-refresh',
                    },
                ),
                db=AsyncMock(),
                redis=redis,
            )
        self.assertEqual(response.access_token, 'access')
        assert refresh.await_args is not None
        self.assertEqual(
            refresh.await_args.args[0].refresh_token,
            'source-refresh',
        )

        with self.assertRaises(HTTPException) as raised:
            await oauth.token(
                _Request(
                    {
                        'grant_type': 'refresh_token',
                        'client_id': 'unknown',
                    },
                ),
                db=AsyncMock(),
                redis=redis,
            )
        self.assertEqual(raised.exception.detail, 'invalid_oauth_client')

        with self.assertRaises(HTTPException) as raised:
            await oauth.token(
                _Request({'grant_type': 'password'}),
                db=AsyncMock(),
                redis=redis,
            )
        self.assertEqual(raised.exception.detail, 'unsupported_grant_type')

    async def test_me_handles_invalid_and_active_users(self) -> None:
        db = AsyncMock()
        user = SimpleNamespace(id=7)
        db.scalar.return_value = None
        with self.assertRaises(HTTPException) as raised:
            await oauth.me(db=db, user=user)
        self.assertEqual(raised.exception.detail, 'invalid_user')

        loaded = SimpleNamespace(
            id=7,
            username='alice',
            role='admin',
            group_id=3,
            status='active',
            profile=SimpleNamespace(given_name='Alice', family_name='Chen'),
        )
        db.scalar.return_value = loaded
        with patch.object(
            oauth, '_load_feature_names', AsyncMock(
                return_value=['streaming'],
            ),
        ):
            response = await oauth.me(db=db, user=user)
        self.assertEqual(response.display_name, 'Alice Chen')
        self.assertEqual(response.feature_names, ['streaming'])

        loaded.profile = SimpleNamespace(given_name='', family_name='')
        with patch.object(
            oauth, '_load_feature_names', AsyncMock(return_value=[]),
        ):
            response = await oauth.me(db=db, user=user)
        self.assertEqual(response.display_name, 'alice')

    async def test_revoke_handles_refresh_access_and_authorization(
        self,
    ) -> None:
        redis = AsyncMock()
        with patch.object(oauth, 'logout_user', AsyncMock()) as logout:
            with patch.object(
                oauth, 'revoke_media_for_parent', AsyncMock(),
            ) as revoke_media:
                await oauth.revoke(
                    _Request(
                        {
                            'token': 'refresh-token',
                            'token_type_hint': 'refresh_token',
                        },
                        authorization='Bearer access-token',
                    ),
                    redis=redis,
                )
                await oauth.revoke(
                    _Request(
                        {
                            'token': 'access-token',
                            'token_type_hint': 'access_token',
                        },
                    ),
                    redis=redis,
                )
        self.assertEqual(logout.await_count, 3)
        self.assertEqual(revoke_media.await_count, 3)


if __name__ == '__main__':
    unittest.main()
