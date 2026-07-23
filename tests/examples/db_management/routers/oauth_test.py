from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import patch
from urllib.parse import parse_qs
from urllib.parse import urlsplit

from fastapi import FastAPI
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
