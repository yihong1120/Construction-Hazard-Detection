from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import patch

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Response
from fastapi.testclient import TestClient

from examples.auth.database import get_db
from examples.auth.jwt_config import jwt_access
from examples.auth.jwt_config import jwt_refresh
from examples.auth.redis_pool import get_redis_pool
from examples.auth.session_store import AUTH_SESSION_TTL_SECONDS
from examples.auth.session_store import create_auth_session
from examples.bff import router as bff
from examples.bff import session_services
from tests.examples.auth.session_store_test import FakeRedis


def _access_subject() -> dict[str, object]:
    return {
        'username': 'alice',
        'user_id': 1,
        'role': 'user',
        'jti': 'access-jti',
        'features': [],
    }


def _refresh_subject() -> dict[str, str]:
    return {
        'username': 'alice',
        'family_id': 'refresh-family',
        'token_id': 'refresh-token-id',
    }


class BffRouterTest(unittest.TestCase):
    def setUp(self) -> None:
        app = FastAPI()
        app.include_router(bff.router)
        self.redis = FakeRedis()
        self.db = AsyncMock()
        self.db.scalar.return_value = SimpleNamespace(
            id=1,
            username='alice',
            role='user',
            group_id=2,
            status='active',
            profile=SimpleNamespace(given_name='Alice', family_name='Chen'),
        )
        app.dependency_overrides[get_redis_pool] = lambda: self.redis
        app.dependency_overrides[get_db] = lambda: self.db
        self.client = TestClient(app)
        self.origin = 'https://changdar-server.mooo.com'

    @patch('examples.bff.session_services.login_user')
    def test_login_and_session_never_expose_tokens(
        self,
        login_user: AsyncMock,
    ) -> None:
        access = jwt_access.create_access_token(_access_subject())
        refresh = jwt_refresh.create_access_token(_refresh_subject())
        login_user.return_value = {
            'access_token': access,
            'refresh_token': refresh,
            'user_id': 1,
            'feature_names': ['stream'],
        }
        response = self.client.post(
            '/bff/auth/login',
            headers={'Origin': self.origin},
            json={
                'identifier': 'alice',
                'password': 'secret',
                'hcaptcha_token': 'captcha',
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertNotIn('access_token', response.text)
        self.assertNotIn('refresh_token', response.text)
        self.assertEqual(response.json()['user']['display_name'], 'Alice Chen')
        self.assertIn('HttpOnly', response.headers['set-cookie'])
        self.assertIn('Secure', response.headers['set-cookie'])
        session_id = response.cookies[session_services.SESSION_COOKIE]

        session = self.client.get(
            '/bff/auth/session',
            cookies={session_services.SESSION_COOKIE: session_id},
        )
        self.assertEqual(session.status_code, 200)
        self.assertEqual(session.json()['feature_names'], ['stream'])
        self.assertNotIn('token', session.text.lower())
        self.assertIn('HttpOnly', session.headers['set-cookie'])
        self.assertIn('Secure', session.headers['set-cookie'])

    def test_session_refreshes_server_token_and_rolls_cookie(self) -> None:
        """An active web session refreshes server-side credentials only."""
        access = jwt_access.create_access_token(_access_subject())
        refresh = jwt_refresh.create_access_token(_refresh_subject())
        session_id, session = asyncio.run(
            create_auth_session(
                self.redis,  # type: ignore[arg-type]
                {
                    'access_token': access,
                    'refresh_token': refresh,
                    'feature_names': [],
                },
                {
                    'id': 1,
                    'username': 'alice',
                    'display_name': 'Alice Chen',
                    'role': 'user',
                    'group_id': 2,
                    'status': 'active',
                },
            ),
        )

        with patch(
            'examples.bff.session_services.get_proxy_access_token',
            new_callable=AsyncMock,
            return_value=('rotated-access-token', session),
        ) as get_proxy_access_token:
            response = self.client.get(
                '/bff/auth/session',
                cookies={session_services.SESSION_COOKIE: session_id},
            )

        self.assertEqual(response.status_code, 200)
        get_proxy_access_token.assert_awaited_once_with(self.redis, session_id)
        self.assertIn('HttpOnly', response.headers['set-cookie'])
        self.assertEqual(
            self.redis.ttls[f'bff:session:{session["session_id_hash"]}'],
            AUTH_SESSION_TTL_SECONDS,
        )

    @patch('examples.bff.session_services.login_user')
    def test_csrf_and_origin_protect_logout(
        self,
        login_user: AsyncMock,
    ) -> None:
        login_user.return_value = {
            'access_token': jwt_access.create_access_token(
                _access_subject(),
            ),
            'refresh_token': jwt_refresh.create_access_token(
                _refresh_subject(),
            ),
            'user_id': 1,
            'feature_names': [],
        }
        login = self.client.post(
            '/bff/auth/login',
            headers={'Origin': self.origin},
            json={'identifier': 'alice', 'password': 'secret'},
        )
        session_id = login.cookies[session_services.SESSION_COOKIE]
        cookies = {session_services.SESSION_COOKIE: session_id}

        denied = self.client.post('/bff/auth/logout', cookies=cookies)
        self.assertEqual(denied.status_code, 403)

        csrf = self.client.get(
            '/bff/auth/csrf',
            cookies=cookies,
        ).json()['csrf_token']
        with patch(
            'examples.bff.session_services.logout_user',
            new_callable=AsyncMock,
        ):
            logged_out = self.client.post(
                '/bff/auth/logout',
                cookies=cookies,
                headers={
                    'Origin': self.origin,
                    'X-CSRF-Token': csrf,
                },
            )
        self.assertEqual(logged_out.status_code, 204)

    def test_service_proxy_path_does_not_repeat_api(self) -> None:
        access = jwt_access.create_access_token(_access_subject())
        refresh = jwt_refresh.create_access_token(_refresh_subject())
        session_id, _ = asyncio.run(
            create_auth_session(
                self.redis,  # type: ignore[arg-type]
                {
                    'access_token': access,
                    'refresh_token': refresh,
                    'feature_names': [],
                },
                {'id': 1, 'username': 'alice'},
            ),
        )
        cookies = {session_services.SESSION_COOKIE: session_id}

        with patch(
            'examples.bff.session_services.proxy_request',
            new_callable=AsyncMock,
            return_value=Response(status_code=204),
        ) as proxy_request:
            response = self.client.get(
                '/bff/chat/messages',
                cookies=cookies,
            )

        self.assertEqual(response.status_code, 204)
        assert proxy_request.await_args is not None
        self.assertEqual(proxy_request.await_args.args[3], 'chat/messages')
        self.assertIn('HttpOnly', response.headers['set-cookie'])

        old_path = self.client.get(
            '/bff/api/chat/messages',
            cookies=cookies,
        )
        self.assertEqual(old_path.status_code, 404)
        self.assertEqual(old_path.json()['detail'], 'bff_route_not_allowed')

    def test_session_helpers_reject_expired_sessions_and_missing_users(
        self,
    ) -> None:
        """BFF session helpers do not continue with deleted server-side
        data."""
        request = SimpleNamespace(cookies={})
        with patch(
            'examples.bff.session_services.get_auth_session',
            new_callable=AsyncMock,
            return_value=None,
        ):
            with self.assertRaises(HTTPException) as expired:
                asyncio.run(session_services._session(request, self.redis))
        self.assertEqual(expired.exception.status_code, 401)

        self.db.scalar.return_value = None
        with self.assertRaises(HTTPException) as missing_user:
            asyncio.run(session_services._user_summary(self.db, 999))
        self.assertEqual(missing_user.exception.detail, 'user_not_found')

    def test_user_summary_uses_username_when_profile_is_absent(self) -> None:
        """Build a BFF session for an account created without a profile."""
        self.db.scalar.return_value = SimpleNamespace(
            id=1,
            username='service-account',
            role='user',
            group_id=None,
            status='active',
            profile=None,
        )

        summary = asyncio.run(session_services._user_summary(self.db, 1))

        self.assertEqual(summary.display_name, 'service-account')

    def test_proxy_post_requires_csrf_token(self) -> None:
        """Mutating BFF proxy requests always pass through CSRF enforcement."""
        access = jwt_access.create_access_token(_access_subject())
        refresh = jwt_refresh.create_access_token(_refresh_subject())
        session_id, _ = asyncio.run(
            create_auth_session(
                self.redis,  # type: ignore[arg-type]
                {
                    'access_token': access,
                    'refresh_token': refresh,
                    'feature_names': [],
                },
                {'id': 1, 'username': 'alice'},
            ),
        )

        response = self.client.post(
            '/bff/chat/messages',
            cookies={session_services.SESSION_COOKIE: session_id},
            headers={'Origin': self.origin},
        )

        self.assertEqual(response.status_code, 403)


if __name__ == '__main__':
    unittest.main()
