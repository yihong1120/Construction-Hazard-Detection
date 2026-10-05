from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import patch
from uuid import UUID

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Response
from fastapi.testclient import TestClient

from examples.auth.database import get_db
from examples.auth.deployment_context import DeploymentBinding
from examples.auth.redis_pool import get_redis_pool
from examples.auth.session_store import AUTH_SESSION_TTL_SECONDS
from examples.auth.session_store import create_auth_session
from examples.bff import router as bff
from examples.bff import session_services
from examples.bff.user_summary import load_user_summary
from tests.examples.auth.session_store_test import FakeRedis
from tests.oidc_fixtures import fixture_token

_DEPLOYMENT = DeploymentBinding(
    tenant_id=UUID('00000000-0000-0000-0000-000000000001'),
    deployment_id=UUID('00000000-0000-0000-0000-000000000002'),
    api_base_url='https://api.example.com',
    config_revision=1,
)


def _access_subject() -> dict[str, object]:
    """Perform access subject.

    Returns:
        The callable result.
    """
    return {
        'username': 'alice',
        'user_id': 1,
        'role': 'user',
        'jti': 'access-jti',
        'features': [],
        'tenant_id': str(_DEPLOYMENT.tenant_id),
        'deployment_id': str(_DEPLOYMENT.deployment_id),
        'config_revision': _DEPLOYMENT.config_revision,
    }


def _refresh_subject() -> dict[str, object]:
    """Perform refresh subject.

    Returns:
        The callable result.
    """
    return {
        'username': 'alice',
        'family_id': 'refresh-family',
        'token_id': 'refresh-token-id',
        'tenant_id': str(_DEPLOYMENT.tenant_id),
        'deployment_id': str(_DEPLOYMENT.deployment_id),
        'config_revision': _DEPLOYMENT.config_revision,
    }


class BffRouterTest(unittest.TestCase):
    """Provide BffRouterTest."""

    def setUp(self) -> None:
        """Perform setUp."""
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
        deployment_resolver = patch.object(
            session_services,
            'resolve_request_deployment',
            new=AsyncMock(return_value=_DEPLOYMENT),
        )
        deployment_resolver.start()
        self.addCleanup(deployment_resolver.stop)

    def test_session_refreshes_server_token_and_rolls_cookie(self) -> None:
        """An active web session refreshes server-side credentials only."""
        access = fixture_token(
            _access_subject(),
            issuer=_DEPLOYMENT.issuer,
            audience=_DEPLOYMENT.audience,
        )
        refresh = fixture_token(
            _refresh_subject(),
            issuer=_DEPLOYMENT.issuer,
            audience=_DEPLOYMENT.audience,
        )
        session_id, session = asyncio.run(
            create_auth_session(
                self.redis,  # type: ignore[arg-type]
                {
                    'access_token': access,
                    'refresh_token': refresh,
                    'feature_names': [],
                    'deployment': _DEPLOYMENT.as_response(),
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
        get_proxy_access_token.assert_awaited_once_with(
            self.redis,
            session_id,
            deployment=_DEPLOYMENT,
        )
        self.assertIn('HttpOnly', response.headers['set-cookie'])
        self.assertEqual(
            self.redis.ttls[f"bff:session:{session['session_id_hash']}"],
            AUTH_SESSION_TTL_SECONDS,
        )

    def test_oidc_logout_returns_an_opaque_global_logout_url(self) -> None:
        """An OIDC BFF logout clears local state before browser navigation."""
        access = fixture_token(
            _access_subject(),
            issuer=_DEPLOYMENT.issuer,
            audience=_DEPLOYMENT.audience,
        )
        refresh = fixture_token(
            _refresh_subject(),
            issuer=_DEPLOYMENT.issuer,
            audience=_DEPLOYMENT.audience,
        )
        session_id, _ = asyncio.run(
            create_auth_session(
                self.redis,  # type: ignore[arg-type]
                {
                    'access_token': access,
                    'auth_provider': 'oidc',
                    'feature_names': [],
                    'refresh_token': refresh,
                    'deployment': _DEPLOYMENT.as_response(),
                },
                {'id': 1, 'username': 'alice'},
            ),
        )
        cookies = {session_services.SESSION_COOKIE: session_id}
        csrf = self.client.get('/bff/auth/csrf', cookies=cookies).json()[
            'csrf_token'
        ]
        with patch(
            'examples.bff.session_services.create_oidc_global_logout_url',
            new_callable=AsyncMock,
            return_value='/bff/auth/oidc/logout?state=opaque-state',
        ):
            response = self.client.post(
                '/bff/auth/logout',
                cookies=cookies,
                headers={
                    'Origin': self.origin,
                    'X-CSRF-Token': csrf,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {'global_logout_url': '/bff/auth/oidc/logout?state=opaque-state'},
        )
        self.assertNotIn('token', response.text.lower())

    def test_detection_catalog_uses_authenticated_session(self) -> None:
        """Web model discovery uses an authenticated BFF session."""
        access = fixture_token(
            _access_subject(),
            issuer=_DEPLOYMENT.issuer,
            audience=_DEPLOYMENT.audience,
        )
        refresh = fixture_token(
            _refresh_subject(),
            issuer=_DEPLOYMENT.issuer,
            audience=_DEPLOYMENT.audience,
        )
        session_id, _ = asyncio.run(
            create_auth_session(
                self.redis,  # type: ignore[arg-type]
                {
                    'access_token': access,
                    'refresh_token': refresh,
                    'feature_names': [],
                    'deployment': _DEPLOYMENT.as_response(),
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
                '/bff/detection/models',
                cookies=cookies,
            )

        self.assertEqual(response.status_code, 204)
        assert proxy_request.await_args is not None
        self.assertEqual(proxy_request.await_args.args[3], 'detection/models')
        self.db.close.assert_awaited_once()
        self.assertIn('HttpOnly', response.headers['set-cookie'])

    def test_service_proxy_path_does_not_repeat_api(self) -> None:
        """Test service proxy path does not repeat api."""
        access = fixture_token(
            _access_subject(),
            issuer=_DEPLOYMENT.issuer,
            audience=_DEPLOYMENT.audience,
        )
        refresh = fixture_token(
            _refresh_subject(),
            issuer=_DEPLOYMENT.issuer,
            audience=_DEPLOYMENT.audience,
        )
        session_id, _ = asyncio.run(
            create_auth_session(
                self.redis,  # type: ignore[arg-type]
                {
                    'access_token': access,
                    'refresh_token': refresh,
                    'feature_names': [],
                    'deployment': _DEPLOYMENT.as_response(),
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
        self.db.close.assert_awaited_once()
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
            asyncio.run(load_user_summary(self.db, 999))
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

        summary = asyncio.run(load_user_summary(self.db, 1))

        self.assertEqual(summary.display_name, 'service-account')

    def test_session_deployment_rejects_invalid_or_changed_bindings(
        self,
    ) -> None:
        """Browser sessions cannot outlive invalid or replaced deployments."""
        request = SimpleNamespace()
        with self.assertRaises(HTTPException) as invalid:
            asyncio.run(
                session_services._require_session_deployment(
                    request,
                    self.db,
                    {},
                ),
            )
        self.assertEqual(invalid.exception.status_code, 409)

        changed = DeploymentBinding(
            tenant_id=_DEPLOYMENT.tenant_id,
            deployment_id=_DEPLOYMENT.deployment_id,
            api_base_url='https://replacement.example.com',
            config_revision=2,
        )
        with patch.object(
            session_services,
            'resolve_request_deployment',
            AsyncMock(return_value=changed),
        ):
            with self.assertRaises(HTTPException) as mismatch:
                asyncio.run(
                    session_services._require_session_deployment(
                        request,
                        self.db,
                        {'deployment': _DEPLOYMENT.as_response()},
                    ),
                )
        self.assertEqual(mismatch.exception.status_code, 409)

    def test_proxy_post_requires_csrf_token(self) -> None:
        """Mutating BFF proxy requests always pass through CSRF enforcement."""
        access = fixture_token(
            _access_subject(),
            issuer=_DEPLOYMENT.issuer,
            audience=_DEPLOYMENT.audience,
        )
        refresh = fixture_token(
            _refresh_subject(),
            issuer=_DEPLOYMENT.issuer,
            audience=_DEPLOYMENT.audience,
        )
        session_id, _ = asyncio.run(
            create_auth_session(
                self.redis,  # type: ignore[arg-type]
                {
                    'access_token': access,
                    'refresh_token': refresh,
                    'feature_names': [],
                    'deployment': _DEPLOYMENT.as_response(),
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

    def test_device_invitation_proxy_uses_session_and_csrf(self) -> None:
        """The browser reaches invitation management only through the BFF."""
        access = fixture_token(
            _access_subject(),
            issuer=_DEPLOYMENT.issuer,
            audience=_DEPLOYMENT.audience,
        )
        refresh = fixture_token(
            _refresh_subject(),
            issuer=_DEPLOYMENT.issuer,
            audience=_DEPLOYMENT.audience,
        )
        session_id, session = asyncio.run(
            create_auth_session(
                self.redis,  # type: ignore[arg-type]
                {
                    'access_token': access,
                    'refresh_token': refresh,
                    'feature_names': [],
                    'deployment': _DEPLOYMENT.as_response(),
                },
                {'id': 1, 'username': 'alice'},
            ),
        )
        with patch(
            'examples.bff.session_services.proxy_request',
            new_callable=AsyncMock,
            return_value=Response(status_code=200, content=b'{}'),
        ) as proxy_request:
            response = self.client.post(
                '/bff/db_management/deployment-enrollment-codes',
                cookies={session_services.SESSION_COOKIE: session_id},
                headers={
                    'Origin': self.origin,
                    'X-CSRF-Token': str(session['csrf_secret']),
                },
                json={'expires_in_minutes': 30},
            )

        self.assertEqual(response.status_code, 200)
        assert proxy_request.await_args is not None
        self.assertEqual(
            proxy_request.await_args.args[3],
            'db_management/deployment-enrollment-codes',
        )

    def test_web_playback_proxy_uses_db_management_and_csrf(self) -> None:
        """The browser sends playback controls through the BFF allowlist."""
        access = fixture_token(
            _access_subject(),
            issuer=_DEPLOYMENT.issuer,
            audience=_DEPLOYMENT.audience,
        )
        refresh = fixture_token(
            _refresh_subject(),
            issuer=_DEPLOYMENT.issuer,
            audience=_DEPLOYMENT.audience,
        )
        session_id, session = asyncio.run(
            create_auth_session(
                self.redis,  # type: ignore[arg-type]
                {
                    'access_token': access,
                    'refresh_token': refresh,
                    'feature_names': [],
                    'deployment': _DEPLOYMENT.as_response(),
                },
                {'id': 1, 'username': 'alice'},
            ),
        )
        with patch(
            'examples.bff.session_services.proxy_request',
            new_callable=AsyncMock,
            return_value=Response(status_code=200, content=b'{}'),
        ) as proxy_request:
            response = self.client.post(
                '/bff/db_management/api/playback/sessions',
                cookies={session_services.SESSION_COOKIE: session_id},
                headers={
                    'Origin': self.origin,
                    'X-CSRF-Token': str(session['csrf_secret']),
                },
                json={'site': 'Site A', 'camera': 'Camera 1'},
            )

        self.assertEqual(response.status_code, 200)
        assert proxy_request.await_args is not None
        self.assertEqual(
            proxy_request.await_args.args[3],
            'db_management/api/playback/sessions',
        )


if __name__ == '__main__':
    unittest.main()
