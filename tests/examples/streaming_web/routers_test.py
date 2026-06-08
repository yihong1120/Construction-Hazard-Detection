from __future__ import annotations

import unittest
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi.testclient import TestClient
from jose import jwt

from examples.auth.database import get_db
from examples.auth.jwt_config import jwt_access
from examples.auth.jwt_config import JwtAuthorizationCredentials
from examples.auth.redis_pool import get_redis_pool
from examples.auth.redis_pool import get_redis_pool_ws
from examples.streaming_web import routers
from examples.streaming_web.routers import _build_stream_listing
from examples.streaming_web.routers import router


class TestRouters(unittest.IsolatedAsyncioTestCase):
    """Test suite for FastAPI routers.

    This suite validates behaviour for the following endpoints:

    - GET /api/labels
    - GET /api/streams/{label}
    - GET /api/metadata/stream-id/{label}/{stream_id}
    - WS  /api/ws/metadata-id/{label}/{stream_id}

    Attributes:
        app: The in-memory FastAPI application under test.
        fake_redis: Async mock used to stand in for the Redis pool.
        mock_db_session: Async mock simulating the database session.
        client: Test client for driving HTTP and WebSocket requests.
    """

    app: FastAPI
    fake_redis: AsyncMock
    mock_db_session: AsyncMock
    client: TestClient

    def setUp(self) -> None:
        """Initialise the app and mock dependencies.

        This method wires dependency overrides so that networked
        integrations (Redis, DB, rate limiting, and JWT credentials)
        are replaced with safe, deterministic test doubles.

        Returns:
            None
        """
        self.app: FastAPI = FastAPI()
        self.app.include_router(router, prefix='/api')

        # Override Redis dependencies with an async mock
        self.fake_redis = AsyncMock()
        self.app.dependency_overrides[get_redis_pool] = lambda: self.fake_redis
        self.app.dependency_overrides[get_redis_pool_ws] = (
            lambda: self.fake_redis
        )

        # Bypass JWT authentication with a mock credentials object
        mock_credentials = SimpleNamespace(subject={'username': 'testuser'})
        self.app.dependency_overrides[jwt_access] = lambda: mock_credentials

        # Mock the database session
        self.mock_db_session = AsyncMock()
        self.app.dependency_overrides[get_db] = lambda: self.mock_db_session

        # Set up default mock user and result for database queries
        self.setup_default_db_mocks()

        self.client = TestClient(self.app)

    def setup_default_db_mocks(self) -> None:
        """Set up default mock user and site for database queries.

        Prepares a default user in the mocked session so tests that do not
        explicitly tailor the user can proceed without additional setup.

        Returns:
            None
        """
        # Create default mock site and user
        mock_site = MagicMock()
        mock_site.name = 'label1'
        mock_user = MagicMock()
        mock_user.role = 'admin'
        mock_user.id = 1
        mock_user.group_id = 1
        mock_user.sites = [mock_site]

        mock_user_result = MagicMock()
        mock_user_result.scalar_one_or_none.return_value = mock_user
        (
            mock_user_result.unique.return_value.scalars.return_value
            .one_or_none.return_value
        ) = mock_user
        mock_sites_result = MagicMock()
        mock_sites_result.scalars.return_value.all.return_value = [mock_site]
        (
            mock_sites_result.scalars.return_value.unique.return_value
            .all.return_value
        ) = [mock_site]

        self.mock_db_session.execute.side_effect = [
            mock_user_result,
            mock_sites_result,
        ]

    def tearDown(self) -> None:
        """Clear all dependency overrides after each test.

        Ensures state does not leak between test cases.

        Returns:
            None
        """
        self.app.dependency_overrides.clear()

    # -----------------------------
    # Test GET /api/labels
    # -----------------------------
    @patch(
        'examples.streaming_web.routers.get_user_and_sites',
        new_callable=AsyncMock,
    )
    def test_get_labels_success(
        self,
        mock_get_user_and_sites: AsyncMock,
    ) -> None:
        """Test successful retrieval of labels."""
        mock_user = MagicMock()
        mock_get_user_and_sites.return_value = (mock_user, [], 'super_admin')
        labels_result = MagicMock()
        labels_result.scalars.return_value.all.return_value = [
            'label1',
            'label2',
        ]
        self.mock_db_session.execute = AsyncMock(return_value=labels_result)

        response = self.client.get('/api/labels')

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'labels': ['label1', 'label2']})

    @patch(
        'examples.streaming_web.routers.get_user_and_sites',
        new_callable=AsyncMock,
    )
    def test_get_labels_with_non_admin_user(
        self, mock_get_user_and_sites: AsyncMock,
    ) -> None:
        """Test label filtering for non-admin users."""
        mock_user = MagicMock()
        mock_get_user_and_sites.return_value = (
            mock_user,
            ['label1'],
            'user',
        )
        labels_result = MagicMock()
        labels_result.scalars.return_value.all.return_value = [
            'label1',
            'label2',
        ]
        self.mock_db_session.execute = AsyncMock(return_value=labels_result)

        response = self.client.get('/api/labels')

        self.assertEqual(response.status_code, 200)
        # Non-admin user should only see their allowed labels
        self.assertEqual(response.json(), {'labels': ['label1']})

    @patch(
        'examples.streaming_web.routers.get_user_and_sites',
        new_callable=AsyncMock,
    )
    def test_get_labels_error(
        self,
        mock_get_user_and_sites: AsyncMock,
    ) -> None:
        """Test error handling in labels endpoint."""
        mock_user = MagicMock()
        mock_get_user_and_sites.return_value = (mock_user, [], 'super_admin')
        self.mock_db_session.execute = AsyncMock(
            side_effect=Exception('DB error'),
        )

        response = self.client.get('/api/labels')

        self.assertEqual(response.status_code, 500)
        self.assertIn('Failed to fetch labels', response.json()['detail'])

    def test_get_labels_invalid_token(self) -> None:
        """Test labels endpoint with invalid token."""
        # Override JWT to return invalid credentials
        mock_credentials = SimpleNamespace(subject={})
        self.app.dependency_overrides[jwt_access] = lambda: mock_credentials

        response = self.client.get('/api/labels')

        # The endpoint catches the HTTPException and returns 500
        self.assertEqual(response.status_code, 500)
        self.assertIn('Invalid token', response.json()['detail'])

    @patch(
        'examples.streaming_web.routers._get_configured_media_streams',
        new_callable=AsyncMock,
    )
    def test_get_streams_returns_empty_without_configured_streams(
        self,
        mock_get_configured_streams: AsyncMock,
    ) -> None:
        """Return only DB-configured streams without scanning Redis keys."""
        mock_get_configured_streams.return_value = []

        response = self.client.get('/api/streams/label1')

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'streams': []})

    def test_stream_listing_uses_clean_hls_by_default(self) -> None:
        """Use clean HLS by default; overlay playback is negotiated later."""
        stream = _build_stream_listing('label1', 'Cam1', 'Q2FtMQ')

        self.assertEqual(
            stream['playback_url'],
            '/hazard/media/hazard_bGFiZWwx_Q2FtMQ/index.m3u8',
        )
        self.assertEqual(
            stream['hls_url'],
            '/hazard/media/hazard_bGFiZWwx_Q2FtMQ/index.m3u8',
        )
        self.assertEqual(
            stream['overlay_playback_endpoint'],
            '/hazard/api/stream-playback',
        )
        self.assertEqual(stream['require_annotated_playback'], 'false')

    def test_stream_listing_can_enable_annotated_requirement_by_env(
        self,
    ) -> None:
        """Allow deployments to force annotated playback explicitly."""
        with patch.dict(
            'os.environ',
            {'MEDIA_REQUIRE_ANNOTATED_PLAYBACK': 'true'},
        ):
            stream = _build_stream_listing('label1', 'Cam1', 'Q2FtMQ')

        self.assertEqual(
            stream['playback_url'],
            '/hazard/media/'
            'hazard_bGFiZWwx_Q2FtMQ_annotated_emgtVFc/index.m3u8',
        )
        self.assertEqual(stream['require_annotated_playback'], 'true')

    def test_overlay_languages_returns_frontend_contract(self) -> None:
        """Expose canonical codes and translations for Flutter clients."""
        response = self.client.get('/api/overlay-languages')

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body['default_language'], 'zh-TW')
        self.assertEqual(
            body['playback_endpoint'],
            '/hazard/api/stream-playback',
        )
        self.assertIn('zh-TW', body['allowed_language_codes'])
        self.assertEqual(body['aliases']['zh_TW'], 'zh-TW')

        languages = {
            item['code']: item
            for item in body['languages']
        }
        self.assertEqual(languages['en']['notification_code'], 'en-GB')
        self.assertEqual(languages['ja']['notification_code'], 'ja-JP')
        self.assertIn(
            'warning_no_hardhat',
            languages['zh-TW']['notification_templates'],
        )
        self.assertIn(
            'no-hardhat',
            languages['zh-TW']['class_labels'],
        )

    def test_stream_playback_languages_alias_contract(self) -> None:
        """Expose language contract under the stream-playback namespace too."""
        response = self.client.get('/api/stream-playback/languages')

        self.assertEqual(response.status_code, 200)
        self.assertIn('supported_languages', response.json())

    @patch('examples.streaming_web.routers.get_user_and_sites')
    def test_stream_playback_clean_returns_ready_url(
        self,
        mock_get_user_and_sites: AsyncMock,
    ) -> None:
        """Overlay off returns the clean stream without creating demand."""
        mock_get_user_and_sites.return_value = (None, ['label1'], 'admin')
        stream_result = MagicMock()
        stream_result.scalars.return_value.all.return_value = ['Cam1']
        self.mock_db_session.execute.side_effect = None
        self.mock_db_session.execute.return_value = stream_result

        response = self.client.post(
            '/api/stream-playback',
            json={
                'label': 'label1',
                'stream_id': 'Q2FtMQ',
                'overlay': False,
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body['overlay'])
        self.assertEqual(body['status'], 'ready')
        self.assertEqual(
            body['playback_url'],
            '/hazard/media/hazard_bGFiZWwx_Q2FtMQ/index.m3u8',
        )
        self.fake_redis.set.assert_not_called()

    @patch('examples.streaming_web.routers.get_user_and_sites')
    def test_stream_playback_overlay_registers_shared_demand(
        self,
        mock_get_user_and_sites: AsyncMock,
    ) -> None:
        """Overlay requests create one shared demand key per language."""
        mock_get_user_and_sites.return_value = (None, ['label1'], 'admin')
        stream_result = MagicMock()
        stream_result.scalars.return_value.all.return_value = ['Cam1']
        self.mock_db_session.execute.side_effect = None
        self.mock_db_session.execute.return_value = stream_result

        async def empty_scan_iter(**_kwargs) -> None:
            """Support empty_scan_iter."""
            if False:
                yield b''

        self.fake_redis.scan_iter = empty_scan_iter
        self.fake_redis.exists = AsyncMock(return_value=0)

        response = self.client.post(
            '/api/stream-playback',
            json={
                'label': 'label1',
                'stream_id': 'Q2FtMQ',
                'overlay': True,
                'language': 'zh_TW',
            },
        )

        self.assertEqual(response.status_code, 202)
        body = response.json()
        self.assertTrue(body['overlay'])
        self.assertEqual(body['status'], 'starting')
        self.assertEqual(body['language'], 'zh-TW')
        self.assertEqual(
            body['playback_url'],
            '/hazard/media/'
            'hazard_bGFiZWwx_Q2FtMQ_annotated_emgtVFc/index.m3u8',
        )
        self.fake_redis.set.assert_awaited_once()

    @patch('examples.streaming_web.routers.get_user_and_sites')
    def test_stream_playback_overlay_ready_returns_200(
        self,
        mock_get_user_and_sites: AsyncMock,
    ) -> None:
        """Ready overlay paths return 200 so the player can load them."""
        mock_get_user_and_sites.return_value = (None, ['label1'], 'admin')
        stream_result = MagicMock()
        stream_result.scalars.return_value.all.return_value = ['Cam1']
        self.mock_db_session.execute.side_effect = None
        self.mock_db_session.execute.return_value = stream_result

        async def empty_scan_iter(**_kwargs) -> None:
            """Support empty_scan_iter."""
            if False:
                yield b''

        self.fake_redis.scan_iter = empty_scan_iter
        self.fake_redis.exists = AsyncMock(return_value=1)

        response = self.client.post(
            '/api/stream-playback',
            json={
                'label': 'label1',
                'key': 'Cam1',
                'overlay': 'backend',
                'language': 'en',
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['status'], 'ready')

    # -----------------------------
    # Test WebSocket endpoints (basic endpoint existence)
    # -----------------------------
    def test_websocket_endpoints_exist(self) -> None:
        """Test that WebSocket endpoints are properly defined in the router."""
        # Check that the router has the WebSocket routes
        routes = [route for route in self.app.routes]
        websocket_paths = [
            route.path for route in routes
            if hasattr(route, 'path') and route.path.startswith('/api/ws/')
        ]

        expected_paths = [
            '/api/ws/metadata-id/{label}/{stream_id}',
        ]

        for expected_path in expected_paths:
            self.assertIn(expected_path, websocket_paths)

    @patch(
        'examples.streaming_web.routers.get_public_ice_servers',
        return_value=[{'urls': ['turn:example.test:3478'], 'username': 'u'}],
    )
    def test_webrtc_ice_servers_requires_auth_and_returns_servers(
        self,
        mock_get_public_ice_servers: MagicMock,
    ) -> None:
        """Return ICE servers scoped to the authenticated viewer."""
        response = self.client.get('/api/webrtc/ice-servers')

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                'iceServers': [
                    {'urls': ['turn:example.test:3478'], 'username': 'u'},
                ],
            },
        )
        mock_get_public_ice_servers.assert_called_once_with('testuser')

    @patch('examples.streaming_web.routers.get_user_and_sites')
    def test_issue_media_session_sets_http_only_cookie(
        self,
        mock_get_user_and_sites: AsyncMock,
    ) -> None:
        """Issue a dedicated media session cookie from a valid access token."""
        mock_get_user_and_sites.return_value = (
            SimpleNamespace(status='active'),
            ['label1'],
            'admin',
        )

        response = self.client.post('/api/media-session')

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body['token_type'], 'hazard_media_session')
        self.assertEqual(body['expires_in'], 4500)
        self.assertNotIn('media_session_token', body)
        self.assertIn('httponly', response.headers['set-cookie'].lower())
        self.assertIn('hazard_media_session=', response.headers['set-cookie'])

    @patch('examples.streaming_web.routers.get_user_and_sites')
    def test_media_auth_allows_authorised_site(
        self,
        mock_get_user_and_sites: AsyncMock,
    ) -> None:
        """Allows Nginx auth_request for a user-owned media path."""
        mock_get_user_and_sites.return_value = (None, ['label1'], 'admin')
        token = jwt_access.create_access_token({'username': 'testuser'})

        response = self.client.get(
            '/api/media-auth',
            headers={
                'Authorization': f'Bearer {token}',
                'X-Original-URI': (
                    '/hazard/media/'
                    'hazard_bGFiZWwx_Q2FtMQ_annotated_emgtVFc/index.m3u8'
                ),
            },
        )

        self.assertEqual(response.status_code, 204)
        mock_get_user_and_sites.assert_awaited_once()

    @patch('examples.streaming_web.routers.get_user_and_sites')
    def test_media_auth_accepts_media_session_cookie(
        self,
        mock_get_user_and_sites: AsyncMock,
    ) -> None:
        """Allow HLS requests using the dedicated media session cookie."""
        mock_get_user_and_sites.return_value = (
            SimpleNamespace(status='active'),
            ['label1'],
            'admin',
        )
        with patch(
            'examples.streaming_web.routers.'
            'MEDIA_SESSION_COOKIE_SECURE',
            False,
        ):
            self.client.post('/api/media-session')

        response = self.client.get(
            '/api/media-auth',
            headers={
                'X-Original-URI': (
                    '/hazard/media/'
                    'hazard_bGFiZWwx_Q2FtMQ_annotated_ZW4/segment0.ts'
                ),
            },
        )

        self.assertEqual(response.status_code, 204)
        self.assertEqual(
            response.headers['x-media-auth-mode'], 'media_session',
        )

    @patch('examples.streaming_web.routers.get_user_and_sites')
    def test_media_auth_denies_wrong_site(
        self,
        mock_get_user_and_sites: AsyncMock,
    ) -> None:
        """Denies MediaMTX requests outside the viewer's site scope."""
        mock_get_user_and_sites.return_value = (None, ['other-site'], 'admin')
        token = jwt_access.create_access_token({'username': 'testuser'})

        response = self.client.get(
            '/api/media-auth',
            headers={
                'Authorization': f'Bearer {token}',
                'X-Original-URI': (
                    '/hazard/media/'
                    'hazard_bGFiZWwx_Q2FtMQ_annotated_emgtVFc/segment0.ts'
                ),
            },
        )

        self.assertEqual(response.status_code, 403)

    @patch('examples.streaming_web.routers.get_user_and_sites')
    def test_media_auth_accepts_cookie_token(
        self,
        mock_get_user_and_sites: AsyncMock,
    ) -> None:
        """Supports native HLS requests with cookies."""
        mock_get_user_and_sites.return_value = (None, ['label1'], 'admin')
        token = jwt_access.create_access_token({'username': 'testuser'})

        response = self.client.get(
            '/api/media-auth',
            cookies={'hazard_access_token': token},
            headers={
                'X-Original-URI': (
                    '/hazard/media/webrtc/hazard_bGFiZWwx_Q2FtMQ/whep'
                ),
            },
        )

        self.assertEqual(response.status_code, 204)

    @patch('examples.streaming_web.routers.get_user_and_sites')
    def test_media_auth_accepts_original_uri_query_token(
        self,
        mock_get_user_and_sites: AsyncMock,
    ) -> None:
        """Supports native HLS URLs that can only carry token in the URL."""
        mock_get_user_and_sites.return_value = (None, ['label1'], 'admin')
        token = jwt_access.create_access_token({'username': 'testuser'})

        response = self.client.get(
            '/api/media-auth',
            headers={
                'X-Original-URI': (
                    '/hazard/media/'
                    'hazard_bGFiZWwx_Q2FtMQ_annotated_ZW4/index.m3u8'
                    f'?token={token}'
                ),
            },
        )

        self.assertEqual(response.status_code, 204)

    @patch('examples.streaming_web.routers.get_user_and_sites')
    def test_media_auth_allows_recently_expired_token_grace(
        self,
        mock_get_user_and_sites: AsyncMock,
    ) -> None:
        """Allow signed access tokens inside the HLS grace window."""
        mock_get_user_and_sites.return_value = (None, ['label1'], 'admin')
        token = jwt_access.create_access_token(
            {'username': 'testuser'},
            expires_delta=timedelta(seconds=-60),
        )

        response = self.client.get(
            '/api/media-auth',
            headers={
                'Authorization': f'Bearer {token}',
                'X-Original-URI': (
                    '/hazard/media/'
                    'hazard_bGFiZWwx_Q2FtMQ_annotated_ZW4/segment0.ts'
                ),
            },
        )

        self.assertEqual(response.status_code, 204)

    def test_media_auth_rejects_stale_expired_token_with_reason(self) -> None:
        """Return expired_token when a signed token is beyond media grace."""
        token = jwt_access.create_access_token(
            {'username': 'testuser'},
            expires_delta=timedelta(seconds=-3600),
        )

        response = self.client.get(
            '/api/media-auth',
            headers={
                'Authorization': f'Bearer {token}',
                'X-Original-URI': (
                    '/hazard/media/'
                    'hazard_bGFiZWwx_Q2FtMQ_annotated_ZW4/segment0.ts'
                ),
            },
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()['detail'], 'expired_token')
        self.assertEqual(
            response.headers['x-media-auth-error'],
            'expired_token',
        )

    def test_media_auth_rejects_invalid_token_with_reason(self) -> None:
        """Return invalid_token for bad JWTs so clients can branch cleanly."""
        response = self.client.get(
            '/api/media-auth',
            headers={
                'Authorization': 'Bearer not-a-jwt',
                'X-Original-URI': (
                    '/hazard/media/'
                    'hazard_bGFiZWwx_Q2FtMQ_annotated_ZW4/segment0.ts'
                ),
            },
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()['detail'], 'invalid_token')
        self.assertEqual(
            response.headers['www-authenticate'],
            'Bearer error="invalid_token"',
        )

    def test_media_auth_rejects_invalid_session_reason(self) -> None:
        """Return media-session-specific reason for bad session cookies."""
        response = self.client.get(
            '/api/media-auth',
            cookies={'hazard_media_session': 'not-a-session'},
            headers={
                'X-Original-URI': (
                    '/hazard/media/'
                    'hazard_bGFiZWwx_Q2FtMQ_annotated_ZW4/segment0.ts'
                ),
            },
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()['detail'], 'invalid_media_session')

    @patch('examples.streaming_web.routers.get_user_and_sites')
    def test_media_auth_rejects_inactive_user(
        self,
        mock_get_user_and_sites: AsyncMock,
    ) -> None:
        """Grace-period tokens still require an active user account."""
        mock_get_user_and_sites.return_value = (
            SimpleNamespace(status='inactive'),
            ['label1'],
            'admin',
        )
        token = jwt_access.create_access_token({'username': 'testuser'})

        response = self.client.get(
            '/api/media-auth',
            headers={
                'Authorization': f'Bearer {token}',
                'X-Original-URI': (
                    '/hazard/media/'
                    'hazard_bGFiZWwx_Q2FtMQ_annotated_ZW4/segment0.ts'
                ),
            },
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()['detail'], 'inactive_user')

    def test_language_helpers_cover_alias_and_defaults(self) -> None:
        """Language helper branches stay stable for clients."""
        with patch.dict(
            'os.environ',
            {
                'MEDIA_OVERLAY_ALLOWED_LANGUAGES': 'fr',
                'MEDIA_DEFAULT_OVERLAY_LANGUAGE': 'zh-TW',
            },
        ):
            self.assertEqual(routers._allowed_overlay_languages(), ('fr',))
            self.assertEqual(routers._default_overlay_language(), 'fr')

        original = routers.OVERLAY_TO_NOTIFICATION_LANGUAGE.get('fr')
        routers.OVERLAY_TO_NOTIFICATION_LANGUAGE['fr'] = 'missing'
        try:
            self.assertEqual(
                routers._notification_language_code('fr'),
                'en-GB',
            )
        finally:
            if original is None:
                routers.OVERLAY_TO_NOTIFICATION_LANGUAGE.pop('fr', None)
            else:
                routers.OVERLAY_TO_NOTIFICATION_LANGUAGE['fr'] = original

        details = routers.OVERLAY_LANGUAGE_DETAILS['fr']
        routers.OVERLAY_LANGUAGE_DETAILS['fr'] = {
            'name': 'French',
            'native_name': 'Français',
            'aliases': 'fr-FR',
        }
        try:
            self.assertEqual(routers._language_alias_map()['fr'], 'fr')
        finally:
            routers.OVERLAY_LANGUAGE_DETAILS['fr'] = details

    @patch('examples.streaming_web.routers.get_user_and_sites')
    async def test_authorise_label_access_rejects_missing_or_denied_user(
        self,
        mock_get_user_and_sites: AsyncMock,
    ) -> None:
        """Exercise this test."""
        credentials = cast(
            JwtAuthorizationCredentials,
            SimpleNamespace(subject={}),
        )
        with self.assertRaises(HTTPException) as invalid_ctx:
            await routers._authorise_label_access(
                credentials,
                self.mock_db_session,
                'label1',
            )
        self.assertEqual(invalid_ctx.exception.status_code, 401)

        credentials = cast(
            JwtAuthorizationCredentials,
            SimpleNamespace(subject={'username': 'alice'}),
        )
        mock_get_user_and_sites.return_value = (None, ['other'], 'user')
        with self.assertRaises(HTTPException) as denied_ctx:
            await routers._authorise_label_access(
                credentials,
                self.mock_db_session,
                'label1',
            )
        self.assertEqual(denied_ctx.exception.status_code, 403)

    def test_media_token_extractors_cover_headers_queries_and_cookies(
        self,
    ) -> None:
        """Exercise this test."""
        request = cast(
            Request,
            SimpleNamespace(
                headers={'authorization': 'Bearer access-token'},
                query_params={},
                cookies={},
            ),
        )
        self.assertEqual(
            routers._extract_media_auth_token(request),
            'access-token',
        )

        request = cast(
            Request,
            SimpleNamespace(
                headers={
                    'x-forwarded-uri': '/hazard/media/path?token=from-uri',
                },
                query_params={},
                cookies={},
            ),
        )
        self.assertEqual(
            routers._extract_media_auth_token(request),
            'from-uri',
        )

        request = cast(
            Request,
            SimpleNamespace(
                headers={},
                query_params={'media_session': 'from-query'},
                cookies={},
            ),
        )
        self.assertEqual(
            routers._extract_media_session_token(request),
            'from-query',
        )

        request = cast(
            Request,
            SimpleNamespace(
                headers={
                    'x-original-uri': (
                        '/hazard/media/path?media_session=from-uri'
                    ),
                },
                query_params={},
                cookies={},
            ),
        )
        self.assertEqual(
            routers._extract_media_session_token(request),
            'from-uri',
        )

    def test_decode_media_session_token_rejects_bad_payloads(self) -> None:
        """Exercise this test."""
        expired_token = jwt.encode(
            {
                'typ': 'hazard_media_session',
                'sub': 'alice',
                'exp': datetime.now(timezone.utc) - timedelta(minutes=1),
            },
            routers.settings.authjwt_secret_key,
            algorithm=routers.settings.ALGORITHM,
        )
        with self.assertRaises(HTTPException) as expired:
            routers._decode_media_session_token(expired_token)
        self.assertEqual(expired.exception.detail, 'expired_media_session')

        wrong_type_token = jwt.encode(
            {
                'typ': 'other',
                'sub': 'alice',
                'exp': datetime.now(timezone.utc) + timedelta(minutes=1),
            },
            routers.settings.authjwt_secret_key,
            algorithm=routers.settings.ALGORITHM,
        )
        with self.assertRaises(HTTPException) as wrong_type:
            routers._decode_media_session_token(wrong_type_token)
        self.assertEqual(wrong_type.exception.detail, 'invalid_media_session')

        no_subject_token = jwt.encode(
            {
                'typ': 'hazard_media_session',
                'sub': '',
                'exp': datetime.now(timezone.utc) + timedelta(minutes=1),
            },
            routers.settings.authjwt_secret_key,
            algorithm=routers.settings.ALGORITHM,
        )
        with self.assertRaises(HTTPException) as no_subject:
            routers._decode_media_session_token(no_subject_token)
        self.assertEqual(no_subject.exception.detail, 'invalid_media_session')

    def test_decode_media_auth_token_rejects_invalid_expired_payload(
        self,
    ) -> None:
        """Exercise this test."""
        with (
            patch.object(
                routers.jwt_access,
                'decode_token',
                side_effect=[
                    routers.ExpiredSignatureError(),
                    {'subject': {'username': 'alice'}, 'exp': 'bad'},
                ],
            ),
            self.assertRaises(HTTPException) as ctx,
        ):
            routers._decode_media_auth_token('expired')

        self.assertEqual(ctx.exception.detail, 'invalid_token')

        with (
            patch.object(
                routers.jwt_access,
                'decode_token',
                side_effect=[
                    routers.ExpiredSignatureError(),
                    routers.InvalidTokenError(),
                ],
            ),
            self.assertRaises(HTTPException) as invalid_ctx,
        ):
            routers._decode_media_auth_token('expired')

        self.assertEqual(invalid_ctx.exception.detail, 'invalid_token')

    def test_resolve_media_auth_identity_accepts_sub_and_missing_token(
        self,
    ) -> None:
        """Exercise this test."""
        request = cast(
            Request,
            SimpleNamespace(
                headers={},
                query_params={'token': 'access'},
                cookies={},
            ),
        )
        with patch.object(
            routers.jwt_access,
            'decode_token',
            return_value={'sub': 'alice'},
        ):
            self.assertEqual(
                routers._resolve_media_auth_identity(request),
                ('alice', 'access'),
            )

        request = cast(
            Request,
            SimpleNamespace(
                headers={},
                query_params={'token': 'access'},
                cookies={},
            ),
        )
        with patch.object(
            routers.jwt_access,
            'decode_token',
            return_value={'subject': {}, 'sub': ''},
        ):
            with self.assertRaises(HTTPException) as invalid:
                routers._resolve_media_auth_identity(request)
        self.assertEqual(invalid.exception.detail, 'invalid_token')

        request = cast(
            Request,
            SimpleNamespace(headers={}, query_params={}, cookies={}),
        )
        with self.assertRaises(HTTPException) as ctx:
            routers._resolve_media_auth_identity(request)
        self.assertEqual(ctx.exception.detail, 'missing_token')

    def test_media_path_helpers_cover_invalid_and_webrtc_paths(self) -> None:
        """Exercise this test."""
        self.assertEqual(
            routers._extract_media_path_from_uri('/hazard/live'),
            '',
        )
        self.assertEqual(routers._extract_media_path_from_uri('/no/media'), '')
        self.assertEqual(
            routers._extract_media_path_from_uri(
                '/hazard/media/webrtc/hazard_site_cam/whep',
            ),
            'hazard_site_cam',
        )
        self.assertEqual(routers._decode_redis_key(b'abc'), 'abc')
        self.assertEqual(routers._decode_redis_key(123), '123')

    async def test_active_overlay_languages_skips_bad_keys(self) -> None:
        """Exercise this test."""
        async def scan_iter(**_kwargs) -> None:
            """Support scan_iter."""
            yield b'media_overlay_demand:hazard_site_cam:not-base64'
            yield b'media_overlay_demand:hazard_site_cam:emgtVFc'

        rds = MagicMock()
        rds.scan_iter = scan_iter

        languages = await routers._active_overlay_languages(
            rds,
            'hazard_site_cam',
        )

        self.assertEqual(languages, {'zh-TW'})

    async def test_touch_overlay_demand_from_media_path_ignores_bad_inputs(
        self,
    ) -> None:
        """Exercise this test."""
        rds = AsyncMock()

        await routers._touch_overlay_demand_from_media_path(rds, 'clean-path')
        rds.set.assert_not_called()

        with patch.dict(
            'os.environ',
            {'MEDIA_OVERLAY_ALLOWED_LANGUAGES': 'en'},
        ):
            await routers._touch_overlay_demand_from_media_path(
                rds,
                'hazard_site_cam_annotated_emgtVFc',
            )
        rds.set.assert_not_called()

    async def test_touch_overlay_demand_from_media_path_logs_set_error(
        self,
    ) -> None:
        """Exercise this test."""
        rds = AsyncMock()
        rds.set.side_effect = RuntimeError('redis down')

        await routers._touch_overlay_demand_from_media_path(
            rds,
            'hazard_site_cam_annotated_emgtVFc',
        )

        rds.set.assert_awaited_once()

    def test_normalise_stream_id_handles_decoders(self) -> None:
        """Exercise this test."""
        encoded = routers.Utils.encode('Cam1')
        self.assertEqual(routers._normalise_stream_id(encoded), 'Cam1')
        self.assertEqual(routers._normalise_stream_id('Q2FtMQ'), 'Cam1')
        self.assertEqual(
            routers._normalise_stream_id('plain-name'),
            'plain-name',
        )

    async def test_resolve_configured_stream_name_rejects_missing_and_unknown(
        self,
    ) -> None:
        """Exercise this test."""
        with self.assertRaises(HTTPException) as missing:
            await routers._resolve_configured_stream_name(
                self.mock_db_session,
                'label1',
                None,
                None,
            )
        self.assertEqual(missing.exception.status_code, 422)

        result = MagicMock()
        result.scalars.return_value.all.return_value = ['Cam1']
        self.mock_db_session.execute = AsyncMock(return_value=result)
        with self.assertRaises(HTTPException) as unknown:
            await routers._resolve_configured_stream_name(
                self.mock_db_session,
                'label1',
                'Q2FtMg',
                None,
            )
        self.assertEqual(unknown.exception.status_code, 404)

    def test_overlay_languages_rejects_missing_subject(self) -> None:
        """Exercise this test."""
        self.app.dependency_overrides[jwt_access] = lambda: SimpleNamespace(
            subject={},
        )

        response = self.client.get('/api/overlay-languages')

        self.assertEqual(response.status_code, 401)

    @patch('examples.streaming_web.routers.get_user_and_sites')
    def test_issue_media_session_rejects_invalid_or_inactive_user(
        self,
        mock_get_user_and_sites: AsyncMock,
    ) -> None:
        """Exercise this test."""
        self.app.dependency_overrides[jwt_access] = lambda: SimpleNamespace(
            subject={},
        )
        response = self.client.post('/api/media-session')
        self.assertEqual(response.status_code, 401)

        self.app.dependency_overrides[jwt_access] = lambda: SimpleNamespace(
            subject={'username': 'testuser'},
        )
        mock_get_user_and_sites.return_value = (
            SimpleNamespace(status='inactive'),
            ['label1'],
            'admin',
        )
        response = self.client.post('/api/media-session')
        self.assertEqual(response.status_code, 401)

    @patch('examples.streaming_web.routers.get_user_and_sites')
    def test_issue_media_session_can_expose_token(
        self,
        mock_get_user_and_sites: AsyncMock,
    ) -> None:
        """Exercise this test."""
        mock_get_user_and_sites.return_value = (
            SimpleNamespace(status='active'),
            ['label1'],
            'admin',
        )
        with patch(
            'examples.streaming_web.routers.'
            'MEDIA_SESSION_EXPOSE_TOKEN',
            True,
        ):
            response = self.client.post('/api/media-session')

        self.assertEqual(response.status_code, 200)
        self.assertIn('media_session_token', response.json())

    @patch('examples.streaming_web.routers.get_user_and_sites')
    def test_media_auth_rejects_invalid_media_path(
        self,
        mock_get_user_and_sites: AsyncMock,
    ) -> None:
        """Exercise this test."""
        mock_get_user_and_sites.return_value = (None, ['label1'], 'admin')
        token = jwt_access.create_access_token({'username': 'testuser'})

        response = self.client.get(
            '/api/media-auth',
            headers={
                'Authorization': f'Bearer {token}',
                'X-Original-URI': '/hazard/media/not-hazard/index.m3u8',
            },
        )

        self.assertEqual(response.status_code, 403)

    @patch('examples.streaming_web.routers.get_user_and_sites')
    def test_stream_playback_rejects_unsupported_language(
        self,
        mock_get_user_and_sites: AsyncMock,
    ) -> None:
        """Exercise this test."""
        mock_get_user_and_sites.return_value = (None, ['label1'], 'admin')
        stream_result = MagicMock()
        stream_result.scalars.return_value.all.return_value = ['Cam1']
        self.mock_db_session.execute.side_effect = None
        self.mock_db_session.execute.return_value = stream_result

        with patch.dict(
            'os.environ',
            {'MEDIA_OVERLAY_ALLOWED_LANGUAGES': 'en'},
        ):
            response = self.client.post(
                '/api/stream-playback',
                json={
                    'label': 'label1',
                    'stream_id': 'Q2FtMQ',
                    'overlay': True,
                    'language': 'zh-TW',
                },
            )

        self.assertEqual(response.status_code, 422)

    @patch('examples.streaming_web.routers.get_user_and_sites')
    def test_stream_playback_rejects_overlay_language_limit(
        self,
        mock_get_user_and_sites: AsyncMock,
    ) -> None:
        """Exercise this test."""
        mock_get_user_and_sites.return_value = (None, ['label1'], 'admin')
        stream_result = MagicMock()
        stream_result.scalars.return_value.all.return_value = ['Cam1']
        self.mock_db_session.execute.side_effect = None
        self.mock_db_session.execute.return_value = stream_result

        with patch(
            'examples.streaming_web.routers._active_overlay_languages',
            new=AsyncMock(return_value={'zh-TW', 'ja', 'vi', 'id', 'fr'}),
        ):
            response = self.client.post(
                '/api/stream-playback',
                json={
                    'label': 'label1',
                    'stream_id': 'Q2FtMQ',
                    'overlay': True,
                    'language': 'en',
                },
            )

        self.assertEqual(response.status_code, 429)

    async def test_configured_streams_returns_empty_on_db_error(self) -> None:
        """Exercise this test."""
        self.mock_db_session.execute = AsyncMock(
            side_effect=RuntimeError('db down'),
        )

        self.assertEqual(
            await routers._get_configured_media_streams(
                self.mock_db_session,
                'label1',
            ),
            [],
        )

    async def test_configured_streams_returns_stream_listings(self) -> None:
        """Exercise this test."""
        result = MagicMock()
        result.scalars.return_value.all.return_value = ['Cam1']
        self.mock_db_session.execute = AsyncMock(return_value=result)

        streams = await routers._get_configured_media_streams(
            self.mock_db_session,
            'label1',
        )

        self.assertEqual(len(streams), 1)
        self.assertEqual(streams[0]['key'], 'Cam1')

    def test_webrtc_ice_servers_rejects_missing_subject(self) -> None:
        """Exercise this test."""
        self.app.dependency_overrides[jwt_access] = lambda: SimpleNamespace(
            subject={},
        )

        response = self.client.get('/api/webrtc/ice-servers')

        self.assertEqual(response.status_code, 401)

    async def test_metadata_stream_id_returns_sse_response(self) -> None:
        """Exercise this test."""
        request = MagicMock()
        response = await routers.metadata_stream_id(
            request,
            'label1',
            'Q2FtMQ',
            self.fake_redis,
        )

        self.assertEqual(
            response.media_type,
            'text/event-stream',
        )

    async def test_websocket_metadata_stream_id_delegates_to_handler(
        self,
    ) -> None:
        """Exercise this test."""
        with patch(
            'examples.streaming_web.routers.'
            'handle_metadata_stream_id_ws',
            new=AsyncMock(),
        ) as handler:
            await routers.websocket_metadata_stream_id(
                websocket=MagicMock(),
                label='label1',
                stream_id='Q2FtMQ',
                rds=self.fake_redis,
                db=self.mock_db_session,
            )

        handler.assert_awaited_once()


if __name__ == '__main__':
    unittest.main()


'''
pytest \
    --cov=examples.streaming_web.routers \
    --cov-report=term-missing \
    tests/examples/streaming_web/routers_test_new.py
'''
