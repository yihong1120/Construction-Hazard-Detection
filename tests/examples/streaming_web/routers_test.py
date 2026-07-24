from __future__ import annotations

import json
import unittest
from datetime import datetime
from datetime import timezone
from types import SimpleNamespace
from typing import cast
from unittest.mock import ANY
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi.testclient import TestClient
from starlette.routing import Match

from examples.auth.database import get_db
from examples.auth.jwt_config import jwt_access
from examples.auth.jwt_config import JwtAuthorizationCredentials
from examples.auth.redis_pool import get_redis_pool
from examples.auth.redis_pool import get_redis_pool_ws
from examples.streaming_web import routers
from examples.streaming_web.routers import _build_stream_listing
from examples.streaming_web.routers import router
from examples.streaming_web.schemas import StreamPlaybackBatchRequest
from examples.streaming_web.schemas import StreamPlaybackRequest


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

    @patch('examples.streaming_web.routers.get_user_and_sites')
    def test_get_streams_returns_empty_without_configured_streams(
        self,
        mock_get_user_and_sites: AsyncMock,
    ) -> None:
        """Return only DB-configured streams without scanning Redis keys."""
        mock_get_user_and_sites.return_value = (None, ['label1'], 'admin')
        stream_result = MagicMock()
        stream_result.scalars.return_value.all.return_value = []
        self.mock_db_session.execute.side_effect = None
        self.mock_db_session.execute.return_value = stream_result

        response = self.client.get('/api/streams/label1')

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'streams': []})

    @patch('examples.streaming_web.routers.get_user_and_sites')
    def test_get_streams_returns_session_playback_urls(
        self,
        mock_get_user_and_sites: AsyncMock,
    ) -> None:
        """Stream listings return stable session URLs instead of direct HLS."""
        mock_get_user_and_sites.return_value = (None, ['label1'], 'admin')
        stream_result = MagicMock()
        stream_result.scalars.return_value.all.return_value = ['Cam1']
        self.mock_db_session.execute.side_effect = None
        self.mock_db_session.execute.return_value = stream_result

        response = self.client.get('/api/streams/label1')

        self.assertEqual(response.status_code, 200)
        stream = response.json()['streams'][0]
        self.assertEqual(stream['profile'], 'clean')
        self.assertIn(
            '/hazard/api/stream-playback/sessions/',
            stream['playback_url'],
        )
        self.assertEqual(
            stream['media_hls_url'],
            '/hazard/media/hazard_bGFiZWwx_Q2FtMQ/index.m3u8',
        )

    def test_stream_listing_uses_clean_hls_by_default(self) -> None:
        """Use clean HLS by default; overlay playback is negotiated later."""
        stream = _build_stream_listing('label1', 'Cam1', 'Q2FtMQ')

        self.assertEqual(
            stream['playback_url'],
            '/hazard/media/hazard_bGFiZWwx_Q2FtMQ/index.m3u8',
        )
        self.assertEqual(
            stream['media_hls_url'],
            '/hazard/media/hazard_bGFiZWwx_Q2FtMQ/index.m3u8',
        )
        self.assertEqual(stream['profile'], 'clean')

    def test_visible_stream_query_excludes_disabled_recognition_streams(
        self,
    ) -> None:
        """Only recognition-enabled streams appear in walls."""
        statement = routers._visible_stream_names_query('label1')
        where_clause = str(statement.whereclause)

        self.assertIn(
            'stream_configs.recognition_enabled IS true',
            where_clause,
        )

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
        """Overlay off returns the clean stream and creates clean demand."""
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
                'profile': 'clean',
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body['status'], 'ready')
        self.assertEqual(body['state'], 'ready')
        self.assertEqual(body['profile'], 'clean')
        self.assertIn(
            '/hazard/api/stream-playback/sessions/',
            body['playback_url'],
        )
        self.assertEqual(
            body['media_hls_url'],
            '/hazard/media/hazard_bGFiZWwx_Q2FtMQ/index.m3u8',
        )
        self.fake_redis.set.assert_any_await(
            'media_clean_demand:hazard_bGFiZWwx_Q2FtMQ',
            ANY,
            ex=90,
        )

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
                'profile': 'overlay',
                'language': 'zh-TW',
            },
        )

        self.assertEqual(response.status_code, 202)
        body = response.json()
        self.assertEqual(body['status'], 'starting')
        self.assertEqual(body['state'], 'starting')
        self.assertEqual(body['profile'], 'overlay')
        self.assertEqual(body['language'], 'zh-TW')
        self.assertIn(
            '/hazard/api/stream-playback/sessions/',
            body['playback_url'],
        )
        self.assertEqual(
            body['media_hls_url'],
            '/hazard/media/'
            'hazard_bGFiZWwx_Q2FtMQ_annotated_emgtVFc/index.m3u8',
        )
        self.assertFalse(body['overlay_ready'])
        self.assertGreaterEqual(self.fake_redis.set.await_count, 2)

    @patch('examples.streaming_web.routers.get_user_and_sites')
    def test_stream_playback_preview_overlay_uses_isolated_rendition(
        self,
        mock_get_user_and_sites: AsyncMock,
    ) -> None:
        """A wall rendition never aliases the detail annotated path."""
        mock_get_user_and_sites.return_value = (None, ['label1'], 'admin')
        stream_result = MagicMock()
        stream_result.scalars.return_value.all.return_value = ['Cam1']
        self.mock_db_session.execute.side_effect = None
        self.mock_db_session.execute.return_value = stream_result

        async def empty_scan_iter(**_kwargs) -> None:
            if False:
                yield b''

        self.fake_redis.scan_iter = empty_scan_iter
        self.fake_redis.exists = AsyncMock(return_value=0)
        response = self.client.post(
            '/api/stream-playback',
            json={
                'label': 'label1',
                'key': 'Cam1',
                'profile': 'overlay',
                'rendition': 'preview',
                'language': 'zh-TW',
            },
        )

        self.assertEqual(response.status_code, 202)
        body = response.json()
        self.assertEqual(body['rendition'], 'preview')
        self.assertEqual(
            body['media_path'],
            'hazard_bGFiZWwx_Q2FtMQ_preview_annotated_emgtVFc',
        )
        self.fake_redis.set.assert_any_await(
            'media_overlay_demand:hazard_bGFiZWwx_Q2FtMQ_preview:emgtVFc',
            ANY,
            ex=90,
        )

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
                'profile': 'overlay',
                'language': 'en',
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body['status'], 'ready')
        self.assertEqual(body['state'], 'ready')
        self.assertEqual(body['profile'], 'overlay')
        self.assertTrue(body['overlay_ready'])
        self.assertIn(
            '/hazard/api/stream-playback/sessions/',
            body['playback_url'],
        )
        self.assertEqual(
            body['media_hls_url'],
            '/hazard/media/'
            'hazard_bGFiZWwx_Q2FtMQ_annotated_ZW4/index.m3u8',
        )

    def test_stream_playback_session_playlist_rewrites_fragment_auth(
        self,
    ) -> None:
        """Stable session playlists keep mt on every HLS fragment URL."""
        session = {
            'session_id': 'session-1',
            'username': 'testuser',
            'label': 'label1',
            'stream_name': 'Cam1',
            'stream_id': 'Q2FtMQ',
            'profile': 'clean',
            'language': None,
            'base_media_path': 'hazard_bGFiZWwx_Q2FtMQ',
        }
        self.fake_redis.get = AsyncMock(return_value=json.dumps(session))
        self.fake_redis.expire = AsyncMock()

        with patch(
            'examples.streaming_web.routers._fetch_internal_hls_playlist',
            new=AsyncMock(
                return_value=('#EXTM3U\n#EXTINF:2,\nseg0.ts\n', None),
            ),
        ) as fetch_playlist:
            response = self.client.get(
                '/api/stream-playback/sessions/session-1/index.m3u8'
                '?mt=opaque-token&_HLS_msn=3',
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn(
            '/hazard/media/hazard_bGFiZWwx_Q2FtMQ/seg0.ts?mt=opaque-token',
            response.text,
        )
        fetch_playlist.assert_awaited_once_with(
            'hazard_bGFiZWwx_Q2FtMQ',
            media_query='_HLS_msn=3',
        )
        self.fake_redis.expire.assert_awaited_once()

    def test_stream_playback_session_playlist_forwards_hls_session_cookie(
        self,
    ) -> None:
        """The client needs MediaMTX's HLS session for child playlists."""
        session = {
            'session_id': 'session-1',
            'username': 'testuser',
            'label': 'label1',
            'stream_name': 'Cam1',
            'stream_id': 'Q2FtMQ',
            'profile': 'clean',
            'language': None,
            'base_media_path': 'hazard_bGFiZWwx_Q2FtMQ',
        }
        self.fake_redis.get = AsyncMock(return_value=json.dumps(session))
        self.fake_redis.expire = AsyncMock()

        with patch(
            'examples.streaming_web.routers._fetch_internal_hls_playlist',
            new=AsyncMock(
                return_value=(
                    '#EXTM3U\n#EXTINF:2,\nseg0.ts\n',
                    routers._media_hls_session_cookie(
                        'hazard_bGFiZWwx_Q2FtMQ',
                        'session-123',
                    ),
                ),
            ),
        ):
            response = self.client.get(
                '/api/stream-playback/sessions/session-1/index.m3u8'
                '?mt=opaque-token',
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn('hlsSession=session-123', response.headers['set-cookie'])
        self.assertIn(
            'Path=/hazard/media/hazard_bGFiZWwx_Q2FtMQ/',
            response.headers['set-cookie'],
        )
        self.assertIn('Secure', response.headers['set-cookie'])
        self.assertIn('HttpOnly', response.headers['set-cookie'])
        self.assertIn('SameSite=None', response.headers['set-cookie'])
        self.assertIn('Partitioned', response.headers['set-cookie'])

    async def test_internal_hls_playlist_follows_mediamtx_cookie_redirect(
        self,
    ) -> None:
        """MediaMTX's cookie-check redirect must be followed internally."""
        upstream_response = MagicMock()
        upstream_response.status_code = 200
        upstream_response.text = '#EXTM3U\n#EXTINF:2,\nseg0.ts\n'
        upstream_response.cookies = {'hlsSession': 'session-456'}
        upstream_client = MagicMock()
        upstream_client.get = AsyncMock(return_value=upstream_response)
        upstream_context = MagicMock()
        upstream_context.__aenter__ = AsyncMock(
            return_value=upstream_client,
        )
        upstream_context.__aexit__ = AsyncMock(return_value=False)

        with patch(
            'examples.streaming_web.routers.httpx.AsyncClient',
            return_value=upstream_context,
        ) as async_client:
            playlist = await routers._fetch_internal_hls_playlist(
                'hazard_bGFiZWwx_Q2FtMQ',
                media_query='_HLS_msn=3',
            )

        self.assertEqual(
            playlist,
            (
                upstream_response.text,
                'hlsSession=session-456; '
                'Path=/hazard/media/hazard_bGFiZWwx_Q2FtMQ/; '
                'Secure; HttpOnly; SameSite=None; Partitioned',
            ),
        )
        async_client.assert_called_once_with(
            timeout=routers.MEDIA_INTERNAL_HLS_TIMEOUT_SECONDS,
            follow_redirects=True,
        )
        upstream_client.get.assert_awaited_once_with(
            f'{routers.MEDIA_INTERNAL_HLS_BASE_URL}/'
            'hazard_bGFiZWwx_Q2FtMQ/index.m3u8?_HLS_msn=3',
        )

    async def test_internal_hls_playlist_rejects_empty_response(
        self,
    ) -> None:
        """An empty body is not a usable HLS playlist."""
        upstream_response = MagicMock()
        upstream_response.status_code = 200
        upstream_response.text = ''
        upstream_client = MagicMock()
        upstream_client.get = AsyncMock(return_value=upstream_response)
        upstream_context = MagicMock()
        upstream_context.__aenter__ = AsyncMock(
            return_value=upstream_client,
        )
        upstream_context.__aexit__ = AsyncMock(return_value=False)

        with patch(
            'examples.streaming_web.routers.httpx.AsyncClient',
            return_value=upstream_context,
        ), self.assertRaises(HTTPException) as context:
            await routers._fetch_internal_hls_playlist(
                'hazard_bGFiZWwx_Q2FtMQ',
                media_query='',
            )

        self.assertEqual(context.exception.status_code, 503)
        self.assertEqual(context.exception.detail, 'media_playlist_not_ready')

    def test_stream_playback_session_playlist_waits_for_fresh_session(
        self,
    ) -> None:
        """Fresh on-demand sessions wait briefly before fetching HLS."""
        session = {
            'session_id': 'session-1',
            'username': 'testuser',
            'label': 'label1',
            'stream_name': 'Cam1',
            'stream_id': 'Q2FtMQ',
            'profile': 'clean',
            'language': None,
            'base_media_path': 'hazard_bGFiZWwx_Q2FtMQ',
            'created_at': datetime.now(timezone.utc).isoformat(),
        }
        self.fake_redis.get = AsyncMock(return_value=json.dumps(session))
        self.fake_redis.expire = AsyncMock()

        with (
            patch(
                'examples.streaming_web.routers.'
                'STREAM_PLAYBACK_STARTUP_WAIT_SECONDS',
                0.25,
            ),
            patch(
                'examples.streaming_web.routers.asyncio.sleep',
                new_callable=AsyncMock,
            ) as sleep,
            patch(
                'examples.streaming_web.routers.'
                '_fetch_internal_hls_playlist',
                new=AsyncMock(return_value=('#EXTM3U\nseg0.ts\n', None)),
            ),
        ):
            response = self.client.get(
                '/api/stream-playback/sessions/session-1/index.m3u8'
                '?mt=opaque-token',
            )

        self.assertEqual(response.status_code, 200)
        sleep.assert_awaited_once()

    @patch('examples.streaming_web.routers.get_user_and_sites')
    def test_stream_playback_batch_creates_site_sessions(
        self,
        mock_get_user_and_sites: AsyncMock,
    ) -> None:
        """Batch endpoint returns stable playback URLs for a site overview."""
        mock_get_user_and_sites.return_value = (None, ['label1'], 'admin')
        stream_result = MagicMock()
        stream_result.scalars.return_value.all.return_value = ['Cam1', 'Cam2']
        self.mock_db_session.execute.side_effect = None
        self.mock_db_session.execute.return_value = stream_result

        async def empty_scan_iter(**_kwargs) -> None:
            if False:
                yield b''

        self.fake_redis.scan_iter = empty_scan_iter
        self.fake_redis.exists = AsyncMock(return_value=0)

        response = self.client.post(
            '/api/stream-playback/batch',
            json={
                'label': 'label1',
                'profile': 'overlay',
                'language': 'zh-TW',
            },
        )

        self.assertEqual(response.status_code, 202)
        body = response.json()
        self.assertEqual(body['count'], 2)
        self.assertEqual(body['max_streams'], 24)
        self.assertEqual(
            body['stream_playback_endpoint'],
            '/hazard/api/stream-playback',
        )
        self.assertEqual(
            body['batch_endpoint'],
            '/hazard/api/stream-playback/batch',
        )
        self.assertEqual(len(body['items']), 2)
        for item in body['items']:
            self.assertEqual(item['profile'], 'overlay')
            self.assertIn(
                '/hazard/api/stream-playback/sessions/',
                item['playback_url'],
            )

    def test_stream_playback_batch_rejects_more_than_24_site_streams(
        self,
    ) -> None:
        """Site overview rejects locations with more than 24 streams."""
        stream_result = MagicMock()
        stream_result.scalars.return_value.all.return_value = [
            f'Cam{index}' for index in range(25)
        ]
        self.mock_db_session.execute.side_effect = None
        self.mock_db_session.execute.return_value = stream_result

        response = self.client.post(
            '/api/stream-playback/batch',
            json={
                'label': 'label1',
                'profile': 'overlay',
                'language': 'zh-TW',
            },
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(
            response.json()['detail'],
            {
                'code': 'stream_batch_limit_exceeded',
                'count': 25,
                'max_streams': 24,
            },
        )

    def test_stream_playback_batch_rejects_more_than_24_explicit_streams(
        self,
    ) -> None:
        """Explicit batch requests share the same 24-stream wall limit."""
        response = self.client.post(
            '/api/stream-playback/batch',
            json={
                'streams': [
                    {'label': 'label1', 'key': f'Cam{index}'}
                    for index in range(25)
                ],
                'profile': 'overlay',
                'language': 'zh-TW',
            },
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(
            response.json()['detail'],
            {
                'code': 'stream_batch_limit_exceeded',
                'count': 25,
                'max_streams': 24,
            },
        )

    def test_batch_explicit_streams_inherit_profile_when_omitted(self) -> None:
        """Explicit batch streams inherit profile unless they set their own."""
        batch = StreamPlaybackBatchRequest(
            label='label1',
            profile='overlay',
            language='zh-TW',
            transport='hls',
        )

        inherited = routers._inherit_batch_playback_defaults(
            StreamPlaybackRequest(key='Cam1'),
            batch,
        )
        self.assertEqual(inherited.profile, 'overlay')
        self.assertEqual(inherited.language, 'zh-TW')

        explicit_clean = routers._inherit_batch_playback_defaults(
            StreamPlaybackRequest(
                key='Cam2',
                profile='clean',
                language=None,
            ),
            batch,
        )
        self.assertEqual(explicit_clean.profile, 'clean')
        self.assertIsNone(explicit_clean.language)

    @patch('examples.streaming_web.routers.get_user_and_sites')
    def test_stream_playback_release_accepts_session_id(
        self,
        mock_get_user_and_sites: AsyncMock,
    ) -> None:
        """Release a playback session without label or stream fields."""
        mock_get_user_and_sites.return_value = (None, ['label1'], 'admin')
        session = {
            'session_id': 'session-1',
            'username': 'testuser',
            'profile': 'overlay',
            'language': 'zh-TW',
            'base_media_path': 'hazard_bGFiZWwx_Q2FtMQ',
        }
        self.fake_redis.get = AsyncMock(return_value=json.dumps(session))

        async def empty_scan_iter(**_kwargs) -> None:
            if False:
                yield b''

        self.fake_redis.scan_iter = empty_scan_iter

        response = self.client.post(
            '/api/stream-playback/release',
            json={'session_id': 'session-1'},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body['status'], 'released')
        self.assertEqual(body['session_id'], 'session-1')
        self.fake_redis.delete.assert_any_await(
            'stream_playback_session:session-1',
        )
        self.fake_redis.delete.assert_any_await(
            'media_overlay_demand:hazard_bGFiZWwx_Q2FtMQ:emgtVFc',
        )

    @patch('examples.streaming_web.routers.get_user_and_sites')
    def test_stream_playback_release_requires_session_id(
        self,
        mock_get_user_and_sites: AsyncMock,
    ) -> None:
        """Release requires a concrete playback session."""
        mock_get_user_and_sites.return_value = (None, ['label1'], 'admin')
        stream_result = MagicMock()
        stream_result.scalars.return_value.all.return_value = ['Cam1']
        self.mock_db_session.execute.side_effect = None
        self.mock_db_session.execute.return_value = stream_result

        response = self.client.post(
            '/api/stream-playback/release',
            json={
                'label': 'label1',
                'stream_id': 'Q2FtMQ',
                'language': 'zh_TW',
            },
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()['detail'], 'session_id_required')

    # -----------------------------
    # Test WebSocket endpoints (basic endpoint existence)
    # -----------------------------
    def test_websocket_endpoints_exist(self) -> None:
        """Test that WebSocket endpoints are properly defined in the router."""
        # FastAPI 0.139 keeps included routers as a route container, so the
        # outer routes list no longer exposes each WebSocket route directly.
        scope = {
            'type': 'websocket',
            'path': '/api/ws/metadata-id/label1/Q2FtMQ',
            'root_path': '',
            'scheme': 'ws',
            'headers': [],
            'query_string': b'',
            'client': ('testclient', 50000),
            'server': ('testserver', 80),
            'subprotocols': [],
        }

        self.assertIn(
            Match.FULL,
            [route.matches(scope)[0] for route in self.app.routes],
        )

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

    @patch(
        'examples.streaming_web.routers.get_media_session',
        new_callable=AsyncMock,
    )
    def test_media_auth_denies_wrong_site(
        self,
        mock_get_media_session: AsyncMock,
    ) -> None:
        """Denies MediaMTX requests outside the media capability scope."""
        mock_get_media_session.return_value = {
            'id': 'media-session-1',
            'username': 'testuser',
            'site': 'other-site',
            'cameras': ['Cam1'],
            'profile': 'overlay',
        }

        response = self.client.get(
            '/api/media-auth',
            headers={
                'X-Original-URI': (
                    '/hazard/media/'
                    'hazard_bGFiZWwx_Q2FtMQ_annotated_emgtVFc/segment0.ts'
                    '?mt=opaque-token'
                ),
            },
        )

        self.assertEqual(response.status_code, 403)

    @patch(
        'examples.streaming_web.routers.get_media_session',
        new_callable=AsyncMock,
    )
    def test_media_auth_refreshes_playback_session_for_media_path(
        self,
        mock_get_media_session: AsyncMock,
    ) -> None:
        """HLS media reads keep their playback session alive."""
        media_path = 'hazard_bGFiZWwx_Q2FtMQ'
        mock_get_media_session.return_value = {
            'id': 'media-session-1',
            'username': 'testuser',
            'site': 'label1',
            'cameras': ['Cam1'],
            'profile': 'clean',
            'quality': 'detail',
        }
        session = {
            'session_id': 'session-1',
            'username': 'testuser',
            'label': 'label1',
            'stream_name': 'Cam1',
            'stream_id': 'Q2FtMQ',
            'profile': 'clean',
            'base_media_path': media_path,
        }

        async def scan_iter(**_kwargs):
            yield (
                'stream_playback_media_session:'
                f'{media_path}:session-1'
            )

        self.fake_redis.scan_iter = scan_iter
        self.fake_redis.get = AsyncMock(return_value=json.dumps(session))
        self.fake_redis.expire = AsyncMock()

        response = self.client.get(
            '/api/media-auth',
            headers={
                'X-Original-URI': (
                    f'/hazard/media/{media_path}/video1_seg1956.mp4'
                    '?mt=opaque-token'
                ),
            },
        )

        self.assertEqual(response.status_code, 204)
        self.fake_redis.expire.assert_any_await(
            'stream_playback_session:session-1',
            routers.STREAM_PLAYBACK_SESSION_TTL_SECONDS,
        )
        self.fake_redis.expire.assert_any_await(
            f'stream_playback_media_session:{media_path}:session-1',
            routers.STREAM_PLAYBACK_SESSION_TTL_SECONDS,
        )

    @patch(
        'examples.streaming_web.routers.get_media_session',
        new_callable=AsyncMock,
    )
    def test_media_auth_accepts_opaque_mt_query_token(
        self,
        mock_get_media_session: AsyncMock,
    ) -> None:
        """Signed playback URLs authorise HLS reads with the mt query token."""
        mock_get_media_session.return_value = {
            'id': 'media-session-1',
            'username': 'testuser',
            'site': 'label1',
            'cameras': ['Cam1'],
            'profile': 'clean',
            'quality': 'detail',
        }

        response = self.client.get(
            '/api/media-auth',
            headers={
                'X-Original-URI': (
                    '/hazard/media/'
                    'hazard_bGFiZWwx_Q2FtMQ/index.m3u8?mt=opaque-token'
                ),
            },
        )

        self.assertEqual(response.status_code, 204)
        self.assertEqual(
            response.headers['x-media-auth-mode'],
            'opaque_media_session',
        )
        mock_get_media_session.assert_awaited_once_with(
            self.fake_redis,
            'opaque-token',
        )

    def test_media_auth_rejects_missing_media_token(self) -> None:
        """Media auth no longer accepts main JWTs or cookies as fallback."""
        response = self.client.get(
            '/api/media-auth',
            headers={
                'X-Original-URI': (
                    '/hazard/media/'
                    'hazard_bGFiZWwx_Q2FtMQ_annotated_ZW4/segment0.ts'
                ),
            },
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()['detail'], 'missing_media_token')
        self.assertEqual(
            response.headers['x-media-auth-error'],
            'missing_media_token',
        )

    @patch(
        'examples.streaming_web.routers.get_media_session',
        new_callable=AsyncMock,
    )
    def test_media_auth_rejects_expired_media_token(
        self,
        mock_get_media_session: AsyncMock,
    ) -> None:
        """Unknown opaque media tokens are rejected directly."""
        mock_get_media_session.return_value = None

        response = self.client.get(
            '/api/media-auth',
            headers={
                'X-Original-URI': (
                    '/hazard/media/'
                    'hazard_bGFiZWwx_Q2FtMQ_annotated_ZW4/segment0.ts'
                    '?mt=expired'
                ),
            },
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()['detail'], 'expired_media_session')
        self.assertEqual(
            response.headers['www-authenticate'],
            'Bearer error="expired_media_session"',
        )

    @patch(
        'examples.streaming_web.routers.get_media_session',
        new_callable=AsyncMock,
    )
    def test_media_auth_rejects_inactive_user(
        self,
        mock_get_media_session: AsyncMock,
    ) -> None:
        """Inactive users are rejected without DB I/O."""
        mock_get_media_session.return_value = {
            'id': 'media-session-1',
            'username': 'testuser',
            'site': 'label1',
            'cameras': ['Cam1'],
            'profile': 'overlay',
            'quality': 'detail',
            'user_active': False,
        }

        response = self.client.get(
            '/api/media-auth',
            headers={
                'X-Original-URI': (
                    '/hazard/media/'
                    'hazard_bGFiZWwx_Q2FtMQ_annotated_ZW4/segment0.ts'
                    '?mt=opaque-token'
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

    def test_opaque_media_token_extractor_reads_query_and_original_uri(
        self,
    ) -> None:
        """Only opaque playback media tokens are accepted for HLS auth."""
        request = cast(
            Request,
            SimpleNamespace(
                headers={},
                query_params={'mt': 'from-query'},
                cookies={},
            ),
        )
        self.assertEqual(
            routers._extract_opaque_media_token(request),
            'from-query',
        )

        request = cast(
            Request,
            SimpleNamespace(
                headers={
                    'x-original-uri': (
                        '/hazard/media/path?media_token=from-uri'
                    ),
                },
                query_params={},
                cookies={},
            ),
        )
        self.assertEqual(
            routers._extract_opaque_media_token(request),
            'from-uri',
        )

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
    def test_media_auth_rejects_invalid_media_path(
        self,
        mock_get_user_and_sites: AsyncMock,
    ) -> None:
        """Exercise this test."""
        mock_get_user_and_sites.return_value = (None, ['label1'], 'admin')

        response = self.client.get(
            '/api/media-auth',
            headers={
                'X-Original-URI': (
                    '/hazard/media/not-hazard/index.m3u8?mt=opaque-token'
                ),
            },
        )

        self.assertEqual(response.status_code, 403)

    @patch('examples.streaming_web.routers.get_user_and_sites')
    def test_media_auth_accepts_batch_scope_and_rejects_unlisted_camera(
        self,
        mock_get_user_and_sites: AsyncMock,
    ) -> None:
        """Media auth checks scoped Redis capabilities without database I/O."""
        mock_get_user_and_sites.return_value = (
            SimpleNamespace(status='active'),
            ['Site A'],
            'admin',
        )
        session = {
            'username': 'testuser',
            'site': 'Site A',
            'camera': None,
            'cameras': ['Cam 1', 'Cam 2'],
            'scope': 'batch',
            'profile': 'overlay',
            'quality': 'detail',
        }
        with (
            patch.object(
                routers,
                'get_media_session',
                new=AsyncMock(return_value=session),
            ),
            patch.object(
                routers,
                '_touch_media_demand_from_media_path',
                new=AsyncMock(),
            ),
            patch.object(
                routers,
                '_refresh_playback_sessions_for_media_path',
                new=AsyncMock(),
            ),
        ):
            allowed = self.client.get(
                '/api/media-auth',
                headers={
                    'X-Original-URI': (
                        '/hazard/media/'
                        'hazard_U2l0ZSBB_Q2FtIDE_annotated_emgtVFc/'
                        'index.m3u8?mt=batch-token'
                    ),
                },
            )
            denied = self.client.get(
                '/api/media-auth',
                headers={
                    'X-Original-URI': (
                        '/hazard/media/'
                        'hazard_U2l0ZSBB_Q2FtIDM_annotated_emgtVFc/'
                        'index.m3u8?mt=batch-token'
                    ),
                },
            )

        self.assertEqual(allowed.status_code, 204)
        self.assertEqual(
            allowed.headers['x-media-auth-mode'],
            'opaque_media_session',
        )
        self.assertEqual(denied.status_code, 403)
        self.assertEqual(denied.json()['detail'], 'media_scope_denied')
        mock_get_user_and_sites.assert_not_awaited()
        self.mock_db_session.execute.assert_not_awaited()

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
                    'profile': 'overlay',
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
                    'profile': 'overlay',
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

    async def test_configured_streams_overlay_requests_demand(self) -> None:
        """Overlay overview listings request demand before exposing status."""
        result = MagicMock()
        result.scalars.return_value.all.return_value = ['Cam1']
        self.mock_db_session.execute = AsyncMock(return_value=result)
        self.fake_redis.exists = AsyncMock(return_value=0)

        streams = await routers._get_configured_media_streams(
            self.mock_db_session,
            'label1',
            rds=self.fake_redis,
            overlay_mode='backend',
            overlay_language='zh-TW',
        )

        self.assertEqual(len(streams), 1)
        stream = streams[0]
        self.fake_redis.set.assert_awaited_once()
        self.assertFalse(stream['overlay_ready'])
        self.assertEqual(stream['status'], 'starting')
        self.assertEqual(stream['profile'], 'overlay')
        self.assertEqual(
            stream['playback_url'],
            '/hazard/media/'
            'hazard_bGFiZWwx_Q2FtMQ_annotated_emgtVFc/index.m3u8',
        )
        self.assertEqual(
            stream['media_hls_url'],
            '/hazard/media/'
            'hazard_bGFiZWwx_Q2FtMQ_annotated_emgtVFc/index.m3u8',
        )

    def test_webrtc_ice_servers_rejects_missing_subject(self) -> None:
        """Exercise this test."""
        self.app.dependency_overrides[jwt_access] = lambda: SimpleNamespace(
            subject={},
        )

        response = self.client.get('/api/webrtc/ice-servers')

        self.assertEqual(response.status_code, 401)

    @patch(
        'examples.streaming_web.routers._authorise_label_access',
        new_callable=AsyncMock,
    )
    async def test_metadata_stream_id_returns_sse_response(
        self,
        mock_authorise_label_access: AsyncMock,
    ) -> None:
        """Exercise this test."""
        request = MagicMock()
        credentials = cast(
            JwtAuthorizationCredentials,
            SimpleNamespace(subject={'username': 'testuser'}),
        )
        response = await routers.metadata_stream_id(
            request,
            'label1',
            'Q2FtMQ',
            credentials=credentials,
            db=self.mock_db_session,
            rds=self.fake_redis,
        )

        self.assertEqual(
            response.media_type,
            'text/event-stream',
        )
        self.assertNotIn('connection', response.headers)
        mock_authorise_label_access.assert_awaited_once_with(
            credentials,
            self.mock_db_session,
            'label1',
        )
        self.mock_db_session.close.assert_awaited_once()

    def test_metadata_stream_id_rejects_missing_subject(self) -> None:
        """Reject SSE clients that do not carry a valid JWT subject."""
        self.app.dependency_overrides[jwt_access] = lambda: SimpleNamespace(
            subject={},
        )

        response = self.client.get('/api/metadata/stream-id/label1/Q2FtMQ')

        self.assertEqual(response.status_code, 401)

    @patch(
        'examples.streaming_web.routers.get_user_and_sites',
        new_callable=AsyncMock,
    )
    def test_metadata_stream_id_rejects_unauthorised_label(
        self,
        mock_get_user_and_sites: AsyncMock,
    ) -> None:
        """Reject SSE clients outside the requested site's access scope."""
        mock_get_user_and_sites.return_value = (None, ['other-label'], 'user')

        response = self.client.get('/api/metadata/stream-id/label1/Q2FtMQ')

        self.assertEqual(response.status_code, 403)

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
