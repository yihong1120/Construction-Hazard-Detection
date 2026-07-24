from __future__ import annotations

import asyncio
import unittest
from datetime import timedelta
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
from examples.auth.session_store import auth_session_key
from examples.auth.session_store import create_auth_session
from examples.auth.session_store import create_media_session
from examples.auth.session_store import get_media_session
from examples.db_management.routers import playback
from tests.examples.auth.session_store_test import FakeRedis


class PlaybackRouterTest(unittest.TestCase):
    def setUp(self) -> None:
        app = FastAPI()
        app.include_router(playback.router)
        self.redis = FakeRedis()
        self.db = AsyncMock()
        app.dependency_overrides[get_redis_pool] = lambda: self.redis
        app.dependency_overrides[get_db] = lambda: self.db
        self.client = TestClient(app)
        self.access_token = jwt_access.create_access_token({
            'username': 'ChangDar',
            'user_id': 1,
        })
        self.headers = {'Authorization': f'Bearer {self.access_token}'}

    def test_native_single_playback_returns_signed_detail_url(self) -> None:
        upstream = {
            'session_id': 'stream-session-1',
            'key': 'Cam1',
            'label': 'Site A',
            'profile': 'clean',
            'state': 'ready',
            'status': 'ready',
            'media_hls_url': '/hazard/media/site_cam1/index.m3u8',
            'playback_url': (
                '/hazard/api/stream-playback/sessions/'
                'stream-session-1/index.m3u8'
            ),
        }

        with patch.object(
            playback,
            '_post_streaming_playback',
            new_callable=AsyncMock,
            return_value=(upstream, 200),
        ) as post_streaming:
            response = self.client.post(
                '/api/playback/sessions',
                headers=self.headers,
                json={
                    'site': 'Site A',
                    'camera': 'Cam1',
                    'profile': 'clean',
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body['mode'], 'single')
        self.assertEqual(body['quality'], 'detail')
        self.assertEqual(body['camera'], 'Cam1')
        self.assertEqual(body['token_transport'], 'query')
        self.assertEqual(
            body['single_endpoint'],
            '/hazard/api/db_management/api/playback/sessions',
        )
        self.assertEqual(
            body['wall_endpoint'],
            '/hazard/api/db_management/api/playback/walls',
        )
        self.assertNotIn('media_token', body)

        media_token = parse_qs(urlsplit(body['hls_url']).query)['mt'][0]
        self.assertIn(
            '/hazard/api/stream-playback/sessions/',
            body['hls_url'],
        )
        self.assertIn('/hazard/media/site_cam1/', body['media_hls_url'])
        media = asyncio.run(
            get_media_session(
                self.redis,  # type: ignore[arg-type]
                media_token,
            ),
        )
        assert media is not None
        self.assertEqual(media['parent'], 'native:user:1')
        self.assertEqual(media['camera'], 'Cam1')
        self.assertEqual(media['quality'], 'detail')
        post_streaming.assert_awaited_once()
        self.assertEqual(
            post_streaming.await_args.kwargs['payload']['profile'],
            'clean',
        )
        self.assertEqual(
            post_streaming.await_args.kwargs['payload']['rendition'],
            'detail',
        )

    def test_native_wall_playback_returns_shared_preview_token(self) -> None:
        upstream = {
            'items': [
                {
                    'session_id': 'stream-session-1',
                    'key': 'Cam1',
                    'media_hls_url': '/hazard/media/site_cam1/index.m3u8',
                    'playback_url': (
                        '/hazard/api/stream-playback/sessions/'
                        'stream-session-1/index.m3u8'
                    ),
                    'state': 'ready',
                    'status': 'ready',
                    'profile': 'overlay',
                    'language': 'zh-TW',
                },
                {
                    'session_id': 'stream-session-2',
                    'key': 'Cam2',
                    'media_hls_url': '/hazard/media/site_cam2/index.m3u8',
                    'playback_url': (
                        '/hazard/api/stream-playback/sessions/'
                        'stream-session-2/index.m3u8'
                    ),
                    'state': 'ready',
                    'status': 'ready',
                    'profile': 'overlay',
                    'language': 'zh-TW',
                },
            ],
            'count': 2,
            'max_streams': 24,
        }

        with patch.object(
            playback,
            '_post_streaming_playback',
            new_callable=AsyncMock,
            return_value=(upstream, 200),
        ) as post_streaming:
            response = self.client.post(
                '/api/playback/walls',
                headers=self.headers,
                json={
                    'site': 'Site A',
                    'cameras': ['Cam1', 'Cam2'],
                    'profile': 'overlay',
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body['mode'], 'multi_stream')
        self.assertEqual(body['layout'], 'responsive')
        self.assertEqual(body['quality'], 'preview')
        self.assertEqual(body['count'], 2)
        media_tokens = {
            parse_qs(urlsplit(item['preview_hls_url']).query)['mt'][0]
            for item in body['items']
        }
        self.assertEqual(len(media_tokens), 1)
        media = asyncio.run(
            get_media_session(
                self.redis,  # type: ignore[arg-type]
                next(iter(media_tokens)),
            ),
        )
        assert media is not None
        self.assertEqual(media['scope'], 'batch')
        self.assertEqual(media['cameras'], ['Cam1', 'Cam2'])
        self.assertEqual(media['quality'], 'preview')
        self.assertEqual(
            media['demand_keys'],
            [
                'media_overlay_demand:hazard_U2l0ZSBB_Q2FtMQ_preview:emgtVFc',
                'media_overlay_demand:hazard_U2l0ZSBB_Q2FtMg_preview:emgtVFc',
            ],
        )
        self.assertEqual(
            post_streaming.await_args.kwargs['payload']['streams'][0],
            {
                'label': 'Site A',
                'key': 'Cam1',
                'profile': 'overlay',
                'rendition': 'preview',
                'language': None,
                'transport': 'hls',
            },
        )

    def test_web_bff_session_uses_same_playback_endpoint(self) -> None:
        access = jwt_access.create_access_token(
            {'username': 'alice', 'user_id': 2},
            timedelta(minutes=15),
        )
        refresh = jwt_refresh.create_access_token(
            {'username': 'alice'},
            timedelta(days=30),
        )
        session_id, session = asyncio.run(
            create_auth_session(
                self.redis,  # type: ignore[arg-type]
                {
                    'access_token': access,
                    'refresh_token': refresh,
                    'feature_names': [],
                },
                {
                    'id': 2,
                    'username': 'alice',
                    'display_name': 'Alice',
                    'role': 'user',
                    'group_id': 1,
                    'status': 'active',
                },
            ),
        )
        upstream = {
            'session_id': 'stream-session-1',
            'key': 'Cam1',
            'media_hls_url': '/hazard/media/site_cam1/index.m3u8',
            'state': 'ready',
            'status': 'ready',
            'profile': 'clean',
        }

        with patch.object(
            playback,
            '_post_streaming_playback',
            new_callable=AsyncMock,
            return_value=(upstream, 200),
        ):
            response = self.client.post(
                '/api/playback/sessions',
                cookies={'__Host-vn_session': session_id},
                headers={
                    'Origin': 'https://changdar-server.mooo.com',
                    'X-CSRF-Token': str(session['csrf_secret']),
                },
                json={
                    'site': 'Site A',
                    'camera': 'Cam1',
                    'profile': 'clean',
                },
            )

        self.assertEqual(response.status_code, 200)
        media_token = parse_qs(urlsplit(response.json()['hls_url']).query)[
            'mt'
        ][0]
        media = asyncio.run(
            get_media_session(
                self.redis,  # type: ignore[arg-type]
                media_token,
            ),
        )
        assert media is not None
        self.assertEqual(media['parent'], auth_session_key(session_id))
        self.assertEqual(media['platform'], 'web')

    def test_renew_extends_existing_wall_without_recreating_playback(
        self,
    ) -> None:
        """Lease renewal keeps every wall HLS URL and media token stable."""
        _, media = asyncio.run(
            create_media_session(
                self.redis,  # type: ignore[arg-type]
                user_id=1,
                username='ChangDar',
                site='Site A',
                cameras=['Cam1', 'Cam2'],
                profile='overlay',
                parent='native:user:1',
                platform='native',
                quality='preview',
                purpose='playback',
            ),
        )

        with patch.object(
            playback,
            '_post_streaming_playback',
            new_callable=AsyncMock,
        ) as post_streaming:
            response = self.client.post(
                '/api/playback/sessions/renew',
                headers=self.headers,
                json={'id': media['id']},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body['id'], media['id'])
        self.assertEqual(body['mode'], 'multi_stream')
        self.assertTrue(body['renewed'])
        self.assertFalse(body['hls_urls_changed'])
        self.assertEqual(body['expires_in'], 600)
        self.assertEqual(
            body['renew_endpoint'],
            '/hazard/api/db_management/api/playback/sessions/renew',
        )
        post_streaming.assert_not_awaited()


if __name__ == '__main__':
    unittest.main()
