from __future__ import annotations

import asyncio
import unittest
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch
from urllib.parse import parse_qs
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.testclient import TestClient
from jwt.exceptions import InvalidTokenError

from examples.auth.database import get_db
from examples.auth.jwt_config import jwt_access
from examples.auth.jwt_config import jwt_refresh
from examples.auth.jwt_config import JwtAuthorizationCredentials
from examples.auth.redis_pool import get_redis_pool
from examples.auth.session_store import auth_session_key
from examples.auth.session_store import create_auth_session
from examples.auth.session_store import create_media_session
from examples.auth.session_store import get_media_session
from examples.db_management.routers import playback
from examples.db_management.schemas.playback import PlaybackRenewRequest
from examples.db_management.schemas.playback import PlaybackWallRequest
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
        self.access_token = jwt_access.create_access_token(
            {
                'username': 'ChangDar',
                'user_id': 1,
            },
        )
        self.headers = {'Authorization': f"Bearer {self.access_token}"}

    def test_native_single_playback_returns_signed_detail_url(self) -> None:
        upstream = {
            'session_id': 'stream-session-1',
            'key': 'Cam1',
            'label': 'Site A',
            'profile': 'clean',
            'rendition': 'detail',
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
        self.assertEqual(
            media['playback_sessions']['stream-session-1']['stream_name'],
            'Cam1',
        )
        post_streaming.assert_awaited_once()
        assert post_streaming.await_args is not None
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
        assert post_streaming.await_args is not None
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


def _request(
    authorization: str = '',
    session_id: str | None = None,
) -> SimpleNamespace:
    cookies = {}
    if session_id is not None:
        cookies[playback.SESSION_COOKIE] = session_id
    return SimpleNamespace(
        headers={
            'authorization': authorization,
            'x-csrf-token': 'csrf-token',
        },
        cookies=cookies,
    )


def _http_context(response: object) -> tuple[MagicMock, MagicMock]:
    client = MagicMock()
    client.post = AsyncMock(return_value=response)
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=client)
    context.__aexit__ = AsyncMock(return_value=None)
    return client, context


class TestPlaybackCoverage(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.db = MagicMock()
        self.db.scalar = AsyncMock()
        self.redis = MagicMock()
        self.principal = playback.PlaybackPrincipal(
            username='alice',
            user_id=7,
            parent='native:user:7',
            platform='native',
            access_token='access-token',
        )

    async def test_bearer_decode_and_subject_helpers(self) -> None:
        self.assertEqual(
            playback._bearer_token(
                _request('Bearer token'),
            ),
            'token',
        )
        self.assertIsNone(playback._bearer_token(_request('Basic token')))
        self.assertEqual(playback._subject_user_id({'user_id': '7'}), 7)
        self.assertIsNone(playback._subject_user_id({'user_id': 'invalid'}))

        with patch.object(
            playback.jwt_access,
            'decode_token',
            side_effect=InvalidTokenError('bad token'),
        ):
            with self.assertRaises(HTTPException) as invalid:
                await playback._decode_access_token('bad', self.redis)
        self.assertEqual(invalid.exception.status_code, 401)

        with (
            patch.object(
                playback.jwt_access,
                'decode_token',
                return_value={'sub': 'alice'},
            ),
            patch.object(
                playback,
                'is_access_token_revoked',
                new=AsyncMock(return_value=False),
            ),
        ):
            credentials = await playback._decode_access_token(
                'token',
                self.redis,
            )
        self.assertEqual(credentials.subject, {'username': 'alice'})

        with (
            patch.object(
                playback.jwt_access,
                'decode_token',
                return_value={'sub': 7},
            ),
            patch.object(
                playback,
                'is_access_token_revoked',
                new=AsyncMock(return_value=False),
            ),
        ):
            with self.assertRaises(HTTPException) as empty_subject:
                await playback._decode_access_token('token', self.redis)
        self.assertEqual(empty_subject.exception.status_code, 401)

    async def test_bearer_decode_rejects_revoked_and_redis_failures(
        self,
    ) -> None:
        """Playback tokens fail closed when revocation cannot be verified."""
        payload = {'subject': {'username': 'alice'}}
        with (
            patch.object(
                playback.jwt_access,
                'decode_token',
                return_value=payload,
            ),
            patch.object(
                playback,
                'is_access_token_revoked',
                new=AsyncMock(return_value=True),
            ),
            self.assertRaises(HTTPException) as revoked,
        ):
            await playback._decode_access_token('token', self.redis)
        self.assertEqual(revoked.exception.status_code, 401)

        with (
            patch.object(
                playback.jwt_access,
                'decode_token',
                return_value=payload,
            ),
            patch.object(
                playback,
                'is_access_token_revoked',
                new=AsyncMock(side_effect=playback.RedisError('offline')),
            ),
            self.assertRaises(HTTPException) as unavailable,
        ):
            await playback._decode_access_token('token', self.redis)
        self.assertEqual(unavailable.exception.status_code, 503)

    def test_streaming_descriptors_ignore_each_invalid_required_value(
        self,
    ) -> None:
        """Only complete descriptors may restore HLS sessions."""
        base: dict[str, object] = {
            'session_id': 'session',
            'label': 'SiteA',
            'key': 'Camera1',
            'profile': 'clean',
            'rendition': 'detail',
        }
        for field in ('session_id', 'label', 'key', 'profile', 'rendition'):
            item: dict[str, object] = dict(base)
            item[field] = ''
            self.assertEqual(
                playback._streaming_session_descriptors([item]), {},
            )

    async def test_user_lookup_and_principal_resolution_failures(self) -> None:
        self.db.scalar.return_value = '9'
        self.assertEqual(
            await playback._load_user_id_by_username(self.db, 'alice'),
            9,
        )
        self.db.scalar.return_value = 'invalid'
        with self.assertRaises(HTTPException) as invalid_user:
            await playback._load_user_id_by_username(self.db, 'alice')
        self.assertEqual(invalid_user.exception.detail, 'invalid_user')

        with patch.object(
            playback,
            '_decode_access_token',
            return_value=JwtAuthorizationCredentials(subject={}),
        ):
            with self.assertRaises(HTTPException) as invalid_bearer:
                await playback._resolve_playback_principal(
                    _request('Bearer token'),
                    self.db,
                    self.redis,
                )
        self.assertEqual(invalid_bearer.exception.status_code, 401)

        with patch.object(
            playback, 'get_auth_session', new=AsyncMock(return_value=None),
        ):
            with self.assertRaises(HTTPException) as expired_session:
                await playback._resolve_playback_principal(
                    _request(), self.db, self.redis,
                )
        self.assertEqual(
            expired_session.exception.detail,
            'app_session_expired',
        )

        with patch.object(
            playback,
            'get_auth_session',
            new=AsyncMock(return_value={'user': {'id': 1}}),
        ):
            with patch.object(playback, 'check_csrf'):
                with patch.object(
                    playback,
                    'get_proxy_access_token',
                    new=AsyncMock(return_value=('web-token', 'refresh')),
                ):
                    with patch.object(
                        playback,
                        '_decode_access_token',
                        return_value=JwtAuthorizationCredentials(subject={}),
                    ):
                        with self.assertRaises(
                            HTTPException,
                        ) as invalid_web_token:
                            await playback._resolve_playback_principal(
                                _request(
                                    session_id='session',
                                ),
                                self.db,
                                self.redis,
                            )
        self.assertEqual(invalid_web_token.exception.status_code, 401)

    async def test_principal_resolution_uses_database_and_bff_fallbacks(
        self,
    ) -> None:
        credentials = JwtAuthorizationCredentials(
            subject={'username': 'alice'},
        )
        with patch.object(
            playback, '_decode_access_token', return_value=credentials,
        ):
            with patch.object(
                playback,
                '_load_user_id_by_username',
                new=AsyncMock(return_value=9),
            ):
                principal = await playback._resolve_playback_principal(
                    _request('Bearer token'),
                    self.db,
                    self.redis,
                )
        self.assertEqual(principal.parent, 'native:user:9')

        with patch.object(
            playback,
            'get_auth_session',
            new=AsyncMock(return_value={'user': {'id': 'bad'}}),
        ):
            with patch.object(playback, 'check_csrf'):
                with patch.object(
                    playback,
                    'get_proxy_access_token',
                    new=AsyncMock(return_value=('web-token', 'refresh')),
                ):
                    with patch.object(
                        playback,
                        '_decode_access_token',
                        return_value=credentials,
                    ):
                        with patch.object(
                            playback,
                            '_load_user_id_by_username',
                            new=AsyncMock(return_value=11),
                        ):
                            principal = (
                                await playback._resolve_playback_principal(
                                    _request(
                                        session_id='session',
                                    ),
                                    self.db,
                                    self.redis,
                                )
                            )
        self.assertEqual(principal.platform, 'web')
        self.assertEqual(principal.user_id, 11)

    async def test_streaming_upstream_errors_and_detail_parsing(self) -> None:
        response = MagicMock()
        response.json.side_effect = ValueError('not json')
        response.text = 'plain error'
        self.assertEqual(playback._streaming_detail(response), 'plain error')
        response.json.side_effect = None
        response.json.return_value = {'detail': 'forbidden'}
        self.assertEqual(playback._streaming_detail(response), 'forbidden')
        response.json.return_value = ['unexpected']
        self.assertEqual(playback._streaming_detail(response), ['unexpected'])

        _, timeout_context = _http_context(MagicMock())
        timeout_context.__aenter__.side_effect = httpx.TimeoutException(
            'timeout',
        )
        with patch.object(
            playback.httpx, 'AsyncClient', return_value=timeout_context,
        ):
            with self.assertRaises(HTTPException) as unavailable:
                await playback._post_streaming_playback(
                    '/stream',
                    principal=self.principal,
                    payload={},
                )
        self.assertEqual(unavailable.exception.status_code, 502)

        failure_response = MagicMock(status_code=403)
        failure_response.json.return_value = {'detail': 'forbidden'}
        _, failure_context = _http_context(failure_response)
        with patch.object(
            playback.httpx, 'AsyncClient', return_value=failure_context,
        ):
            with self.assertRaises(HTTPException) as failed_upstream:
                await playback._post_streaming_playback(
                    '/stream',
                    principal=self.principal,
                    payload={},
                )
        self.assertEqual(failed_upstream.exception.detail, 'forbidden')

        invalid_response = MagicMock(status_code=200)
        invalid_response.json.side_effect = ValueError('bad json')
        _, invalid_context = _http_context(invalid_response)
        with patch.object(
            playback.httpx, 'AsyncClient', return_value=invalid_context,
        ):
            with self.assertRaises(HTTPException) as bad_json:
                await playback._post_streaming_playback(
                    '/stream',
                    principal=self.principal,
                    payload={},
                )
        self.assertEqual(
            bad_json.exception.detail,
            'invalid_streaming_upstream_response',
        )

        list_response = MagicMock(status_code=200)
        list_response.json.return_value = []
        _, list_context = _http_context(list_response)
        with patch.object(
            playback.httpx, 'AsyncClient', return_value=list_context,
        ):
            with self.assertRaises(HTTPException) as bad_body:
                await playback._post_streaming_playback(
                    '/stream',
                    principal=self.principal,
                    payload={},
                )
        self.assertEqual(bad_body.exception.status_code, 502)

        success_response = MagicMock(status_code=201)
        success_response.json.return_value = {'key': 'Cam1'}
        _, success_context = _http_context(success_response)
        with patch.object(
            playback.httpx, 'AsyncClient', return_value=success_context,
        ):
            body, status_code = await playback._post_streaming_playback(
                '/stream',
                principal=self.principal,
                payload={'key': 'Cam1'},
            )
        self.assertEqual((body, status_code), ({'key': 'Cam1'}, 201))

    def test_signed_urls_profiles_and_wall_payload(self) -> None:
        self.assertEqual(playback._normalise_profile(None), 'clean')
        with self.assertRaises(HTTPException):
            playback._normalise_profile('unknown')
        signed = playback._signed_stream_item(
            {
                'media_hls_url': '/media/index.m3u8?mt=old&keep=value',
                'playback_url': '/play/index.m3u8?media_token=old',
            },
            'new-token',
        )
        assert isinstance(signed['media_hls_url'], str)
        assert isinstance(signed['playback_url'], str)
        self.assertIn('mt=new-token', signed['media_hls_url'])
        self.assertIn('mt=new-token', signed['playback_url'])
        self.assertEqual(signed['hls_url'], signed['playback_url'])
        self.assertEqual(
            playback._signed_stream_item(
                {'media_hls_url': '/media'},
                'token',
            )['hls_url'],
            '/media?mt=token',
        )

        self.assertEqual(
            playback._playback_demand_keys(
                site='Site',
                cameras=['Cam'],
                profile='clean',
                quality='preview',
                language=None,
            )[0].split(':', 1)[0],
            'media_clean_demand',
        )
        self.assertEqual(
            playback._playback_demand_keys(
                site='Site',
                cameras=['Cam'],
                profile='overlay',
                quality='detail',
                language='en',
            )[0].split(':', 1)[0],
            'media_overlay_demand',
        )
        with self.assertRaises(ValueError):
            playback._playback_demand_keys(
                site='Site',
                cameras=['Cam'],
                profile='bad',
                quality='detail',
                language=None,
            )

        wall_payload = playback._wall_upstream_payload(
            PlaybackWallRequest(site='Site', cameras=['Cam1']),
            'overlay',
        )
        self.assertNotIn('label', wall_payload)
        streams = wall_payload['streams']
        assert isinstance(streams, list)
        assert streams
        first_stream = streams[0]
        assert isinstance(first_stream, dict)
        self.assertEqual(first_stream['key'], 'Cam1')

    def test_wall_camera_validator_handles_missing_blank_and_duplicate_names(
        self,
    ) -> None:
        """Wall requests may omit cameras but cannot contain invalid names."""
        self.assertIsNone(PlaybackWallRequest(site='Site').cameras)
        self.assertIsNone(PlaybackWallRequest.validate_cameras(None))
        with self.assertRaises(ValueError):
            PlaybackWallRequest(site='Site', cameras=['Cam 1', '  '])
        with self.assertRaises(ValueError):
            PlaybackWallRequest(site='Site', cameras=['Cam 1', 'Cam 1'])

    async def test_wall_validation_and_session_lifecycle_errors(self) -> None:
        payload = PlaybackWallRequest(site='Site')
        with patch.object(
            playback,
            '_post_streaming_playback',
            new=AsyncMock(return_value=({}, 200)),
        ):
            with self.assertRaises(HTTPException) as invalid_items:
                await playback._create_wall_playback(
                    payload=payload,
                    principal=self.principal,
                    redis=self.redis,
                )
        self.assertEqual(invalid_items.exception.status_code, 502)

        with patch.object(
            playback,
            '_post_streaming_playback',
            new=AsyncMock(return_value=({'items': [{}]}, 200)),
        ):
            with self.assertRaises(HTTPException) as no_cameras:
                await playback._create_wall_playback(
                    payload=payload,
                    principal=self.principal,
                    redis=self.redis,
                )
        self.assertEqual(no_cameras.exception.detail, 'cameras_not_found')

        with patch.object(
            playback,
            '_resolve_playback_principal',
            new=AsyncMock(return_value=self.principal),
        ):
            with patch.object(
                playback,
                'renew_media_session',
                new=AsyncMock(return_value=None),
            ):
                with self.assertRaises(HTTPException) as expired:
                    await playback.renew_playback_session(
                        PlaybackRenewRequest(
                            id='session',
                        ),
                        _request(),
                        self.db,
                        self.redis,
                    )
        self.assertEqual(expired.exception.detail, 'expired_media_session')

        with patch.object(
            playback,
            '_resolve_playback_principal',
            new=AsyncMock(return_value=self.principal),
        ):
            with patch.object(
                playback,
                'delete_media_session',
                new=AsyncMock(return_value=False),
            ):
                with self.assertRaises(HTTPException) as absent:
                    await playback.delete_playback_session(
                        'session',
                        _request(),
                        self.db,
                        self.redis,
                    )
        self.assertEqual(absent.exception.detail, 'session_not_found')

        with patch.object(
            playback,
            '_resolve_playback_principal',
            new=AsyncMock(return_value=self.principal),
        ):
            with patch.object(
                playback,
                'delete_media_session',
                new=AsyncMock(return_value=True),
            ):
                response = await playback.delete_playback_session(
                    'session',
                    _request(),
                    self.db,
                    self.redis,
                )
        self.assertEqual(response.status_code, 204)


if __name__ == '__main__':
    unittest.main()
