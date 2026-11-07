from __future__ import annotations

import asyncio
import json
import unittest
from datetime import datetime
from datetime import timezone
from functools import partial
from types import SimpleNamespace
from typing import Any
from typing import cast
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import HTTPException

import examples.streaming_web.media_paths as playback_media_paths
import examples.streaming_web.playback_config as playback_config
import examples.streaming_web.playback_languages as playback_languages
import examples.streaming_web.playback_policy as playback_policy
import examples.streaming_web.playback_selection as playback_selection
import examples.streaming_web.playback_session_store as playback_store
import examples.streaming_web.schemas as playback_schemas
from examples.auth.jwt_config import JwtAuthorizationCredentials
from examples.db_management.schemas.auth import AccessTokenSubject
from examples.streaming_web import playback_hls
from examples.streaming_web import routers
from examples.streaming_web import streaming_api_service
from examples.streaming_web import streaming_metadata_service
from examples.streaming_web.schemas import StreamPlaybackBatchRequest
from examples.streaming_web.schemas import StreamPlaybackRequest
from tests.redis_fixtures_test import spy_redis


async def _demand_and_media_session_indexes_read_scores(
    rds,
    /,
    *args,
    **kwargs,
):
    if args[1] == '-inf':
        return []
    return await rds.zrangebyscore(*args, **kwargs)


async def _demand_and_media_session_indexes_read_many(rds, /, *args):
    return await rds.mget(*args)


async def _refreshes_and_prunes_invalid_media_read_scores(
    rds,
    /,
    *args,
    **kwargs,
):
    if args[1] == '-inf':
        return []
    return await rds.zrangebyscore(*args, **kwargs)


async def _refreshes_and_prunes_invalid_media_read_many(
    rds,
    /,
    *args,
):
    return await rds.mget(*args)


async def _backend_overlay_metadata_builds_demand_events() -> Any:
    yield b'data: {}\n\n'


async def _upstream_missing_playlist_is_retryable_check():
    client = SimpleNamespace(
        get=AsyncMock(
            return_value=SimpleNamespace(status_code=404),
        ),
    )
    try:
        await playback_hls.fetch_internal_hls_playlist(
            'hazard_test',
            '',
            client,
        )
    except HTTPException as exc:
        assert exc.status_code == 503
        assert exc.detail == 'media_playlist_not_ready'
        assert exc.headers['Retry-After'] == '2'
    else:
        raise AssertionError('Expected retryable startup response')


def _credentials(
    subject: dict[str, object],
) -> JwtAuthorizationCredentials:
    """Create incomplete credentials for authentication guard tests."""
    return JwtAuthorizationCredentials(
        subject=cast(AccessTokenSubject, subject),
    )


class TestStreamingRouterHlsHelpers(unittest.IsolatedAsyncioTestCase):
    """Provide TestStreamingRouterHlsHelpers."""

    def test_overlay_and_media_session_helpers_cover_invalid_inputs(
        self,
    ) -> None:
        """Test overlay and media session helpers cover invalid inputs."""
        options = playback_languages._overlay_language_options(
            ('en',),
        )
        self.assertEqual(options[0].code, 'en')
        self.assertEqual(
            playback_hls.media_session_demand_ttl(
                {'expires_at': 0},
            ),
            playback_hls.MEDIA_PUBLISHER_IDLE_GRACE_SECONDS,
        )
        self.assertIsNone(
            playback_hls.media_hls_session_cookie(
                'hazard_site_cam',
                None,
            ),
        )
        with self.assertRaisesRegex(HTTPException, 'invalid_hls_session'):
            playback_hls.media_hls_session_cookie(
                'hazard_site_cam',
                'invalid value',
            )

        with self.assertRaises(HTTPException) as raised:
            playback_policy.normalise_playback_profile('webrtc')
        self.assertEqual(raised.exception.detail, 'unsupported_profile')
        with self.assertRaises(HTTPException) as raised:
            playback_policy.normalise_playback_rendition('thumbnail')
        self.assertEqual(raised.exception.detail, 'unsupported_rendition')

    def test_hls_uri_and_playlist_rewriting_preserve_media_query_and_auth(
        self,
    ) -> None:
        """Test hls uri and playlist rewriting preserve media query and
        auth."""
        media_path = 'hazard_site_camera'
        auth_query = 'mt=opaque-token'
        self.assertEqual(
            playback_hls.rewrite_hls_uri('segment.ts', media_path, ''),
            'segment.ts',
        )
        self.assertIn(
            '/hazard/media/hazard_site_camera/segment.ts?mt=opaque-token',
            playback_hls.rewrite_hls_uri(
                'segment.ts',
                media_path,
                auth_query,
            ),
        )
        self.assertIn(
            '/hazard/media/hazard_site_camera/part.ts?mt=opaque-token',
            playback_hls.rewrite_hls_uri(
                f"{media_path}/part.ts",
                media_path,
                auth_query,
            ),
        )
        self.assertIn(
            (
                '/hazard/media/hazard_site_camera/absolute.ts?foo=bar&mt='
                'opaque-token'
            ),
            playback_hls.rewrite_hls_uri(
                'https://media.example/other/absolute.ts?foo=bar',
                media_path,
                auth_query,
            ),
        )
        self.assertEqual(
            playback_hls.rewrite_hls_uri(
                f"/hazard/media/{media_path}/already.ts",
                media_path,
                auth_query,
            ),
            f"/hazard/media/{media_path}/already.ts?mt=opaque-token",
        )

        playlist = '\n#EXT-X-KEY:METHOD=AES-128,URI="key.bin"\nsegment.ts\n'
        rewritten = playback_hls.rewrite_hls_playlist_media_urls(
            playlist,
            media_path=media_path,
            auth_query=auth_query,
        )
        self.assertTrue(rewritten.startswith('\n#EXT-X-KEY'))
        self.assertIn(
            'URI="/hazard/media/hazard_site_camera/key.bin?mt=opaque-toke'
            'n"',
            rewritten,
        )
        self.assertIn(
            '/hazard/media/hazard_site_camera/segment.ts?mt=opaque-token',
            rewritten,
        )

    def test_media_path_and_session_payload_helpers_handle_bad_values(
        self,
    ) -> None:
        """Test media path and session payload helpers handle bad values."""
        self.assertEqual(
            playback_hls.extract_media_path_from_uri('/not-media/path'),
            '',
        )
        self.assertEqual(
            playback_hls.extract_media_path_from_uri(
                '/hazard/media/webrtc/hazard_site_cam/whep',
            ),
            'hazard_site_cam',
        )
        self.assertFalse(
            playback_hls.media_path_matches_site(
                'hazard_other_cam',
                'site',
            ),
        )
        self.assertIsNone(
            playback_store.decode_playback_session_payload(
                None,
            ),
        )
        with self.assertRaises(ValueError):
            playback_store.decode_playback_session_payload(
                b'not-json',
            )
        self.assertEqual(
            playback_store.decode_playback_session_payload(
                b'{"profile":"clean"}',
            ),
            {'profile': 'clean'},
        )
        with self.assertRaises(TypeError):
            playback_store.decode_playback_session_payload(
                cast(bytes, 123),
            )

    def test_media_session_scope_and_selected_path_helpers(self) -> None:
        """Test media session scope and selected path helpers."""
        base_path = playback_media_paths.build_media_path('SiteA', 'Camera1')
        preview_path = playback_media_paths.build_preview_media_path(base_path)
        overlay_path = playback_media_paths.build_annotated_media_path(
            base_path,
            'en',
        )
        self.assertTrue(
            playback_hls.opaque_media_session_allows_path(
                {
                    'site': 'SiteA',
                    'camera': 'Camera1',
                    'cameras': ['Camera1'],
                    'quality': 'detail',
                    'profile': 'clean',
                },
                base_path,
            ),
        )

        self.assertTrue(
            playback_hls.opaque_media_session_allows_path(
                {
                    'site': 'SiteA',
                    'cameras': ['Camera1'],
                    'quality': 'preview',
                    'profile': 'clean',
                },
                preview_path,
            ),
        )
        self.assertTrue(
            playback_hls.opaque_media_session_allows_path(
                {
                    'site': 'SiteA',
                    'camera': 'Camera1',
                    'cameras': ['Camera1'],
                    'quality': 'detail',
                    'profile': 'overlay',
                },
                overlay_path,
            ),
        )
        self.assertFalse(
            playback_hls.opaque_media_session_allows_path(
                {
                    'site': 'SiteA',
                    'cameras': [],
                    'quality': 'detail',
                    'profile': 'unknown',
                },
                base_path,
            ),
        )
        self.assertFalse(
            playback_hls.opaque_media_session_allows_path(
                {'site': 'SiteA', 'quality': 'invalid', 'camera': 'Camera1'},
                base_path,
            ),
        )
        with self.assertRaises(KeyError):
            playback_policy.session_selected_media_path(
                cast(playback_schemas.PlaybackSession, {}),
            )
        self.assertEqual(
            playback_policy.session_selected_media_path(
                cast(
                    playback_schemas.PlaybackSession,
                    {'base_media_path': base_path, 'profile': 'clean'},
                ),
            ),
            base_path,
        )
        self.assertEqual(
            playback_policy.session_selected_media_path(
                cast(
                    playback_schemas.PlaybackSession,
                    {
                        'base_media_path': base_path,
                        'profile': 'overlay',
                        'overlay_media_path': overlay_path,
                    },
                ),
            ),
            overlay_path,
        )

    async def test_demand_and_media_session_indexes_propagate_failures(
        self,
    ) -> None:
        """Test demand and media session indexes propagate failures."""
        base_path = playback_media_paths.build_media_path('SiteA', 'Camera1')
        overlay_path = playback_media_paths.build_annotated_media_path(
            base_path,
            'en',
        )
        rds = AsyncMock()
        pipeline = MagicMock()
        pipeline.__aenter__ = AsyncMock(return_value=pipeline)
        pipeline.__aexit__ = AsyncMock(return_value=None)
        pipeline.execute = AsyncMock()
        pipeline.watch = AsyncMock()

        pipeline.get = AsyncMock(return_value=None)

        read_scores = partial(
            _demand_and_media_session_indexes_read_scores,
            rds,
        )

        read_many = partial(_demand_and_media_session_indexes_read_many, rds)

        pipeline.zrangebyscore = AsyncMock(side_effect=read_scores)
        pipeline.mget = AsyncMock(side_effect=read_many)
        rds.pipeline = MagicMock(return_value=pipeline)
        with patch.object(
            playback_policy,
            'touch_overlay_demand',
            AsyncMock(),
        ) as touch_overlay:
            await playback_policy.touch_media_demand_from_media_path(
                rds,
                overlay_path,
                ttl_seconds=17,
            )
        touch_overlay.assert_awaited_once_with(
            rds,
            base_path,
            'en',
            ttl_seconds=17,
        )

        with patch.object(
            playback_policy,
            'touch_clean_demand',
            AsyncMock(side_effect=RuntimeError('redis offline')),
        ):
            with self.assertRaisesRegex(RuntimeError, 'redis offline'):
                await playback_policy.touch_media_demand_from_media_path(
                    rds,
                    base_path,
                )

        with self.assertRaises(KeyError):
            await playback_store.delete_playback_session(
                rds,
                cast(playback_schemas.PlaybackSession, {}),
            )
        with self.assertRaises(KeyError):
            await playback_store.register_playback_session_media_path(
                rds,
                cast(playback_schemas.PlaybackSession, {}),
                base_path,
            )

        rds.expire.side_effect = RuntimeError('redis offline')
        with self.assertRaisesRegex(RuntimeError, 'redis offline'):
            await playback_store.refresh_playback_session_ttl(
                rds,
                'session-1',
            )

    async def test_refreshes_and_prunes_invalid_media_session_indexes(
        self,
    ) -> None:
        """Test refreshes and prunes invalid media session indexes."""
        rds = MagicMock()
        pipeline = MagicMock()
        pipeline.__aenter__ = AsyncMock(return_value=pipeline)
        pipeline.__aexit__ = AsyncMock(return_value=None)
        pipeline.execute = AsyncMock()
        pipeline.watch = AsyncMock()

        pipeline.get = AsyncMock(return_value=None)

        read_scores = partial(
            _refreshes_and_prunes_invalid_media_read_scores,
            rds,
        )

        read_many = partial(_refreshes_and_prunes_invalid_media_read_many, rds)

        pipeline.zrangebyscore = AsyncMock(side_effect=read_scores)
        pipeline.mget = AsyncMock(side_effect=read_many)
        rds.pipeline.return_value = pipeline
        rds.set = AsyncMock(return_value=True)
        rds.zrangebyscore = AsyncMock(return_value=[b'missing', b'mismatch'])
        rds.mget = AsyncMock(
            return_value=[
                None,
                b'{"profile":"clean","base_media_path":"other-path"}',
            ],
        )
        await playback_store.refresh_playback_sessions_for_media_path(
            rds,
            'hazard_site_cam',
        )
        pipeline.zrem.assert_any_call(
            'stream_playback_media_session:hazard_site_cam',
            'missing',
            'mismatch',
        )

        rds.set = AsyncMock(return_value=True)
        rds.zrangebyscore = AsyncMock(side_effect=RuntimeError('zset failed'))
        with self.assertRaisesRegex(RuntimeError, 'zset failed'):
            await playback_store.refresh_playback_sessions_for_media_path(
                rds,
                'hazard_site_cam',
            )

    async def test_playback_session_errors_and_startup_input_validation(
        self,
    ) -> None:
        """Test playback session errors and startup input validation."""
        rds = spy_redis()
        with self.assertRaises(HTTPException) as raised:
            await playback_store.create_or_update_playback_session(
                rds,
                'alice',
                playback_store.PlaybackSessionPlan(
                    'SiteA',
                    'Camera1',
                    'clean',
                    'detail',
                    None,
                    'missing',
                ),
            )
        self.assertEqual(raised.exception.detail, 'session_not_found')
        rds.data[playback_store.playback_session_key('forbidden')] = (
            b'{"username":"bob"}'
        )
        with self.assertRaises(HTTPException) as raised:
            await playback_store.create_or_update_playback_session(
                rds,
                'alice',
                playback_store.PlaybackSessionPlan(
                    'SiteA',
                    'Camera1',
                    'clean',
                    'detail',
                    None,
                    'forbidden',
                ),
            )
        self.assertEqual(raised.exception.detail, 'session_forbidden')

        with patch.object(
            playback_config,
            'STREAM_PLAYBACK_STARTUP_WAIT_SECONDS',
            0.0,
        ):
            await playback_selection.wait_for_session_startup(
                cast(playback_schemas.PlaybackSession, {}),
            )
        with patch.object(
            playback_config,
            'STREAM_PLAYBACK_STARTUP_WAIT_SECONDS',
            1.0,
        ):
            with self.assertRaises(ValueError):
                await playback_selection.wait_for_session_startup(
                    cast(
                        playback_schemas.PlaybackSession,
                        {'created_at': 'not-a-date'},
                    ),
                )

    async def test_label_access_rejects_inactive_users(self) -> None:
        """Test label access rejects inactive users."""
        credentials = _credentials({'username': 'alice'})
        inactive_user = SimpleNamespace(status='suspended')
        with patch.object(
            playback_hls,
            'load_user_access_context',
            AsyncMock(return_value=(inactive_user, ['SiteA'], 'user')),
        ):
            with self.assertRaises(HTTPException) as raised:
                await playback_hls.authorise_label_access(
                    credentials,
                    AsyncMock(),
                    'SiteA',
                )
        self.assertEqual(raised.exception.detail, 'inactive_user')

    async def test_internal_hls_playlist_maps_network_status_and_empty_errors(
        self,
    ) -> None:
        """Test internal hls playlist maps network status and empty errors."""
        context = MagicMock()
        client = AsyncMock()
        context.__aenter__ = AsyncMock(return_value=client)
        context.__aexit__ = AsyncMock(return_value=None)

        client.get.side_effect = playback_hls.httpx.TimeoutException(
            'timed out',
        )
        with patch.object(
            playback_hls.httpx,
            'AsyncClient',
            return_value=context,
        ):
            with self.assertRaises(HTTPException) as raised:
                await playback_hls.fetch_internal_hls_playlist(
                    'hazard_site_cam',
                    media_query='',
                )
        self.assertEqual(raised.exception.status_code, 502)

        client.get.side_effect = None
        client.get.return_value = SimpleNamespace(
            status_code=503,
            text='unavailable',
            cookies={},
        )
        with patch.object(
            playback_hls.httpx,
            'AsyncClient',
            return_value=context,
        ):
            with self.assertRaises(HTTPException) as raised:
                await playback_hls.fetch_internal_hls_playlist(
                    'hazard_site_cam',
                    media_query='quality=low',
                )
        self.assertEqual(raised.exception.status_code, 503)

        client.get.return_value = SimpleNamespace(
            status_code=200,
            text='  ',
            cookies={},
        )
        with patch.object(
            playback_hls.httpx,
            'AsyncClient',
            return_value=context,
        ):
            with self.assertRaises(HTTPException) as raised:
                await playback_hls.fetch_internal_hls_playlist(
                    'hazard_site_cam',
                    media_query='',
                )
        self.assertEqual(raised.exception.detail, 'media_playlist_not_ready')

    async def test_internal_hls_playlist_uses_lifespan_client(self) -> None:
        """A supplied shared client avoids creating a per-request transport."""
        client = AsyncMock()
        client.get.return_value = SimpleNamespace(
            status_code=200,
            text='#EXTM3U',
            cookies={},
        )

        playlist_result = await playback_hls.fetch_internal_hls_playlist(
            'hazard_site_cam',
            media_query='quality=low',
            http_client=client,
        )
        playlist, session_cookie = playlist_result

        self.assertEqual(playlist, '#EXTM3U')
        self.assertIsNone(session_cookie)
        client.get.assert_awaited_once()

    async def test_session_scan_startup_and_playlist_input_guards(
        self,
    ) -> None:
        """Playback helpers fail closed when sessions or media auth are
        invalid."""
        rds = SimpleNamespace(
            zremrangebyscore=AsyncMock(),
            zcard=AsyncMock(return_value=1),
        )
        self.assertTrue(
            await playback_store.has_other_playback_session(
                rds,
                base_media_path='hazard_site_cam',
                profile='overlay',
                language='en',
            ),
        )
        rds.zremrangebyscore.assert_awaited_once()
        rds.zcard.assert_awaited_once_with(
            playback_store.playback_demand_session_key(
                'hazard_site_cam',
                'overlay',
                'en',
            ),
        )

        created_at = datetime.now(timezone.utc).isoformat()
        with (
            patch.object(
                playback_config,
                'STREAM_PLAYBACK_STARTUP_WAIT_SECONDS',
                30.0,
            ),
            patch.object(
                asyncio,
                'sleep',
                AsyncMock(),
            ) as sleep,
        ):
            await playback_selection.wait_for_session_startup(
                cast(
                    playback_schemas.PlaybackSession,
                    {'created_at': created_at},
                ),
            )
        sleep.assert_awaited_once()

        request = SimpleNamespace(url=SimpleNamespace(query=''))
        with patch.object(
            playback_store,
            'load_playback_session',
            AsyncMock(return_value=None),
        ):
            with self.assertRaisesRegex(HTTPException, 'session_not_found'):
                await routers.stream_playback_session_playlist(
                    'missing',
                    request,
                    rds,
                )

        with patch.object(
            playback_store,
            'load_playback_session',
            AsyncMock(return_value={'session_id': 'present'}),
        ):
            with self.assertRaisesRegex(HTTPException, 'missing_media_token'):
                await routers.stream_playback_session_playlist(
                    'present',
                    request,
                    rds,
                )

    async def test_playback_batch_and_endpoint_auth_guards(self) -> None:
        """Batch expansion and playback endpoints reject missing identity
        safely."""
        explicit_profile = StreamPlaybackRequest(profile='clean')
        self.assertTrue(
            streaming_api_service._model_field_was_set(
                explicit_profile,
                'profile',
            ),
        )
        self.assertFalse(
            streaming_api_service._model_field_was_set(
                explicit_profile,
                'language',
            ),
        )

        with self.assertRaisesRegex(HTTPException, 'label_required'):
            await streaming_api_service._build_batch_playback_requests(
                StreamPlaybackBatchRequest(),
                AsyncMock(),
            )

        oversized = [
            StreamPlaybackRequest(key=f"Camera{index}")
            for index in range(
                streaming_api_service.MAX_STREAM_PLAYBACK_BATCH_STREAMS + 1,
            )
        ]
        with self.assertRaisesRegex(
            HTTPException,
            'stream_batch_limit_exceeded',
        ):
            streaming_api_service._enforce_stream_playback_batch_limit(
                oversized,
            )

        credentials = SimpleNamespace(subject={})
        request = StreamPlaybackRequest(label='SiteA', key='Camera1')
        with self.assertRaisesRegex(HTTPException, 'Invalid token'):
            await routers.request_stream_playback(
                request,
                credentials,
                AsyncMock(),
                AsyncMock(),
            )
        with self.assertRaisesRegex(HTTPException, 'Invalid token'):
            await routers.request_stream_playback_batch(
                StreamPlaybackBatchRequest(label='SiteA'),
                credentials,
                AsyncMock(),
                AsyncMock(),
            )
        with self.assertRaisesRegex(HTTPException, 'Invalid token'):
            await routers.release_stream_playback(
                request,
                credentials,
                AsyncMock(),
            )
        with self.assertRaisesRegex(HTTPException, 'Invalid token'):
            await routers.get_streams_for_label_route(
                'SiteA',
                credentials=credentials,
                db=AsyncMock(),
                rds=AsyncMock(),
            )
        with self.assertRaisesRegex(HTTPException, 'Invalid token'):
            await routers.get_webrtc_ice_servers(credentials)

    async def test_playback_release_playlist_and_listing_edge_paths(
        self,
    ) -> None:
        """Release cleanup and listing failures preserve their API
        contracts."""
        credentials = _credentials({'username': 'alice'})
        rds = AsyncMock()
        with patch.object(
            playback_store,
            'load_playback_session',
            AsyncMock(return_value=None),
        ):
            with self.assertRaisesRegex(HTTPException, 'session_not_found'):
                await routers.release_stream_playback(
                    StreamPlaybackRequest(session_id='missing'),
                    credentials,
                    rds,
                )

        with patch.object(
            playback_store,
            'load_playback_session',
            AsyncMock(return_value={'username': 'bob'}),
        ):
            with self.assertRaisesRegex(HTTPException, 'session_forbidden'):
                await routers.release_stream_playback(
                    StreamPlaybackRequest(session_id='forbidden'),
                    credentials,
                    rds,
                )

        clean_session = {
            'username': 'alice',
            'profile': 'clean',
            'base_media_path': 'hazard_site_camera',
        }
        with (
            patch.object(
                playback_store,
                'load_playback_session',
                AsyncMock(return_value=clean_session),
            ),
            patch.object(
                playback_store,
                'delete_playback_session',
                AsyncMock(return_value=True),
            ) as delete_session,
            patch.object(
                playback_store,
                'has_other_playback_session',
                AsyncMock(return_value=False),
            ),
        ):
            response = await routers.release_stream_playback(
                StreamPlaybackRequest(session_id='clean'),
                credentials,
                rds,
            )
        self.assertEqual(response.status_code, 200)
        delete_session.assert_awaited_once_with(rds, clean_session)

        request = SimpleNamespace(url=SimpleNamespace(query='mt=token'))
        with (
            patch.object(
                playback_store,
                'load_playback_session',
                AsyncMock(return_value={'session_id': 'valid'}),
            ),
            patch.object(
                playback_store,
                'refresh_playback_session_ttl',
                AsyncMock(),
            ),
            patch.object(
                playback_selection,
                'select_session_playback',
                AsyncMock(return_value={'hls_url': '/invalid/path'}),
            ),
            patch.object(
                playback_selection,
                'wait_for_session_startup',
                AsyncMock(),
            ),
        ):
            with self.assertRaisesRegex(
                HTTPException,
                'invalid_media_playlist',
            ):
                await routers.stream_playback_session_playlist(
                    'valid',
                    request,
                    rds,
                )

        with self.assertRaisesRegex(HTTPException, 'label_required'):
            await streaming_api_service.negotiate_stream_playback(
                StreamPlaybackRequest(),
                username='alice',
                credentials=credentials,
                db=AsyncMock(),
                rds=rds,
            )

        with (
            patch.object(
                streaming_api_service,
                'authorise_label_access',
                AsyncMock(),
            ),
            patch.object(
                streaming_api_service,
                'normalise_overlay_mode',
                return_value='backend',
            ),
            patch.object(
                streaming_api_service,
                'normalise_label_language',
                return_value='xx',
            ),
            patch.object(
                playback_languages,
                '_allowed_overlay_languages',
                return_value={'en'},
            ),
        ):
            with self.assertRaisesRegex(HTTPException, 'unsupported_language'):
                await routers.get_streams_for_label_route(
                    'SiteA',
                    overlay='backend',
                    language='xx',
                    credentials=credentials,
                    db=AsyncMock(),
                    rds=rds,
                )

        db = AsyncMock()
        db.execute.side_effect = RuntimeError('database unavailable')
        with patch.object(
            streaming_api_service,
            'authorise_label_access',
            AsyncMock(),
        ):
            with self.assertRaisesRegex(RuntimeError, 'database unavailable'):
                await routers.get_streams_for_label_route(
                    'SiteA',
                    credentials=credentials,
                    db=db,
                    rds=rds,
                )

    async def test_backend_overlay_metadata_builds_demand_and_ready_payload(
        self,
    ) -> None:
        """Backend overlays create SSE demand metadata for the requested
        language."""

        events = _backend_overlay_metadata_builds_demand_events

        db = AsyncMock()
        rds = spy_redis()
        request = MagicMock()
        credentials = SimpleNamespace(subject={'username': 'alice'})
        generator = MagicMock(return_value=events())
        with (
            patch.object(
                streaming_metadata_service,
                'authorise_label_access',
                AsyncMock(),
            ),
            patch.object(
                streaming_metadata_service,
                'metadata_stream_generator',
                generator,
            ),
        ):
            response = await routers.metadata_stream_id(
                request,
                'SiteA',
                'Q2FtMQ==',
                overlay='backend',
                language='en',
                credentials=credentials,
                db=db,
                rds=rds,
            )

        self.assertEqual(response.media_type, 'text/event-stream')
        db.close.assert_awaited_once()
        lease = generator.call_args.args[1].overlay
        self.assertEqual(lease.ready_payload['profile'], 'overlay')
        self.assertEqual(lease.ready_payload['language'], 'en')
        self.assertIn('media_overlay_demand:', lease.demand_key)

    async def test_remaining_playback_session_paths_fail_closed_or_refresh(
        self,
    ) -> None:
        """Session scans skip stale rows and valid updates remove old
        indexes."""
        base_path = playback_media_paths.build_media_path('SiteA', 'Camera1')
        self.assertFalse(
            playback_hls.opaque_media_session_allows_path(
                {
                    'site': 'SiteA',
                    'camera': 'Camera1',
                    'cameras': ['Camera1'],
                    'quality': 'detail',
                    'profile': 'unknown',
                },
                base_path,
            ),
        )

        rds = spy_redis()
        existing = playback_store.new_playback_session(
            session_id='existing',
            username='alice',
            label='SiteA',
            stream_name='Camera1',
            profile='overlay',
            rendition='detail',
            language='en',
        )
        await rds.set(
            playback_store.playback_session_key('existing'),
            json.dumps(existing).encode(),
        )
        await playback_store.register_playback_session_demand(rds, existing)
        plan = playback_store.PlaybackSessionPlan(
            'SiteA',
            'Camera1',
            'clean',
            'detail',
            None,
            'existing',
        )
        session = await playback_store.create_or_update_playback_session(
            rds,
            'alice',
            plan,
        )
        self.assertEqual(session['session_id'], 'existing')
        self.assertNotEqual(session['revision'], existing['revision'])
        self.assertEqual(
            await rds.zcard(
                playback_store.playback_demand_session_key(
                    base_path,
                    'overlay',
                    'en',
                ),
            ),
            0,
        )

        rds.zremrangebyscore = AsyncMock()
        rds.zcard = AsyncMock(return_value=1)
        self.assertTrue(
            await playback_store.has_other_playback_session(
                rds,
                base_media_path=base_path,
                profile='clean',
            ),
        )
        rds.zcard.assert_awaited_once_with(
            playback_store.playback_demand_session_key(
                base_path,
                'clean',
            ),
        )

        with (
            patch.object(
                streaming_metadata_service,
                'authorise_label_access',
                AsyncMock(),
            ),
            patch.object(
                streaming_metadata_service,
                'normalise_label_language',
                return_value='xx',
            ),
            patch.object(
                playback_languages,
                '_allowed_overlay_languages',
                return_value={'en'},
            ),
        ):
            with self.assertRaisesRegex(HTTPException, 'unsupported_language'):
                await routers.metadata_stream_id(
                    MagicMock(),
                    'SiteA',
                    'Q2FtMQ==',
                    overlay='backend',
                    language='xx',
                    credentials=SimpleNamespace(subject={'username': 'alice'}),
                    db=AsyncMock(),
                    rds=AsyncMock(),
                )


if __name__ == '__main__':
    unittest.main()


def test_upstream_missing_playlist_is_retryable_startup():
    """A missing MediaMTX path must not look like a lost application
    session.
    """

    check = _upstream_missing_playlist_is_retryable_check

    asyncio.run(check())
