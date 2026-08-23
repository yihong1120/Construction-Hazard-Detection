from __future__ import annotations

import unittest
from datetime import datetime
from datetime import timezone
from types import SimpleNamespace
from typing import Any
from typing import cast
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import HTTPException

from examples.auth.jwt_config import JwtAuthorizationCredentials
from examples.db_management.schemas.auth import AccessTokenSubject
from examples.streaming_web import playback_hls
from examples.streaming_web import playback_service
from examples.streaming_web import routers
from examples.streaming_web import streaming_api_service
from examples.streaming_web import streaming_metadata_service
from examples.streaming_web.media_paths import build_clean_demand_key
from examples.streaming_web.schemas import StreamPlaybackBatchRequest
from examples.streaming_web.schemas import StreamPlaybackRequest


def _credentials(
    subject: dict[str, object],
) -> JwtAuthorizationCredentials:
    """Create incomplete credentials for authentication guard tests."""
    return JwtAuthorizationCredentials(
        subject=cast(AccessTokenSubject, subject),
    )


class TestStreamingRouterHlsHelpers(unittest.IsolatedAsyncioTestCase):

    """Provide TestStreamingRouterHlsHelpers.
    """

    def test_overlay_and_media_session_helpers_cover_invalid_inputs(
        self,
    ) -> None:
        """Test overlay and media session helpers cover invalid inputs.
        """
        options = playback_service._overlay_language_options(('en',))
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
            playback_service._normalise_playback_profile('webrtc')
        self.assertEqual(raised.exception.detail, 'unsupported_profile')
        with self.assertRaises(HTTPException) as raised:
            playback_service._normalise_playback_rendition('thumbnail')
        self.assertEqual(raised.exception.detail, 'unsupported_rendition')

    def test_hls_uri_and_playlist_rewriting_preserve_media_query_and_auth(
        self,
    ) -> None:
        """Test hls uri and playlist rewriting preserve media query and auth.
        """
        media_path = 'hazard_site_camera'
        auth_query = 'mt=opaque-token'
        self.assertEqual(
            playback_hls.rewrite_hls_uri('segment.ts', media_path, ''),
            'segment.ts',
        )
        self.assertIn(
            '/hazard/media/hazard_site_camera/segment.ts?mt=opaque-token',
            playback_hls.rewrite_hls_uri(
                'segment.ts', media_path, auth_query,
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
                '/hazard/media/hazard_site_camera/absolute.ts?foo=bar&'
                'mt=opaque-token'
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
            'URI="/hazard/media/hazard_site_camera/key.bin?mt=opaque-token"',
            rewritten,
        )
        self.assertIn(
            '/hazard/media/hazard_site_camera/segment.ts?mt=opaque-token',
            rewritten,
        )

    def test_media_path_and_session_payload_helpers_handle_bad_values(
        self,
    ) -> None:
        """Test media path and session payload helpers handle bad values.
        """
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
            playback_service._decode_playback_session_payload(None),
        )
        with self.assertRaises(ValueError):
            playback_service._decode_playback_session_payload(b'not-json')
        self.assertEqual(
            playback_service._decode_playback_session_payload(
                b'{"profile":"clean"}',
            ),
            {'profile': 'clean'},
        )
        with self.assertRaises(TypeError):
            playback_service._decode_playback_session_payload(
                cast(bytes, 123),
            )

    def test_media_session_scope_and_selected_path_helpers(self) -> None:
        """Test media session scope and selected path helpers.
        """
        base_path = playback_service.build_media_path('SiteA', 'Camera1')
        preview_path = playback_service.build_preview_media_path(base_path)
        overlay_path = playback_service.build_annotated_media_path(
            base_path, 'en',
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
            playback_service._session_selected_media_path(
                cast(playback_service.PlaybackSession, {}),
            )
        self.assertEqual(
            playback_service._session_selected_media_path(
                cast(
                    playback_service.PlaybackSession,
                    {'base_media_path': base_path, 'profile': 'clean'},
                ),
            ),
            base_path,
        )
        self.assertEqual(
            playback_service._session_selected_media_path(
                cast(
                    playback_service.PlaybackSession,
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
        """Test demand and media session indexes propagate failures.
        """
        base_path = playback_service.build_media_path('SiteA', 'Camera1')
        overlay_path = playback_service.build_annotated_media_path(
            base_path, 'en',
        )
        rds = AsyncMock()
        pipeline = MagicMock()
        pipeline.__aenter__ = AsyncMock(return_value=pipeline)
        pipeline.__aexit__ = AsyncMock(return_value=None)
        pipeline.execute = AsyncMock()
        rds.pipeline = MagicMock(return_value=pipeline)
        with patch.object(
            playback_service, 'touch_overlay_demand', AsyncMock(),
        ) as touch_overlay:
            await playback_service._touch_media_demand_from_media_path(
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
            playback_service,
            'touch_clean_demand',
            AsyncMock(side_effect=RuntimeError('redis offline')),
        ):
            with self.assertRaisesRegex(RuntimeError, 'redis offline'):
                await playback_service._touch_media_demand_from_media_path(
                    rds,
                    base_path,
                )

        with self.assertRaises(KeyError):
            await playback_service._delete_playback_session_media_indexes(
                rds,
                cast(playback_service.PlaybackSession, {}),
            )
        with self.assertRaises(KeyError):
            await playback_service._register_playback_session_media_path(
                rds,
                cast(playback_service.PlaybackSession, {}),
                base_path,
            )

        rds.expire.side_effect = RuntimeError('redis offline')
        with self.assertRaisesRegex(RuntimeError, 'redis offline'):
            await playback_service._refresh_playback_session_ttl(
                rds,
                'session-1',
            )

    async def test_refreshes_and_prunes_invalid_media_session_indexes(
        self,
    ) -> None:
        """Test refreshes and prunes invalid media session indexes.
        """
        rds = MagicMock()
        pipeline = MagicMock()
        pipeline.__aenter__ = AsyncMock(return_value=pipeline)
        pipeline.__aexit__ = AsyncMock(return_value=None)
        pipeline.execute = AsyncMock()
        rds.pipeline.return_value = pipeline
        rds.set = AsyncMock(return_value=True)
        rds.zrangebyscore = AsyncMock(return_value=[b'missing', b'mismatch'])
        rds.mget = AsyncMock(
            return_value=[
                None,
                b'{"profile":"clean","base_media_path":"other-path"}',
            ],
        )
        await playback_service._refresh_playback_sessions_for_media_path(
            rds,
            'hazard_site_cam',
        )
        pipeline.zrem.assert_called_once_with(
            'stream_playback_media_session:hazard_site_cam',
            'missing',
            'mismatch',
        )

        rds.set = AsyncMock(return_value=True)
        rds.zrangebyscore = AsyncMock(side_effect=RuntimeError('zset failed'))
        with self.assertRaisesRegex(RuntimeError, 'zset failed'):
            await playback_service._refresh_playback_sessions_for_media_path(
                rds,
                'hazard_site_cam',
            )

    async def test_playback_session_errors_and_startup_input_validation(
        self,
    ) -> None:
        """Test playback session errors and startup input validation.
        """
        rds = AsyncMock()
        with patch.object(
            playback_service, '_load_playback_session', AsyncMock(
                return_value=None,
            ),
        ):
            with self.assertRaises(HTTPException) as raised:
                await playback_service._create_or_update_playback_session(
                    rds,
                    session_id='missing',
                    username='alice',
                    label='SiteA',
                    stream_name='Camera1',
                    profile='clean',
                    rendition='detail',
                    language=None,
                )
        self.assertEqual(raised.exception.detail, 'session_not_found')

        with patch.object(
            playback_service,
            '_load_playback_session',
            AsyncMock(return_value={'username': 'bob'}),
        ):
            with self.assertRaises(HTTPException) as raised:
                await playback_service._create_or_update_playback_session(
                    rds,
                    session_id='forbidden',
                    username='alice',
                    label='SiteA',
                    stream_name='Camera1',
                    profile='clean',
                    rendition='detail',
                    language=None,
                )
        self.assertEqual(raised.exception.detail, 'session_forbidden')

        with patch.object(
            playback_service, 'STREAM_PLAYBACK_STARTUP_WAIT_SECONDS', 0.0,
        ):
            await playback_service._wait_for_session_startup(
                cast(playback_service.PlaybackSession, {}),
            )
        with patch.object(
            playback_service, 'STREAM_PLAYBACK_STARTUP_WAIT_SECONDS', 1.0,
        ):
            with self.assertRaises(ValueError):
                await playback_service._wait_for_session_startup(
                    cast(
                        playback_service.PlaybackSession,
                        {'created_at': 'not-a-date'},
                    ),
                )

    async def test_label_access_rejects_inactive_users(self) -> None:
        """Test label access rejects inactive users.
        """
        credentials = _credentials({'username': 'alice'})
        inactive_user = SimpleNamespace(status='suspended')
        with patch.object(
            playback_hls,
            'load_user_access_context',
            AsyncMock(return_value=(inactive_user, ['SiteA'], 'user')),
        ):
            with self.assertRaises(HTTPException) as raised:
                await playback_hls.authorise_label_access(
                    credentials, AsyncMock(), 'SiteA',
                )
        self.assertEqual(raised.exception.detail, 'inactive_user')

    async def test_internal_hls_playlist_maps_network_status_and_empty_errors(
        self,
    ) -> None:
        """Test internal hls playlist maps network status and empty errors.
        """
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
                    'hazard_site_cam', media_query='',
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
                    'hazard_site_cam', media_query='quality=low',
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
                    'hazard_site_cam', media_query='',
                )
        self.assertEqual(raised.exception.detail, 'media_playlist_not_ready')

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
            await playback_service._has_other_playback_session(
                rds,
                base_media_path='hazard_site_cam',
                profile='overlay',
                language='en',
            ),
        )
        rds.zremrangebyscore.assert_awaited_once()
        rds.zcard.assert_awaited_once_with(
            playback_service._playback_demand_session_key(
                'hazard_site_cam',
                'overlay',
                'en',
            ),
        )

        created_at = datetime.now(timezone.utc).isoformat()
        with (
            patch.object(
                playback_service,
                'STREAM_PLAYBACK_STARTUP_WAIT_SECONDS',
                30.0,
            ),
            patch.object(
                playback_service.asyncio,
                'sleep',
                AsyncMock(),
            ) as sleep,
        ):
            await playback_service._wait_for_session_startup(
                cast(
                    playback_service.PlaybackSession,
                    {'created_at': created_at},
                ),
            )
        sleep.assert_awaited_once()

        request = SimpleNamespace(url=SimpleNamespace(query=''))
        with patch.object(
            playback_service,
            '_load_playback_session',
            AsyncMock(return_value=None),
        ):
            with self.assertRaisesRegex(HTTPException, 'session_not_found'):
                await routers.stream_playback_session_playlist(
                    'missing', request, rds,
                )

        with patch.object(
            playback_service,
            '_load_playback_session',
            AsyncMock(return_value={'session_id': 'present'}),
        ):
            with self.assertRaisesRegex(HTTPException, 'missing_media_token'):
                await routers.stream_playback_session_playlist(
                    'present', request, rds,
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
            HTTPException, 'stream_batch_limit_exceeded',
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
                request, credentials, AsyncMock(),
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
            playback_service,
            '_load_playback_session',
            AsyncMock(return_value=None),
        ):
            with self.assertRaisesRegex(HTTPException, 'session_not_found'):
                await routers.release_stream_playback(
                    StreamPlaybackRequest(session_id='missing'),
                    credentials,
                    rds,
                )

        with patch.object(
            playback_service,
            '_load_playback_session',
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
                playback_service,
                '_load_playback_session',
                AsyncMock(return_value=clean_session),
            ),
            patch.object(
                playback_service,
                '_delete_playback_session_media_indexes',
                AsyncMock(),
            ),
            patch.object(
                playback_service,
                '_has_other_playback_session',
                AsyncMock(return_value=False),
            ),
        ):
            response = await routers.release_stream_playback(
                StreamPlaybackRequest(session_id='clean'),
                credentials,
                rds,
            )
        self.assertEqual(response.status_code, 200)
        rds.delete.assert_any_await(
            build_clean_demand_key('hazard_site_camera'),
        )

        request = SimpleNamespace(url=SimpleNamespace(query='mt=token'))
        with (
            patch.object(
                playback_service,
                '_load_playback_session',
                AsyncMock(return_value={'session_id': 'valid'}),
            ),
            patch.object(
                playback_service,
                '_refresh_playback_session_ttl',
                AsyncMock(),
            ),
            patch.object(
                playback_service,
                '_select_session_playback',
                AsyncMock(return_value={'hls_url': '/invalid/path'}),
            ),
            patch.object(
                playback_service,
                '_wait_for_session_startup', AsyncMock(),
            ),
        ):
            with self.assertRaisesRegex(
                HTTPException, 'invalid_media_playlist',
            ):
                await routers.stream_playback_session_playlist(
                    'valid', request, rds,
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
                'authorise_label_access', AsyncMock(),
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
                playback_service,
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

        async def events() -> Any:
            """Perform events.

            Returns:
                The callable result.
            """
            yield b'data: {}\n\n'

        db = AsyncMock()
        rds = AsyncMock()
        request = MagicMock()
        credentials = SimpleNamespace(subject={'username': 'alice'})
        generator = MagicMock(return_value=events())
        with (
            patch.object(
                streaming_metadata_service,
                'authorise_label_access', AsyncMock(),
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
        kwargs = generator.call_args.kwargs
        self.assertEqual(kwargs['overlay_ready_payload']['profile'], 'overlay')
        self.assertEqual(kwargs['overlay_ready_payload']['language'], 'en')
        self.assertIn('media_overlay_demand:', kwargs['overlay_demand_key'])

    async def test_remaining_playback_session_paths_fail_closed_or_refresh(
        self,
    ) -> None:
        """Session scans skip stale rows and valid updates remove old
        indexes."""
        base_path = playback_service.build_media_path('SiteA', 'Camera1')
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

        rds = AsyncMock()
        existing = {'username': 'alice'}
        with (
            patch.object(
                playback_service,
                '_load_playback_session',
                AsyncMock(return_value=existing),
            ),
            patch.object(
                playback_service,
                '_delete_playback_session_media_indexes',
                AsyncMock(),
            ) as delete_indexes,
        ):
            session = await playback_service._create_or_update_playback_session(
                rds,
                session_id='existing',
                username='alice',
                label='SiteA',
                stream_name='Camera1',
                profile='clean',
                rendition='detail',
                language=None,
            )
        self.assertEqual(session['session_id'], 'existing')
        delete_indexes.assert_awaited_once_with(rds, existing)

        rds.zremrangebyscore = AsyncMock()
        rds.zcard = AsyncMock(return_value=1)
        self.assertTrue(
            await playback_service._has_other_playback_session(
                rds,
                base_media_path=base_path,
                profile='clean',
            ),
        )
        rds.zcard.assert_awaited_once_with(
            playback_service._playback_demand_session_key(
                base_path,
                'clean',
            ),
        )

        with (
            patch.object(
                streaming_metadata_service,
                'authorise_label_access', AsyncMock(),
            ),
            patch.object(
                streaming_metadata_service,
                'normalise_label_language',
                return_value='xx',
            ),
            patch.object(
                playback_service,
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
