from __future__ import annotations

import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import HTTPException

from examples.auth.jwt_config import JwtAuthorizationCredentials
from examples.streaming_web import routers
from examples.streaming_web.schemas import StreamPlaybackBatchRequest
from examples.streaming_web.schemas import StreamPlaybackRequest


class TestStreamingRouterHlsHelpers(unittest.IsolatedAsyncioTestCase):
    def test_overlay_and_media_session_helpers_cover_invalid_inputs(
        self,
    ) -> None:
        options = routers._overlay_language_option_payloads(('en',))
        self.assertEqual(options[0]['code'], 'en')
        self.assertEqual(
            routers._media_session_demand_ttl(
                {'expires_at': 'bad'},
            ),
            routers.MEDIA_PUBLISHER_IDLE_GRACE_SECONDS,
        )
        self.assertIsNone(
            routers._media_hls_session_cookie(
                'hazard_site_cam',
                None,
            ),
        )
        self.assertIsNone(
            routers._media_hls_session_cookie(
                'hazard_site_cam',
                'invalid value',
            ),
        )

        with self.assertRaises(HTTPException) as raised:
            routers._normalise_playback_profile('webrtc')
        self.assertEqual(raised.exception.detail, 'unsupported_profile')
        with self.assertRaises(HTTPException) as raised:
            routers._normalise_playback_rendition('thumbnail')
        self.assertEqual(raised.exception.detail, 'unsupported_rendition')

    def test_hls_uri_and_playlist_rewriting_preserve_media_query_and_auth(
        self,
    ) -> None:
        media_path = 'hazard_site_camera'
        auth_query = 'mt=opaque-token'
        self.assertEqual(
            routers._rewrite_hls_uri('segment.ts', media_path, ''),
            'segment.ts',
        )
        self.assertIn(
            '/hazard/media/hazard_site_camera/segment.ts?mt=opaque-token',
            routers._rewrite_hls_uri('segment.ts', media_path, auth_query),
        )
        self.assertIn(
            '/hazard/media/hazard_site_camera/part.ts?mt=opaque-token',
            routers._rewrite_hls_uri(
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
            routers._rewrite_hls_uri(
                'https://media.example/other/absolute.ts?foo=bar',
                media_path,
                auth_query,
            ),
        )
        self.assertEqual(
            routers._rewrite_hls_uri(
                f"/hazard/media/{media_path}/already.ts",
                media_path,
                auth_query,
            ),
            f"/hazard/media/{media_path}/already.ts?mt=opaque-token",
        )

        playlist = '\n#EXT-X-KEY:METHOD=AES-128,URI="key.bin"\nsegment.ts\n'
        rewritten = routers._rewrite_hls_playlist_media_urls(
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
        self.assertEqual(
            routers._extract_media_path_from_uri('/not-media/path'),
            '',
        )
        self.assertEqual(
            routers._extract_media_path_from_uri(
                '/hazard/media/webrtc/hazard_site_cam/whep',
            ),
            'hazard_site_cam',
        )
        self.assertFalse(
            routers._media_path_matches_site(
                'hazard_other_cam',
                'site',
            ),
        )
        self.assertIsNone(routers._decode_playback_session_payload(None))
        self.assertIsNone(
            routers._decode_playback_session_payload(b'not-json'),
        )
        self.assertIsNone(
            routers._decode_playback_session_payload('["not", "a", "dict"]'),
        )
        self.assertEqual(
            routers._decode_playback_session_payload(b'{"profile":"clean"}'),
            {'profile': 'clean'},
        )
        self.assertIsNone(routers._decode_playback_session_payload(123))

    def test_media_session_scope_and_selected_path_helpers(self) -> None:
        base_path = routers.build_media_path('SiteA', 'Camera1')
        preview_path = routers.build_preview_media_path(base_path)
        overlay_path = routers.build_annotated_media_path(base_path, 'en')
        self.assertTrue(
            routers._opaque_media_session_allows_path(
                {
                    'site': 'SiteA',
                    'camera': 'Camera1',
                    'quality': 'detail',
                    'profile': 'clean',
                },
                base_path,
            ),
        )

        self.assertTrue(
            routers._opaque_media_session_allows_path(
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
            routers._opaque_media_session_allows_path(
                {
                    'site': 'SiteA',
                    'camera': 'Camera1',
                    'quality': 'detail',
                    'profile': 'overlay',
                },
                overlay_path,
            ),
        )
        self.assertFalse(
            routers._opaque_media_session_allows_path(
                {'site': 'SiteA', 'quality': 'detail', 'profile': 'unknown'},
                base_path,
            ),
        )
        self.assertFalse(
            routers._opaque_media_session_allows_path(
                {'site': 'SiteA', 'quality': 'invalid', 'camera': 'Camera1'},
                base_path,
            ),
        )
        self.assertIsNone(routers._session_selected_media_path({}))
        self.assertEqual(
            routers._session_selected_media_path(
                {'base_media_path': base_path},
            ),
            base_path,
        )
        self.assertEqual(
            routers._session_selected_media_path(
                {
                    'base_media_path': base_path,
                    'profile': 'overlay',
                    'overlay_media_path': overlay_path,
                },
            ),
            overlay_path,
        )

    async def test_restore_playback_session_rejects_invalid_capability_data(
        self,
    ) -> None:
        """A media token can restore only a complete matching session scope."""
        request = SimpleNamespace(query_params={}, headers={})
        rds = AsyncMock()
        descriptor: dict[str, object] = {
            'label': 'SiteA',
            'stream_name': 'Camera1',
            'profile': 'clean',
            'rendition': 'detail',
        }
        base: dict[str, object] = {
            'username': 'alice',
            'site': 'SiteA',
            'cameras': ['Camera1'],
            'profile': 'clean',
            'quality': 'detail',
            'playback_sessions': {'session': descriptor},
        }
        with patch.object(
            routers,
            'get_media_session',
            new=AsyncMock(return_value=None),
        ) as get_media:
            assert await routers._restore_playback_session(
                rds,
                'session',
                request,
            ) is None

            get_media.return_value = {'username': 'alice'}
            assert await routers._restore_playback_session(
                rds,
                'session',
                request,
            ) is None

            get_media.return_value = {'playback_sessions': {'session': []}}
            assert await routers._restore_playback_session(
                rds,
                'session',
                request,
            ) is None

            for field in (
                'label',
                'stream_name',
                'profile',
                'rendition',
            ):
                malformed = {
                    **base,
                    'playback_sessions': {
                        'session': {
                            **descriptor,
                            field: '',
                        },
                    },
                }
                get_media.return_value = malformed
                assert await routers._restore_playback_session(
                    rds,
                    'session',
                    request,
                ) is None

            wrong_scope = {
                **base,
                'playback_sessions': {
                    'session': {
                        **descriptor,
                        'label': 'OtherSite',
                    },
                },
            }
            get_media.return_value = wrong_scope
            assert await routers._restore_playback_session(
                rds,
                'session',
                request,
            ) is None

            get_media.return_value = {**base, 'quality': 'unexpected'}
            assert await routers._restore_playback_session(
                rds,
                'session',
                request,
            ) is None

            get_media.return_value = {**base, 'username': ''}
            assert await routers._restore_playback_session(
                rds,
                'session',
                request,
            ) is None

            invalid_language = {
                **base,
                'profile': 'overlay',
                'playback_sessions': {
                    'session': {
                        **descriptor,
                        'profile': 'overlay',
                        'language': 7,
                    },
                },
            }
            get_media.return_value = invalid_language
            assert await routers._restore_playback_session(
                rds,
                'session',
                request,
            ) is None

            preview = {
                **base,
                'quality': 'preview',
                'playback_sessions': {
                    'session': {
                        **descriptor,
                        'rendition': 'preview',
                    },
                },
            }
            get_media.return_value = preview
            restored = await routers._restore_playback_session(
                rds,
                'session',
                request,
            )
            assert restored is not None
            self.assertTrue(
                str(restored['base_media_path']).endswith('_preview'),
            )

            with patch.object(
                routers,
                '_opaque_media_session_allows_path',
                return_value=False,
            ):
                get_media.return_value = base
                assert await routers._restore_playback_session(
                    rds,
                    'session',
                    request,
                ) is None

    async def test_demand_and_media_session_indexes_tolerate_failures(
        self,
    ) -> None:
        base_path = routers.build_media_path('SiteA', 'Camera1')
        overlay_path = routers.build_annotated_media_path(base_path, 'en')
        rds = AsyncMock()
        with patch.object(
            routers, '_touch_overlay_demand', AsyncMock(),
        ) as touch_overlay:
            await routers._touch_media_demand_from_media_path(
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
            routers,
            '_touch_clean_demand',
            AsyncMock(side_effect=RuntimeError('redis offline')),
        ):
            await routers._touch_media_demand_from_media_path(rds, base_path)

        await routers._delete_playback_session_media_indexes(rds, {})
        await routers._register_playback_session_media_path(rds, {}, base_path)

        rds.expire.side_effect = RuntimeError('redis offline')
        await routers._refresh_playback_session_ttl(rds, 'session-1')

    async def test_refreshes_and_prunes_invalid_media_session_indexes(
        self,
    ) -> None:
        async def keys():
            yield b'stream_playback_media_session:hazard_site_cam:missing'
            yield b'stream_playback_media_session:hazard_site_cam:mismatch'

        rds = MagicMock()
        rds.scan_iter.return_value = keys()
        rds.delete = AsyncMock()
        rds.expire = AsyncMock()
        with patch.object(
            routers,
            '_load_playback_session',
            AsyncMock(side_effect=[None, {'base_media_path': 'other-path'}]),
        ):
            await routers._refresh_playback_sessions_for_media_path(
                rds,
                'hazard_site_cam',
            )
        self.assertEqual(rds.delete.await_count, 2)

        async def failing_keys():
            raise RuntimeError('scan failed')
            yield b''

        rds.scan_iter.return_value = failing_keys()
        await routers._refresh_playback_sessions_for_media_path(
            rds, 'hazard_site_cam',
        )

    async def test_playback_session_errors_and_startup_input_validation(
        self,
    ) -> None:
        rds = AsyncMock()
        with patch.object(
            routers, '_load_playback_session', AsyncMock(return_value=None),
        ):
            with self.assertRaises(HTTPException) as raised:
                await routers._create_or_update_playback_session(
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
            routers,
            '_load_playback_session',
            AsyncMock(return_value={'username': 'bob'}),
        ):
            with self.assertRaises(HTTPException) as raised:
                await routers._create_or_update_playback_session(
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
            routers, 'STREAM_PLAYBACK_STARTUP_WAIT_SECONDS', 0.0,
        ):
            await routers._wait_for_session_startup({})
        with patch.object(
            routers, 'STREAM_PLAYBACK_STARTUP_WAIT_SECONDS', 1.0,
        ):
            await routers._wait_for_session_startup(
                {'created_at': 'not-a-date'},
            )

    async def test_label_access_rejects_inactive_users(self) -> None:
        credentials = JwtAuthorizationCredentials(
            subject={'username': 'alice'},
        )
        inactive_user = SimpleNamespace(status='suspended')
        with patch.object(
            routers,
            'get_user_and_sites',
            AsyncMock(return_value=(inactive_user, ['SiteA'], 'user')),
        ):
            with self.assertRaises(HTTPException) as raised:
                await routers._authorise_label_access(
                    credentials, AsyncMock(), 'SiteA',
                )
        self.assertEqual(raised.exception.detail, 'inactive_user')

    async def test_internal_hls_playlist_maps_network_status_and_empty_errors(
        self,
    ) -> None:
        context = MagicMock()
        client = AsyncMock()
        context.__aenter__ = AsyncMock(return_value=client)
        context.__aexit__ = AsyncMock(return_value=None)

        client.get.side_effect = routers.httpx.TimeoutException('timed out')
        with patch.object(routers.httpx, 'AsyncClient', return_value=context):
            with self.assertRaises(HTTPException) as raised:
                await routers._fetch_internal_hls_playlist(
                    'hazard_site_cam', media_query='',
                )
        self.assertEqual(raised.exception.status_code, 502)

        client.get.side_effect = None
        client.get.return_value = SimpleNamespace(
            status_code=503,
            text='unavailable',
            cookies={},
        )
        with patch.object(routers.httpx, 'AsyncClient', return_value=context):
            with self.assertRaises(HTTPException) as raised:
                await routers._fetch_internal_hls_playlist(
                    'hazard_site_cam', media_query='quality=low',
                )
        self.assertEqual(raised.exception.status_code, 503)

        client.get.return_value = SimpleNamespace(
            status_code=200,
            text='  ',
            cookies={},
        )
        with patch.object(routers.httpx, 'AsyncClient', return_value=context):
            with self.assertRaises(HTTPException) as raised:
                await routers._fetch_internal_hls_playlist(
                    'hazard_site_cam', media_query='',
                )
        self.assertEqual(raised.exception.detail, 'media_playlist_not_ready')

    async def test_session_scan_startup_and_playlist_input_guards(
        self,
    ) -> None:
        """Playback helpers fail closed when sessions or media auth are
        invalid."""

        async def matching_keys():
            yield b'stream_playback_session:other'

        rds = SimpleNamespace(
            scan_iter=lambda **_kwargs: matching_keys(),
            get=AsyncMock(
                return_value=(
                    b'{"profile":"overlay","base_media_path":'
                    b'"hazard_site_cam",'
                    b'"language":"en"}'
                ),
            ),
        )
        self.assertTrue(
            await routers._has_other_overlay_sessions(
                rds,
                released_session_id='released',
                base_media_path='hazard_site_cam',
                language='en',
            ),
        )

        async def failing_keys():
            raise RuntimeError('redis unavailable')
            yield b''

        rds.scan_iter = lambda **_kwargs: failing_keys()
        with patch('builtins.print'):
            self.assertTrue(
                await routers._has_other_clean_sessions(
                    rds,
                    released_session_id='released',
                    base_media_path='hazard_site_cam',
                ),
            )

        created_at = datetime.now().replace(tzinfo=None).isoformat()
        with (
            patch.object(
                routers,
                'STREAM_PLAYBACK_STARTUP_WAIT_SECONDS',
                30.0,
            ),
            patch.object(routers.asyncio, 'sleep', AsyncMock()) as sleep,
        ):
            await routers._wait_for_session_startup({'created_at': created_at})
        sleep.assert_awaited_once()

        request = SimpleNamespace(url=SimpleNamespace(query=''))
        with patch.object(
            routers,
            '_load_playback_session',
            AsyncMock(return_value=None),
        ):
            with self.assertRaisesRegex(HTTPException, 'session_not_found'):
                await routers.stream_playback_session_playlist(
                    'missing', request, rds,
                )

        with patch.object(
            routers,
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
        legacy = SimpleNamespace(__fields_set__={'profile'})
        self.assertTrue(routers._model_field_was_set(legacy, 'profile'))
        self.assertFalse(routers._model_field_was_set(legacy, 'language'))

        with self.assertRaisesRegex(HTTPException, 'label_required'):
            await routers._build_batch_playback_requests(
                StreamPlaybackBatchRequest(),
                AsyncMock(),
            )

        oversized = [
            StreamPlaybackRequest(key=f"Camera{index}")
            for index in range(routers.MAX_STREAM_PLAYBACK_BATCH_STREAMS + 1)
        ]
        with self.assertRaisesRegex(
            HTTPException, 'stream_batch_limit_exceeded',
        ):
            routers._enforce_stream_playback_batch_limit(oversized)

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
        credentials = JwtAuthorizationCredentials(
            subject={'username': 'alice'},
        )
        rds = AsyncMock()
        with patch.object(
            routers,
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
            routers,
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
                routers,
                '_load_playback_session',
                AsyncMock(return_value=clean_session),
            ),
            patch.object(
                routers,
                '_delete_playback_session_media_indexes',
                AsyncMock(),
            ),
            patch.object(
                routers,
                '_has_other_clean_sessions',
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
            routers.build_clean_demand_key('hazard_site_camera'),
        )

        request = SimpleNamespace(url=SimpleNamespace(query='mt=token'))
        with (
            patch.object(
                routers,
                '_load_playback_session',
                AsyncMock(return_value={'session_id': 'valid'}),
            ),
            patch.object(
                routers,
                '_refresh_playback_session_ttl',
                AsyncMock(),
            ),
            patch.object(
                routers,
                '_select_session_playback',
                AsyncMock(return_value={'hls_url': '/invalid/path'}),
            ),
            patch.object(routers, '_wait_for_session_startup', AsyncMock()),
        ):
            with self.assertRaisesRegex(
                HTTPException, 'invalid_media_playlist',
            ):
                await routers.stream_playback_session_playlist(
                    'valid', request, rds,
                )

        with self.assertRaisesRegex(HTTPException, 'label_required'):
            await routers._negotiate_stream_playback(
                StreamPlaybackRequest(),
                username='alice',
                credentials=credentials,
                db=AsyncMock(),
                rds=rds,
            )

        with (
            patch.object(routers, '_authorise_label_access', AsyncMock()),
            patch.object(
                routers,
                'normalise_overlay_mode',
                return_value='backend',
            ),
            patch.object(
                routers,
                'normalise_label_language',
                return_value='xx',
            ),
            patch.object(
                routers,
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
        with patch.object(routers, '_authorise_label_access', AsyncMock()):
            response = await routers.get_streams_for_label_route(
                'SiteA',
                credentials=credentials,
                db=db,
                rds=rds,
            )
        self.assertEqual(response.body, b'{"streams":[]}')

    async def test_backend_overlay_metadata_builds_demand_and_ready_payload(
        self,
    ) -> None:
        """Backend overlays create SSE demand metadata for the requested
        language."""

        async def events():
            yield b'data: {}\n\n'

        db = AsyncMock()
        rds = AsyncMock()
        request = MagicMock()
        credentials = SimpleNamespace(subject={'username': 'alice'})
        generator = MagicMock(return_value=events())
        with (
            patch.object(routers, '_authorise_label_access', AsyncMock()),
            patch.object(routers, 'metadata_stream_generator', generator),
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
        base_path = routers.build_media_path('SiteA', 'Camera1')
        self.assertFalse(
            routers._opaque_media_session_allows_path(
                {
                    'site': 'SiteA',
                    'camera': 'Camera1',
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
                routers,
                '_load_playback_session',
                AsyncMock(return_value=existing),
            ),
            patch.object(
                routers,
                '_delete_playback_session_media_indexes',
                AsyncMock(),
            ) as delete_indexes,
        ):
            session = await routers._create_or_update_playback_session(
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

        async def stale_keys():
            yield routers._playback_session_key('released').encode()
            yield b'stream_playback_session:stale'

        stale_rds = SimpleNamespace(
            scan_iter=lambda **_kwargs: stale_keys(),
            get=AsyncMock(return_value=b'not-json'),
        )
        self.assertFalse(
            await routers._has_other_overlay_sessions(
                stale_rds,
                released_session_id='released',
                base_media_path=base_path,
                language='en',
            ),
        )

        async def clean_keys():
            yield b'stream_playback_session:clean'

        rds.scan_iter = lambda **_kwargs: clean_keys()
        rds.get = AsyncMock(
            return_value=(
                b'{"profile":"clean","base_media_path":"'
                + base_path.encode()
                + b'"}'
            ),
        )
        self.assertTrue(
            await routers._has_other_clean_sessions(
                rds,
                released_session_id='released',
                base_media_path=base_path,
            ),
        )

        async def stale_clean_keys():
            yield routers._playback_session_key('released').encode()
            yield b'stream_playback_session:stale'

        rds.scan_iter = lambda **_kwargs: stale_clean_keys()
        rds.get = AsyncMock(return_value=b'not-json')
        self.assertFalse(
            await routers._has_other_clean_sessions(
                rds,
                released_session_id='released',
                base_media_path=base_path,
            ),
        )

        async def failing_keys():
            raise RuntimeError('redis unavailable')
            yield b''

        rds.scan_iter = lambda **_kwargs: failing_keys()
        with patch('builtins.print'):
            self.assertTrue(
                await routers._has_other_overlay_sessions(
                    rds,
                    released_session_id='released',
                    base_media_path=base_path,
                    language='en',
                ),
            )

        with (
            patch.object(routers, '_authorise_label_access', AsyncMock()),
            patch.object(
                routers,
                'normalise_label_language',
                return_value='xx',
            ),
            patch.object(
                routers,
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
