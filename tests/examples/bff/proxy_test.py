from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock
from unittest.mock import patch

from fastapi import HTTPException
from redis.exceptions import ConnectionError as RedisConnectionError

from examples.bff.proxy import _get_proxy_access_token_or_503
from examples.bff.proxy import _is_sse_request
from examples.bff.proxy import _log_sse_error_events
from examples.bff.proxy import resolve_upstream


class BffServicesTest(unittest.TestCase):
    def test_deployed_web_service_aliases_match_canonical_routes(self) -> None:
        aliases = {
            'detect': 'detection',
            'db_management': 'management',
            'file_manage': 'files',
            'streaming_web': 'streaming',
        }

        for legacy, canonical in aliases.items():
            with self.subTest(legacy=legacy):
                self.assertEqual(
                    resolve_upstream(f'{legacy}/resource'),
                    resolve_upstream(f'{canonical}/resource'),
                )

    def test_fcm_service_is_allowlisted(self) -> None:
        base, suffix = resolve_upstream('fcm/notifications/unread_count')

        self.assertTrue(base)
        self.assertEqual(suffix, 'notifications/unread_count')

    def test_metadata_path_uses_sse_streaming_proxy(self) -> None:
        request = type(
            'Request', (), {
                'headers': {'accept': '*/*'},
            },
        )()

        self.assertTrue(
            _is_sse_request(
                request,  # type: ignore[arg-type]
                'metadata/stream-id/site/cam',
            ),
        )

    def test_accept_event_stream_uses_sse_streaming_proxy(self) -> None:
        request = type(
            'Request', (), {
                'headers': {'accept': 'text/event-stream'},
            },
        )()

        self.assertTrue(
            _is_sse_request(
                request,  # type: ignore[arg-type]
                'streams/site',
            ),
        )

    def test_redis_sse_error_is_logged_with_upstream_status(self) -> None:
        buffer = bytearray()

        with self.assertLogs('uvicorn.error', level='WARNING') as logs:
            _log_sse_error_events(
                buffer,
                b'event: redis_error\ndata: {"source":"redis",',
                request_path='/bff/streaming_web/metadata/stream-id/site/cam',
                upstream_status=200,
            )
            self.assertEqual(
                buffer,
                b'event: redis_error\ndata: {"source":"redis",',
            )
            _log_sse_error_events(
                buffer,
                b'"code":"redis_unavailable"}\n\n',
                request_path='/bff/streaming_web/metadata/stream-id/site/cam',
                upstream_status=200,
            )

        self.assertIn('BFF SSE Redis error', logs.output[0])
        self.assertIn('upstream_status=200', logs.output[0])
        self.assertIn('code=redis_unavailable', logs.output[0])

    def test_bff_redis_connection_failure_returns_503(self) -> None:
        async def get_token() -> HTTPException:
            with (
                patch(
                    'examples.bff.proxy.get_proxy_access_token',
                    new=AsyncMock(side_effect=RedisConnectionError('down')),
                ),
                self.assertLogs('uvicorn.error', level='WARNING') as logs,
            ):
                with self.assertRaises(HTTPException) as raised:
                    await _get_proxy_access_token_or_503(
                        AsyncMock(),
                        'session-id',
                        '/bff/streaming_web/metadata/stream-id/site/cam',
                    )
            self.assertIn('BFF Redis connection failed', logs.output[0])
            return raised.exception

        error = asyncio.run(get_token())

        self.assertEqual(error.status_code, 503)
        self.assertEqual(error.detail, 'bff_redis_unavailable')


if __name__ == '__main__':
    unittest.main()
