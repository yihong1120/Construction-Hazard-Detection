from __future__ import annotations

import asyncio
import json
import os
from functools import partial
from typing import Any
from unittest import IsolatedAsyncioTestCase
from unittest import TestCase
from unittest.mock import AsyncMock
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.testclient import TestClient

from examples.violation_records.response_debug import ViolationResponseDebug


def _logs_serialized_json_only_for_detail(id: int):
    return {'id': id, 'display_name': '人員', 'bbox': {'x': 0.2}}


def _logs_serialized_json_only_for_media():
    return Response(b'image', media_type='image/png')


def _new_middleware_loads_wildcard_or_detail(id: int):
    return {'id': id}


async def _capture_bound_does_not_truncate_app(
    body,
    /,
    scope,
    receive,
    send,
):
    await send(
        {
            'type': 'http.response.start',
            'status': 201,
            'headers': [
                (b'content-type', b'application/json; charset=utf-8'),
            ],
        },
    )
    await send(
        {
            'type': 'http.response.body',
            'body': body[:5],
            'more_body': True,
        },
    )
    await send({'type': 'http.response.body', 'body': body[5:]})


class _InterleavedResponsesKeepIndependentBodiesCallbacks:
    arrived: Any
    ready: Any

    async def _call_app(self, scope, receive, send):

        await send(
            {
                'type': 'http.response.start',
                'status': 200,
                'headers': [(b'content-type', b'application/json')],
            },
        )
        await send(
            {
                'type': 'http.response.body',
                'body': b'{"id":',
                'more_body': True,
            },
        )
        self.arrived += 1
        if self.arrived == 2:
            self.ready.set()
        await self.ready.wait()
        await send(
            {
                'type': 'http.response.body',
                'body': scope['path'].rsplit('/', 1)[1].encode() + b'}',
            },
        )


class ResponseDebugTest(TestCase):
    def test_logs_serialized_json_only_for_selected_paths(self):
        app = FastAPI()
        app.add_middleware(ViolationResponseDebug)

        app.get('/violations/{id}')(
            app.get('/violations/{id}/feedback-options')(
                app.get('/violations/{id}/review-options')(
                    _logs_serialized_json_only_for_detail,
                ),
            ),
        )

        app.get('/get_violation_image')(_logs_serialized_json_only_for_media)

        with (
            patch.dict(os.environ, VIOLATION_DEBUG_RESPONSE_IDS='155157'),
            patch(
                'examples.violation_records.response_debug.logger.info',
            ) as log,
            TestClient(app) as client,
        ):
            for suffix in ('', '/feedback-options', '/review-options'):
                response = client.get(
                    '/violations/155157' + suffix,
                    headers={'Authorization': 'Bearer secret'},
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(log.call_args.args[-1], response.text)
                self.assertNotIn('secret', str(log.call_args))
            self.assertEqual(log.call_count, 3)
            client.get('/violations/123')
            client.get('/get_violation_image')
            self.assertEqual(log.call_count, 3)

    def test_new_middleware_loads_wildcard_or_disabled_policy(self):
        for selected, expected in [('*', 9), ('', 0)]:
            app = FastAPI()
            app.add_middleware(ViolationResponseDebug)

            app.get('/violations/{id}')(
                app.get('/violations/{id}/feedback-options')(
                    app.get('/violations/{id}/review-options')(
                        _new_middleware_loads_wildcard_or_detail,
                    ),
                ),
            )

            with (
                patch.dict(os.environ, VIOLATION_DEBUG_RESPONSE_IDS=selected),
                patch(
                    'examples.violation_records.response_debug.logger.info',
                ) as log,
                TestClient(app) as client,
            ):
                for record_id in (155157, 155158, 999999):
                    for suffix in ('', '/feedback-options', '/review-options'):
                        client.get(f'/violations/{record_id}{suffix}')
                self.assertEqual(log.call_count, expected)


class ResponseCaptureIsolationTest(IsolatedAsyncioTestCase):
    async def test_interleaved_responses_keep_independent_bodies(self):
        _callbacks = _InterleavedResponsesKeepIndependentBodiesCallbacks()
        _callbacks.arrived = 0
        _callbacks.ready = asyncio.Event()

        app = _callbacks._call_app

        with (
            patch.dict(os.environ, VIOLATION_DEBUG_RESPONSE_IDS='*'),
            patch(
                'examples.violation_records.response_debug.logger.info',
            ) as log,
        ):
            middleware = ViolationResponseDebug(app)
            senders = [AsyncMock(), AsyncMock()]
            async with asyncio.timeout(1):
                await asyncio.gather(
                    *[
                        middleware(
                            {
                                'type': 'http',
                                'method': 'GET',
                                'path': f'/violations/{index}',
                            },
                            AsyncMock(),
                            sender,
                        )
                        for index, sender in enumerate(senders, 1)
                    ],
                )
        captured = {
            call.args[1]: json.loads(call.args[-1])
            for call in log.call_args_list
        }
        self.assertEqual(
            captured,
            {'/violations/1': {'id': 1}, '/violations/2': {'id': 2}},
        )
        for index, sender in enumerate(senders, 1):
            body = b''.join(
                call.args[0].get('body', b'')
                for call in sender.await_args_list
            )
            self.assertEqual(json.loads(body), {'id': index})

    async def test_capture_bound_does_not_truncate_the_response(self):
        body = b'{"message":"long response"}'

        app = partial(_capture_bound_does_not_truncate_app, body)

        with (
            patch.dict(os.environ, VIOLATION_DEBUG_RESPONSE_IDS='1'),
            patch('examples.violation_records.response_debug._MAX_BYTES', 8),
            patch(
                'examples.violation_records.response_debug.logger.info',
            ) as log,
        ):
            send = AsyncMock()
            await ViolationResponseDebug(app)(
                {'type': 'http', 'method': 'GET', 'path': '/violations/1'},
                AsyncMock(),
                send,
            )
        self.assertEqual(log.call_count, 1)
        self.assertEqual(log.call_args.args[2:4], (201, True))
        self.assertEqual(log.call_args.args[-1], body[:8].decode())
        forwarded = b''.join(
            call.args[0].get('body', b'') for call in send.await_args_list
        )
        self.assertEqual(forwarded, body)
