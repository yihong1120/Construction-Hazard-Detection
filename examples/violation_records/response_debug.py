"""Opt-in logging of serialized JSON for selected violation GET requests."""
from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger('uvicorn.error')
_PATH = re.compile(
    r'^/violations/(\d+)(?:/(?:feedback-options|review-options))?$',
)
_MAX_BYTES = 2 * 1024 * 1024


class ViolationResponseDebug:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        match = _PATH.fullmatch(scope.get('path', ''))
        selected = {
            v.strip()
            for v in os.getenv('VIOLATION_DEBUG_RESPONSE_IDS', '').split(',')
            if v.strip()
        }
        if (
            scope['type'] != 'http'
            or scope.get('method') != 'GET'
            or not match
            or ('*' not in selected and match[1] not in selected)
        ):
            return await self.app(scope, receive, send)
        body = bytearray()
        status, is_json, truncated = 0, False, False

        async def capture(message):
            nonlocal status, is_json, truncated
            if message['type'] == 'http.response.start':
                status = message['status']
                content_type = dict(message.get('headers', [])).get(
                    b'content-type', b'',
                )
                is_json = (
                    content_type.split(b';')[0].strip() == b'application/json'
                )
            elif message['type'] == 'http.response.body' and is_json:
                chunk = message.get('body', b'')
                remaining = _MAX_BYTES - len(body)
                body.extend(chunk[:remaining])
                truncated = truncated or len(chunk) > remaining
            await send(message)
            if (
                message['type'] == 'http.response.body'
                and is_json
                and not message.get('more_body', False)
            ):
                logger.info(
                    '[violation-response] GET %s status=%s '
                    'truncated=%s body=%s',
                    scope['path'],
                    status,
                    truncated,
                    body.decode('utf-8', errors='replace'),
                )

        await self.app(scope, receive, capture)
