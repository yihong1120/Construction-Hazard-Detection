from __future__ import annotations

import logging
import re

from starlette.types import Message
from starlette.types import Send

from examples.violation_records.debug_settings import ResponseDebugSettings

logger = logging.getLogger('uvicorn.error')
_PATH = re.compile(
    r'^/violations/(\d+)(?:/(?:feedback-options|review-options))?$',
)
_MAX_BYTES = 2 * 1024 * 1024


class ViolationResponseDebug:
    def __init__(self, app):
        self.app = app
        self.selected = ResponseDebugSettings().selected

    async def __call__(self, scope, receive, send):
        match = _PATH.fullmatch(scope.get('path', ''))
        selected = self.selected
        if (
            scope['type'] != 'http'
            or scope.get('method') != 'GET'
            or not match
            or ('*' not in selected and match[1] not in selected)
        ):
            return await self.app(scope, receive, send)

        await self.app(scope, receive, _ResponseCapture(send, scope['path']))


class _ResponseCapture:
    def __init__(self, send: Send, path: str):
        self.send = send
        self.path = path
        self.body = bytearray()
        self.status = 0
        self.is_json = False
        self.truncated = False

    async def __call__(self, message: Message) -> None:
        if message['type'] == 'http.response.start':
            self.status = message['status']
            content_type = dict(message.get('headers', [])).get(
                b'content-type',
                b'',
            )
            self.is_json = (
                content_type.split(b';')[0].strip() == b'application/json'
            )
        elif message['type'] == 'http.response.body' and self.is_json:
            chunk = message.get('body', b'')
            remaining = _MAX_BYTES - len(self.body)
            self.body.extend(chunk[:remaining])
            self.truncated = self.truncated or len(chunk) > remaining
        await self.send(message)
        if (
            message['type'] == 'http.response.body'
            and self.is_json
            and not message.get('more_body', False)
        ):
            logger.info(
                '[violation-response] GET %s status=%s '
                'truncated=%s body=%s',
                self.path,
                self.status,
                self.truncated,
                self.body.decode('utf-8', errors='replace'),
            )
