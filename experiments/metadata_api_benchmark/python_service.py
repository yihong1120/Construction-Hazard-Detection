"""Run the Python frame-delivery path without application setup services."""
from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from urllib.parse import urlparse

import redis.asyncio as redis
import uvicorn
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import WebSocket
from fastapi.responses import StreamingResponse

from examples.streaming_web.metadata_fanout import MetadataFanout
from examples.streaming_web.metadata_keys import build_metadata_key
from examples.streaming_web.metadata_wire import build_metadata_payload
from examples.streaming_web.metadata_wire import encode_sse_event


def create_app(redis_url: str, streams: int, shared: bool) -> FastAPI:
    connection = redis.Redis.from_url(redis_url)
    fanout = MetadataFanout()
    subscribers = 0

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            async with fanout.lifespan():
                yield
        finally:
            await connection.aclose()

    app = FastAPI(lifespan=lifespan)

    @app.get('/health')
    async def health() -> dict[str, str | int]:
        return {'status': 'ok', 'subscribers': subscribers}

    @app.get('/sse/{camera}')
    async def sse(camera: int) -> StreamingResponse:
        if not 0 <= camera < streams:
            raise HTTPException(status_code=404)

        async def events() -> AsyncIterator[bytes]:
            nonlocal subscribers
            subscription = await fanout.subscribe(
                connection, build_metadata_key(
                    'benchmark', f'camera-{camera:03d}',
                ),
            )
            subscribers += 1
            try:
                yield b'retry: 15000\n: connected\n\n'
                while True:
                    try:
                        async with asyncio.timeout(15):
                            frame = await subscription.get()
                    except TimeoutError:
                        yield b': heartbeat\n\n'
                        continue
                    if isinstance(frame, Exception):
                        raise frame
                    if shared:
                        yield frame.sse
                    else:
                        payload = build_metadata_payload(frame.record)
                        payload['id'] = frame.record['id']
                        yield encode_sse_event(payload)
            finally:
                subscribers -= 1
                await subscription.close()

        return StreamingResponse(
            events(),
            media_type='text/event-stream',
            headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
        )

    @app.websocket('/ws/{camera}')
    async def websocket(socket: WebSocket, camera: int) -> None:
        nonlocal subscribers
        if not 0 <= camera < streams:
            await socket.close(code=1008)
            return
        subscription = await fanout.subscribe(
            connection, build_metadata_key(
                'benchmark', f'camera-{camera:03d}',
            ),
        )
        subscribers += 1
        await socket.accept()

        async def receive() -> None:
            while True:
                message = await socket.receive()
                if message['type'] == 'websocket.disconnect':
                    return
                if message.get('text') == '{"action":"ping"}':
                    await socket.send_text('{"action":"pong"}')

        async def send() -> None:
            while True:
                frame = await subscription.get()
                if isinstance(frame, Exception):
                    raise frame
                text = (
                    frame.websocket if shared else
                    json.dumps(
                        build_metadata_payload(frame.record),
                        separators=(',', ':'),
                        ensure_ascii=False,
                    )
                )
                await socket.send_text(text)

        tasks = [asyncio.create_task(receive()), asyncio.create_task(send())]
        try:
            done, _ = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                task.result()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            subscribers -= 1
            await subscription.close()

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--redis-url', required=True)
    parser.add_argument('--streams', type=int, default=32)
    parser.add_argument('--shared', action='store_true')
    args = parser.parse_args()
    address = urlparse(args.redis_url)
    if address.scheme != 'redis' or address.hostname != '127.0.0.1':
        parser.error('an isolated loopback Redis URL is required')
    uvicorn.run(
        create_app(args.redis_url, args.streams, args.shared),
        host='127.0.0.1',
        port=args.port,
        workers=1,
        loop='asyncio',
        http='h11',
        ws='wsproto',
        access_log=False,
        log_level='warning',
    )


if __name__ == '__main__':
    main()
