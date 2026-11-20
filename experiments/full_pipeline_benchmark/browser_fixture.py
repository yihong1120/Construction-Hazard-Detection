"""Loopback-only synthetic playback bridge for the real browser experiment."""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import uuid
from pathlib import Path
from urllib.parse import urljoin
from urllib.parse import urlparse

from fastapi import Request


def main():
    import httpx
    import redis.asyncio as redis
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, Response, StreamingResponse
    from fastapi.staticfiles import StaticFiles

    from examples.streaming_web.media_paths import build_clean_demand_key
    from examples.streaming_web.media_paths import build_overlay_demand_key
    from examples.streaming_web.metadata_keys import build_metadata_key
    from examples.streaming_web.metadata_wire import build_metadata_payload
    from examples.streaming_web.redis_service import (
        fetch_latest_metadata_for_key,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--media-base', required=True)
    parser.add_argument('--overlay', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    assert not config['redis'].get('prefix')
    address = os.environ[config['redis']['url_env']]
    assert urlparse(address).hostname in ('127.0.0.1', 'localhost')
    db = redis.Redis.from_url(address)
    cameras = {s['id']: s for s in config['streams'][:4]}
    assets = Path(__file__).resolve().parents[2] / 'native/pipeline'
    sessions = {}

    async def renew():
        while True:
            for stream in cameras.values():
                path = urlparse(stream['output']['clean']).path.lstrip('/')
                await db.set(build_clean_demand_key(path), '1', ex=10)
                await db.set(build_overlay_demand_key(path, 'en'), '1', ex=10)
            await asyncio.sleep(2)

    @contextlib.asynccontextmanager
    async def lifespan(_):
        task = asyncio.create_task(renew())
        yield
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await db.aclose()

    app = FastAPI(lifespan=lifespan)
    app.mount('/web', StaticFiles(directory=assets / 'web'))
    app.mount('/assets', StaticFiles(directory=assets / 'assets'))

    @app.get('/')
    async def page():
        text = Path(__file__).with_name('browser.html').read_text()
        return HTMLResponse(
            text.replace(
                '__SETTINGS__', json.dumps({
                    'ids': list(cameras), 'overlay': args.overlay,
                }),
            ),
        )

    @app.post('/whep/{identity}')
    async def whep(identity: str, request: Request):
        if identity not in cameras:
            raise HTTPException(404)
        output = cameras[identity]['output']
        path = urlparse(output['clean' if args.overlay else 'annotated']).path
        url = args.media_base + path + '/whep'
        async with httpx.AsyncClient(trust_env=False, timeout=15) as client:
            response = await client.post(
                url, content=await request.body(),
                headers={'Content-Type': 'application/sdp'},
            )
        headers = {}
        if response.status_code == 201:
            token = uuid.uuid4().hex
            sessions[token] = urljoin(url, response.headers['location'])
            headers['Location'] = '/session/' + token
        return Response(
            response.content, status_code=response.status_code,
            media_type='application/sdp', headers=headers,
        )

    @app.delete('/session/{token}')
    async def close(token: str):
        url = sessions.pop(token, None)
        if url:
            async with httpx.AsyncClient(trust_env=False, timeout=5) as client:
                await client.delete(url)
        return Response(status_code=204)

    @app.get('/metadata/{identity}')
    async def metadata(identity: str):
        if identity not in cameras:
            raise HTTPException(404)
        stream = cameras[identity]
        key = build_metadata_key(stream['site'], stream['name'])

        async def generate():
            last = '0-0'
            while True:
                item = await fetch_latest_metadata_for_key(db, key, last)
                if item is not None:
                    last = item['id']
                    payload = json.dumps(build_metadata_payload(item))
                    yield 'event: metadata\ndata: ' + payload + '\n\n'
        return StreamingResponse(generate(), media_type='text/event-stream')

    uvicorn.run(app, host='127.0.0.1', port=28777, log_level='warning')


if __name__ == '__main__':
    main()
