from __future__ import annotations

import asyncio

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from starlette.responses import JSONResponse
from starlette.responses import RedirectResponse


async def local_logout_url(redis):
    from examples.auth.session_store import create_oidc_logout_state
    from urllib.parse import urlencode

    state = await create_oidc_logout_state(redis)
    return '/bff/auth/oidc/logout?' + urlencode({'state': state})


async def local_logout_response(redis, state):
    from examples.auth.session_store import consume_oidc_logout_state

    if not await consume_oidc_logout_state(redis, state):
        return JSONResponse(
            {'detail': 'oidc_logout_state_invalid'},
            status_code=400,
        )
    return RedirectResponse(
        'https://dev.changdar-server.mooo.com/login',
        status_code=303,
    )


async def _marker(request, call_next):
    response = await call_next(request)
    response.headers['X-Construction-Environment'] = 'dev'
    return response


def marked(app):
    app.middleware('http')(_marker)
    return app


async def _disabled_identity(*args, **kwargs):
    raise HTTPException(403, 'shared_identity_changes_disabled_in_dev')


async def _protect_shared_identity(request, call_next):
    if request.url.path == '/bff/auth/account':
        return JSONResponse(
            {'detail': 'shared_identity_changes_disabled_in_dev'},
            status_code=403,
        )
    if request.url.path == '/bff/auth/oidc/logout':
        if request.method != 'GET':
            return JSONResponse(
                {'detail': 'method_not_allowed'},
                status_code=405,
            )
        return await local_logout_response(
            request.app.state.redis_client.client,
            request.query_params.get('state'),
        )
    return await call_next(request)


async def _management_health(request: Request):
    from examples.auth.database import engine
    from sqlalchemy import text

    async with engine.connect() as connection:
        await connection.execute(text('SELECT 1'))
    await request.app.state.redis_client.client.ping()
    return {'status': True, 'environment': 'dev'}


def management():
    from examples.db_management.app import app
    from examples.bff import proxy
    from examples.bff import session_services
    from examples.db_management.services import (
        keycloak_user_management_services as identities,
    )

    proxy.BFF_UPSTREAMS['detect'] = proxy.BFF_UPSTREAMS['detection']
    # Identity changes remain blocked for every future admin route in dev.
    identities.keycloak_admin_request = _disabled_identity
    session_services.create_oidc_global_logout_url = local_logout_url
    app.middleware('http')(_protect_shared_identity)
    app.add_api_route(
        '/health',
        _management_health,
        methods=['GET'],
        include_in_schema=False,
    )
    return marked(app)


def violations():
    from examples.violation_records.app import app

    return marked(app)


class _InferenceLimit:
    def __init__(self):
        self.gate = asyncio.Semaphore(1)

    async def dispatch(self, request, call_next):
        if request.method == 'POST' and request.url.path in {
            '/detect',
            '/v2/detect',
        }:
            async with self.gate:
                return await call_next(request)
        return await call_next(request)


async def _health():
    return {'status': True, 'environment': 'dev'}


def detection():
    from deploy.dev.run import STATE

    (STATE / 'work/models/pt').mkdir(parents=True, exist_ok=True, mode=0o700)
    from examples.YOLO_server_api.app import app

    app.middleware('http')(_InferenceLimit().dispatch)
    app.add_api_route(
        '/health',
        _health,
        methods=['GET'],
        include_in_schema=False,
    )
    return marked(app)


def streaming():
    from examples.streaming_web.app import app

    return marked(app)


async def _stop_external_push(request, call_next):
    if request.method == 'POST':
        return JSONResponse(
            {'detail': 'external_push_disabled_in_dev'},
            status_code=403,
        )
    return await call_next(request)


def notifications():
    from examples.auth.lifespan import global_lifespan
    from examples.local_notification_server.routers import router

    # Device/inbox storage works without loading production push credentials.
    app = FastAPI(lifespan=global_lifespan)
    app.include_router(router)
    app.middleware('http')(_stop_external_push)
    return marked(app)


async def _unavailable(path: str):
    return JSONResponse(
        {
            ('detail'): (
                'This integration is not connected in dev; use '
                'ai-dev for AI chat.'
            ),
        },
        status_code=503,
    )


def unavailable():
    app = FastAPI()
    app.add_api_route(
        '/{path:path}',
        _unavailable,
        methods=['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'HEAD'],
    )
    return marked(app)
