"""Optional MCP app using the construction API's identity policy."""
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends
from fastapi import FastAPI
from fastapi import Request
from fastapi import Security
from mcp.server.transport_security import TransportSecuritySettings
from starlette.responses import Response

from examples.auth.database import get_db
from examples.auth.jwt_config import jwt_access
from examples.auth.jwt_config import JwtAuthorizationCredentials
from examples.auth.lifespan import global_lifespan
from examples.mcp_server.agent_reads.tools import build_server


class AgentReadsResponse(Response):
    """Let the SDK write the response while dependencies stay alive."""

    def __init__(self, transport: Any) -> None:
        super().__init__()
        self.transport = transport

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        await self.transport(scope, receive, send)


class AgentReadsLifespan:
    def __init__(self, original: Any, transport: Any) -> None:
        self.original = original
        self.transport = transport

    @asynccontextmanager
    async def run(self, app: FastAPI) -> AsyncIterator[None]:
        async with self.original(app):
            async with self.transport.router.lifespan_context(self.transport):
                yield


async def authenticated_transport(
    request: Request,
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
    db: Any = Depends(get_db),
) -> Response:
    request.state.agent_read_identity = credentials
    request.state.agent_read_database = db
    return AgentReadsResponse(request.app.state.agent_reads_transport)


def install(app: FastAPI, path: str = '/mcp') -> None:
    """Install into an application already owning DB/Redis/OIDC resources."""
    if getattr(app.state, 'agent_reads_transport', None) is not None:
        return
    server = build_server()
    transport = server.streamable_http_app(
        streamable_http_path=path,
        stateless_http=True,
        json_response=True,
        max_request_body_size=65_536,
        # jwt_access already resolves the request's registered Host/scheme and
        # requires a real OIDC bearer before the SDK receives the request.
        # A loopback SDK Host allowlist would reject the registered origin.
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=False,
        ),
    )
    app.state.agent_reads_transport = transport
    app.add_api_route(
        path,
        authenticated_transport,
        methods=['GET', 'POST', 'DELETE'],
        include_in_schema=False,
    )
    app.router.lifespan_context = AgentReadsLifespan(
        app.router.lifespan_context, transport,
    ).run


def create_app() -> FastAPI:
    """Run via Uvicorn factory; the base HTTP API needs no MCP extra."""
    app = FastAPI(lifespan=global_lifespan)
    install(app)
    return app
