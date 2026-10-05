from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from examples.db_management.routers.oauth import router


def test_retired_native_oauth_grants_are_absent():
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    for path in ['/oauth/authorize', '/oauth/token', '/oauth/revoke']:
        assert client.get(path).status_code == 404
        assert client.post(path).status_code == 404
    assert '/me' in {route.path for route in router.routes}
