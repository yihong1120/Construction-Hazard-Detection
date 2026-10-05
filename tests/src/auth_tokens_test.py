from __future__ import annotations

import asyncio
import os
import time
from functools import wraps
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import jwt
import pytest

from src.auth_tokens import AuthenticationConfigurationError
from src.auth_tokens import TokenManager


def async_test(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))
    return wrapped


@async_test
async def test_oidc_acquires_once_and_renews_without_legacy_refresh():
    token = jwt.encode(
        {'exp': time.time() + 300},
        'test-secret-at-least-32-bytes-long-for-hs256',
    )
    response = MagicMock(status=200)
    response.json = AsyncMock(
        return_value={
            'access_token': token,
            'token_type': 'Bearer',
        },
    )
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=None)
    session = MagicMock()
    session.post.return_value = response
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    env = {
        'WORKER_OIDC_CLIENT_ID': 'worker',
        'WORKER_OIDC_CLIENT_SECRET': 'secret',
        'WORKER_OIDC_TOKEN_ENDPOINT': (
            'https://sso.example/realms/test/protocol/openid-connect/token'
        ),
    }
    with (
        patch.dict(os.environ, env, clear=True),
        patch.object(TokenManager, '_create_session', return_value=session),
    ):
        shared = {}
        manager = TokenManager(shared_token=shared)
        assert await asyncio.gather(
            manager.get_valid_token(), manager.get_valid_token(),
        ) == [token, token]
        assert session.post.call_count == 1
        assert shared['access_token'] == token
        await manager.refresh_token()
        assert session.post.call_count == 2
        assert (
            session.post.call_args.kwargs['data']['grant_type']
            == 'client_credentials'
        )
        assert session.post.call_args.kwargs['allow_redirects'] is False
        assert 'refresh_token' not in shared


@async_test
@pytest.mark.parametrize(
    'endpoint',
    [
        '',
        'http://sso.example/token',
        'https://user:pass@sso.example/token',
    ],
)
async def test_invalid_configuration_does_not_retry(endpoint):
    with (
        patch.dict(
            os.environ,
            {
                'WORKER_OIDC_TOKEN_ENDPOINT': endpoint,
                'WORKER_OIDC_CLIENT_ID': 'worker',
                'WORKER_OIDC_CLIENT_SECRET': 'secret',
            },
            clear=True,
        ),
        patch.object(TokenManager, '_create_session') as session,
    ):
        with pytest.raises(AuthenticationConfigurationError):
            await TokenManager().get_valid_token()
        session.assert_not_called()


@async_test
async def test_provider_rejection_does_not_retry_or_expose_response():
    response = MagicMock(status=401)
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=None)
    session = MagicMock()
    session.post.return_value = response
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    with (
        patch.dict(
            os.environ,
            {
                'WORKER_OIDC_CLIENT_ID': 'worker',
                'WORKER_OIDC_CLIENT_SECRET': 'secret',
                'WORKER_OIDC_TOKEN_ENDPOINT': 'http://127.0.0.1:8081/token',
            },
            clear=True,
        ),
        patch.object(TokenManager, '_create_session', return_value=session),
    ):
        with pytest.raises(AuthenticationConfigurationError, match='HTTP 401'):
            await TokenManager().get_valid_token()
        assert session.post.call_count == 1
        response.text.assert_not_called()


@async_test
async def test_preflight_closes_redis_and_stops_on_failure():
    import main
    manager = MagicMock()
    manager.redis.ping = AsyncMock(
        side_effect=RuntimeError('redis unavailable'),
    )
    manager.redis.aclose = AsyncMock()
    with (
        patch.object(main, 'RedisManager', return_value=manager),
        patch.object(main, 'TokenManager') as tokens,
    ):
        with pytest.raises(RuntimeError, match='redis unavailable'):
            await main.validate_runtime_dependencies()
        manager.redis.aclose.assert_awaited_once()
        tokens.assert_not_called()


@async_test
@pytest.mark.parametrize('reject_token', [False, True])
async def test_preflight_validates_oidc_without_removed_login_switch(
        reject_token,
):
    import main
    manager = MagicMock()
    manager.redis.ping = AsyncMock(return_value=True)
    manager.redis.aclose = AsyncMock()
    token_call = AsyncMock(return_value='test-token')
    if reject_token:
        token_call.side_effect = AuthenticationConfigurationError(
            'invalid worker',
        )
    with (
        patch.object(main, 'RedisManager', return_value=manager),
        patch.object(TokenManager, 'get_valid_token', token_call),
    ):
        if reject_token:
            with pytest.raises(
                AuthenticationConfigurationError, match='invalid worker',
            ):
                await main.validate_runtime_dependencies()
        else:
            await main.validate_runtime_dependencies()
        manager.redis.ping.assert_awaited_once()
        manager.redis.aclose.assert_awaited_once()
        token_call.assert_awaited_once()
