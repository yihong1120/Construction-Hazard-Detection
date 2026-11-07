from __future__ import annotations

import asyncio
import os
import time
from functools import partial
from functools import wraps
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import httpx
import jwt
import pytest

from src.auth_tokens import AuthenticationConfigurationError
from src.auth_tokens import TokenManager
from src.notifiers.fcm_notifier import FCMSender
from src.violation_sender import ViolationSender


def _test_wrapped(func, /, *args, **kwargs):
    return asyncio.run(func(*args, **kwargs))


async def _all_acquisition_paths_share_refresh_respond(
    requests,
    /,
    request,
):
    requests.append(request)
    await asyncio.sleep(0)
    return httpx.Response(
        200,
        json={
            'access_token': access_token(len(requests)),
            'token_type': 'Bearer',
        },
    )


def _provider_failure_is_one_attempt_respond(
    requests,
    status,
    /,
    request,
):
    requests.append(request)
    return httpx.Response(status, text='private-provider-body')


def _invalid_provider_response_is_configuration_Response(
    payload,
    /,
    request,
):
    return httpx.Response(200, json=payload)


def async_test(func):
    wrapped = wraps(func)(partial(_test_wrapped, func))

    return wrapped


_ENV = {
    'WORKER_OIDC_CLIENT_ID': 'worker',
    'WORKER_OIDC_CLIENT_SECRET': 'private-secret',
    'WORKER_OIDC_TOKEN_ENDPOINT': 'https://identity.test/token',
}


def access_token(number):
    return jwt.encode(
        {'exp': time.time() + 300, 'jti': str(number)},
        'test-secret-at-least-32-bytes-long-for-hs256',
    )


@async_test
async def test_all_acquisition_paths_share_refresh_and_pool():
    requests = []

    respond = partial(_all_acquisition_paths_share_refresh_respond, requests)

    manager = TokenManager()
    with patch.dict(os.environ, _ENV, clear=True):
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(respond),
        ) as client:
            manager._client = client
            values = await asyncio.gather(
                *(manager.get_valid_token() for _ in range(4)),
            )
            await asyncio.gather(
                *(manager.ensure_token_valid() for _ in range(4)),
            )
            assert len(set(values)) == 1 and len(requests) == 1
            old = values[0]
            await asyncio.gather(
                *(manager.handle_401(old) for _ in range(4)),
            )
            assert len(requests) == 2
            assert await manager.get_valid_token() != old
            assert b'grant_type=client_credentials' in requests[0].content
            assert b'refresh_token' not in requests[0].content
            assert manager._client is client


@async_test
@pytest.mark.parametrize(
    'endpoint',
    [
        '',
        'http://identity.test/token',
        'https://user:pass@identity.test/token',
    ],
)
async def test_invalid_configuration_never_creates_transport(endpoint):
    with (
        patch.dict(
            os.environ,
            {**_ENV, 'WORKER_OIDC_TOKEN_ENDPOINT': endpoint},
            clear=True,
        ),
        patch.object(TokenManager, '_get_client') as client,
    ):
        with pytest.raises(AuthenticationConfigurationError):
            await TokenManager().get_valid_token()
        client.assert_not_called()


@async_test
@pytest.mark.parametrize(
    'status, error',
    [(401, AuthenticationConfigurationError), (503, RuntimeError)],
)
async def test_provider_failure_is_one_attempt_and_contains_no_private_body(
    status,
    error,
):
    requests = []

    respond = partial(
        _provider_failure_is_one_attempt_respond,
        requests,
        status,
    )

    with patch.dict(os.environ, _ENV, clear=True):
        manager = TokenManager()
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(respond),
        ) as client:
            manager._client = client
            with pytest.raises(error, match='HTTP ' + str(status)) as raised:
                await manager.get_valid_token()
            assert 'private-provider-body' not in str(raised.value)
            assert len(requests) == 1


@async_test
@pytest.mark.parametrize(
    'payload',
    [
        {},
        {'access_token': 'token'},
        {'access_token': '', 'token_type': 'Bearer'},
    ],
)
async def test_invalid_provider_response_is_configuration_error(payload):
    with patch.dict(os.environ, _ENV, clear=True):
        manager = TokenManager()
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                partial(
                    _invalid_provider_response_is_configuration_Response,
                    payload,
                ),
            ),
        ) as client:
            manager._client = client
            with pytest.raises(AuthenticationConfigurationError):
                await manager.get_valid_token()


@async_test
async def test_sender_owns_only_its_private_provider():
    shared = TokenManager()
    client = await shared._get_client()
    assert not client.trust_env
    first, second = ViolationSender(token_manager=shared), FCMSender(
        token_manager=shared,
    )
    await first.close()
    await second.close()
    assert not client.is_closed
    await shared.close()
    assert client.is_closed
    private = ViolationSender()
    owned = await private.token_manager._get_client()
    await private.close()
    assert owned.is_closed


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
                AuthenticationConfigurationError,
                match='invalid worker',
            ):
                await main.validate_runtime_dependencies()
        else:
            await main.validate_runtime_dependencies()
        manager.redis.ping.assert_awaited_once()
        manager.redis.aclose.assert_awaited_once()
        token_call.assert_awaited_once()
