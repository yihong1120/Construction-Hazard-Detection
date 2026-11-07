from __future__ import annotations

import ast
import os
import unittest
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from examples.auth.settings_fields import DatabaseSettings
from examples.bff.settings import CookieSettings
from examples.bff.settings import ProxySettings
from examples.mcp_server.settings import McpTransportSettings
from src.auth_tokens import TokenManager
from src.redis_client import RedisManager
from src.settings.capture import CaptureSettings
from src.settings.connections import FcmTransportSettings
from src.settings.connections import ModelFetchSettings
from src.settings.connections import RedisSettings
from src.settings.connections import StartupSettings
from src.settings.inference import ApiInferenceSettings
from src.settings.inference import LocalInferenceSettings
from src.settings.inference import WorkerSettings
from src.settings.media import MediaGatewaySettings
from src.settings.media import MediaPublishSettings
from src.settings.media import NvencSettings
from src.settings.notifications import LineSettings
from src.settings.notifications import WeChatSettings


@pytest.mark.parametrize(
    'settings,key,value',
    [
        (CaptureSettings, 'STREAM_CAPTURE_READ_TIMEOUT_MS', '0'),
        (CaptureSettings, 'STREAM_CAPTURE_REOPEN_DELAY_SECONDS', 'NaN'),
        (DatabaseSettings, 'DB_POOL_TIMEOUT_SECONDS', 'inf'),
        (RedisSettings, 'REDIS_PORT', '65536'),
        (ProxySettings, 'BFF_UPSTREAM_TIMEOUT_SECONDS', '-1'),
        (ProxySettings, 'BFF_REFRESH_BEFORE_SECONDS', '-1'),
        (ProxySettings, 'BFF_DETECTION_API_URL', 'file:///etc/passwd'),
        (StartupSettings, 'AUTO_CREATE_SCHEMA', 'typo'),
        (CookieSettings, 'BFF_SESSION_COOKIE_SECURE', 'typo'),
        (LocalInferenceSettings, 'DETECT_LOCAL_IMGSZ', '0'),
        (WorkerSettings, 'YOLO_WORKER_TIMEOUT_SECONDS', 'NaN'),
        (NvencSettings, 'MEDIA_NVENC_MAX_SESSIONS', '0'),
        (McpTransportSettings, 'MCP_PORT', '0'),
        (McpTransportSettings, 'MCP_TRANSPORT', 'typo'),
    ],
)
def test_invalid_settings_are_rejected_without_resources(settings, key, value):
    with patch.dict(os.environ, {key: value}, clear=True):
        with pytest.raises(ValidationError):
            settings()


def test_explicit_parameters_override_environment_and_keep_empty_password():
    with patch.dict(
        os.environ,
        {
            'REDIS_HOST': 'environment',
            'REDIS_PORT': 'invalid',
            'REDIS_PASSWORD': 'environment-secret',
            'FCM_API_URL': 'invalid',
        },
        clear=True,
    ), patch('src.redis_client.redis.Redis') as client:
        manager = RedisManager('explicit', 1234, '')
        client.assert_called_once_with(
            host='explicit',
            port=1234,
            password='',
            decode_responses=False,
        )
        assert manager.settings.host == 'explicit'
        assert (
            FcmTransportSettings.from_overrides(
                api_url='https://explicit.example/',
            ).api_url
            == 'https://explicit.example'
        )


def test_explicit_value_overrides_all_legacy_environment_aliases():
    with patch.dict(
        os.environ,
        {
            'DETECT_INGRESS_CONCURRENCY': 'invalid',
            'YOLO_API_QUEUE_CAPACITY': 'invalid',
        },
        clear=True,
    ):
        config = ApiInferenceSettings.from_overrides(queue_capacity=2)
        assert config.queue_capacity == 2


@pytest.mark.parametrize(
    'url', [
        'https://models.example/get_new_model/',
        'https://models.example/get_new_model?channel=a/',
    ],
)
def test_model_download_retains_complete_endpoint(url):
    with patch.dict(os.environ, {}, clear=True):
        assert ModelFetchSettings.from_overrides(api_url=url).api_url == url


def test_invalid_redis_settings_fail_before_constructing_client():
    with patch.dict(os.environ, {'REDIS_PORT': 'invalid'}, clear=True):
        with patch('src.redis_client.redis.Redis') as client:
            with pytest.raises(ValidationError):
                RedisManager()
            client.assert_not_called()


def test_redis_url_encodes_exact_secret_and_ipv6():
    with patch.dict(os.environ, {}, clear=True):
        config = RedisSettings.from_overrides(
            host='::1',
            password=' a:@/#? ',
        )
    assert config.password.get_secret_value() == ' a:@/#? '
    assert config.url == 'redis://:%20a%3A%40%2F%23%3F%20@[::1]:6379/0'
    assert 'a:@/#?' not in repr(config)
    assert 'password' not in config.model_dump()


def test_optional_providers_do_not_require_unrelated_credentials():
    with patch.dict(os.environ, {}, clear=True):
        assert not LineSettings().channel_access_token.get_secret_value()
        assert not WeChatSettings().corp_secret.get_secret_value()
        assert FcmTransportSettings().api_url == 'http://127.0.0.1:8003'


def test_host_cookie_requires_secure_transport():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValidationError):
            CookieSettings(session_cookie_secure=False)


def test_capture_reconnect_budget_must_cover_initial_delay():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValidationError):
            CaptureSettings(
                reopen_delay_seconds=10, max_reopen_delay_seconds=5,
            )


@pytest.mark.parametrize('ttl', ['3', '15', '45'])
def test_api_and_worker_share_readiness_ttl(ttl):
    with patch.dict(
        os.environ, {'MEDIA_OVERLAY_READY_TTL_SECONDS': ttl}, clear=True,
    ):
        assert (
            MediaGatewaySettings().ready_ttl_seconds
            == MediaPublishSettings().ready_ttl_seconds
        )


@pytest.mark.ml
def test_capture_holds_timeouts_across_environment_changes():
    from src.stream_capture import StreamCapture

    with patch.dict(
        os.environ,
        {
            'STREAM_CAPTURE_OPEN_TIMEOUT_MS': '1234',
            'STREAM_CAPTURE_READ_TIMEOUT_MS': '5678',
        },
        clear=True,
    ):
        capture = StreamCapture('rtsp://camera.example/live')
        os.environ['STREAM_CAPTURE_READ_TIMEOUT_MS'] = 'invalid'
        with patch('src.stream_capture.cv2.VideoCapture') as create:
            capture._create_capture(capture.stream_url)
        assert create.call_args.args[2][1::2] == [1234, 5678]


def test_business_modules_do_not_read_environment_configuration():
    root = Path(__file__).resolve().parents[2]
    forbidden = []
    for directory in ('src', 'examples'):
        for path in (root / directory).rglob('*.py'):
            if 'settings' in path.parts or path.name.endswith(
                ('settings.py', 'settings_fields.py'),
            ):
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == 'os'
                    and node.func.attr == 'getenv'
                ):
                    forbidden.append(f'{path.relative_to(root)}:{node.lineno}')
    assert not forbidden, forbidden


class WorkerConfigurationSnapshotTest(unittest.IsolatedAsyncioTestCase):
    async def test_refresh_retains_owner_credentials(self):
        manager = TokenManager()
        self.addAsyncCleanup(manager.close)
        response = Mock(status_code=200)
        response.json.return_value = {
            'access_token': 'token',
            'token_type': 'Bearer',
        }
        client = AsyncMock()
        client.post.return_value = response
        with patch.dict(
            os.environ,
            {
                'WORKER_OIDC_TOKEN_ENDPOINT': 'https://identity.example/token',
                'WORKER_OIDC_CLIENT_ID': 'original',
                'WORKER_OIDC_CLIENT_SECRET': ' original-secret ',
            },
            clear=True,
        ), patch.object(
            manager, '_get_client', AsyncMock(return_value=client),
        ):
            await manager.authenticate()
            os.environ['WORKER_OIDC_CLIENT_ID'] = 'changed'
            os.environ['WORKER_OIDC_CLIENT_SECRET'] = 'changed-secret'
            await manager.authenticate(force=True)
        assert client.post.await_count == 2
        for call in client.post.await_args_list:
            assert call.kwargs['data']['client_id'] == 'original'
            assert call.kwargs['data']['client_secret'] == ' original-secret '
        assert 'original-secret' not in repr(manager._oidc_settings)
