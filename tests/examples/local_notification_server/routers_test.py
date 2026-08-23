from __future__ import annotations

import unittest
from collections.abc import AsyncIterator
from datetime import datetime
from datetime import timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.testclient import TestClient

from examples.auth.database import get_db
from examples.auth.jwt_config import jwt_access
from examples.auth.models import FcmDeviceToken
from examples.auth.redis_pool import get_redis_pool
from examples.local_notification_server import (
    notification_delivery_service,
)
from examples.local_notification_server import routers as router_endpoints
from examples.local_notification_server import (
    site_preference_service,
)
from examples.local_notification_server.fcm_service import FcmSendResult
from examples.local_notification_server.routers import delete_notification
from examples.local_notification_server.routers import (
    get_notification_device_status,
)
from examples.local_notification_server.routers import (
    get_notification_unread_count,
)
from examples.local_notification_server.routers import list_notifications
from examples.local_notification_server.routers import (
    mark_all_notifications_read,
)
from examples.local_notification_server.routers import mark_notification_read
from examples.local_notification_server.routers import router
from examples.local_notification_server.routers import send_test_notification
from examples.local_notification_server.schemas import (
    SiteNotificationPreferenceIn,
)
from examples.local_notification_server.schemas import (
    SiteNotificationPreferenceUpdateRequest,
)
from examples.local_notification_server.schemas import SiteNotifyRequest
from examples.local_notification_server.services import fcm_token_hash

routers = notification_delivery_service


def mock_jwt_access() -> MagicMock:
    """Mock JWT credentials to avoid the need for a real token in tests.

    Returns:
        MagicMock: A mock object with dummy jti and sub attributes.
    """
    return MagicMock(jti='dummy-jti', sub='dummy-sub')


class TestLocalNotificationServer(unittest.TestCase):
    """Unit test suite for routes in the local notification server."""

    def setUp(self) -> None:
        """Set up a FastAPI app, test client, and mock dependencies before each
        test."""
        self.app: FastAPI = FastAPI()
        self.app.include_router(router, prefix='/fcm')
        self.client: TestClient = TestClient(self.app)
        self.mock_session: AsyncMock = AsyncMock()
        self.mock_session.add = MagicMock()
        self.mock_session.add_all = MagicMock()
        empty_rows = MagicMock()
        empty_rows.scalars.return_value.all.return_value = []
        self.mock_session.execute = AsyncMock(return_value=empty_rows)
        self.mock_session.commit = AsyncMock()

        # Redis mock: use MagicMock for correct pipeline chain
        self.mock_redis: MagicMock = MagicMock()
        self.mock_redis.get = AsyncMock(return_value=None)
        self.mock_redis.mget = AsyncMock(return_value=[b'1'])
        self.mock_redis.exists = AsyncMock(return_value=0)
        self.mock_redis.smembers = AsyncMock(return_value=set())
        self.mock_redis.set = AsyncMock()
        self.mock_redis.delete = AsyncMock()
        self.mock_redis.hget = AsyncMock(return_value=None)
        self.mock_redis.hgetall = AsyncMock(return_value={})
        self.mock_redis.hset = AsyncMock()
        self.mock_redis.hdel = AsyncMock()
        self.mock_redis.srem = AsyncMock()
        # pipeline mock will be set in each test as needed

        async def override_get_db() -> AsyncIterator[AsyncMock]:
            """Override dependency for database session, returning a mock
            session object.

            Yields:
                AsyncMock: The mocked database session.
            """
            yield self.mock_session

        async def override_get_redis_pool() -> MagicMock:
            """Override dependency for Redis connection, returning a mock Redis
            object.

            Returns:
                MagicMock: The mocked Redis connection.
            """
            return self.mock_redis

        # Override the dependencies with the mocks
        self.app.dependency_overrides[get_db] = override_get_db
        self.app.dependency_overrides[get_redis_pool] = override_get_redis_pool
        self.app.dependency_overrides[jwt_access] = mock_jwt_access

    def tearDown(self) -> None:
        """Clear overrides after each test to avoid leakage between test
        cases."""
        self.app.dependency_overrides.clear()

    # ------------------------------------------------------------------------
    # Helper functions for simulating database queries
    # ------------------------------------------------------------------------
    def mock_no_user_in_db(self) -> None:
        """Simulate a scenario where the queried user is not found in the
        database."""
        result: MagicMock = MagicMock()
        result.unique.return_value = result
        result.scalar_one_or_none.return_value = None
        result.scalar.return_value = None  # Add this for the scalar() check
        self.mock_session.execute = AsyncMock(return_value=result)

    def mock_user_in_db(self, user_id: int) -> MagicMock:
        """Simulate a scenario where the queried user is found in the database.

        Args:
            user_id (int): The ID of the user to mock.

        Returns:
            MagicMock: The mocked user object.
        """
        mock_user: MagicMock = MagicMock()
        mock_user.id = user_id

        result: MagicMock = MagicMock()
        result.unique.return_value = result
        result.scalar_one_or_none.return_value = mock_user
        result.scalar.return_value = user_id  # Add this for the scalar() check
        self.mock_session.execute = AsyncMock(return_value=result)
        self.mock_session.scalar = AsyncMock(return_value=None)
        return mock_user

    def mock_site_in_db(
        self,
        site_name: str,
        users: list[MagicMock] | None = None,
    ) -> MagicMock:
        """Simulate a scenario where a Site is found in the database.

        Args:
            site_name (str):
                The name of the site to mock.
            users (list[MagicMock] | None):
                A list of mocked user objects associated with the site.
                Defaults to an empty list if None.
        Returns:
            MagicMock: The mocked site object containing the given users.
        """
        if users is None:
            users = []

        mock_site: MagicMock = MagicMock()
        mock_site.name = site_name
        mock_site.users = users

        site_lookup_result: MagicMock = MagicMock()
        site_lookup_result.first.return_value = (1, 1)

        user_ids_result: MagicMock = MagicMock()
        user_ids_result.scalars.return_value.all.return_value = [
            user.id for user in users
        ]

        self.mock_session.execute = AsyncMock(
            side_effect=[site_lookup_result, user_ids_result],
        )
        return mock_site

    # ------------------------------------------------------------------------
    # Tests for PUT /devices - registering an FCM device
    # ------------------------------------------------------------------------
    def test_register_device_requires_authenticated_user(self) -> None:
        """Reject token registration when the authenticated user is absent.

        Expect a 401 error response.
        """
        self.mock_no_user_in_db()
        # Patch pipeline to avoid await error
        pipe_mock = MagicMock()
        pipe_mock.hset = MagicMock()
        pipe_mock.expire = MagicMock()
        pipe_mock.execute = AsyncMock()
        self.mock_redis.pipeline.return_value = pipe_mock
        data: dict[str, object] = {
            'device_token': 'test-token-999',
            'device_lang': 'en-GB',
            'platform': 'web',
        }
        response = self.client.put('/fcm/devices', json=data)
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json(), {'detail': 'User not found'})

    def test_register_device_success(self) -> None:
        """Test successful token storage for an existing user."""
        self.mock_user_in_db(user_id=123)
        # Patch pipeline
        token_pipe_mock = MagicMock()
        token_pipe_mock.hset = MagicMock()
        token_pipe_mock.expire = MagicMock()
        token_pipe_mock.execute = AsyncMock()
        token_pipe_mock.sadd = MagicMock()
        self.mock_redis.pipeline.return_value = token_pipe_mock
        data: dict[str, object] = {
            'device_token': 'my-test-token',
            'device_lang': 'en-GB',
            'platform': 'web',
        }
        response = self.client.put('/fcm/devices', json=data)
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload['ok'])
        self.assertTrue(payload['updated'])
        self.assertEqual(payload['user_id'], 123)
        self.assertEqual(payload['device_lang'], 'en-GB')
        self.assertNotIn('token_hash', payload)
        self.assertIn('registered_at', payload)
        self.assertIn('last_seen_at', payload)
        token_pipe_mock.hset.assert_any_call(
            'fcm_tokens:123',
            'my-test-token',
            'en-GB',
        )
        token_pipe_mock.sadd.assert_called_once_with(
            'fcm_token_index:123',
            fcm_token_hash('my-test-token'),
        )
        self.assertEqual(token_pipe_mock.hset.call_count, 2)
        self.mock_session.commit.assert_awaited()

    def test_register_device_requires_platform(self) -> None:
        """Token registration rejects a request without its platform."""
        self.mock_user_in_db(user_id=123)
        data: dict[str, object] = {
            'device_token': 'my-test-token',
            'device_lang': 'en-GB',
        }

        response = self.client.put('/fcm/devices', json=data)

        self.assertEqual(response.status_code, 422)
        self.mock_redis.pipeline.assert_not_called()

    def test_register_device_with_canonical_device_lang(self) -> None:
        """Test token storage with a canonical device language."""
        self.mock_user_in_db(user_id=123)
        token_pipe_mock = MagicMock()
        token_pipe_mock.hset = MagicMock()
        token_pipe_mock.expire = MagicMock()
        token_pipe_mock.execute = AsyncMock()
        token_pipe_mock.sadd = MagicMock()
        self.mock_redis.pipeline.return_value = token_pipe_mock
        data: dict[str, object] = {
            'device_token': 'test-token',
            'device_lang': 'zh-TW',
            'platform': 'web',
        }
        response = self.client.put('/fcm/devices', json=data)
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload['ok'])
        self.assertTrue(payload['updated'])
        self.assertEqual(payload['device_lang'], 'zh-TW')
        self.assertNotIn('token_hash', payload)
        token_pipe_mock.hset.assert_any_call(
            'fcm_tokens:123',
            'test-token',
            'zh-TW',
        )
        self.assertEqual(token_pipe_mock.hset.call_count, 2)

    def test_register_device_with_unsupported_device_lang(self) -> None:
        """Unsupported device languages are rejected instead of retry."""
        self.mock_user_in_db(user_id=123)
        data: dict[str, object] = {
            'device_token': 'test-token',
            'device_lang': 'unknown',
            'platform': 'web',
        }

        response = self.client.put('/fcm/devices', json=data)

        self.assertEqual(response.status_code, 422)
        self.mock_redis.pipeline.assert_not_called()

    # ------------------------------------------------------------------------
    # Tests for DELETE /devices - removing an FCM token from Redis
    # ------------------------------------------------------------------------
    def test_unregister_device_requires_authenticated_user(self) -> None:
        """Reject device removal when the authenticated user is absent.

        Expect a 401 error response.
        """
        self.mock_no_user_in_db()
        pipe_mock = MagicMock()
        pipe_mock.hdel = MagicMock()
        pipe_mock.hlen = MagicMock()
        pipe_mock.execute = AsyncMock()
        self.mock_redis.pipeline.return_value = pipe_mock
        data: dict[str, object] = {
            'device_token': 'unknown-token',
        }
        response = self.client.request(
            'DELETE',
            '/fcm/devices',
            json=data,
        )
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json(), {'detail': 'User not found'})

    def test_unregister_device_not_in_redis(self) -> None:
        """Test attempting to delete a token that does not exist in Redis."""
        user = MagicMock()
        user.id = 10
        user_result = MagicMock()
        user_result.unique.return_value = user_result
        user_result.scalar_one_or_none.return_value = user
        update_result = MagicMock()
        update_result.rowcount = 0
        self.mock_session.execute = AsyncMock(
            side_effect=[user_result, update_result],
        )
        pipe_mock = MagicMock()
        pipe_mock.hdel = MagicMock()
        pipe_mock.hlen = MagicMock()
        pipe_mock.execute = AsyncMock(return_value=[0, 1])
        self.mock_redis.pipeline.return_value = pipe_mock
        data: dict[str, object] = {
            'device_token': 'non-existent-token',
        }
        response = self.client.request(
            'DELETE',
            '/fcm/devices',
            json=data,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                'message': 'Token not found.',
            },
        )

    def test_unregister_device_success(self) -> None:
        """Test successfully deleting an existing token in Redis."""
        user = MagicMock()
        user.id = 10
        user_result = MagicMock()
        user_result.unique.return_value = user_result
        user_result.scalar_one_or_none.return_value = user
        update_result = MagicMock()
        update_result.rowcount = 1
        self.mock_session.execute = AsyncMock(
            side_effect=[user_result, update_result],
        )
        pipe_mock = MagicMock()
        pipe_mock.hdel = MagicMock()
        pipe_mock.hlen = MagicMock()
        pipe_mock.execute = AsyncMock(return_value=[1, 1])
        self.mock_redis.pipeline.return_value = pipe_mock
        data: dict[str, object] = {
            'device_token': 'existing-token',
        }
        response = self.client.request(
            'DELETE',
            '/fcm/devices',
            json=data,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'message': 'Token deleted.'})
        pipe_mock.hdel.assert_called_once_with(
            'fcm_tokens:10',
            'existing-token',
        )

    def test_unregister_device_deletes_key_when_no_tokens_remain(self) -> None:
        """Test that the Redis key is deleted when no tokens remain after
        deletion."""
        user = MagicMock()
        user.id = 10
        user_result = MagicMock()
        user_result.unique.return_value = user_result
        user_result.scalar_one_or_none.return_value = user
        update_result = MagicMock()
        update_result.rowcount = 1
        self.mock_session.execute = AsyncMock(
            side_effect=[user_result, update_result],
        )
        pipe_mock = MagicMock()
        pipe_mock.hdel = MagicMock()
        pipe_mock.hlen = MagicMock()
        pipe_mock.execute = AsyncMock(return_value=[1, 0])
        self.mock_redis.pipeline.return_value = pipe_mock
        self.mock_redis.delete = AsyncMock()
        data: dict[str, object] = {
            'device_token': 'existing-token',
        }
        response = self.client.request(
            'DELETE',
            '/fcm/devices',
            json=data,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'message': 'Token deleted.'})
        self.assertEqual(self.mock_redis.delete.await_count, 2)
        self.mock_redis.delete.assert_any_await('fcm_tokens:10')

    # ------------------------------------------------------------------------
    # Tests for /send_fcm_notification (POST) - sending notifications
    # ------------------------------------------------------------------------
    @patch(
        'examples.local_notification_server.notification_delivery_service.'
        'get_site_notification_user_ids_cached',
        new_callable=AsyncMock,
        return_value=None,
    )
    def test_send_fcm_notification_site_not_found(
        self,
        mock_get_user_ids: AsyncMock,
    ) -> None:
        """Test send fcm notification site not found.

        Args:
            mock_get_user_ids: Value used by this callable.
        """
        _ = mock_get_user_ids
        self.mock_redis.set = AsyncMock(return_value=True)

        data: dict[str, object] = {
            'site': 'MissingSite',
            'stream_name': 'TestStream',
            'image_path': None,
            'violation_id': None,
            'body': {'warning_no_hardhat': {'count': 1}},
            'type': 'site_alert',
            'title': 'Site alert',
            'deep_link': '/sites/missing',
            'metadata': {},
        }
        headers: dict[str, str] = {'Authorization': 'Bearer dummy-token'}
        response = self.client.post(
            '/fcm/send_fcm_notification',
            json=data,
            headers=headers,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                'success': False,
                'message': "Site 'MissingSite' not found.",
            },
        )

    @patch(
        'examples.local_notification_server.notification_delivery_service.'
        'get_site_notification_user_ids_cached',
        new_callable=AsyncMock,
        return_value=[],
    )
    def test_send_fcm_notification_site_no_users(
        self,
        mock_get_user_ids: AsyncMock,
    ) -> None:
        """Test send fcm notification site no users.

        Args:
            mock_get_user_ids: Value used by this callable.
        """
        _ = mock_get_user_ids
        self.mock_redis.set = AsyncMock(return_value=True)

        data: dict[str, object] = {
            'site': 'EmptySite',
            'stream_name': 'EmptyStream',
            'image_path': None,
            'violation_id': None,
            'body': {'warning_no_hardhat': {'count': 1}},
            'type': 'site_alert',
            'title': 'Site alert',
            'deep_link': '/sites/empty',
            'metadata': {},
        }
        headers: dict[str, str] = {'Authorization': 'Bearer dummy-token'}
        response = self.client.post(
            '/fcm/send_fcm_notification',
            json=data,
            headers=headers,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                'success': False,
                'message': "Site 'EmptySite' has no subscribed users.",
            },
        )

    @patch(
        'examples.local_notification_server.notification_delivery_service.'
        'get_site_notification_user_ids_cached',
        new_callable=AsyncMock,
        return_value=[1, 2],
    )
    def test_send_fcm_notification_no_tokens(
        self,
        mock_get_user_ids: AsyncMock,
    ) -> None:
        """Test sending a notification where users exist, but none have tokens
        in Redis."""
        _ = mock_get_user_ids
        self.mock_redis.set = AsyncMock(return_value=True)

        # Redis pipeline mock
        pipe_mock: MagicMock = MagicMock()
        pipe_mock.hgetall = MagicMock()
        pipe_mock.execute = AsyncMock(return_value=[{}, {}])
        self.mock_redis.pipeline.return_value = pipe_mock

        data: dict[str, object] = {
            'site': 'SiteWithNoTokens',
            'stream_name': 'SiteStream',
            'image_path': None,
            'violation_id': None,
            'body': {'warning_no_hardhat': {'count': 1}},
            'type': 'site_alert',
            'title': 'Site alert',
            'deep_link': '/sites/no-tokens',
            'metadata': {},
        }
        headers: dict[str, str] = {'Authorization': 'Bearer dummy-token'}
        response = self.client.post(
            '/fcm/send_fcm_notification',
            json=data,
            headers=headers,
        )

        self.assertEqual(response.status_code, 200)
        resp_json = response.json()
        self.assertFalse(resp_json['success'])
        self.assertEqual(
            resp_json['message'],
            "Site 'SiteWithNoTokens' has no device tokens.",
        )
        self.assertEqual(resp_json['stats']['total_batches'], 0)
        self.assertEqual(resp_json['stats']['successful_batches'], 0)
        self.assertEqual(
            resp_json['stats']['preflight']['recipient_users'],
            2,
        )
        self.assertEqual(resp_json['stats']['preflight']['unique_tokens'], 0)

    @patch(
        'examples.local_notification_server.push_dispatch.'
        'send_fcm_notification_service',
        new_callable=AsyncMock,
    )
    def test_send_fcm_notification_success(
        self,
        mock_send_fcm: AsyncMock,
    ) -> None:
        """Test successfully sending a notification when a site, users, and
        user tokens in Redis are all available.

        Args:
            mock_send_fcm (AsyncMock): Mocked FCM notification sending service.
        """
        self.mock_redis.set = AsyncMock(return_value=True)

        pipe_mock: MagicMock = MagicMock()
        pipe_mock.hgetall = MagicMock()
        pipe_mock.execute = AsyncMock(
            return_value=[{b'tokenA': b'en-GB', b'tokenB': b'zh-TW'}],
        )
        self.mock_redis.pipeline.return_value = pipe_mock

        mock_send_fcm.return_value = FcmSendResult(1, 0)

        data: dict[str, object] = {
            'site': 'MySite',
            'stream_name': 'MainStream',
            'image_path': None,
            'violation_id': 999,
            'body': {'warning_no_hardhat': {'count': 1}},
            'type': 'violation',
            'title': 'Violation alert',
            'deep_link': '/violations?violation_id=999',
            'metadata': {'violation_id': 999},
        }
        headers: dict[str, str] = {'Authorization': 'Bearer dummy-token'}
        with patch(
            'examples.local_notification_server.notification_delivery_service.'
            'get_site_notification_user_ids_cached',
            new=AsyncMock(return_value=[42]),
        ):
            response = self.client.post(
                '/fcm/send_fcm_notification',
                json=data,
                headers=headers,
            )

        self.assertEqual(response.status_code, 200)
        resp_json = response.json()
        self.assertTrue(resp_json['success'])
        self.assertIn('batches succeeded', resp_json['message'])
        self.assertIn('stats', resp_json)
        self.assertEqual(
            mock_send_fcm.await_count,
            2,
            'Expected 2 calls to send_fcm_notification_service '
            'for two languages.',
        )

    @patch(
        'examples.local_notification_server.push_dispatch.'
        'send_fcm_notification_service',
        new_callable=AsyncMock,
    )
    def test_send_fcm_notification_all_fail(
        self,
        mock_send_fcm: AsyncMock,
    ) -> None:
        """Test send fcm notification all fail.

        Args:
            mock_send_fcm: Value used by this callable.
        """
        self.mock_redis.set = AsyncMock(return_value=True)

        pipe_mock: MagicMock = MagicMock()
        pipe_mock.hgetall = MagicMock()
        pipe_mock.execute = AsyncMock(
            return_value=[{b'tA': b'en-GB', b'tB': b'zh-TW'}],
        )
        self.mock_redis.pipeline.return_value = pipe_mock

        mock_send_fcm.return_value = FcmSendResult(0, 1)

        data: dict[str, object] = {
            'site': 'MySite',
            'stream_name': 'FailStream',
            'violation_id': 123,
            'body': {'warning_no_hardhat': {'count': 1}},
            'type': 'violation',
            'title': 'Violation alert',
            'deep_link': '/violations?violation_id=123',
            'metadata': {'violation_id': 123},
        }
        headers: dict[str, str] = {'Authorization': 'Bearer dummy-token'}
        with patch(
            'examples.local_notification_server.notification_delivery_service.'
            'get_site_notification_user_ids_cached',
            new=AsyncMock(return_value=[99]),
        ):
            response = self.client.post(
                '/fcm/send_fcm_notification',
                json=data,
                headers=headers,
            )

        self.assertEqual(response.status_code, 200)
        resp_json = response.json()
        self.assertFalse(resp_json['success'])
        self.assertIn('batches succeeded', resp_json['message'])
        self.assertIn('stats', resp_json)
        self.assertEqual(mock_send_fcm.await_count, 2)

    @patch(
        'examples.local_notification_server.push_dispatch.'
        'send_fcm_notification_service',
        new_callable=AsyncMock,
    )
    def test_send_fcm_notification_timeout(
        self,
        mock_send_fcm: AsyncMock,
    ) -> None:
        """Test FCM notification sending timeout branch."""
        _ = mock_send_fcm
        self.mock_redis.set = AsyncMock(return_value=True)
        pipe_mock: MagicMock = MagicMock()
        pipe_mock.hgetall = MagicMock()
        pipe_mock.execute = AsyncMock(return_value=[{b'token': b'en-GB'}])
        self.mock_redis.pipeline.return_value = pipe_mock
        # Patch asyncio.wait_for to raise TimeoutError
        import asyncio as real_asyncio

        with patch('asyncio.wait_for', side_effect=real_asyncio.TimeoutError):
            with patch(
                'examples.local_notification_server.notification_delivery_service.'
                'get_site_notification_user_ids_cached',
                new=AsyncMock(return_value=[1]),
            ):
                data = {
                    'site': 'TimeoutSite',
                    'stream_name': 'TimeoutStream',
                    'body': {'warning_no_hardhat': {'count': 1}},
                    'type': 'violation',
                    'title': 'Violation alert',
                    'deep_link': '/violations/timeout',
                    'metadata': {},
                }
                headers = {'Authorization': 'Bearer dummy-token'}
                response = self.client.post(
                    '/fcm/send_fcm_notification',
                    json=data,
                    headers=headers,
                )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()['success'])
        self.assertIn('timed out', response.json()['message'])

    @patch(
        'examples.local_notification_server.push_dispatch.'
        'send_fcm_notification_service',
        new_callable=AsyncMock,
    )
    def test_send_fcm_notification_exception(
        self,
        mock_send_fcm: AsyncMock,
    ) -> None:
        """Test FCM notification sending exception branch."""
        _ = mock_send_fcm
        self.mock_redis.set = AsyncMock(return_value=True)
        pipe_mock: MagicMock = MagicMock()
        pipe_mock.hgetall = MagicMock()
        pipe_mock.execute = AsyncMock(return_value=[{b'token': b'en-GB'}])
        self.mock_redis.pipeline.return_value = pipe_mock
        # Patch asyncio.gather to raise Exception
        with patch('asyncio.wait_for', side_effect=Exception('fail!')):
            with patch(
                'examples.local_notification_server.notification_delivery_service.'
                'get_site_notification_user_ids_cached',
                new=AsyncMock(return_value=[1]),
            ):
                data = {
                    'site': 'ExceptionSite',
                    'stream_name': 'ExceptionStream',
                    'body': {'warning_no_hardhat': {'count': 1}},
                    'type': 'violation',
                    'title': 'Violation alert',
                    'deep_link': '/violations/exception',
                    'metadata': {},
                }
                headers = {'Authorization': 'Bearer dummy-token'}
                response = self.client.post(
                    '/fcm/send_fcm_notification',
                    json=data,
                    headers=headers,
                )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()['success'])
        self.assertIn(
            'Failed to send FCM notifications.',
            response.json()['message'],
        )

    def test_send_fcm_notification_duplicate_skipped(self) -> None:
        """It skips duplicated sends claimed by another notification server."""
        self.mock_redis.set = AsyncMock(return_value=False)

        data: dict[str, object] = {
            'site': 'MySite',
            'stream_name': 'MainStream',
            'image_path': None,
            'violation_id': 999,
            'body': {'warning_no_hardhat': {'count': 1}},
            'type': 'violation',
            'title': 'Violation alert',
            'deep_link': '/violations?violation_id=999',
            'metadata': {'violation_id': 999},
        }
        headers: dict[str, str] = {'Authorization': 'Bearer dummy-token'}
        response = self.client.post(
            '/fcm/send_fcm_notification',
            json=data,
            headers=headers,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                'success': True,
                'message': 'Duplicate notification skipped.',
            },
        )
        self.mock_session.execute.assert_not_called()

    def test_send_fcm_notification_rejects_empty_body(self) -> None:
        """Notification input rejects an empty warning payload."""
        data: dict[str, object] = {
            'site': 'AnySite',
            'stream_name': 'AnyStream',
            'image_path': None,
            'violation_id': None,
            'body': {},  # Empty dict
        }
        headers: dict[str, str] = {'Authorization': 'Bearer dummy-token'}
        response = self.client.post(
            '/fcm/send_fcm_notification',
            json=data,
            headers=headers,
        )

        self.assertEqual(response.status_code, 422)


class TestNotificationCenterRoutes(unittest.IsolatedAsyncioTestCase):
    """Unit tests for notification-center route helpers."""

    def setUp(self) -> None:
        """Prepare a current user and async session mock."""
        self.db: AsyncMock = AsyncMock()
        self.user: MagicMock = MagicMock()
        self.user.id = 9

    def _notification(self, **overrides: object) -> SimpleNamespace:
        """Build a lightweight notification object for serialization."""
        values: dict[str, object] = {
            'id': 1,
            'user_id': 9,
            'type': 'violation',
            'title': 'Alert',
            'body': 'Site - Cam\nWarning',
            'deep_link': '/violations?violation_id=1',
            'is_read': False,
            'created_at': datetime.now(timezone.utc),
            'metadata_json': {'violation_id': 1},
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    async def test_list_notifications_filters_and_paginates(self) -> None:
        """List notifications returns one keyset page and cursor."""
        item_result = MagicMock()
        item_result.scalars.return_value.all.return_value = [
            self._notification(),
        ]
        self.db.execute = AsyncMock(return_value=item_result)

        result = await list_notifications(
            status='unread',
            notification_type='violation',
            page_size=20,
            cursor=None,
            db=self.db,
            me=self.user,
        )

        self.assertIsNone(result.next_cursor)
        self.assertEqual(
            result.items[0].deep_link,
            '/violations?violation_id=1',
        )
        self.assertEqual(self.db.execute.await_count, 1)

    async def test_get_notification_unread_count(self) -> None:
        """Unread badge returns only the current user's unread total."""
        result_mock = MagicMock()
        result_mock.scalar.return_value = 7
        self.db.execute = AsyncMock(return_value=result_mock)

        result = await get_notification_unread_count(self.db, self.user)

        self.assertEqual(result.unread_count, 7)

    async def test_get_notification_device_status(self) -> None:
        """Device-status exposes current token diagnostic metadata."""
        token_hash = fcm_token_hash('token-a')
        token = FcmDeviceToken(
            user_id=9,
            device_token_encrypted='encrypted',
            device_token_hash=token_hash,
            platform='web',
            device_lang='zh-TW',
            permission_status='granted',
            created_at=datetime(2026, 6, 27, tzinfo=timezone.utc),
            last_seen_at=datetime(2026, 6, 27, 0, 1, tzinfo=timezone.utc),
            web_vapid_key_available=True,
            web_service_worker_registered=True,
        )
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = [token]
        self.db.execute = AsyncMock(return_value=result_mock)

        result = await get_notification_device_status(self.db, self.user)

        self.assertTrue(result.has_fcm_token)
        self.assertEqual(result.token_count, 1)
        self.assertEqual(result.devices[0].token_hash, token_hash)
        self.assertEqual(result.devices[0].platform, 'web')
        self.assertEqual(result.devices[0].device_lang, 'zh-TW')

    @patch(
        'examples.local_notification_server.notification_delivery_service.'
        'send_fcm_notification_service',
        new_callable=AsyncMock,
    )
    async def test_send_test_notification_success(
        self,
        mock_send: AsyncMock,
    ) -> None:
        """Test notification sends to the current user's stored tokens."""
        mock_send.return_value = FcmSendResult(1, 0)
        rds = MagicMock()
        db = AsyncMock()

        with (
            patch(
                'examples.local_notification_server.notification_delivery_service.'
                'ensure_fcm_token_cache_for_users',
                new=AsyncMock(return_value=1),
            ),
            patch(
                'examples.local_notification_server.notification_delivery_service.'
                'load_active_fcm_device_tokens',
                new=AsyncMock(return_value=['token-a']),
            ),
            patch(
                'examples.local_notification_server.notification_delivery_service.'
                'mark_fcm_tokens_success',
                new=AsyncMock(),
            ) as mock_mark_success,
        ):
            result = await send_test_notification(rds, db, self.user)

        self.assertTrue(result.success)
        self.assertEqual(result.attempted_tokens, 1)
        self.assertEqual(result.success_count, 1)
        mock_send.assert_awaited_once()
        mock_mark_success.assert_awaited_once_with(
            9,
            ['token-a'],
            rds,
            db=db,
        )

    async def test_send_test_notification_without_token(self) -> None:
        """Test notification reports missing FCM token clearly."""
        rds = MagicMock()
        db = AsyncMock()

        with (
            patch(
                'examples.local_notification_server.notification_delivery_service.'
                'ensure_fcm_token_cache_for_users',
                new=AsyncMock(return_value=0),
            ),
            patch(
                'examples.local_notification_server.notification_delivery_service.'
                'load_active_fcm_device_tokens',
                new=AsyncMock(return_value=[]),
            ),
        ):
            result = await send_test_notification(rds, db, self.user)

        self.assertFalse(result.success)
        self.assertEqual(result.attempted_tokens, 0)
        self.assertIn('No FCM token', result.message)

    async def test_mark_notification_read(self) -> None:
        """Marking one notification updates only an owned record."""
        notification = self._notification(is_read=False)
        self.db.scalar = AsyncMock(return_value=notification)
        self.db.commit = AsyncMock()
        self.db.refresh = AsyncMock()

        result = await mark_notification_read(1, self.db, self.user)

        self.assertTrue(notification.is_read)
        self.assertTrue(result.is_read)
        self.db.commit.assert_awaited_once()
        self.db.refresh.assert_awaited_once_with(notification)

    async def test_mark_notification_read_not_found(self) -> None:
        """Unknown or unowned notifications return 404."""
        self.db.scalar = AsyncMock(return_value=None)

        with self.assertRaises(HTTPException) as ctx:
            await mark_notification_read(404, self.db, self.user)

        self.assertEqual(ctx.exception.status_code, 404)

    async def test_mark_all_notifications_read(self) -> None:
        """Read-all returns the number of updated rows."""
        result_mock = MagicMock()
        result_mock.rowcount = 3
        self.db.execute = AsyncMock(return_value=result_mock)
        self.db.commit = AsyncMock()

        result = await mark_all_notifications_read(self.db, self.user)

        self.assertEqual(result.updated_count, 3)
        self.db.commit.assert_awaited_once()

    async def test_delete_notification(self) -> None:
        """Deleting one notification removes only an owned record."""
        notification = self._notification()
        self.db.scalar = AsyncMock(return_value=notification)
        self.db.delete = AsyncMock()
        self.db.commit = AsyncMock()

        result = await delete_notification(1, self.db, self.user)

        self.assertEqual(result, {'message': 'Notification deleted.'})
        self.db.delete.assert_awaited_once_with(notification)
        self.db.commit.assert_awaited_once()


if __name__ == '__main__':
    unittest.main()

"""Pytest \

--cov=examples.local_notification_server.routers \
--cov-report=term-missing \
tests/examples/local_notification_server/routers_test.py
"""


class TestNotificationRouterBranches(unittest.IsolatedAsyncioTestCase):

    """Provide TestNotificationRouterBranches.
    """

    def setUp(self) -> None:
        """Perform setUp.
        """
        self.db = MagicMock()
        self.db.execute = AsyncMock()
        self.db.commit = AsyncMock()
        self.user: Any = SimpleNamespace(
            id=7,
            username='operator',
            role='admin',
        )
        self.redis = MagicMock()

    async def test_notification_reports_non_sendable_tokens(self) -> None:
        """A preflight result distinguishes unavailable from absent tokens."""
        request = SiteNotifyRequest(
            site='Roadwork',
            stream_name='Cam 1',
            body={'warning_no_hardhat': {'count': 1}},
            type='site_alert',
            title='Site alert',
            deep_link='/sites/roadwork',
            metadata={},
        )
        with (
            patch.object(
                routers,
                '_claim_notification_send',
                new=AsyncMock(return_value=True),
            ),
            patch.object(
                routers,
                'get_site_notification_user_ids_cached',
                new=AsyncMock(return_value=[7]),
            ),
            patch.object(
                routers,
                'create_notification_records_for_users',
                new=AsyncMock(return_value=1),
            ),
            patch.object(
                routers,
                'ensure_fcm_token_cache_for_users',
                new=AsyncMock(),
            ),
            patch.object(
                routers,
                'iter_push_tasks_streaming',
                return_value=[],
            ),
            patch.object(
                routers,
                'execute_push_tasks_bounded_streaming',
                new=AsyncMock(return_value=(True, 0, 0, None)),
            ),
            patch.object(
                routers,
                'preflight_from_token_stats',
                return_value={'unique_tokens': 1},
            ),
        ):
            result = await router_endpoints.send_fcm_notification(
                request,
                self.db,
                MagicMock(),
                self.redis,
            )

        self.assertFalse(result['success'])
        self.assertEqual(
            result['message'],
            "Site 'Roadwork' has no sendable device tokens.",
        )

    async def test_list_notifications_accepts_read_filter(self) -> None:
        """Read notifications use the dedicated read condition."""
        item_result = MagicMock()
        item_result.scalars.return_value.all.return_value = []
        self.db.execute.return_value = item_result

        result = await router_endpoints.list_notifications(
            status='read',
            page_size=10,
            cursor=None,
            db=self.db,
            me=self.user,
        )

        self.assertEqual(result.items, [])
        self.assertIsNone(result.next_cursor)

    async def test_structured_fcm_results_track_success_failure_and_invalidity(
        self,
    ) -> None:
        """Structured Firebase responses update delivery state accurately."""
        tokens = ['valid-token', 'invalid-token']
        with (
            patch.object(
                routers,
                'ensure_fcm_token_cache_for_users',
                new=AsyncMock(),
            ),
            patch.object(
                routers,
                'load_active_fcm_device_tokens',
                new=AsyncMock(return_value=tokens),
            ),
            patch.object(
                routers,
                'send_fcm_notification_service',
                new=AsyncMock(
                    side_effect=[
                        FcmSendResult(success_count=2, failure_count=0),
                        FcmSendResult(
                            success_count=0,
                            failure_count=2,
                            invalid_tokens=('invalid-token',),
                        ),
                    ],
                ),
            ),
            patch.object(
                routers,
                'mark_fcm_tokens_success',
                new=AsyncMock(),
            ) as mark_success,
            patch.object(
                routers,
                'mark_fcm_tokens_failure',
                new=AsyncMock(),
            ) as mark_failure,
            patch.object(
                routers,
                'mark_invalid_fcm_tokens_for_users',
                new=AsyncMock(),
            ) as mark_invalid,
        ):
            successful = await router_endpoints.send_test_notification(
                self.redis,
                self.db,
                self.user,
            )
            failed = await router_endpoints.send_test_notification(
                self.redis,
                self.db,
                self.user,
            )

        self.assertTrue(successful.success)
        self.assertFalse(failed.success)
        self.assertEqual(failed.invalid_tokens, 1)
        mark_success.assert_awaited_once_with(
            7,
            tokens,
            self.redis,
            db=self.db,
        )
        mark_failure.assert_awaited_once_with(
            7,
            tokens,
            self.redis,
            'fcm_error',
            db=self.db,
        )
        mark_invalid.assert_awaited_once_with(
            [7],
            {'invalid-token'},
            self.redis,
            db=self.db,
        )

    async def test_fcm_failure_is_recorded(self) -> None:
        """Failed FCM sends retain token failure bookkeeping."""
        with (
            patch.object(
                routers,
                'ensure_fcm_token_cache_for_users',
                new=AsyncMock(),
            ),
            patch.object(
                routers,
                'load_active_fcm_device_tokens',
                new=AsyncMock(return_value=['token']),
            ),
            patch.object(
                routers,
                'send_fcm_notification_service',
                new=AsyncMock(return_value=FcmSendResult(0, 1)),
            ),
            patch.object(
                routers,
                'mark_fcm_tokens_failure',
                new=AsyncMock(),
            ) as mark_failure,
        ):
            result = await router_endpoints.send_test_notification(
                self.redis,
                self.db,
                self.user,
            )

        self.assertFalse(result.success)
        mark_failure.assert_awaited_once()

    async def test_notification_scope_handles_admin_and_missing_group(
        self,
    ) -> None:
        """Super admins see all sites; group-less users have no management
        scope."""
        sites = [SimpleNamespace(id=1, name='All sites')]
        super_admin = SimpleNamespace(
            id=1,
            username='ChangDar',
            role='admin',
            group_id=None,
        )
        group_less = SimpleNamespace(
            id=2,
            username='operator',
            role='admin',
            group_id=None,
        )
        with patch.object(
            site_preference_service,
            'list_sites',
            new=AsyncMock(return_value=sites),
        ) as list_sites:
            result = await site_preference_service._list_notification_scope_sites(
                self.db,
                super_admin,
            )

        self.assertEqual(result, sites)
        list_sites.assert_awaited_once_with(self.db)
        with self.assertRaises(HTTPException) as raised:
            await site_preference_service._list_notification_scope_sites(
                self.db,
                group_less,
            )
        self.assertEqual(raised.exception.status_code, 403)

    async def test_notification_scope_filters_regular_user_by_group(
        self,
    ) -> None:
        """Regular managers receive only sites belonging to their group."""
        user = SimpleNamespace(
            id=3,
            username='manager',
            role='admin',
            group_id=12,
        )
        sites = [SimpleNamespace(id=4, name='Group site')]
        with patch.object(
            site_preference_service,
            'list_sites',
            new=AsyncMock(return_value=sites),
        ) as list_sites:
            result = await site_preference_service._list_notification_scope_sites(
                self.db, user,
            )

        self.assertEqual(result, sites)
        list_sites.assert_awaited_once_with(self.db, group_id=12)

    async def test_list_preferences_uses_explicit_and_effective_values(
        self,
    ) -> None:
        """Explicit preferences override effective site-access defaults."""
        site = SimpleNamespace(
            id=1,
            name='Roadwork',
            groups=[SimpleNamespace(name='Team A')],
        )
        pref_rows = MagicMock()
        pref_rows.all.return_value = [(1, False)]
        self.db.execute.return_value = pref_rows
        with (
            patch.object(
                site_preference_service,
                '_list_notification_scope_sites',
                new=AsyncMock(return_value=[site]),
            ),
            patch.object(
                site_preference_service,
                'list_effective_sites_for_user',
                new=AsyncMock(return_value=[site]),
            ),
        ):
            result = await site_preference_service.list_site_notification_preferences(
                self.db,
                self.user,
            )

        self.assertEqual(len(result), 1)
        self.assertFalse(result[0].is_enabled)
        self.assertEqual(result[0].group_name, 'Team A')

    async def test_empty_preference_scope_needs_no_database_queries(
        self,
    ) -> None:
        """Users with no visible sites receive an empty preference list."""
        with patch.object(
            site_preference_service,
            '_list_notification_scope_sites',
            new=AsyncMock(return_value=[]),
        ):
            result = await site_preference_service.list_site_notification_preferences(
                self.db,
                self.user,
            )

        self.assertEqual(result, [])
        self.db.execute.assert_not_awaited()

    async def test_preference_update_ignores_sites_omitted_from_payload(
        self,
    ) -> None:
        """A partial request preserves a second allowed site's preference."""
        sites = [
            SimpleNamespace(id=1, name='Included site'),
            SimpleNamespace(id=2, name='Unchanged site'),
        ]
        pref_result = MagicMock()
        pref_result.scalars.return_value.all.return_value = []
        self.db.execute.return_value = pref_result
        payload = SiteNotificationPreferenceUpdateRequest(
            preferences=[
                SiteNotificationPreferenceIn(site_id=1, is_enabled=True),
            ],
        )
        with (
            patch.object(
                site_preference_service,
                '_list_notification_scope_sites',
                new=AsyncMock(return_value=sites),
            ),
            patch.object(
                site_preference_service,
                'list_effective_sites_for_user',
                new=AsyncMock(return_value=[sites[0]]),
            ),
            patch.object(
                site_preference_service,
                'list_site_notification_preferences',
                new=AsyncMock(return_value=[]),
            ),
        ):
            result = await site_preference_service.update_site_notification_preferences(
                payload,
                self.db,
                self.user,
                self.redis,
            )

        self.assertEqual(result, [])
        self.assertEqual(self.db.add.call_count, 1)
        self.db.commit.assert_awaited_once()

    async def test_preference_update_rejects_out_of_scope_requests(
        self,
    ) -> None:
        """Out-of-scope site preference updates are rejected."""
        site = SimpleNamespace(id=1, name='Allowed site')
        invalid = SiteNotificationPreferenceUpdateRequest(
            preferences=[
                SiteNotificationPreferenceIn(site_id=2, is_enabled=True),
            ],
        )
        with (
            patch.object(
                site_preference_service,
                '_list_notification_scope_sites',
                new=AsyncMock(return_value=[site]),
            ),
        ):
            with self.assertRaises(HTTPException) as raised:
                await site_preference_service.update_site_notification_preferences(
                    invalid,
                    self.db,
                    self.user,
                    self.redis,
                )

        self.assertEqual(raised.exception.status_code, 403)

    async def test_preference_update_changes_value_and_refreshes_cache(
        self,
    ) -> None:
        """Changed explicit preferences update their row and recipient
        cache."""
        site = SimpleNamespace(id=1, name='Changed site')
        preference = SimpleNamespace(site_id=1, is_enabled=False)
        pref_result = MagicMock()
        pref_result.scalars.return_value.all.return_value = [preference]
        self.db.execute.return_value = pref_result
        payload = SiteNotificationPreferenceUpdateRequest(
            preferences=[
                SiteNotificationPreferenceIn(site_id=1, is_enabled=True),
            ],
        )
        with (
            patch.object(
                site_preference_service,
                '_list_notification_scope_sites',
                new=AsyncMock(return_value=[site]),
            ),
            patch.object(
                site_preference_service,
                'list_effective_sites_for_user',
                new=AsyncMock(return_value=[]),
            ),
            patch.object(
                site_preference_service,
                'refresh_site_notification_user_cache',
                new=AsyncMock(),
            ) as refresh_cache,
            patch.object(
                site_preference_service,
                'list_site_notification_preferences',
                new=AsyncMock(return_value=[]),
            ),
        ):
            await site_preference_service.update_site_notification_preferences(
                payload,
                self.db,
                self.user,
                self.redis,
            )

        self.assertTrue(preference.is_enabled)
        refresh_cache.assert_awaited_once_with(
            'Changed site',
            self.db,
            self.redis,
        )
