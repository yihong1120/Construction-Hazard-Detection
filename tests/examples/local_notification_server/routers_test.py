from __future__ import annotations

import unittest
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from examples.auth.database import get_db
from examples.auth.jwt_config import jwt_access
from examples.auth.redis_pool import get_redis_pool
from examples.local_notification_server.routers import router


def mock_jwt_access() -> MagicMock:
    """
    Mock JWT credentials to avoid the need for a real token in tests.

    Returns:
        MagicMock: A mock object with dummy jti and sub attributes.
    """
    return MagicMock(jti='dummy-jti', sub='dummy-sub')


class TestLocalNotificationServer(unittest.TestCase):
    """
    Unit test suite for routes in the local notification server.
    """

    def setUp(self) -> None:
        """
        Set up a FastAPI app, test client,
        and mock dependencies before each test.
        """
        self.app: FastAPI = FastAPI()
        self.app.include_router(router, prefix='/fcm')
        self.client: TestClient = TestClient(self.app)
        self.mock_session: AsyncMock = AsyncMock()

        # Redis mock: use MagicMock for correct pipeline chain
        self.mock_redis: MagicMock = MagicMock()
        self.mock_redis.hset = AsyncMock()
        self.mock_redis.hdel = AsyncMock()
        # pipeline mock will be set in each test as needed

        async def override_get_db() -> AsyncIterator[AsyncMock]:
            """
            Override dependency for database session,
            returning a mock session object.

            Yields:
                AsyncMock: The mocked database session.
            """
            yield self.mock_session

        async def override_get_redis_pool() -> MagicMock:
            """
            Override dependency for Redis connection,
            returning a mock Redis object.

            Returns:
                MagicMock: The mocked Redis connection.
            """
            return self.mock_redis

        # Override the dependencies with the mocks
        self.app.dependency_overrides[get_db] = override_get_db
        self.app.dependency_overrides[get_redis_pool] = override_get_redis_pool
        self.app.dependency_overrides[jwt_access] = mock_jwt_access

    def tearDown(self) -> None:
        """
        Clear overrides after each test to avoid leakage between test cases.
        """
        self.app.dependency_overrides.clear()

    # ------------------------------------------------------------------------
    # Helper functions for simulating database queries
    # ------------------------------------------------------------------------
    def mock_no_user_in_db(self) -> None:
        """
        Simulate a scenario where the queried user is not found
        in the database.
        """
        result: MagicMock = MagicMock()
        result.unique.return_value = result
        result.scalar_one_or_none.return_value = None
        result.scalar.return_value = None  # Add this for the scalar() check
        self.mock_session.execute = AsyncMock(return_value=result)

    def mock_user_in_db(self, user_id: int) -> MagicMock:
        """
        Simulate a scenario where the queried user is found in the database.

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
        return mock_user

    def mock_site_in_db(
        self,
        site_name: str,
        users: list[MagicMock] | None = None,
    ) -> MagicMock:
        """
        Simulate a scenario where a Site is found in the database.

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

        result: MagicMock = MagicMock()
        result.unique.return_value = result
        result.scalar_one_or_none.return_value = mock_site
        self.mock_session.execute = AsyncMock(return_value=result)
        return mock_site

    # ------------------------------------------------------------------------
    # Tests for /store_token (POST) - storing an FCM token
    # ------------------------------------------------------------------------
    def test_store_fcm_token_user_not_found(self) -> None:
        """
        Test storing a token when the user is not found in the database.
        Expect a 404 error response.
        """
        self.mock_no_user_in_db()
        # Patch pipeline to avoid await error
        pipe_mock = MagicMock()
        pipe_mock.hset = MagicMock()
        pipe_mock.expire = MagicMock()
        pipe_mock.execute = AsyncMock()
        self.mock_redis.pipeline.return_value = pipe_mock
        data: dict[str, object] = {
            'user_id': 999,
            'device_token': 'test-token-999',
        }
        response = self.client.post('/fcm/store_token', json=data)
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {'detail': 'User not found'})

    def test_store_fcm_token_success(self) -> None:
        """Test successful token storage for an existing user."""
        self.mock_user_in_db(user_id=123)
        # Patch pipeline
        pipe_mock = MagicMock()
        pipe_mock.hset = MagicMock()
        pipe_mock.expire = MagicMock()
        pipe_mock.execute = AsyncMock()
        self.mock_redis.pipeline.return_value = pipe_mock
        data: dict[str, object] = {
            'user_id': 123,
            'device_token': 'my-test-token',
        }
        response = self.client.post('/fcm/store_token', json=data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(), {
                'message': 'Token stored successfully.',
            },
        )
        pipe_mock.hset.assert_called_once_with(
            'fcm_tokens:123', 'my-test-token', 'en-GB',
        )

    def test_store_fcm_token_with_device_lang(self) -> None:
        """
        Test token storage with a specific device language specified.
        """
        self.mock_user_in_db(user_id=123)
        pipe_mock = MagicMock()
        pipe_mock.hset = MagicMock()
        pipe_mock.expire = MagicMock()
        pipe_mock.execute = AsyncMock()
        self.mock_redis.pipeline.return_value = pipe_mock
        data: dict[str, object] = {
            'user_id': 123, 'device_token': 'test-token', 'device_lang': 'zh',
        }
        response = self.client.post('/fcm/store_token', json=data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(), {
                'message': 'Token stored successfully.',
            },
        )
        pipe_mock.hset.assert_called_once_with(
            'fcm_tokens:123', 'test-token', 'zh',
        )

    # ------------------------------------------------------------------------
    # Tests for /delete_token (DELETE) - removing an FCM token from Redis
    # ------------------------------------------------------------------------
    def test_delete_fcm_token_user_not_found(self) -> None:
        """
        Test deleting a token when the user is not found in the database.
        The route returns 200 with a message indicating the user is not found.
        """
        self.mock_no_user_in_db()
        pipe_mock = MagicMock()
        pipe_mock.hdel = MagicMock()
        pipe_mock.hlen = MagicMock()
        pipe_mock.execute = AsyncMock()
        self.mock_redis.pipeline.return_value = pipe_mock
        data: dict[str, object] = {
            'user_id': 999,
            'device_token': 'unknown-token',
        }
        response = self.client.request(
            'DELETE', '/fcm/delete_token', json=data,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'message': 'User not found.'})

    def test_delete_fcm_token_not_in_redis(self) -> None:
        """
        Test attempting to delete a token that does not exist in Redis.
        """
        self.mock_user_in_db(user_id=10)
        pipe_mock = MagicMock()
        pipe_mock.hdel = MagicMock()
        pipe_mock.hlen = MagicMock()
        pipe_mock.execute = AsyncMock(return_value=[0, 1])
        self.mock_redis.pipeline.return_value = pipe_mock
        data: dict[str, object] = {
            'user_id': 10,
            'device_token': 'non-existent-token',
        }
        response = self.client.request(
            'DELETE', '/fcm/delete_token', json=data,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(), {
                'message': 'Token not found in Redis hash.',
            },
        )

    def test_delete_fcm_token_success(self) -> None:
        """
        Test successfully deleting an existing token in Redis.
        """
        self.mock_user_in_db(user_id=10)
        pipe_mock = MagicMock()
        pipe_mock.hdel = MagicMock()
        pipe_mock.hlen = MagicMock()
        pipe_mock.execute = AsyncMock(return_value=[1, 1])
        self.mock_redis.pipeline.return_value = pipe_mock
        data: dict[str, object] = {
            'user_id': 10,
            'device_token': 'existing-token',
        }
        response = self.client.request(
            'DELETE', '/fcm/delete_token', json=data,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'message': 'Token deleted.'})
        pipe_mock.hdel.assert_called_once_with(
            'fcm_tokens:10', 'existing-token',
        )

    def test_delete_fcm_token_delete_key_when_no_tokens(self) -> None:
        """
        Test that the Redis key is deleted
        when no tokens remain after deletion.
        """
        self.mock_user_in_db(user_id=10)
        pipe_mock = MagicMock()
        pipe_mock.hdel = MagicMock()
        pipe_mock.hlen = MagicMock()
        pipe_mock.execute = AsyncMock(return_value=[1, 0])
        self.mock_redis.pipeline.return_value = pipe_mock
        self.mock_redis.delete = AsyncMock()
        data: dict[str, object] = {
            'user_id': 10,
            'device_token': 'existing-token',
        }
        response = self.client.request(
            'DELETE', '/fcm/delete_token', json=data,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'message': 'Token deleted.'})
        self.mock_redis.delete.assert_awaited_once_with('fcm_tokens:10')

    # ------------------------------------------------------------------------
    # Tests for /send_fcm_notification (POST) - sending notifications
    # ------------------------------------------------------------------------
    def test_send_fcm_notification_site_not_found(self) -> None:
        """
        Test sending a notification for a non-existent site.
        Expects a 200 response with success=False and a specific message.
        """
        # Simulate a DB query that returns no Site
        result: MagicMock = MagicMock()
        result.unique.return_value = result
        result.scalar_one_or_none.return_value = None
        self.mock_session.execute = AsyncMock(return_value=result)

        data: dict[str, object] = {
            'site': 'MissingSite',
            'stream_name': 'TestStream',
            'image_path': None,
            'violation_id': None,
            'body': {'en': {'helmet': 1}},  # <-- valid schema
        }
        headers: dict[str, str] = {'Authorization': 'Bearer dummy-token'}
        response = self.client.post(
            '/fcm/send_fcm_notification', json=data, headers=headers,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(), {
                'success': False,
                'message': "Site 'MissingSite' not found or has no users.",
            },
        )

    def test_send_fcm_notification_site_no_users(self) -> None:
        """
        Test sending a notification for a site that has no users.
        Expects a 200 response with success=False and a relevant message.
        """
        self.mock_site_in_db('EmptySite', [])

        data: dict[str, object] = {
            'site': 'EmptySite',
            'stream_name': 'EmptyStream',
            'image_path': None,
            'violation_id': None,
            'body': {'en': {'helmet': 1}},  # <-- valid schema
        }
        headers: dict[str, str] = {'Authorization': 'Bearer dummy-token'}
        response = self.client.post(
            '/fcm/send_fcm_notification', json=data, headers=headers,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(), {
                'success': False,
                'message': "Site 'EmptySite' not found or has no users.",
            },
        )

    def test_send_fcm_notification_no_tokens(self) -> None:
        """
        Test sending a notification where users exist,
        but none have tokens in Redis.
        """
        user1: MagicMock = MagicMock(id=1)
        user2: MagicMock = MagicMock(id=2)
        self.mock_site_in_db('SiteWithNoTokens', [user1, user2])

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
            'body': {'en': {'helmet': 1}},
        }
        headers: dict[str, str] = {'Authorization': 'Bearer dummy-token'}
        response = self.client.post(
            '/fcm/send_fcm_notification', json=data, headers=headers,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(), {
                'success': False,
                'message': "Site 'SiteWithNoTokens' has no device tokens.",
            },
        )

    @patch(
        'examples.local_notification_server.services.'
        'send_fcm_notification_service',
        new_callable=AsyncMock,
    )
    def test_send_fcm_notification_success(
        self,
        mock_send_fcm: AsyncMock,
    ) -> None:
        """
        Test successfully sending a notification when a site, users,
        and user tokens in Redis are all available.

        Args:
            mock_send_fcm (AsyncMock): Mocked FCM notification sending service.
        """
        user: MagicMock = MagicMock(id=42)
        self.mock_site_in_db('MySite', [user])

        pipe_mock: MagicMock = MagicMock()
        pipe_mock.hgetall = MagicMock()
        pipe_mock.execute = AsyncMock(
            return_value=[{b'tokenA': b'en', b'tokenB': b'zh'}],
        )
        self.mock_redis.pipeline.return_value = pipe_mock

        mock_send_fcm.return_value = True

        data: dict[str, object] = {
            'site': 'MySite',
            'stream_name': 'MainStream',
            'image_path': None,
            'violation_id': 999,
            'body': {'en': {'helmet': 1}},
        }
        headers: dict[str, str] = {'Authorization': 'Bearer dummy-token'}
        response = self.client.post(
            '/fcm/send_fcm_notification', json=data, headers=headers,
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
        'examples.local_notification_server.services.'
        'send_fcm_notification_service',
        new_callable=AsyncMock,
    )
    def test_send_fcm_notification_all_fail(
        self,
        mock_send_fcm: AsyncMock,
    ) -> None:
        """
        Test overall failure when all notifications fail to send.

        Args:
            mock_send_fcm (AsyncMock): Mocked FCM notification sending service.
        """
        user: MagicMock = MagicMock(id=99)
        self.mock_site_in_db('MySite', [user])

        pipe_mock: MagicMock = MagicMock()
        pipe_mock.hgetall = MagicMock()
        pipe_mock.execute = AsyncMock(
            return_value=[{b'tA': b'en', b'tB': b'zh'}],
        )
        self.mock_redis.pipeline.return_value = pipe_mock

        mock_send_fcm.return_value = False

        data: dict[str, object] = {
            'site': 'MySite',
            'stream_name': 'FailStream',
            'violation_id': 123,
            'body': {'en': {'helmet': 1}},
        }
        headers: dict[str, str] = {'Authorization': 'Bearer dummy-token'}
        response = self.client.post(
            '/fcm/send_fcm_notification', json=data, headers=headers,
        )

        self.assertEqual(response.status_code, 200)
        resp_json = response.json()
        self.assertFalse(resp_json['success'])
        self.assertIn('batches succeeded', resp_json['message'])
        self.assertIn('stats', resp_json)
        self.assertEqual(mock_send_fcm.await_count, 2)

    @patch(
        'examples.local_notification_server.services.'
        'send_fcm_notification_service',
        new_callable=AsyncMock,
    )
    def test_send_fcm_notification_timeout(
        self, mock_send_fcm: AsyncMock,
    ) -> None:
        """
        Test FCM notification sending timeout branch.
        """
        user: MagicMock = MagicMock(id=1)
        self.mock_site_in_db('TimeoutSite', [user])
        pipe_mock: MagicMock = MagicMock()
        pipe_mock.hgetall = MagicMock()
        pipe_mock.execute = AsyncMock(return_value=[{b'token': b'en'}])
        self.mock_redis.pipeline.return_value = pipe_mock
        # Patch asyncio.wait_for to raise TimeoutError
        import asyncio as real_asyncio
        with patch('asyncio.wait_for', side_effect=real_asyncio.TimeoutError):
            data = {
                'site': 'TimeoutSite',
                'stream_name': 'TimeoutStream',
                'body': {'en': {'helmet': 1}},
            }
            headers = {'Authorization': 'Bearer dummy-token'}
            response = self.client.post(
                '/fcm/send_fcm_notification', json=data, headers=headers,
            )
            self.assertEqual(response.status_code, 200)
            self.assertFalse(response.json()['success'])
            self.assertIn('timed out', response.json()['message'])

    @patch(
        'examples.local_notification_server.services.'
        'send_fcm_notification_service',
        new_callable=AsyncMock,
    )
    def test_send_fcm_notification_exception(
        self, mock_send_fcm: AsyncMock,
    ) -> None:
        """
        Test FCM notification sending exception branch.
        """
        _ = mock_send_fcm  # Mark as used to avoid unused argument warning
        user: MagicMock = MagicMock(id=1)
        self.mock_site_in_db('ExceptionSite', [user])
        pipe_mock: MagicMock = MagicMock()
        pipe_mock.hgetall = MagicMock()
        pipe_mock.execute = AsyncMock(return_value=[{b'token': b'en'}])
        self.mock_redis.pipeline.return_value = pipe_mock
        # Patch asyncio.gather to raise Exception
        with patch('asyncio.wait_for', side_effect=Exception('fail!')):
            data = {
                'site': 'ExceptionSite',
                'stream_name': 'ExceptionStream',
                'body': {'en': {'helmet': 1}},
            }
            headers = {'Authorization': 'Bearer dummy-token'}
            response = self.client.post(
                '/fcm/send_fcm_notification', json=data, headers=headers,
            )
            self.assertEqual(response.status_code, 200)
            self.assertFalse(response.json()['success'])
            self.assertIn(
                'Failed to send FCM notifications.',
                response.json()['message'],
            )

    def test_send_fcm_notification_body_empty(self) -> None:
        """
        Test sending a notification with an empty body.
        Expects a 200 response with success=False and a specific message.
        """
        data: dict[str, object] = {
            'site': 'AnySite',
            'stream_name': 'AnyStream',
            'image_path': None,
            'violation_id': None,
            'body': {},  # Empty dict
        }
        headers: dict[str, str] = {'Authorization': 'Bearer dummy-token'}
        response = self.client.post(
            '/fcm/send_fcm_notification', json=data, headers=headers,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(), {
                'success': False,
                'message': 'Body is empty, nothing to send.',
            },
        )


if __name__ == '__main__':
    unittest.main()

"""
pytest \
    --cov=examples.local_notification_server.routers \
    --cov-report=term-missing \
    tests/examples/local_notification_server/routers_test.py
"""
