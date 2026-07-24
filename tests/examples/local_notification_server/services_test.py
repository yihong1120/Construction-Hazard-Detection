from __future__ import annotations

import asyncio
import unittest
from collections import defaultdict
from collections.abc import Awaitable
from collections.abc import Coroutine
from collections.abc import Iterable
from datetime import datetime
from datetime import timezone
from types import SimpleNamespace
from typing import Any
from typing import DefaultDict
from typing import TypeVar
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from cryptography.fernet import Fernet

from examples.local_notification_server import services as svc
from examples.local_notification_server.schemas import SiteNotifyRequest
from examples.local_notification_server.schemas import TokenRequest

# Generic type variable for async helpers
T = TypeVar('T')


class TestServices(unittest.TestCase):
    """
    Unit tests for notification services utilities.
    """

    def setUp(self) -> None:
        """Initialise a clean cache before each test.

        Ensures tests remain isolated and do not depend on call order.
        """
        self.maxDiff = None

    def test_decode_lang_token_map_basic(self) -> None:
        """It decodes raw Redis maps into a language-to-tokens mapping.

        Ensures empty language entries are skipped instead of retry.
        """
        raw = [
            {b't1': b'en-GB', b't2': b'zh'},
            {b't3': b'', b't4': b'ja'},
        ]
        got = svc._decode_lang_token_map(raw)
        # Convert to plain dict for assertion
        got_dict = {k: list(v) for k, v in got.items()}
        self.assertEqual(got_dict['en-GB'], ['t1'])
        self.assertEqual(got_dict['zh-TW'], ['t2'])
        self.assertEqual(got_dict['ja-JP'], ['t4'])

    def test_decode_lang_token_map_accepts_str_values(self) -> None:
        """It accepts decoded Redis hash values as well as bytes."""
        raw = [
            {'t1': 'en-US', 't2': 'zh_TW'},
        ]

        got = svc._decode_lang_token_map(raw)
        got_dict = {k: list(v) for k, v in got.items()}

        self.assertEqual(got_dict['en-GB'], ['t1'])
        self.assertEqual(got_dict['zh-TW'], ['t2'])

    def test_decode_lang_token_map_normalises_and_deduplicates_tokens(
        self,
    ) -> None:
        """It merges language aliases and skips duplicated device tokens."""
        raw = [
            {b't1': b'zh', b't2': b'zh-TW'},
            {b't1': b'en', b't3': b'en-US'},
        ]

        got = svc._decode_lang_token_map(raw)
        got_dict = {k: list(v) for k, v in got.items()}

        self.assertEqual(got_dict['zh-TW'], ['t1', 't2'])
        self.assertEqual(got_dict['en-GB'], ['t3'])

    def test_get_lang_to_tokens_groups_by_lang(self) -> None:
        """It groups tokens by language using the Redis pipeline."""
        user_ids = [1, 2]
        # Mock redis pipeline
        pipe = MagicMock()
        pipe.hgetall = MagicMock()
        pipe.execute = AsyncMock(
            return_value=[{b'a': b'en-GB'}, {b'b': b'zh'}],
        )
        rds = MagicMock()
        rds.pipeline.return_value = pipe

        got = self._run_async(svc._get_lang_to_tokens(user_ids, rds))
        got_dict = {k: list(v) for k, v in got.items()}
        self.assertEqual(got_dict['en-GB'], ['a'])
        self.assertEqual(got_dict['zh-TW'], ['b'])
        # Ensure hgetall called for each user
        self.assertEqual(pipe.hgetall.call_count, 2)

    def test_get_lang_to_tokens_fetches_in_chunks(self) -> None:
        """It limits each Redis pipeline to a bounded number of users."""
        user_ids = [1, 2, 3]
        pipe_one = MagicMock()
        pipe_one.hgetall = MagicMock()
        pipe_one.execute = AsyncMock(
            return_value=[{b'a': b'en-GB'}, {b'b': b'zh-TW'}],
        )
        pipe_two = MagicMock()
        pipe_two.hgetall = MagicMock()
        pipe_two.execute = AsyncMock(return_value=[{b'c': b'en-GB'}])
        rds = MagicMock()
        rds.pipeline.side_effect = [pipe_one, pipe_two]

        with patch.object(svc, '_token_fetch_chunk_size', 2):
            got = self._run_async(svc._get_lang_to_tokens(user_ids, rds))

        got_dict = {k: list(v) for k, v in got.items()}
        self.assertEqual(got_dict['en-GB'], ['a', 'c'])
        self.assertEqual(got_dict['zh-TW'], ['b'])
        self.assertEqual(rds.pipeline.call_count, 2)
        self.assertEqual(pipe_one.hgetall.call_count, 2)
        self.assertEqual(pipe_two.hgetall.call_count, 1)

    def test_notification_data_includes_deep_link(self) -> None:
        """FCM data and stored notifications share the same deep link."""
        req = SiteNotifyRequest(
            site='S1',
            stream_name='Cam1',
            body={'warning_no_hardhat': {'count': 1}},
            violation_id=123,
        )

        data = svc._notification_data(req)

        self.assertEqual(data['type'], 'violation')
        self.assertEqual(data['violation_id'], '123')
        self.assertEqual(data['deep_link'], '/violations?violation_id=123')

    def test_notification_data_uses_request_deep_link(self) -> None:
        """A request-provided deep link is preserved."""
        req = SiteNotifyRequest(
            site='S1',
            stream_name='Cam1',
            body={'warning_no_hardhat': {'count': 1}},
            type='site_alert',
            deep_link='/sites/1',
        )

        data = svc._notification_data(req)

        self.assertEqual(data['type'], 'site_alert')
        self.assertEqual(data['deep_link'], '/sites/1')

    def test_create_notification_records_for_users(self) -> None:
        """It writes one notification-center record per distinct recipient."""
        req = SiteNotifyRequest(
            site='S1',
            stream_name='Cam1',
            body={'warning_no_hardhat': {'count': 1}},
            image_path='https://example.com/v.jpg',
            violation_id=123,
        )
        db = MagicMock()
        db.add_all = MagicMock()
        db.commit = AsyncMock()

        count = self._run_async(
            svc.create_notification_records_for_users(
                req,
                [5, 5, 6],
                db,
            ),
        )

        self.assertEqual(count, 2)
        records = db.add_all.call_args.args[0]
        self.assertEqual([record.user_id for record in records], [5, 6])
        self.assertEqual(records[0].type, 'violation')
        self.assertEqual(
            records[0].deep_link,
            '/violations?violation_id=123',
        )
        self.assertEqual(records[0].metadata_json['violation_id'], 123)
        db.commit.assert_awaited_once()

    @patch(
        'examples.local_notification_server.services.'
        'send_fcm_notification_service',
        new_callable=AsyncMock,
    )
    def test_iter_push_tasks_streaming_batches_redis_chunks(
        self,
        mock_send: AsyncMock,
    ) -> None:
        """It streams Redis chunks into FCM batches incrementally."""
        req = SiteNotifyRequest(
            site='S1',
            stream_name='Cam1',
            body={'warning_no_hardhat': {'count': 1}},
            image_path=None,
            violation_id=123,
        )
        mock_send.return_value = True

        pipe_one = MagicMock()
        pipe_one.hgetall = MagicMock()
        pipe_one.execute = AsyncMock(
            return_value=[{b'a': b'zh'}, {b'b': b'zh'}],
        )
        pipe_two = MagicMock()
        pipe_two.hgetall = MagicMock()
        pipe_two.execute = AsyncMock(
            return_value=[{b'c': b'zh', b'd': b'unknown'}],
        )
        rds = MagicMock()
        rds.pipeline.side_effect = [pipe_one, pipe_two]

        async def collect() -> list[svc.PushTaskResult]:
            """Support collect."""
            tasks: list[Awaitable[svc.PushTaskResult]] = []
            async for task in svc._iter_push_tasks_streaming(
                req, [1, 2, 3], rds,
            ):
                tasks.append(task)
            import asyncio
            return await asyncio.gather(*tasks)

        with patch.object(svc, '_token_fetch_chunk_size', 2):
            with patch.object(svc, '_fcm_batch_size', 2):
                results = self._run_async(collect())

        self.assertEqual(results, [True, True])
        self.assertEqual(rds.pipeline.call_count, 2)
        self.assertEqual(pipe_one.hgetall.call_count, 2)
        self.assertEqual(pipe_two.hgetall.call_count, 1)
        self.assertEqual(mock_send.await_count, 2)
        self.assertEqual(
            mock_send.call_args_list[0].kwargs['device_tokens'],
            ['a', 'b'],
        )
        self.assertEqual(
            mock_send.call_args_list[1].kwargs['device_tokens'],
            ['c'],
        )

    def test_diagnose_push_preflight_reports_token_state(self) -> None:
        """It explains why recipients do or do not produce push batches."""
        req = SiteNotifyRequest(
            site='S1',
            stream_name='Cam1',
            body={'warning_no_hardhat': {'count': 1}},
        )
        pipe = MagicMock()
        pipe.hgetall = MagicMock()
        pipe.execute = AsyncMock(
            return_value=[
                {b'a': b'zh', b'b': b'unknown'},
                {},
                {b'a': b'en'},
            ],
        )
        rds = MagicMock()
        rds.pipeline.return_value = pipe

        stats = self._run_async(
            svc.diagnose_push_preflight(req, [1, 2, 3], rds),
        )

        self.assertEqual(stats['recipient_users'], 3)
        self.assertEqual(stats['users_with_tokens'], 2)
        self.assertEqual(stats['token_entries'], 3)
        self.assertEqual(stats['unique_tokens'], 2)
        self.assertEqual(stats['duplicate_tokens'], 1)
        self.assertEqual(stats['sendable_tokens'], 1)
        self.assertEqual(stats['unsupported_language_tokens'], 1)
        self.assertEqual(stats['tokens_by_language'], {'zh-TW': 1})
        self.assertEqual(stats['unsupported_languages'], {'unknown': 1})

    def test_get_site_notification_user_ids_cached_hit(self) -> None:
        """It returns indexed user IDs from Redis when present."""
        mock_session = AsyncMock()
        mock_redis = AsyncMock()
        mock_redis.exists.return_value = 1
        mock_redis.smembers.return_value = {b'7', b'3'}

        user_ids = self._run_async(
            svc.get_site_notification_user_ids_cached(
                'SiteA',
                mock_session,
                mock_redis,
            ),
        )

        if user_ids is None:
            self.fail('Expected cached user ids, got None.')
        self.assertCountEqual(user_ids, [3, 7])
        mock_session.execute.assert_not_called()

    def test_get_site_notification_user_ids_cached_miss(self) -> None:
        """It queries DB on index miss and rebuilds the Redis recipient set."""
        site_result = MagicMock()
        site_result.first.return_value = (11, 2)
        ids_result = MagicMock()
        ids_result.scalars.return_value.all.return_value = [8, 9]
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(side_effect=[site_result, ids_result])
        mock_redis = MagicMock()
        mock_redis.exists = AsyncMock(return_value=0)
        mock_redis.set = AsyncMock(return_value=True)
        mock_redis.delete = AsyncMock()
        mock_redis.exists.return_value = 0
        pipe = MagicMock()
        pipe.delete = MagicMock()
        pipe.sadd = MagicMock()
        pipe.set = MagicMock()
        pipe.execute = AsyncMock()
        mock_redis.pipeline.return_value = pipe

        user_ids = self._run_async(
            svc.get_site_notification_user_ids_cached(
                'SiteA',
                mock_session,
                mock_redis,
            ),
        )

        self.assertEqual(user_ids, [8, 9])
        mock_redis.set.assert_awaited_once_with(
            'site_notification_users_lock:SiteA',
            '1',
            ex=svc._recipient_index_lock_seconds,
            nx=True,
        )
        pipe.delete.assert_called_once_with('site_notification_users:SiteA')
        pipe.sadd.assert_called_once_with(
            'site_notification_users:SiteA', 8, 9,
        )
        pipe.set.assert_called_once_with(
            'site_notification_users_ready:SiteA',
            '1',
        )

    def test_get_site_notification_user_ids_cached_miss_not_found(
        self,
    ) -> None:
        """It clears Redis index keys when the site no longer exists."""
        site_result = MagicMock()
        site_result.first.return_value = None
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=site_result)
        mock_redis = MagicMock()
        mock_redis.exists = AsyncMock(return_value=0)
        mock_redis.set = AsyncMock(return_value=True)
        mock_redis.delete = AsyncMock()

        user_ids = self._run_async(
            svc.get_site_notification_user_ids_cached(
                'Missing',
                mock_session,
                mock_redis,
            ),
        )

        self.assertIsNone(user_ids)
        self.assertEqual(mock_redis.delete.await_count, 2)
        mock_redis.delete.assert_any_await(
            'site_notification_users:Missing',
            'site_notification_users_ready:Missing',
            'site_notification_users_lock:Missing',
        )
        mock_redis.delete.assert_any_await(
            'site_notification_users_lock:Missing',
        )

    def test_refresh_site_notification_user_cache_empty_site(self) -> None:
        """It marks an existing site as ready even when no users are
        subscribed.
        """
        site_result = MagicMock()
        site_result.first.return_value = (11, 2)
        ids_result = MagicMock()
        ids_result.scalars.return_value.all.return_value = []
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(side_effect=[site_result, ids_result])
        mock_redis = MagicMock()
        pipe = MagicMock()
        pipe.delete = MagicMock()
        pipe.sadd = MagicMock()
        pipe.set = MagicMock()
        pipe.execute = AsyncMock()
        mock_redis.pipeline.return_value = pipe

        user_ids = self._run_async(
            svc.refresh_site_notification_user_cache(
                'SiteA',
                mock_session,
                mock_redis,
            ),
        )

        self.assertEqual(user_ids, [])
        pipe.delete.assert_called_once_with('site_notification_users:SiteA')
        pipe.sadd.assert_not_called()
        pipe.set.assert_called_once_with(
            'site_notification_users_ready:SiteA',
            '1',
        )

    def test_get_site_notification_user_ids_cached_waits_for_builder(
        self,
    ) -> None:
        """It waits for another worker rebuilding the same Redis index."""
        mock_session = AsyncMock()
        mock_redis = MagicMock()
        mock_redis.exists = AsyncMock(side_effect=[0, 1])
        mock_redis.set = AsyncMock(return_value=False)
        mock_redis.smembers = AsyncMock(return_value={b'4', b'2'})
        with patch('asyncio.sleep', new=AsyncMock()):
            user_ids = self._run_async(
                svc.get_site_notification_user_ids_cached(
                    'SiteA',
                    mock_session,
                    mock_redis,
                ),
            )

        if user_ids is None:
            self.fail('Expected cached user ids, got None.')
        self.assertCountEqual(user_ids, [2, 4])
        mock_session.execute.assert_not_called()

    def test_get_site_notification_user_ids_cached_not_found(self) -> None:
        """It yields ``None`` when the site does not exist."""
        site_result = MagicMock()
        site_result.first.return_value = None
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=site_result)
        mock_redis = MagicMock()
        mock_redis.exists = AsyncMock(return_value=0)
        mock_redis.set = AsyncMock(return_value=True)
        mock_redis.delete = AsyncMock()

        user_ids = self._run_async(
            svc.get_site_notification_user_ids_cached(
                'Missing',
                mock_session,
                mock_redis,
            ),
        )

        self.assertIsNone(user_ids)
        self.assertEqual(mock_redis.delete.await_count, 2)
        mock_redis.delete.assert_any_await(
            'site_notification_users:Missing',
            'site_notification_users_ready:Missing',
            'site_notification_users_lock:Missing',
        )
        mock_redis.delete.assert_any_await(
            'site_notification_users_lock:Missing',
        )

    def test_invalidate_site_notification_user_cache(self) -> None:
        """It deletes Redis index keys for the given site names."""
        mock_redis = AsyncMock()

        self._run_async(
            svc.invalidate_site_notification_user_cache(
                ['SiteA', 'SiteB'],
                mock_redis,
            ),
        )

        mock_redis.delete.assert_awaited_once_with(
            'site_notification_users:SiteA',
            'site_notification_users_ready:SiteA',
            'site_notification_users_lock:SiteA',
            'site_notification_users:SiteB',
            'site_notification_users_ready:SiteB',
            'site_notification_users_lock:SiteB',
        )

    @patch(
        'examples.local_notification_server.services.'
        'send_fcm_notification_service',
        new_callable=AsyncMock,
    )
    def test_build_push_tasks_creates_tasks_per_lang(
        self, mock_send: AsyncMock,
    ) -> None:
        """
        It creates one task per language when batches are below 100 tokens.

        Args:
            mock_send (AsyncMock): The mocked send function.
        """
        # Two languages, each with tokens less than batch size -> 2 tasks
        lang_to_tokens: DefaultDict[str, list[str]] = defaultdict(
            list,
            {
                'en-GB': ['t1', 't2'],
                'zh': ['t3'],
            },
        )
        req = SiteNotifyRequest(
            site='S1',
            stream_name='Cam1',
            body={'en': {'helmet': 1}},
            image_path=None,
            violation_id=123,
        )
        mock_send.return_value = True

        tasks = svc._build_push_tasks(req, lang_to_tokens)
        self.assertEqual(len(tasks), 2)
        # The tasks are coroutines produced by the AsyncMock
        # Trigger them to ensure they are awaitable
        results = self._run_async_many(tasks)
        self.assertEqual(results, [True, True])
        self.assertEqual(mock_send.await_count, 2)

    @patch(
        'examples.local_notification_server.services.'
        'send_fcm_notification_service',
        new_callable=AsyncMock,
    )
    def test_build_push_tasks_batches_over_100(
        self, mock_send: AsyncMock,
    ) -> None:
        """
        It splits tokens into batches of 100 for a given language.

        Args:
            mock_send (AsyncMock): The mocked send function.
        """
        # 205 tokens -> 3 batches at size 100
        tokens = [f"t{i}" for i in range(205)]
        lang_to_tokens: DefaultDict[str, list[str]] = defaultdict(
            list,
            {
                'en-GB': tokens,
            },
        )
        req = SiteNotifyRequest(
            site='S1',
            stream_name='Cam1',
            body={'en': {'helmet': 1}},
            image_path=None,
            violation_id=None,
        )
        mock_send.return_value = True

        tasks = svc._build_push_tasks(req, lang_to_tokens)
        self.assertEqual(len(tasks), 3)
        _ = self._run_async_many(tasks)
        self.assertEqual(mock_send.await_count, 3)

    def test_execute_push_tasks_success_and_mix(self) -> None:
        """It returns mixed boolean results when tasks succeed/fail."""
        # Create two tasks: one True, one False
        t1 = AsyncMock(return_value=True)()
        t2 = AsyncMock(return_value=False)()
        ok, results, err = self._run_async(
            svc._execute_push_tasks([t1, t2], timeout=1.0),
        )
        self.assertTrue(ok)
        self.assertEqual(results, [True, False])
        self.assertIsNone(err)

    def test_execute_push_tasks_bounded_aggregates_counts(self) -> None:
        """It executes an iterable of tasks without collecting results."""
        tasks = [
            AsyncMock(return_value=True)(),
            AsyncMock(return_value=False)(),
            AsyncMock(return_value=True)(),
        ]

        ok, total, successful, err = self._run_async(
            svc._execute_push_tasks_bounded(tasks, timeout=1.0),
        )

        self.assertTrue(ok)
        self.assertEqual(total, 3)
        self.assertEqual(successful, 2)
        self.assertIsNone(err)

    def test_execute_push_tasks_bounded_streaming_aggregates_counts(
        self,
    ) -> None:
        """It executes an async iterable of tasks with bounded concurrency."""
        async def task_stream() -> None:
            """Support task_stream."""
            yield AsyncMock(return_value=True)()
            yield AsyncMock(return_value=False)()
            yield AsyncMock(return_value=True)()

        ok, total, successful, err = self._run_async(
            svc._execute_push_tasks_bounded_streaming(
                task_stream(), timeout=1.0, max_concurrency=2,
            ),
        )

        self.assertTrue(ok)
        self.assertEqual(total, 3)
        self.assertEqual(successful, 2)
        self.assertIsNone(err)

    def test_execute_push_tasks_streaming_reports_invalid_tokens(
        self,
    ) -> None:
        """It forwards invalid FCM tokens to the optional handler."""
        async def task_stream() -> None:
            """Support task_stream."""
            yield AsyncMock(
                return_value=svc.FcmSendResult(
                    success_count=0,
                    failure_count=1,
                    invalid_tokens=('bad-token',),
                ),
            )()

        cleanup = AsyncMock()

        ok, total, successful, err = self._run_async(
            svc._execute_push_tasks_bounded_streaming(
                task_stream(),
                timeout=1.0,
                max_concurrency=1,
                invalid_token_handler=cleanup,
            ),
        )

        self.assertTrue(ok)
        self.assertEqual(total, 1)
        self.assertEqual(successful, 0)
        self.assertIsNone(err)
        cleanup.assert_awaited_once_with(('bad-token',))

    def test_execute_push_tasks_timeout(self) -> None:
        """It returns a timeout message when execution exceeds the limit."""
        with patch(
            'asyncio.wait_for',
            side_effect=__import__('asyncio').TimeoutError,
        ):
            ok, results, err = self._run_async(
                svc._execute_push_tasks([], timeout=0.01),
            )
            self.assertFalse(ok)
            self.assertIsNone(results)
            self.assertEqual(err, 'FCM notification sending timed out.')

    def test_execute_push_tasks_exception(self) -> None:
        """It returns a generic error indicator when an exception occurs."""
        with patch('asyncio.wait_for', side_effect=Exception('boom')):
            ok, results, err = self._run_async(
                svc._execute_push_tasks([], timeout=0.01),
            )
            self.assertFalse(ok)
            self.assertIsNone(results)
            self.assertEqual(err, 'internal_error')

    def _run_async(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run a single awaitable to completion.

        Args:
            coro: The awaitable to execute.

        Returns:
            The awaited result, preserving the underlying type.
        """
        import asyncio
        return asyncio.run(coro)

    def _run_async_many(self, coros: Iterable[Awaitable[T]]) -> list[T]:
        """Run multiple awaitables concurrently and collect their results.

        Args:
            coros: An iterable of awaitables to execute.

        Returns:
            A list of results in the same order as the input awaitables.
        """
        import asyncio

        async def gatherer() -> list[T]:
            """Support gatherer."""
            return await asyncio.gather(*coros)

        return asyncio.run(gatherer())


if __name__ == '__main__':
    unittest.main()

"""
pytest \
    --cov=examples.local_notification_server.services \
    --cov-report=term-missing \
    tests/examples/local_notification_server/services_test.py
"""


def _row(token: str = 'device-token') -> svc.FcmDeviceToken:
    now = datetime(2026, 7, 24, 8, 0, tzinfo=timezone.utc)
    return svc.FcmDeviceToken(
        user_id=7,
        device_token_encrypted=svc.encrypt_fcm_token(token),
        device_token_hash=svc.fcm_token_hash(token),
        platform='web',
        device_lang='en-GB',
        permission_status='granted',
        last_seen_at=now,
        created_at=now,
        updated_at=now,
    )


class TestNotificationServicesCoverage(unittest.IsolatedAsyncioTestCase):
    def test_token_crypto_and_serialisation_helpers(self) -> None:
        valid_key = Fernet.generate_key().decode('utf-8')
        with patch.object(
            svc,
            'settings',
            SimpleNamespace(
                fcm_token_encryption_key=valid_key,
                authjwt_secret_key='fallback-secret',
            ),
        ):
            encrypted = svc.encrypt_fcm_token('device-token')
            self.assertEqual(svc.decrypt_fcm_token(encrypted), 'device-token')
            self.assertEqual(svc.decrypt_fcm_token('not-a-token'), '')

        with patch.object(
            svc,
            'settings',
            SimpleNamespace(
                fcm_token_encryption_key='invalid-key',
                authjwt_secret_key='fallback-secret',
            ),
        ):
            self.assertEqual(
                svc.decrypt_fcm_token(svc.encrypt_fcm_token('fallback-token')),
                'fallback-token',
            )

        self.assertEqual(svc._decode_redis_string(None), '')
        self.assertEqual(svc._decode_redis_string(b'value'), 'value')
        self.assertEqual(svc._decode_redis_bool('TRUE'), True)
        self.assertEqual(svc._decode_redis_bool('false'), False)
        self.assertIsNone(svc._decode_redis_bool(None))
        self.assertEqual(svc._encode_optional_bool(None), '')
        self.assertEqual(svc._encode_optional_bool(True), 'true')
        self.assertEqual(svc._encode_optional_bool(False), 'false')
        self.assertEqual(
            svc._datetime_to_api(datetime(2026, 7, 24, 8, 0)),
            '2026-07-24T08:00:00Z',
        )
        self.assertTrue(svc._utc_now_iso().endswith('Z'))

    def test_token_row_status_result_and_cache_write(self) -> None:
        row = _row()
        row.last_success_at = datetime(2026, 7, 24, 8, 1, tzinfo=timezone.utc)
        row.web_vapid_key_available = True
        row.web_service_worker_registered = False
        status = svc._fcm_token_status_row(row)
        self.assertEqual(status['platform'], 'web')
        self.assertTrue(status['is_active'])
        self.assertEqual(status['last_success_at'], '2026-07-24T08:01:00Z')

        result = SimpleNamespace(
            scalars=lambda: SimpleNamespace(all=lambda: [row]),
        )
        self.assertEqual(svc._result_scalars_all(result), [row])
        self.assertEqual(svc._result_scalars_all(SimpleNamespace()), [])

        class _Awaitable:
            def __await__(self):
                yield
                return None

        async def async_scalars() -> object:
            return SimpleNamespace(all=lambda: [])

        async def async_all() -> list[object]:
            return []

        self.assertEqual(
            svc._result_scalars_all(SimpleNamespace(scalars=async_scalars)),
            [],
        )
        self.assertEqual(
            svc._result_scalars_all(
                SimpleNamespace(
                    scalars=lambda: _Awaitable(),
                ),
            ),
            [],
        )
        self.assertEqual(
            svc._result_scalars_all(SimpleNamespace(scalars=lambda: object())),
            [],
        )
        self.assertEqual(
            svc._result_scalars_all(
                SimpleNamespace(
                    scalars=lambda: SimpleNamespace(all=async_all),
                ),
            ),
            [],
        )
        self.assertEqual(
            svc._result_scalars_all(
                SimpleNamespace(
                    scalars=lambda: SimpleNamespace(all=lambda: _Awaitable()),
                ),
            ),
            [],
        )

        pipe = MagicMock()
        svc._queue_token_cache_write(pipe, row, 'device-token')
        pipe.hset.assert_any_call('fcm_tokens:7', 'device-token', 'en-GB')
        pipe.sadd.assert_called_once_with(
            svc._token_index_key(7),
            svc.fcm_token_hash('device-token'),
        )

    async def test_registers_and_updates_device_tokens(self) -> None:
        request = TokenRequest(
            user_id=7,
            device_token='device-token',
            platform='web',
            permission_status='granted',
            app_version='1.0.0',
            web_vapid_key_available=True,
            web_service_worker_registered=True,
        )
        db = MagicMock()
        db.scalar = AsyncMock(return_value=None)
        db.add = MagicMock()
        db.commit = AsyncMock()
        pipe = MagicMock()
        pipe.execute = AsyncMock()
        rds = MagicMock()
        rds.pipeline.return_value = pipe

        response = await svc.record_fcm_token_registration(
            request,
            'en-GB',
            db,
            rds,
        )
        self.assertEqual(
            response['token_hash'],
            svc.fcm_token_hash('device-token'),
        )
        created = db.add.call_args.args[0]
        self.assertEqual(created.platform, 'web')
        self.assertEqual(created.permission_status, 'granted')
        pipe.execute.assert_awaited_once()

        existing = _row()
        db.scalar.return_value = existing
        await svc.record_fcm_token_registration(request, 'zh-TW', db, rds)
        self.assertEqual(existing.device_lang, 'zh-TW')
        self.assertIsNone(existing.disabled_at)

        async_db = MagicMock()
        async_db.scalar = AsyncMock(return_value=None)
        async_db.add = AsyncMock()
        async_db.commit = AsyncMock()
        await svc.record_fcm_token_registration(request, 'en-GB', async_db, rds)
        async_db.add.assert_awaited_once()

    async def test_device_status_loading_and_cache_refresh(self) -> None:
        row = _row()
        result = SimpleNamespace(
            scalars=lambda: SimpleNamespace(all=lambda: [row, object()]),
        )
        db = MagicMock()
        db.execute = AsyncMock(return_value=result)

        status = await svc.list_fcm_device_status(7, db)
        self.assertEqual(status[0]['token_hash'], row.device_token_hash)
        self.assertEqual(await svc.load_active_fcm_device_tokens(7, db), ['device-token'])

        pipe = MagicMock()
        pipe.execute = AsyncMock()
        rds = MagicMock()
        rds.pipeline.return_value = pipe
        self.assertEqual(await svc.refresh_fcm_token_cache_for_users([], db, rds), 0)
        self.assertEqual(
            await svc.refresh_fcm_token_cache_for_users([7, 7], db, rds),
            1,
        )
        pipe.execute.assert_awaited_once()

    async def test_marks_token_delivery_success_failure_and_invalidity(self) -> None:
        db = MagicMock()
        db.execute = AsyncMock()
        db.commit = AsyncMock()
        pipe = MagicMock()
        pipe.execute = AsyncMock()
        rds = MagicMock()
        rds.pipeline.return_value = pipe
        rds.hgetall = AsyncMock(return_value={b'invalid-token': b'en-GB'})

        await svc.mark_fcm_tokens_success(7, ['ok-token'], rds, db)
        await svc.mark_fcm_tokens_failure(7, ['failed-token'], rds, 'offline', db)
        await svc.mark_invalid_fcm_tokens_for_users(
            [7],
            ['invalid-token'],
            rds,
            db=db,
        )
        await svc.mark_invalid_fcm_tokens_for_users([7], [], rds)

        self.assertGreaterEqual(db.execute.await_count, 3)
        self.assertGreaterEqual(pipe.execute.await_count, 3)
        pipe.hdel.assert_called_once_with('fcm_tokens:7', 'invalid-token')

        rds.hgetall.return_value = {b'other-token': b'en-GB'}
        await svc.mark_invalid_fcm_tokens_for_users(
            [7],
            ['invalid-token'],
            rds,
        )

    async def test_token_deletion_and_cache_builder_fallback(self) -> None:
        db = MagicMock()
        db.execute = AsyncMock(return_value=SimpleNamespace(rowcount=1))
        db.commit = AsyncMock()
        rds = AsyncMock()
        self.assertTrue(
            await svc.delete_fcm_token_metadata(7, 'device-token', db, rds),
        )

        db.execute.return_value = SimpleNamespace(rowcount='unknown')
        self.assertFalse(
            await svc.delete_fcm_token_metadata(7, 'device-token', db, rds),
        )

        rds.exists = AsyncMock(return_value=False)
        rds.set = AsyncMock(return_value=False)
        with patch.object(svc, '_recipient_index_wait_attempts', 0):
            with patch.object(
                svc,
                'refresh_site_notification_user_cache',
                AsyncMock(return_value=[7]),
            ) as refresh:
                self.assertEqual(
                    await svc.get_site_notification_user_ids_cached('S1', db, rds),
                    [7],
                )
        refresh.assert_awaited_once_with('S1', db, rds)

    async def test_notification_content_and_push_task_fallbacks(self) -> None:
        request = SiteNotifyRequest(site='S1', stream_name='Cam1', body={})
        self.assertEqual(svc._translate_title('unsupported'), '')
        with patch.object(svc, '_translate_title', return_value=''):
            self.assertEqual(
                svc._notification_record_title(
                    request,
                ), 'Notification',
            )
        titled_request = SiteNotifyRequest(
            site='S1',
            stream_name='Cam1',
            body={},
            title='Custom title',
        )
        self.assertEqual(
            svc._notification_record_title(
                titled_request,
            ), 'Custom title',
        )
        self.assertEqual(svc._notification_record_body(request), 'S1 - Cam1')

        db = MagicMock()
        db.add_all = AsyncMock()
        db.commit = AsyncMock()
        self.assertEqual(
            await svc.create_notification_records_for_users(request, [], db),
            0,
        )
        self.assertEqual(
            await svc.create_notification_records_for_users(request, [7], db),
            1,
        )
        db.add_all.assert_awaited_once()

        self.assertIsNone(svc._build_push_task(request, 'en-GB', []))
        self.assertIsNone(
            svc._build_push_task(
                request, 'unsupported', ['token'],
            ),
        )
        self.assertIsNone(svc._build_push_task(request, 'en-GB', ['token']))

    async def test_refresh_skips_undecryptable_rows(self) -> None:
        row = _row()
        result = SimpleNamespace(
            scalars=lambda: SimpleNamespace(all=lambda: [row]),
        )
        db = MagicMock()
        db.execute = AsyncMock(return_value=result)
        rds = MagicMock()
        rds.pipeline.return_value = MagicMock()
        with patch.object(svc, 'decrypt_fcm_token', return_value=''):
            self.assertEqual(
                await svc.refresh_fcm_token_cache_for_users([7], db, rds),
                0,
            )

    async def test_preflight_and_streaming_builder_handle_empty_tokens(self) -> None:
        request = SiteNotifyRequest(
            site='S1',
            stream_name='Cam1',
            body={'warning_no_hardhat': {'count': 1}},
        )
        pipe = MagicMock()
        pipe.hgetall = MagicMock()
        pipe.execute = AsyncMock(return_value=[{b'': b'en-GB'}])
        rds = MagicMock()
        rds.pipeline.return_value = pipe
        stats = await svc.diagnose_push_preflight(request, [7], rds)
        self.assertEqual(stats['token_entries'], 0)

        async def complete_task() -> bool:
            return True

        pipe.execute.return_value = [{b'token': b'en-GB'}]
        with patch.object(svc, '_fcm_batch_size', 1):
            with patch.object(svc, '_build_push_task', return_value=complete_task()):
                generator = svc._iter_push_tasks_streaming(request, [7], rds)
                task = await generator.__anext__()
                self.assertTrue(await task)
                with self.assertRaises(StopAsyncIteration):
                    await generator.__anext__()

    async def test_bounded_executors_cleanup_and_error_results(self) -> None:
        cleanup = AsyncMock()

        async def invalid_task() -> svc.FcmSendResult:
            return svc.FcmSendResult(0, 1, ('invalid-token',))

        result = await svc._execute_push_tasks_bounded(
            [invalid_task()],
            invalid_token_handler=cleanup,
        )
        self.assertEqual(result, (True, 1, 0, None))
        cleanup.assert_awaited_once_with(('invalid-token',))

        async def pending_task() -> bool:
            await asyncio.Event().wait()
            return True

        timeout_result = await svc._execute_push_tasks_bounded(
            [pending_task()],
            timeout=0.01,
        )
        self.assertEqual(
            timeout_result,
            (False, None, None, 'FCM notification sending timed out.'),
        )

        async def raising_task() -> bool:
            raise RuntimeError('send failed')

        error_result = await svc._execute_push_tasks_bounded(
            [raising_task()],
        )
        self.assertEqual(error_result, (False, None, None, 'internal_error'))

    async def test_streaming_bounded_executor_handles_empty_timeout_and_error(self) -> None:
        async def empty_stream():
            if False:
                yield asyncio.sleep(0)

        self.assertEqual(
            await svc._execute_push_tasks_bounded_streaming(empty_stream()),
            (True, 0, 0, None),
        )

        async def pending_stream():
            yield asyncio.Event().wait()

        timeout_result = await svc._execute_push_tasks_bounded_streaming(
            pending_stream(),
            timeout=0.01,
        )
        self.assertEqual(
            timeout_result,
            (False, None, None, 'FCM notification sending timed out.'),
        )

        async def failing_stream():
            raise RuntimeError('stream failed')
            yield asyncio.sleep(0)

        error_result = await svc._execute_push_tasks_bounded_streaming(
            failing_stream(),
        )
        self.assertEqual(error_result, (False, None, None, 'internal_error'))


if __name__ == '__main__':
    unittest.main()
