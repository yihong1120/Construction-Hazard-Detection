from __future__ import annotations

import asyncio
import unittest
from collections.abc import AsyncIterator
from collections.abc import Awaitable
from collections.abc import Coroutine
from datetime import datetime
from datetime import timezone
from types import SimpleNamespace
from typing import Any
from typing import TypeVar
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from cryptography.fernet import Fernet
from cryptography.fernet import InvalidToken

from examples.local_notification_server import push_dispatch as dispatch
from examples.local_notification_server import services as svc
from examples.local_notification_server import site_recipient_cache
from examples.local_notification_server.schemas import (
    DeviceRegistrationRequest,
)
from examples.local_notification_server.schemas import SiteNotifyRequest

# Generic type variable for async helpers
T = TypeVar('T')
TEST_FERNET_KEY = Fernet.generate_key().decode('utf-8')
svc.settings.fcm_token_encryption_key = TEST_FERNET_KEY


class TestServices(unittest.TestCase):
    """Unit tests for notification services utilities."""

    def setUp(self) -> None:
        """Initialise a clean cache before each test.

        Ensures tests remain isolated and do not depend on call order.
        """
        self.maxDiff = None

    def test_decode_lang_token_map_basic(self) -> None:
        """It decodes raw Redis maps into a language-to-tokens mapping.

        Redis token cache entries use canonical language codes.
        """
        raw = [
            {b't1': b'en-GB', b't2': b'zh-TW'},
            {b't3': b'fr-FR', b't4': b'ja-JP'},
        ]
        got = dispatch._decode_lang_token_map(raw)
        # Convert to plain dict for assertion
        got_dict = {k: list(v) for k, v in got.items()}
        self.assertEqual(got_dict['en-GB'], ['t1'])
        self.assertEqual(got_dict['zh-TW'], ['t2'])
        self.assertEqual(got_dict['fr-FR'], ['t3'])
        self.assertEqual(got_dict['ja-JP'], ['t4'])

    def test_decode_lang_token_map_preserves_indexed_tokens(
        self,
    ) -> None:
        """It groups each indexed token by its recorded language."""
        raw = [
            {b't1': b'zh-TW', b't2': b'zh-TW'},
            {b't1': b'en-GB', b't3': b'en-GB'},
        ]

        got = dispatch._decode_lang_token_map(raw)
        got_dict = {k: list(v) for k, v in got.items()}

        self.assertEqual(got_dict['zh-TW'], ['t1', 't2'])
        self.assertEqual(got_dict['en-GB'], ['t1', 't3'])

    def test_create_notification_records_for_users(self) -> None:
        """It writes one notification-centre record per indexed recipient."""
        req = SiteNotifyRequest(
            site='S1',
            stream_name='Cam1',
            body={'warning_no_hardhat': {'count': 1}},
            image_path='https://example.com/v.jpg',
            violation_id=123,
            type='violation',
            title='Violation alert',
            deep_link='/violations?violation_id=123',
            metadata={'violation_id': 123},
        )
        db = MagicMock()
        db.execute = AsyncMock()
        db.commit = AsyncMock()

        count = self._run_async(
            dispatch.create_notification_records_for_users(
                req,
                [5, 5, 6],
                db,
            ),
        )

        self.assertEqual(count, 3)
        statement, records = db.execute.await_args.args
        self.assertEqual(statement.table.name, 'notifications')
        self.assertEqual([record['user_id'] for record in records], [5, 5, 6])
        self.assertEqual(records[0]['type'], 'violation')
        self.assertEqual(
            records[0]['deep_link'],
            '/violations?violation_id=123',
        )
        self.assertEqual(records[0]['metadata_json']['violation_id'], 123)
        db.commit.assert_awaited_once()

    @patch(
        'examples.local_notification_server.push_dispatch.'
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
            type='violation',
            title='Violation alert',
            deep_link='/violations?violation_id=123',
            metadata={'violation_id': 123},
        )
        mock_send.return_value = dispatch.FcmSendResult(1, 0)

        pipe_one = MagicMock()
        pipe_one.hgetall = MagicMock()
        pipe_one.execute = AsyncMock(
            return_value=[{b'a': b'zh-TW'}, {b'b': b'zh-TW'}],
        )
        pipe_two = MagicMock()
        pipe_two.hgetall = MagicMock()
        pipe_two.execute = AsyncMock(
            return_value=[{b'c': b'zh-TW'}],
        )
        rds = MagicMock()
        rds.pipeline.side_effect = [pipe_one, pipe_two]

        async def collect() -> list[dispatch.PushTaskResult]:
            """Support collect."""
            tasks: list[Awaitable[dispatch.PushTaskResult]] = []
            async for task in dispatch.iter_push_tasks_streaming(
                req,
                [1, 2, 3],
                rds,
            ):
                tasks.append(task)
            import asyncio

            return await asyncio.gather(*tasks)

        with patch.object(dispatch, '_token_fetch_chunk_size', 2):
            with patch.object(dispatch, '_fcm_batch_size', 2):
                results = self._run_async(collect())

        self.assertEqual(
            results,
            [dispatch.FcmSendResult(1, 0), dispatch.FcmSendResult(1, 0)],
        )
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
            type='violation',
            title='Violation alert',
            deep_link='/violations',
            metadata={},
        )
        pipe = MagicMock()
        pipe.hgetall = MagicMock()
        pipe.execute = AsyncMock(
            return_value=[
                {b'a': b'zh-TW', b'b': b'en-GB'},
                {},
                {b'a': b'en-GB'},
            ],
        )
        rds = MagicMock()
        rds.pipeline.return_value = pipe

        stats = self._run_async(
            dispatch.diagnose_push_preflight(req, [1, 2, 3], rds),
        )

        self.assertEqual(stats['recipient_users'], 3)
        self.assertEqual(stats['users_with_tokens'], 2)
        self.assertEqual(stats['token_entries'], 3)
        self.assertEqual(stats['unique_tokens'], 3)
        self.assertEqual(stats['sendable_tokens'], 3)
        self.assertEqual(
            stats['tokens_by_language'],
            {'en-GB': 2, 'zh-TW': 1},
        )

    def test_get_site_notification_user_ids_cached_hit(self) -> None:
        """It returns indexed user IDs from Redis when present."""
        mock_session = AsyncMock()
        mock_redis = AsyncMock()
        mock_redis.exists.return_value = 1
        mock_redis.smembers.return_value = {b'7', b'3'}

        user_ids = self._run_async(
            site_recipient_cache.get_site_notification_user_ids_cached(
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
        mock_redis.delete = AsyncMock()
        mock_redis.exists.return_value = 0
        pipe = MagicMock()
        pipe.delete = MagicMock()
        pipe.sadd = MagicMock()
        pipe.set = MagicMock()
        pipe.execute = AsyncMock()
        mock_redis.pipeline.return_value = pipe

        user_ids = self._run_async(
            site_recipient_cache.get_site_notification_user_ids_cached(
                'SiteA',
                mock_session,
                mock_redis,
            ),
        )

        self.assertEqual(user_ids, [8, 9])
        mock_redis.set.assert_not_called()
        pipe.delete.assert_called_once_with('site_notification_users:SiteA')
        pipe.sadd.assert_called_once_with(
            'site_notification_users:SiteA',
            8,
            9,
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
            site_recipient_cache.get_site_notification_user_ids_cached(
                'Missing',
                mock_session,
                mock_redis,
            ),
        )

        self.assertIsNone(user_ids)
        self.assertEqual(mock_redis.delete.await_count, 1)
        mock_redis.delete.assert_any_await(
            'site_notification_users:Missing',
            'site_notification_users_ready:Missing',
        )

    def test_refresh_site_notification_user_cache_empty_site(self) -> None:
        """It marks an existing site as ready even when no users are
        subscribed."""
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
            site_recipient_cache.refresh_site_notification_user_cache(
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

    def test_get_site_notification_user_ids_cached_not_found(self) -> None:
        """It yields ``None`` when the site does not exist."""
        site_result = MagicMock()
        site_result.first.return_value = None
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=site_result)
        mock_redis = MagicMock()
        mock_redis.exists = AsyncMock(return_value=0)
        mock_redis.delete = AsyncMock()

        user_ids = self._run_async(
            site_recipient_cache.get_site_notification_user_ids_cached(
                'Missing',
                mock_session,
                mock_redis,
            ),
        )

        self.assertIsNone(user_ids)
        self.assertEqual(mock_redis.delete.await_count, 1)
        mock_redis.delete.assert_any_await(
            'site_notification_users:Missing',
            'site_notification_users_ready:Missing',
        )

    def test_invalidate_site_notification_user_cache(self) -> None:
        """It deletes Redis index keys for the given site names."""
        mock_redis = AsyncMock()

        self._run_async(
            site_recipient_cache.invalidate_site_notification_user_cache(
                ['SiteA', 'SiteB'],
                mock_redis,
            ),
        )

        mock_redis.delete.assert_awaited_once_with(
            'site_notification_users:SiteA',
            'site_notification_users_ready:SiteA',
            'site_notification_users:SiteB',
            'site_notification_users_ready:SiteB',
        )

    def test_execute_push_tasks_bounded_streaming_aggregates_counts(
        self,
    ) -> None:
        """It executes an async iterable of tasks with bounded concurrency."""

        async def task_stream() -> (
            AsyncIterator[Awaitable[dispatch.FcmSendResult]]
        ):
            """Support task_stream."""
            yield AsyncMock(return_value=dispatch.FcmSendResult(1, 0))()
            yield AsyncMock(return_value=dispatch.FcmSendResult(0, 1))()
            yield AsyncMock(return_value=dispatch.FcmSendResult(1, 0))()

        ok, total, successful, err = self._run_async(
            dispatch.execute_push_tasks_bounded_streaming(
                task_stream(),
                AsyncMock(),
                timeout=1.0,
                max_concurrency=2,
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

        async def task_stream() -> (
            AsyncIterator[Awaitable[dispatch.FcmSendResult]]
        ):
            """Support task_stream."""
            yield AsyncMock(
                return_value=dispatch.FcmSendResult(
                    success_count=0,
                    failure_count=1,
                    invalid_tokens=('bad-token',),
                ),
            )()

        cleanup = AsyncMock()

        ok, total, successful, err = self._run_async(
            dispatch.execute_push_tasks_bounded_streaming(
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

    def _run_async(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run a single awaitable to completion.

        Args:
            coro: The awaitable to execute.

        Returns:
            The awaited result, preserving the underlying type.
        """
        import asyncio

        return asyncio.run(coro)


if __name__ == '__main__':
    unittest.main()


def _row(token: str = 'device-token') -> svc.FcmDeviceToken:
    """Perform row.

    Args:
        token: Value used by this callable.

    Returns:
        The callable result.
    """
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
    """Provide TestNotificationServicesCoverage."""

    def test_token_crypto_and_serialisation_helpers(self) -> None:
        """Test token crypto and serialisation helpers."""
        valid_key = Fernet.generate_key().decode('utf-8')
        with patch.object(
            svc,
            'settings',
            SimpleNamespace(
                fcm_token_encryption_key=valid_key,
            ),
        ):
            encrypted = svc.encrypt_fcm_token('device-token')
            self.assertEqual(svc.decrypt_fcm_token(encrypted), 'device-token')
            with self.assertRaises(InvalidToken):
                svc.decrypt_fcm_token('not-a-token')

        with patch.object(
            svc,
            'settings',
            SimpleNamespace(
                fcm_token_encryption_key='invalid-key',
            ),
        ):
            with self.assertRaises(ValueError):
                svc.encrypt_fcm_token('device-token')

        self.assertEqual(svc._decode_redis_string(b'value'), 'value')
        self.assertEqual(
            svc._datetime_to_api(
                datetime(2026, 7, 24, 8, 0, tzinfo=timezone.utc),
            ),
            '2026-07-24T08:00:00Z',
        )

    def test_token_cache_write(self) -> None:
        """Test token cache write."""
        row = _row()
        row.last_success_at = datetime(2026, 7, 24, 8, 1, tzinfo=timezone.utc)

        pipe = MagicMock()
        svc._queue_token_cache_write(pipe, row, 'device-token')
        pipe.hset.assert_any_call('fcm_tokens:7', 'device-token', 'en-GB')
        pipe.sadd.assert_called_once_with(
            svc._token_index_key(7),
            svc.fcm_token_hash('device-token'),
        )

    async def test_registers_and_updates_device_tokens(self) -> None:
        """Test registers and updates device tokens."""
        request = DeviceRegistrationRequest(
            device_token='device-token',
            device_lang='en-GB',
            platform='web',
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
            7,
            request,
            db,
            rds,
        )
        self.assertEqual(
            response['token_hash'],
            svc.fcm_token_hash('device-token'),
        )
        created = db.add.call_args.args[0]
        self.assertEqual(created.platform, 'web')
        self.assertEqual(created.permission_status, 'unknown')
        pipe.execute.assert_awaited_once()

        existing = _row()
        db.scalar.return_value = existing
        updated_request = request.model_copy(
            update={'device_lang': 'zh-TW'},
        )
        await svc.record_fcm_token_registration(
            7,
            updated_request,
            db,
            rds,
        )
        self.assertEqual(existing.device_lang, 'zh-TW')
        self.assertIsNone(existing.disabled_at)

        second_db = MagicMock()
        second_db.scalar = AsyncMock(return_value=None)
        second_db.add = MagicMock()
        second_db.commit = AsyncMock()
        await svc.record_fcm_token_registration(
            7,
            request,
            second_db,
            rds,
        )
        second_db.add.assert_called_once()

    async def test_device_status_loading_and_cache_refresh(self) -> None:
        """Test device status loading and cache refresh."""
        row = _row()
        result = SimpleNamespace(
            scalars=lambda: SimpleNamespace(all=lambda: [row]),
        )
        db = MagicMock()
        db.execute = AsyncMock(return_value=result)

        status = await svc.list_fcm_device_status(7, db)
        self.assertEqual(status[0]['token_hash'], row.device_token_hash)
        self.assertEqual(
            await svc.load_active_fcm_device_tokens(7, db),
            ['device-token'],
        )

        pipe = MagicMock()
        pipe.execute = AsyncMock()
        rds = MagicMock()
        rds.pipeline.return_value = pipe
        self.assertEqual(
            await svc.refresh_fcm_token_cache_for_users([7, 7], db, rds),
            1,
        )
        pipe.delete.assert_any_call('fcm_tokens:7')
        pipe.delete.assert_any_call('fcm_token_index:7')
        pipe.execute.assert_awaited_once()

    async def test_marks_token_delivery_success_failure_and_invalidity(
        self,
    ) -> None:
        """Test marks token delivery success failure and invalidity."""
        db = MagicMock()
        db.execute = AsyncMock()
        db.commit = AsyncMock()
        pipe = MagicMock()
        pipe.execute = AsyncMock()
        rds = MagicMock()
        rds.pipeline.return_value = pipe
        rds.hgetall = AsyncMock(return_value={b'invalid-token': b'en-GB'})

        await svc.mark_fcm_tokens_success(7, ['ok-token'], rds, db)
        await svc.mark_fcm_tokens_failure(
            7,
            ['failed-token'],
            rds,
            'offline',
            db,
        )
        await svc.mark_invalid_fcm_tokens_for_users(
            [7],
            ['invalid-token'],
            rds,
            db=db,
        )

        self.assertGreaterEqual(db.execute.await_count, 3)
        self.assertGreaterEqual(pipe.execute.await_count, 3)
        pipe.hdel.assert_called_once_with('fcm_tokens:7', 'invalid-token')

        rds.hgetall.return_value = {b'other-token': b'en-GB'}
        await svc.mark_invalid_fcm_tokens_for_users(
            [7],
            ['invalid-token'],
            rds,
            db,
        )

    async def test_token_deletion_and_recipient_cache_refresh(self) -> None:
        """Test token deletion and recipient cache refresh."""
        db = MagicMock()
        db.execute = AsyncMock(return_value=SimpleNamespace(rowcount=1))
        db.commit = AsyncMock()
        rds = AsyncMock()
        self.assertTrue(
            await svc.delete_fcm_token_metadata(7, 'device-token', db, rds),
        )

        db.execute.return_value = SimpleNamespace(rowcount=0)
        self.assertFalse(
            await svc.delete_fcm_token_metadata(7, 'device-token', db, rds),
        )

        rds.exists = AsyncMock(return_value=False)
        with patch.object(
            site_recipient_cache,
            'refresh_site_notification_user_cache',
            AsyncMock(return_value=[7]),
        ) as refresh:
            get_cached_user_ids = (
                site_recipient_cache.get_site_notification_user_ids_cached
            )
            self.assertEqual(
                await get_cached_user_ids(
                    'S1',
                    db,
                    rds,
                ),
                [7],
            )
        refresh.assert_awaited_once_with('S1', db, rds)

    async def test_notification_content_and_push_task_contracts(self) -> None:
        """Test notification content and push task contracts."""
        request = SiteNotifyRequest(
            site='S1',
            stream_name='Cam1',
            body={'warning_no_hardhat': {'count': 1}},
            type='violation',
            title='Custom title',
            deep_link='/violations',
            metadata={},
        )
        db = MagicMock()
        db.execute = AsyncMock()
        db.commit = AsyncMock()
        self.assertEqual(
            await dispatch.create_notification_records_for_users(
                request, [7], db,
            ),
            1,
        )
        db.execute.assert_awaited_once()
        records = db.execute.await_args.args[1]
        self.assertEqual(
            records[0]['body'],
            'S1 - Cam1\n警告: 有1人未佩戴安全帽!',
        )

        with patch.object(
            dispatch,
            'send_fcm_notification_service',
            new_callable=AsyncMock,
            return_value=dispatch.FcmSendResult(1, 0),
        ):
            task = dispatch.build_push_task(request, 'en-GB', ['token'])
            self.assertEqual(await task, dispatch.FcmSendResult(1, 0))

    async def test_undecryptable_rows_are_disabled_during_cache_refresh(
        self,
    ) -> None:
        """Test undecryptable rows are disabled during cache refresh."""
        row = _row()
        result = SimpleNamespace(
            scalars=lambda: SimpleNamespace(all=lambda: [row]),
        )
        db = MagicMock()
        db.execute = AsyncMock(return_value=result)
        db.commit = AsyncMock()
        pipe = MagicMock()
        pipe.execute = AsyncMock()
        rds = MagicMock()
        rds.pipeline.return_value = pipe
        with patch.object(svc, 'decrypt_fcm_token', side_effect=InvalidToken):
            self.assertEqual(
                await svc.refresh_fcm_token_cache_for_users([7], db, rds),
                0,
            )

        self.assertIsNotNone(row.disabled_at)
        self.assertEqual(row.failure_reason, 'token_decryption_failed')
        db.commit.assert_awaited_once()
        pipe.delete.assert_any_call(
            svc._token_meta_key(7, row.device_token_hash),
        )
        pipe.execute.assert_awaited_once()

    async def test_undecryptable_rows_are_disabled_when_loading_tokens(
        self,
    ) -> None:
        """Test undecryptable rows are disabled when loading tokens."""
        row = _row()
        result = SimpleNamespace(
            scalars=lambda: SimpleNamespace(all=lambda: [row]),
        )
        db = MagicMock()
        db.execute = AsyncMock(return_value=result)
        db.commit = AsyncMock()

        with patch.object(svc, 'decrypt_fcm_token', side_effect=InvalidToken):
            self.assertEqual(
                await svc.load_active_fcm_device_tokens(7, db),
                [],
            )

        self.assertIsNotNone(row.disabled_at)
        self.assertEqual(row.failure_reason, 'token_decryption_failed')
        db.commit.assert_awaited_once()

    async def test_preflight_and_streaming_builder_use_canonical_tokens(
        self,
    ) -> None:
        """Test preflight and streaming builder use canonical tokens."""
        request = SiteNotifyRequest(
            site='S1',
            stream_name='Cam1',
            body={'warning_no_hardhat': {'count': 1}},
            type='violation',
            title='Violation alert',
            deep_link='/violations',
            metadata={},
        )
        pipe = MagicMock()
        pipe.hgetall = MagicMock()
        pipe.execute = AsyncMock(return_value=[{b'token': b'en-GB'}])
        rds = MagicMock()
        rds.pipeline.return_value = pipe
        stats = await dispatch.diagnose_push_preflight(request, [7], rds)
        self.assertEqual(stats['token_entries'], 1)

        async def complete_task() -> dispatch.FcmSendResult:
            """Perform complete task.

            Returns:
                The callable result.
            """
            return dispatch.FcmSendResult(success_count=1, failure_count=0)

        pipe.execute.return_value = [{b'token': b'en-GB'}]
        generator = dispatch.iter_push_tasks_streaming(
            request,
            [7],
            rds,
            fcm_batch_size=1,
            build_push_task_fn=lambda *_args: complete_task(),
        )
        task = await generator.__anext__()
        self.assertTrue(await task)
        with self.assertRaises(StopAsyncIteration):
            await generator.__anext__()

    async def test_streaming_bounded_executor_handles_empty_timeout_and_error(
        self,
    ) -> None:
        """Test streaming bounded executor handles empty timeout and error."""

        async def empty_stream() -> Any:
            """Perform empty stream.

            Returns:
                The callable result.
            """
            if False:
                yield asyncio.sleep(0)

        self.assertEqual(
            await dispatch.execute_push_tasks_bounded_streaming(
                empty_stream(),
                AsyncMock(),
            ),
            (True, 0, 0, None),
        )

        async def pending_stream() -> Any:
            """Perform pending stream.

            Returns:
                The callable result.
            """
            yield asyncio.Event().wait()

        timeout_result = await dispatch.execute_push_tasks_bounded_streaming(
            pending_stream(),
            AsyncMock(),
            timeout=0.01,
        )
        self.assertEqual(
            timeout_result,
            (False, None, None, 'FCM notification sending timed out.'),
        )

        async def failing_stream() -> Any:
            """Perform failing stream.

            Returns:
                The callable result.
            """
            raise RuntimeError('stream failed')
            yield asyncio.sleep(0)

        error_result = await dispatch.execute_push_tasks_bounded_streaming(
            failing_stream(),
            AsyncMock(),
        )
        self.assertEqual(error_result, (False, None, None, 'internal_error'))


if __name__ == '__main__':
    unittest.main()
