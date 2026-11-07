from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from functools import partial
from functools import partialmethod
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import patch

import httpx

from src.async_tasks import run_coupled_tasks
from src.delivery_result import DeliveryOutcome
from src.delivery_result import DeliveryResult
from src.evidence import EvidenceMetadata
from src.violation_delivery import OutboxDispatcher
from src.violation_delivery import ViolationDelivery
from src.violation_delivery import ViolationEvent
from src.violation_delivery import ViolationOutbox
from src.violation_delivery import ViolationQueue


def _delayed_outbox_claim(box, *, _scope):
    event = _scope.claim(box)
    _scope.started.set()
    _scope.release.wait(timeout=2)
    return event


def _concurrent_puts_obey_capacity_and_submit(testcase, /, index):
    try:
        testcase.box.put(
            ViolationEvent(str(index), metadata(), b'image'),
        )
        return True
    except RuntimeError:
        return False


async def _sweep_observed(deliver, processed, /, *args):
    value = await deliver(*args)
    processed.set()
    return value


async def _cancellation_releases_inflight_lease_and_upload(
    started,
    /,
    *args,
):
    started.set()
    await asyncio.Event().wait()


async def _busy_camera_rejoins_fifo_tail_upload(
    done,
    order,
    /,
    data,
    image,
):
    order.append(data.stream_name)
    if len(order) == 6:
        done.set()
    return '42'


async def _cancellation_during_claim_does_not_observe_cancellation(
    cancelled,
    shield,
    /,
    awaitable,
):
    try:
        return await shield(awaitable)
    except asyncio.CancelledError:
        cancelled.set()
        raise


async def _dispatcher_start_and_close_own_run(
    released,
    started,
    /,
):
    started.set()
    try:
        await asyncio.Event().wait()
    finally:
        released.set()


async def _parent_cancellation_drains_pending_worker_worker(
    released,
    /,
):
    try:
        await asyncio.Event().wait()
    finally:
        released.set()


class _SlowCameraDoesNotBlockFastBacklogCallbacks:
    active: Any
    completed: Any
    done: Any
    peak: Any

    async def _call_upload(self, data, image):

        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            if data.stream_name == 'slow':
                await asyncio.Event().wait()
            await asyncio.sleep(0)
            self.completed += 1
            if self.completed == 3:
                self.done.set()
            return '42'
        finally:
            self.active -= 1


class _CancellationDuringClaimDoesNotAbandonACallbacks:
    claim: Any
    release: Any
    started: Any


def metadata(**changes) -> EvidenceMetadata:
    return EvidenceMetadata.from_legacy_payload(
        {
            'site': 'Site',
            'site_id': 7,
            'stream_name': 'Cam',
            'detection_time': '2026-10-30T12:00:00+08:00',
            'warnings': {'warning_no_hardhat': {'count': 1}},
            'detections': [],
            'cone_polygon': [],
            'pole_polygon': [],
            'model_id': 'detector',
            'model_version': 'v1',
            **changes,
        },
    )


class ViolationOutboxTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.box = ViolationOutbox(
            Path(self.directory.name) / 'queue.sqlite3',
            3,
            65536,
        )

    def test_concurrent_puts_obey_capacity_and_preserve_private_permissions(
        self,
    ) -> None:
        submit = partial(_concurrent_puts_obey_capacity_and_submit, self)

        with ThreadPoolExecutor(max_workers=8) as pool:
            self.assertEqual(sum(pool.map(submit, range(8))), 3)
        self.assertEqual(self.box.path.stat().st_mode & 0o777, 0o600)

    def test_capacity_counts_metadata_and_images(self) -> None:
        self.box.max_bytes = 8
        with self.assertRaisesRegex(RuntimeError, 'capacity'):
            self.box.put(ViolationEvent('1', metadata(), b'image'))
        self.assertIsNone(self.box.ready())

    def test_claim_prevents_concurrent_delivery_and_stale_owners(self) -> None:
        self.box.put(ViolationEvent('1', metadata(), b'image'))
        first = self.box.claim()
        assert first is not None
        other = ViolationOutbox(self.box.path, 3, 65536)
        self.assertIsNone(other.claim())
        with patch('src.violation_delivery.time.time', return_value=10**12):
            second = other.claim()
        assert second is not None
        self.assertNotEqual(first.owner, second.owner)
        self.box.complete(first)
        self.box.retry(first, 0)
        with self.assertRaisesRegex(RuntimeError, 'lease lost'):
            self.box.uploaded(first, '42')
        other.complete(second)
        self.assertIsNone(self.box.ready())

    def test_failed_event_keeps_image_and_is_only_retried_explicitly(
        self,
    ) -> None:
        self.box.put(ViolationEvent('1', metadata(), b'image'))
        event = self.box.claim()
        assert event is not None
        self.box.fail(event, 'HTTP 422')
        self.assertIsNone(self.box.ready())
        self.assertEqual(self.box.failures()[0]['reason'], 'HTTP 422')
        self.assertTrue(self.box.requeue('1'))
        retried = self.box.ready()
        assert retried is not None
        self.assertEqual(retried.image, b'image')
        self.assertFalse(self.box.requeue('missing'))

    def test_old_sqlite_schema_and_payload_are_migrated_without_losing_image(
        self,
    ) -> None:
        with closing(sqlite3.connect(self.box.path)) as db, db:
            db.execute(
                (
                    'CREATE TABLE events(id TEXT PRIMARY KEY,payload '
                    'TEXT,image BLOB,violation_id TEXT,attempts INTEGER '
                    'DEFAULT 0,next_attempt REAL DEFAULT 0)'
                ),
            )
            db.execute(
                'INSERT INTO events(id,payload,image) VALUES(?,?,?)',
                (
                    'old',
                    json.dumps(
                        {
                            'site': 'Site',
                            'stream_name': 'Cam',
                            'warnings': {'warning_no_hardhat': {'count': 1}},
                        },
                    ),
                    b'image',
                ),
            )
        event = self.box.claim()
        assert event is not None
        self.assertEqual(event.metadata.site, 'Site')
        self.assertEqual(event.image, b'image')
        self.assertIn('warning_no_hardhat', event.metadata.warnings_json or '')

    def test_unknown_format_is_quarantined_without_blocking_valid_events(
        self,
    ) -> None:
        self.box.put(ViolationEvent('1', metadata(), b'image'))
        with self.box._connect() as db:
            db.execute(
                'UPDATE events SET payload=? WHERE id=?',
                (json.dumps({'version': 999}), '1'),
            )
        self.box.put(ViolationEvent('2', metadata(), b'image'))
        event = self.box.claim()
        assert event is not None
        self.assertEqual(event.id, '2')
        self.assertEqual(
            self.box.failures()[0]['reason'],
            'invalid_event_format',
        )

    def test_retry_does_not_block_new_events(self) -> None:
        self.box.put(ViolationEvent('1', metadata(), b'image'))
        event = self.box.claim()
        assert event is not None
        self.box.retry(event, 60)
        self.box.put(ViolationEvent('2', metadata(), b'image'))
        next_event = self.box.ready()
        assert next_event is not None
        self.assertEqual(next_event.id, '2')
        with patch('src.violation_delivery.time.time', return_value=10**12):
            retry = self.box.ready()
        assert retry is not None
        self.assertEqual((retry.id, retry.attempts), ('1', 1))


class ViolationDeliveryTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        env = patch.dict(
            os.environ,
            {
                'VIOLATION_OUTBOX_DIRECTORY': self.directory.name,
                'VIOLATION_OUTBOX_POLL_SECONDS': '0.01',
            },
        )
        env.start()
        self.addCleanup(env.stop)
        self.queue = ViolationQueue('disabled-camera')
        self.upload = AsyncMock()
        self.upload.send_violation.return_value = '42'
        self.notify = AsyncMock()
        self.notify.send_fcm_message_to_site.return_value = DeliveryResult(
            DeliveryOutcome.completed,
        )
        self.delivery = ViolationDelivery(
            self.queue.outbox,
            self.upload,
            self.notify,
        )

    async def test_notification_retry_reuses_id_and_releases_image_bytes(
        self,
    ) -> None:
        identifier = await self.queue.submit(metadata(), b'jpeg')
        event = self.queue.outbox.ready()
        assert event is not None
        self.notify.send_fcm_message_to_site.return_value = DeliveryResult(
            DeliveryOutcome.retryable,
        )
        with self.assertRaisesRegex(RuntimeError, 'Notification'):
            await self.delivery.deliver(event)
        pending = self.queue.outbox.ready()
        assert pending is not None
        self.assertEqual((pending.violation_id, pending.image), ('42', b''))
        upload_call = self.upload.send_violation.await_args
        assert upload_call is not None
        self.assertEqual(upload_call.args[0].event_id, identifier)
        new_upload = AsyncMock()
        self.notify.send_fcm_message_to_site.return_value = DeliveryResult(
            DeliveryOutcome.skipped,
        )
        await ViolationDelivery(
            self.queue.outbox,
            new_upload,
            self.notify,
        ).deliver(pending)
        new_upload.send_violation.assert_not_awaited()
        self.assertIsNone(self.queue.outbox.ready())

    async def test_empty_image_never_occupies_queue(self) -> None:
        with self.assertRaisesRegex(ValueError, 'empty image'):
            await self.queue.submit(metadata(), b'')
        self.assertIsNone(self.queue.outbox.ready())

    async def sweep(self) -> None:
        processed = asyncio.Event()
        dispatcher = OutboxDispatcher()
        deliver = dispatcher._deliver_one

        observed = partial(_sweep_observed, deliver, processed)

        with (
            patch(
                'src.violation_delivery.ViolationSender',
                return_value=self.upload,
            ),
            patch(
                'src.violation_delivery.FCMSender',
                return_value=self.notify,
            ),
            patch.object(dispatcher, '_deliver_one', observed),
        ):
            dispatcher.start()
            try:
                await asyncio.wait_for(processed.wait(), 2)
            finally:
                await dispatcher.close()

    async def test_dispatcher_drains_queues_without_running_camera(
        self,
    ) -> None:
        await self.queue.submit(metadata(), b'jpeg')
        self.notify.send_fcm_message_to_site.return_value = DeliveryResult(
            DeliveryOutcome.skipped,
        )
        await self.sweep()
        self.assertIsNone(self.queue.outbox.ready())
        self.upload.close.assert_awaited_once()
        self.notify.close.assert_awaited_once()

    async def test_permanent_http_failure_is_retained_for_repair(self) -> None:
        await self.queue.submit(metadata(), b'jpeg')
        response = httpx.Response(
            422,
            request=httpx.Request(
                'POST',
                'https://example.test/upload',
            ),
        )
        self.upload.send_violation.side_effect = httpx.HTTPStatusError(
            'invalid',
            request=response.request,
            response=response,
        )
        await self.sweep()
        self.assertIsNone(self.queue.outbox.ready())
        self.assertEqual(len(self.queue.outbox.failures()), 1)
        self.notify.send_fcm_message_to_site.assert_not_awaited()

    async def test_transient_failure_reaches_bounded_attempt_limit(
        self,
    ) -> None:
        await self.queue.submit(metadata(), b'jpeg')
        self.upload.send_violation.side_effect = httpx.ConnectError('offline')
        await self.sweep()
        with patch('src.violation_delivery.time.time', return_value=10**12):
            event = self.queue.outbox.ready()
        assert event is not None
        self.assertEqual(event.attempts, 1)
        with patch.dict(
            os.environ,
            {'VIOLATION_OUTBOX_MAX_ATTEMPTS': '2'},
        ), patch('src.violation_delivery.time.time', return_value=10**12):
            await self.sweep()
        self.assertEqual(len(self.queue.outbox.failures()), 1)

    async def test_cancellation_releases_inflight_lease_and_clients(self):
        await self.queue.submit(metadata(), b'jpeg')
        started = asyncio.Event()

        upload = partial(
            _cancellation_releases_inflight_lease_and_upload,
            started,
        )

        self.upload.send_violation.side_effect = upload
        with (
            patch(
                'src.violation_delivery.ViolationSender',
                return_value=self.upload,
            ),
            patch(
                'src.violation_delivery.FCMSender',
                return_value=self.notify,
            ),
        ):
            dispatcher = OutboxDispatcher()
            dispatcher.start()
            await started.wait()
            await dispatcher.close()
        event = self.queue.outbox.claim()
        assert event is not None
        self.assertEqual(event.image, b'jpeg')
        self.upload.close.assert_awaited_once()
        self.notify.close.assert_awaited_once()

    async def test_client_initialisation_failure_closes_open_transport(self):
        with (
            patch(
                'src.violation_delivery.ViolationSender',
                return_value=self.upload,
            ),
            patch(
                'src.violation_delivery.FCMSender',
                side_effect=ValueError('Invalid transport settings'),
            ),
        ):
            with self.assertRaises(ValueError):
                await OutboxDispatcher().run()
        self.upload.close.assert_awaited_once()

    async def test_slow_camera_does_not_block_fast_backlog_or_exceed_pool(
        self,
    ):
        _callbacks = _SlowCameraDoesNotBlockFastBacklogCallbacks()
        with patch.dict(
            os.environ,
            {
                'VIOLATION_OUTBOX_CONCURRENCY': '2',
                'VIOLATION_OUTBOX_EVENT_TIMEOUT_SECONDS': '0.05',
            },
        ):
            dispatcher = OutboxDispatcher()
        fast = ViolationQueue('fast-camera')
        await self.queue.submit(metadata(stream_name='slow'), b'jpeg')
        for _ in range(3):
            await fast.submit(metadata(stream_name='fast'), b'jpeg')
        _callbacks.active = _callbacks.peak = _callbacks.completed = 0
        _callbacks.done = asyncio.Event()

        upload = _callbacks._call_upload

        self.upload.send_violation.side_effect = upload
        with patch(
            'src.violation_delivery.ViolationSender',
            return_value=self.upload,
        ), patch('src.violation_delivery.FCMSender', return_value=self.notify):
            dispatcher.start()
            try:
                await asyncio.wait_for(_callbacks.done.wait(), 2)
                await asyncio.sleep(0.1)
                with self.queue.outbox._connect() as db:
                    attempts, owner = db.execute(
                        'SELECT attempts,lease_owner FROM events',
                    ).fetchone()
                self.assertEqual(attempts, 1)
                self.assertEqual(owner, '')
                self.assertLessEqual(_callbacks.peak, 2)
                self.assertIsNone(fast.outbox.ready())
            finally:
                await dispatcher.close()

    async def test_busy_camera_rejoins_fifo_tail_without_poll_delay(self):
        self.queue.outbox.path = Path(self.directory.name) / 'a.sqlite3'
        cold = ViolationQueue('cold-camera')
        cold.outbox.path = Path(self.directory.name) / 'z.sqlite3'
        for _ in range(5):
            await self.queue.submit(metadata(stream_name='busy'), b'jpeg')
        await cold.submit(metadata(stream_name='cold'), b'jpeg')
        order = []
        done = asyncio.Event()

        upload = partial(_busy_camera_rejoins_fifo_tail_upload, done, order)

        self.upload.send_violation.side_effect = upload
        with patch.dict(
            os.environ,
            {
                'VIOLATION_OUTBOX_CONCURRENCY': '1',
                'VIOLATION_OUTBOX_POLL_SECONDS': '5',
            },
        ), patch(
            'src.violation_delivery.ViolationSender',
            return_value=self.upload,
        ), patch(
            'src.violation_delivery.FCMSender',
            return_value=self.notify,
        ):
            dispatcher = OutboxDispatcher()
            dispatcher.start()
            try:
                await asyncio.wait_for(done.wait(), 2)
                self.assertEqual(order[:2], ['busy', 'cold'])
            finally:
                await dispatcher.close()

    async def test_cancellation_during_claim_does_not_abandon_a_lease(self):
        _callbacks = _CancellationDuringClaimDoesNotAbandonACallbacks()
        await self.queue.submit(metadata(), b'jpeg')
        _callbacks.started, _callbacks.release = (
            threading.Event(),
            threading.Event(),
        )
        _callbacks.claim = ViolationOutbox.claim
        cancelled = asyncio.Event()
        shield = asyncio.shield

        observe_cancellation = partial(
            _cancellation_during_claim_does_not_observe_cancellation,
            cancelled,
            shield,
        )

        delayed_claim = partialmethod(_delayed_outbox_claim, _scope=_callbacks)

        with patch(
            'src.violation_delivery.ViolationSender',
            return_value=self.upload,
        ), patch(
            'src.violation_delivery.FCMSender',
            return_value=self.notify,
        ), patch.object(
            ViolationOutbox,
            'claim',
            delayed_claim,
        ), patch(
            'src.violation_delivery.asyncio.shield',
            side_effect=observe_cancellation,
        ):
            dispatcher = OutboxDispatcher()
            dispatcher.start()
            await asyncio.to_thread(_callbacks.started.wait, 2)
            close = asyncio.create_task(dispatcher.close())
            try:
                await asyncio.wait_for(cancelled.wait(), 2)
            finally:
                _callbacks.release.set()
            await asyncio.wait_for(close, 2)
        event = self.queue.outbox.claim()
        self.assertIsNotNone(event)
        self.upload.send_violation.assert_not_awaited()

    async def test_dispatcher_start_and_close_own_the_background_task(
        self,
    ) -> None:
        dispatcher = OutboxDispatcher()
        started = asyncio.Event()
        released = asyncio.Event()

        run = partial(_dispatcher_start_and_close_own_run, released, started)

        with patch.object(dispatcher, 'run', run):
            dispatcher.start()
            task = dispatcher.task
            dispatcher.start()
            self.assertIs(dispatcher.task, task)
            await started.wait()
            await dispatcher.close()
        self.assertTrue(released.is_set())
        self.assertIsNone(dispatcher.task)


class CoupledTaskLifetimeTest(unittest.IsolatedAsyncioTestCase):
    async def test_parent_cancellation_drains_pending_worker(self) -> None:
        released = asyncio.Event()

        worker = partial(
            _parent_cancellation_drains_pending_worker_worker,
            released,
        )

        child = asyncio.create_task(worker())
        task = asyncio.create_task(run_coupled_tasks({child}))
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(child.done())
        self.assertTrue(released.is_set())


class OutboxArchiveTest(unittest.TestCase):
    def test_failed_archive_does_not_consume_pending_slots(self):
        with tempfile.TemporaryDirectory() as directory:
            box = ViolationOutbox(Path(directory) / 'outbox.sqlite3', 1, 65536)
            box.put(ViolationEvent('failed', metadata(), b'first'))
            event = box.claim()
            box.fail(event, 'permanent')
            box.put(ViolationEvent('next', metadata(), b'next'))
            self.assertEqual(box.ready().id, 'next')
            self.assertEqual(box.failures()[0]['id'], 'failed')
            self.assertEqual(box.inspect('failed')['image_bytes'], 5)
            with self.assertRaisesRegex(RuntimeError, 'capacity'):
                box.requeue('failed')
            box.complete(box.claim())
            self.assertTrue(box.requeue('failed'))
            self.assertEqual(box.failures(), [])
            self.assertEqual(box.ready().image, b'first')

    def test_archive_eviction_enforces_combined_payload_limit(self):
        from src.settings.delivery import OutboxRetention

        with tempfile.TemporaryDirectory() as directory:
            box = ViolationOutbox(
                Path(directory) / 'outbox.sqlite3',
                1,
                2000,
                OutboxRetention(max_events=1, max_bytes=2000),
            )
            for index in range(3):
                box.put(ViolationEvent(str(index), metadata(), b'image'))
                box.fail(box.claim(), 'permanent')
            self.assertEqual([row['id'] for row in box.failures()], ['2'])
            self.assertEqual(box.purge_failures(float('inf')), 1)
            self.assertIsNone(box.inspect('missing'))
            self.assertFalse(box.requeue('missing'))
