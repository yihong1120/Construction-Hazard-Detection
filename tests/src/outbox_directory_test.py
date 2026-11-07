from __future__ import annotations

import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from functools import partial
from pathlib import Path
from unittest.mock import patch

from src.outbox_directory import OutboxDirectory
from src.outbox_directory import WriterLease
from src.settings.delivery import DeliverySettings
from src.violation_delivery import ViolationEvent
from src.violation_delivery import ViolationOutbox
from tests.src.violation_delivery_test import metadata


def _concurrent_camera_writers_share_one_write(
    item,
    testcase,
    /,
    index,
):
    box = ViolationOutbox(
        testcase.root / f'{index}.sqlite3',
        10,
        65536,
    )
    try:
        box.put(item)
        return True
    except RuntimeError:
        return False


def _steady_state_writes_use_incremental_traced(
    connect,
    trace,
    /,
    *args,
    **kwargs,
):
    connection = connect(*args, **kwargs)
    connection.set_trace_callback(trace.append)
    return connection


class OutboxDirectoryTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_concurrent_camera_writers_share_one_budget(self):
        item = ViolationEvent('event', metadata(), b'image')
        size = len(item.serialise().encode()) + len(item.image)
        with patch.dict(
            'os.environ',
            {
                'VIOLATION_OUTBOX_DIRECTORY_MAX_BYTES': str(size * 2),
                'VIOLATION_OUTBOX_MIN_FREE_BYTES': '0',
            },
        ):

            write = partial(
                _concurrent_camera_writers_share_one_write,
                item,
                self,
            )

            with ThreadPoolExecutor(max_workers=6) as workers:
                self.assertEqual(sum(workers.map(write, range(6))), 2)
            index = OutboxDirectory(self.root)
            self.assertEqual(len(index.due(set(), 10)), 2)
            self.assertEqual(len(index.due(set(index.due(set(), 10)), 10)), 0)

    def test_idle_reclamation_respects_writer_lease_and_old_objects_reopen(
        self,
    ):
        box = ViolationOutbox(self.root / 'camera.sqlite3', 2, 65536)
        box.put(ViolationEvent('first', metadata(), b'image'))
        box.complete(box.claim())
        directory = OutboxDirectory(
            self.root,
            DeliverySettings(idle_seconds=0),
        )
        lease = WriterLease(box.path)
        try:
            self.assertEqual(directory.reclaim(), 0)
            self.assertTrue(box.path.exists())
        finally:
            lease.close()
        self.assertEqual(directory.reclaim(), 1)
        self.assertFalse(box.path.exists())
        box.put(ViolationEvent('next', metadata(), b'image'))
        self.assertEqual(box.ready().id, 'next')

    def test_expired_archives_are_cleaned_without_dropping_pending_work(self):
        box = ViolationOutbox(self.root / 'camera.sqlite3', 2, 65536)
        box.put(ViolationEvent('pending', metadata(), b'image'))
        directory = OutboxDirectory(
            self.root,
            DeliverySettings(idle_seconds=0),
        )
        self.assertEqual(directory.reclaim(), 0)
        box.fail(box.claim(), 'permanent')
        with closing(sqlite3.connect(box.path)) as db, db:
            db.execute('UPDATE failed_events SET failed_at=0')
        self.assertEqual(directory.reclaim(), 1)

    def test_dirty_marker_recovers_after_unaccounted_commit(self):
        first = ViolationEvent('first', metadata(), b'image')
        size = len(first.serialise().encode()) + len(first.image)
        box = ViolationOutbox(self.root / 'first.sqlite3', 2, 65536)
        with patch.dict(
            'os.environ',
            {
                'VIOLATION_OUTBOX_DIRECTORY_MAX_BYTES': str(size * 2),
                'VIOLATION_OUTBOX_MIN_FREE_BYTES': '0',
            },
        ):
            box.put(first)
            directory = OutboxDirectory(self.root)
            directory.mark_dirty()
            # Simulate a process dying between committing the queue and index.
            with closing(sqlite3.connect(box.path)) as db, db:
                db.execute(
                    'INSERT INTO events(id,payload,image) VALUES(?,?,?)',
                    ('unaccounted', first.serialise(), first.image),
                )
            other = ViolationOutbox(self.root / 'second.sqlite3', 2, 65536)
            with self.assertRaisesRegex(RuntimeError, 'directory capacity'):
                other.put(first)
            self.assertEqual(len(directory.due(set(), 10)), 1)
            self.assertEqual(box.ready().id, 'first')

    def test_disk_reserve_rejects_new_evidence_without_losing_existing_rows(
        self,
    ):
        box = ViolationOutbox(self.root / 'camera.sqlite3', 2, 65536)
        box.put(ViolationEvent('first', metadata(), b'image'))
        from types import SimpleNamespace

        with patch(
            'src.outbox_directory.shutil.disk_usage',
            return_value=SimpleNamespace(free=0),
        ):
            with self.assertRaisesRegex(RuntimeError, 'disk reserve'):
                box.put(ViolationEvent('second', metadata(), b'image'))
        self.assertEqual(box.ready().id, 'first')


def _process_put(root, limit, number, result):
    with patch.dict(
        'os.environ',
        {
            'VIOLATION_OUTBOX_DIRECTORY_MAX_BYTES': str(limit),
            'VIOLATION_OUTBOX_MIN_FREE_BYTES': '0',
        },
    ):
        box = ViolationOutbox(
            Path(root) / f'process-{number}.sqlite3',
            5,
            65536,
        )
        try:
            box.put(ViolationEvent(str(number), metadata(), b'image'))
            result.put(True)
        except RuntimeError:
            result.put(False)


class DirectoryProcessTest(unittest.TestCase):
    def test_separate_processes_cannot_overrun_directory_budget(self):
        import multiprocessing

        item = ViolationEvent('sample', metadata(), b'image')
        size = len(item.serialise().encode()) + len(item.image)
        with tempfile.TemporaryDirectory() as root:
            context = multiprocessing.get_context('spawn')
            result = context.Queue()
            children = [
                context.Process(
                    target=_process_put,
                    args=(root, size * 2, index, result),
                )
                for index in range(4)
            ]
            try:
                for child in children:
                    child.start()
                outcomes = [result.get(timeout=15) for _ in children]
                self.assertEqual(sum(outcomes), 2)
            finally:
                for child in children:
                    child.join(timeout=3)
                    if child.is_alive():
                        child.kill()
                        child.join(timeout=3)
                result.close()
                result.join_thread()


class OutboxAccountingTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_steady_state_writes_use_incremental_counts_without_rescanning(
        self,
    ):
        box = ViolationOutbox(self.root / 'camera.sqlite3', 20, 65536)
        box.put(ViolationEvent('first', metadata(), b'image'))
        trace = []
        connect = sqlite3.connect

        traced = partial(
            _steady_state_writes_use_incremental_traced,
            connect,
            trace,
        )

        with patch('sqlite3.connect', traced), patch.object(
            Path,
            'glob',
            side_effect=AssertionError('steady-state full scan'),
        ):
            first = box.claim()
            box.uploaded(first, '17')
            box.retry(first, 0)
            first = box.claim()
            box.fail(first, 'test')
            box.requeue(first.id)
            box.complete(box.claim())
            box.put(ViolationEvent('next', metadata(), b'next-image'))
        self.assertFalse(
            any(
                'CREATE ' in sql.upper() or 'SUM(' in sql.upper()
                for sql in trace
            ),
        )
        with closing(connect(box.path)) as db:
            count, size = db.execute(
                'SELECT count(*),sum(length(image)+length(cast(payload '
                'AS BLOB))) FROM events',
            ).fetchone()
            self.assertEqual(
                db.execute(
                    'SELECT pending,pending_bytes FROM outbox_stats',
                ).fetchone(),
                (count, size),
            )
        with closing(connect(self.root / '.outbox-index.db')) as db:
            self.assertEqual(
                db.execute('SELECT bytes FROM budget').fetchone()[0],
                size,
            )

    def test_archive_cleanup_is_bounded_and_respects_live_writer(self):
        box = ViolationOutbox(self.root / 'camera.sqlite3', 20, 65536)
        for index in range(3):
            box.put(ViolationEvent(str(index), metadata(), b'image'))
            box.fail(box.claim(), 'test')
        with closing(sqlite3.connect(box.path)) as db, db:
            db.execute('UPDATE failed_events SET failed_at=0')
        directory = OutboxDirectory(
            self.root,
            DeliverySettings(
                idle_seconds=0,
                archive_cleanup_batch_size=1,
            ),
        )
        lease = WriterLease(box.path)
        try:
            self.assertEqual(directory.reclaim(), 0)
            self.assertEqual(len(box.failures()), 2)
            directory.reclaim()
            self.assertEqual(len(box.failures()), 1)
            directory.reclaim()
            self.assertTrue(box.path.exists())
        finally:
            lease.close()
        self.assertEqual(directory.reclaim(), 1)

    def test_unreadable_orphan_blocks_only_new_reservations(
        self,
    ):
        box = ViolationOutbox(self.root / 'healthy.sqlite3', 20, 65536)
        box.put(ViolationEvent('pending', metadata(), b'image'))
        bad = self.root / 'bad.sqlite3'
        bad.write_bytes(b'not a database')
        directory = OutboxDirectory(self.root)
        directory.discover()
        directory.close()
        self.assertIsNotNone(box.claim())
        with self.assertRaisesRegex(RuntimeError, 'capacity unknown'):
            box.put(ViolationEvent('next', metadata(), b'image'))
        bad.unlink()
        directory.discover()
        directory.close()
        box.put(ViolationEvent('next', metadata(), b'image'))

    def test_empty_legacy_queue_without_archive_table_is_reclaimed(self):
        path = self.root / 'legacy.sqlite3'
        with closing(sqlite3.connect(path)) as db, db:
            db.execute(
                '''CREATE TABLE events(id TEXT, payload TEXT, image BLOB,
                next_attempt REAL DEFAULT 0)''',
            )
        directory = OutboxDirectory(
            self.root,
            DeliverySettings(idle_seconds=0),
        )
        self.assertEqual(directory.due(set(), 1), [])
        self.assertEqual(directory.reclaim(), 1)
        self.assertFalse(path.exists())


class GlobalArchiveCapacityTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.event = ViolationEvent('event', metadata(), b'image')
        self.size = len(self.event.serialise().encode()) + len(
            self.event.image,
        )
        config = patch.dict(
            'os.environ',
            {
                'VIOLATION_OUTBOX_DIRECTORY_MAX_BYTES': str(self.size * 3),
                'VIOLATION_OUTBOX_DIRECTORY_ARCHIVE_MAX_BYTES': str(
                    self.size * 3,
                ),
                'VIOLATION_OUTBOX_MIN_FREE_BYTES': '0',
            },
        )
        config.start()
        self.addCleanup(config.stop)

    def box(self, name):
        return ViolationOutbox(self.root / f'{name}.sqlite3', 10, 65536)

    def archive(self, name):
        box = self.box(name)
        box.put(self.event)
        box.fail(box.claim(), 'permanent')
        return box

    def test_new_camera_reclaims_oldest_archive_from_other_cameras(self):
        older = self.archive('older')
        newer = self.archive('newer')
        pending = self.box('pending')
        pending.put(self.event)
        newest = self.box('new-camera')
        newest.put(self.event)
        self.assertEqual(older.failures(), [])
        self.assertEqual(len(newer.failures()), 1)
        self.assertIsNotNone(pending.ready())
        self.assertIsNotNone(newest.ready())

    def test_archive_budget_retains_pending_events_and_removes_oldest_first(
        self,
    ):
        older = self.archive('older')
        newer = self.archive('newer')
        pending = self.box('pending')
        pending.put(self.event)
        with patch.dict(
            'os.environ',
            {'VIOLATION_OUTBOX_DIRECTORY_ARCHIVE_MAX_BYTES': str(self.size)},
        ):
            self.assertEqual(older.failures(), [])
            self.assertEqual(len(newer.failures()), 1)
            self.assertIsNotNone(pending.ready())
        with sqlite3.connect(self.root / '.outbox-index.db') as index:
            total, archives = index.execute(
                'SELECT bytes,archive_bytes FROM budget',
            ).fetchone()
        self.assertEqual(archives, self.size)
        self.assertEqual(total, self.size * 2)

    def test_all_pending_work_is_preserved_when_no_archives_can_make_room(
        self,
    ):
        for camera in ('a', 'b', 'c'):
            self.box(camera).put(self.event)
        with self.assertRaisesRegex(RuntimeError, 'capacity exhausted'):
            self.box('d').put(self.event)
        for camera in ('a', 'b', 'c'):
            self.assertIsNotNone(self.box(camera).ready())

    def test_failed_global_index_commit_recovers_all_evicted_queue_counters(
        self,
    ):
        older = self.archive('older')
        newer = self.archive('newer')
        settings = DeliverySettings(
            directory=self.root,
            directory_max_bytes=self.size * 3,
            directory_archive_max_bytes=0,
            min_free_bytes=0,
        )
        directory = OutboxDirectory(self.root, settings)
        with self.assertRaisesRegex(RuntimeError, 'crash'):
            with directory.locked() as index:
                directory.trim_archives(index)
                raise RuntimeError('crash')
        with directory.locked() as index:
            self.assertEqual(
                index.execute(
                    'SELECT bytes,archive_bytes FROM budget',
                ).fetchone(),
                (0, 0),
            )
        self.assertFalse((self.root / '.accounting-dirty').exists())
        self.assertEqual(older.failures(), [])
        self.assertEqual(newer.failures(), [])

    def test_rejected_local_admission_keeps_other_camera_archives(self):
        first = self.archive('old-one')
        second = self.archive('old-two')
        full = ViolationOutbox(self.root / 'full.sqlite3', 1, 65536)
        full.put(self.event)
        with self.assertRaisesRegex(
            RuntimeError,
            'Violation outbox capacity exhausted',
        ):
            full.put(
                ViolationEvent('new', self.event.metadata, self.event.image),
            )
        self.assertEqual(len(first.failures()) + len(second.failures()), 2)
        self.assertIsNotNone(full.ready())

    def test_oversized_or_duplicate_event_cannot_trigger_global_eviction(self):
        first = self.archive('old-one')
        second = self.archive('old-two')
        full = self.box('full')
        full.put(self.event)
        for item in (
            self.event,
            ViolationEvent(
                'huge',
                self.event.metadata,
                b'x' * (self.size * 10),
            ),
        ):
            with self.assertRaises((sqlite3.IntegrityError, RuntimeError)):
                full.put(item)
        self.assertEqual(len(first.failures()) + len(second.failures()), 2)
