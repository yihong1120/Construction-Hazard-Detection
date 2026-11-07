from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import AsyncExitStack
from contextlib import closing
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import httpx

from src.async_tasks import finish_on_cancel
from src.auth_tokens import TokenManager
from src.delivery_result import DeliveryOutcome
from src.delivery_result import retryable_status
from src.evidence import EVIDENCE_METADATA
from src.evidence import EvidenceMetadata
from src.notifiers.fcm_notifier import FCMSender
from src.outbox_directory import OutboxDirectory
from src.outbox_directory import WriterLease
from src.outbox_schema import install_outbox_statistics
from src.settings.delivery import DeliverySettings
from src.settings.delivery import OutboxRetention
from src.violation_sender import ViolationSender

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ViolationEvent:
    id: str
    metadata: EvidenceMetadata
    image: bytes
    violation_id: str | None = None
    attempts: int = 0
    owner: str = ''

    def serialise(self) -> str:
        return json.dumps(
            {
                'version': 1,
                'metadata': EVIDENCE_METADATA.dump_python(
                    self.metadata,
                    mode='json',
                ),
            },
        )

    @classmethod
    def from_row(cls, row: tuple) -> ViolationEvent:
        payload = json.loads(row[1])
        if 'version' not in payload:
            metadata = EvidenceMetadata.from_legacy_payload(payload)
        elif payload['version'] == 1:
            metadata = EVIDENCE_METADATA.validate_python(payload['metadata'])
        else:
            raise ValueError('Unsupported evidence event version')
        return cls(row[0], metadata, row[2], row[3], row[4], row[5])


class ViolationOutbox:
    """Bound evidence on disk and retain failed events for inspection."""

    def __init__(
        self,
        path: Path,
        max_events: int,
        max_bytes: int,
        retention: OutboxRetention | None = None,
    ) -> None:
        self.retention = retention or OutboxRetention()
        self.path, self.max_events, self.max_bytes = (
            path,
            max_events,
            max_bytes,
        )
        self._initialised = False
        self._initialise_lock = threading.Lock()

    @contextmanager
    def _local_connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(mode=448, parents=True, exist_ok=True)
        new_file = not self.path.exists()
        if new_file:
            self._initialised = False
        db = sqlite3.connect(self.path, timeout=5)
        try:
            if new_file:
                db.execute('PRAGMA auto_vacuum=INCREMENTAL')
            os.chmod(self.path, 384)
            with self._initialise_lock:
                if not self._initialised:
                    db.execute('PRAGMA journal_mode=WAL')
                    with db:
                        db.execute('BEGIN IMMEDIATE')
                        db.execute(
                            (
                                'CREATE TABLE IF NOT EXISTS events ( id '
                                'TEXT PRIMARY KEY, payload TEXT NOT NULL,'
                                ' image BLOB NOT NULL, violation_id TEXT,'
                                ' attempts INTEGER NOT NULL DEFAULT 0, '
                                'next_attempt REAL NOT NULL DEFAULT 0 )'
                            ),
                        )
                        columns = {
                            row[1]
                            for row in db.execute(
                                'PRAGMA table_info(events)',
                            )
                        }
                        for name, definition in (
                            ('state', "TEXT NOT NULL DEFAULT 'pending'"),
                            ('last_error', "TEXT NOT NULL DEFAULT ''"),
                            ('lease_owner', "TEXT NOT NULL DEFAULT ''"),
                            ('lease_until', 'REAL NOT NULL DEFAULT 0'),
                        ):
                            if name not in columns:
                                db.execute(
                                    f'ALTER TABLE events ADD COLUMN '
                                    f'{name} {definition}',
                                )
                        db.execute(
                            (
                                'CREATE TABLE IF NOT EXISTS '
                                'failed_events (id TEXT PRIMARY KEY, '
                                'payload TEXT NOT NULL, image BLOB NOT '
                                'NULL, violation_id TEXT, attempts '
                                'INTEGER NOT NULL, last_error TEXT NOT '
                                'NULL, failed_at REAL NOT NULL)'
                            ),
                        )
                        db.execute(
                            (
                                'INSERT OR IGNORE INTO failed_events '
                                'SELECT id,payload,image,violation_id,'
                                'attempts,last_error,? FROM events WHERE '
                                "state='failed'"
                            ),
                            (time.time(),),
                        )
                        db.execute("DELETE FROM events WHERE state='failed'")
                        db.execute('PRAGMA journal_size_limit=1048576')
                    with db:
                        install_outbox_statistics(db)
                    self._initialised = True
            with db:
                yield db
        finally:
            db.close()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        directory = OutboxDirectory(self.path.parent)
        with directory.locked() as index:
            directory.mark_dirty(self.path.name)
            succeeded = False
            try:
                with self._local_connect() as db:
                    directory.account(index, db, self.path.name)
                    self._directory_index = index
                    self._directory = directory
                    yield db
                succeeded = True
            finally:
                if self.path.exists():
                    with closing(sqlite3.connect(self.path)) as actual:
                        directory.account(index, actual, self.path.name)
                if succeeded:
                    directory.trim_archives(index)

    @staticmethod
    def _stored_bytes(db: sqlite3.Connection, table: str) -> int:
        column = 'pending_bytes' if table == 'events' else 'archived_bytes'
        return db.execute(
            f'SELECT {column} FROM outbox_stats WHERE id=1',
        ).fetchone()[0]

    def _trim_archive(self, db: sqlite3.Connection, incoming: int = 0) -> None:
        db.execute(
            'DELETE FROM failed_events WHERE failed_at<?',
            (time.time() - self.retention.seconds,),
        )
        pending = self._stored_bytes(db, 'events')
        limit = min(
            self.retention.max_bytes,
            max(
                0,
                self.max_bytes - pending - incoming,
            ),
        )
        while True:
            count = db.execute(
                'SELECT archived FROM outbox_stats WHERE id=1',
            ).fetchone()[0]
            if (
                count <= self.retention.max_events
                and self._stored_bytes(
                    db,
                    'failed_events',
                )
                <= limit
            ):
                break
            db.execute(
                (
                    'DELETE FROM failed_events WHERE id=(SELECT id FROM '
                    'failed_events ORDER BY failed_at,id LIMIT 1)'
                ),
            )

    def _validate_capacity(self, db: sqlite3.Connection, size: int) -> None:
        count = db.execute(
            'SELECT pending FROM outbox_stats WHERE id=1',
        ).fetchone()[0]
        if (
            count >= self.max_events
            or self._stored_bytes(
                db,
                'events',
            )
            + size
            > self.max_bytes
        ):
            raise RuntimeError('Violation outbox capacity exhausted')

    def _check_capacity(self, db: sqlite3.Connection, size: int) -> None:
        self._validate_capacity(db, size)
        self._trim_archive(db, size)

    def put(self, event: ViolationEvent) -> None:
        payload = event.serialise()
        incoming = len(event.image) + len(payload.encode())
        with self._connect() as db:
            # Reject impossible admissions before mutating any archive.
            if db.execute(
                'SELECT 1 FROM events WHERE id=?',
                (event.id,),
            ).fetchone():
                raise sqlite3.IntegrityError(
                    'Evidence event is already queued',
                )
            self._validate_capacity(db, incoming)
            self._directory.reserve(self._directory_index, incoming)
            db.execute('BEGIN IMMEDIATE')
            self._trim_archive(db, incoming)
            db.execute(
                'INSERT INTO events(id,payload,image) VALUES (?,?,?)',
                (event.id, payload, event.image),
            )

    def _archive(
        self,
        db: sqlite3.Connection,
        event_id: str,
        reason: str,
    ) -> None:
        db.execute(
            (
                'INSERT INTO failed_events SELECT id,payload,image,'
                'violation_id,attempts,?,? FROM events WHERE id=? ON '
                'CONFLICT(id) DO UPDATE SET payload=excluded.payload,'
                'image=excluded.image,violation_id=excluded.violation_id,'
                'attempts=excluded.attempts,'
                'last_error=excluded.last_error,'
                'failed_at=excluded.failed_at'
            ),
            (reason[:500], time.time(), event_id),
        )
        db.execute('DELETE FROM events WHERE id=?', (event_id,))
        self._trim_archive(db)

    def _take(self, *, claim: bool) -> ViolationEvent | None:
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            while True:
                row = db.execute(
                    (
                        'SELECT id,payload,image,violation_id,attempts,'
                        "lease_owner FROM events WHERE state='pending' "
                        'AND next_attempt<=? AND lease_until<=? ORDER BY '
                        'rowid LIMIT 1'
                    ),
                    (
                        time.time(),
                        time.time(),
                    ),
                ).fetchone()
                if row is None:
                    return None
                try:
                    event = ViolationEvent.from_row(row)
                except (ValueError, TypeError, KeyError):
                    self._archive(db, row[0], 'invalid_event_format')
                    logger.error(
                        'Invalid persisted evidence event quarantined: %s',
                        row[0],
                    )
                    continue
                if claim:
                    owner = uuid4().hex
                    db.execute(
                        'UPDATE events SET lease_owner=?,lease_until=? '
                        'WHERE id=?',
                        (owner, time.time() + 120, event.id),
                    )
                    event = replace(event, owner=owner)
                return event

    def ready(self) -> ViolationEvent | None:
        return self._take(claim=False)

    def claim(self) -> ViolationEvent | None:
        return self._take(claim=True)

    def uploaded(self, event: ViolationEvent, violation_id: str) -> None:
        with self._connect() as db:
            result = db.execute(
                (
                    "UPDATE events SET violation_id=?,image=X'' WHERE "
                    'id=? AND lease_owner=?'
                ),
                (violation_id, event.id, event.owner),
            )
            if result.rowcount != 1:
                raise RuntimeError('Evidence delivery lease lost')

    def complete(self, event: ViolationEvent) -> None:
        with self._connect() as db:
            db.execute(
                'DELETE FROM events WHERE id=? AND lease_owner=?',
                (event.id, event.owner),
            )

    def retry(
        self,
        event: ViolationEvent,
        delay: float,
        reason: str = '',
    ) -> None:
        with self._connect() as db:
            db.execute(
                (
                    'UPDATE events SET attempts=attempts+1,'
                    'next_attempt=?,last_error=?,lease_until=0,'
                    "lease_owner='' WHERE id=? AND lease_owner=?"
                ),
                (time.time() + delay, reason[:500], event.id, event.owner),
            )

    def fail(self, event: ViolationEvent, reason: str) -> None:
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute(
                'SELECT 1 FROM events WHERE id=? AND lease_owner=?',
                (event.id, event.owner),
            ).fetchone():
                self._archive(db, event.id, reason)

    def failures(self) -> list[dict]:
        with self._connect() as db:
            self._directory.trim_archives(self._directory_index)
            rows = db.execute(
                (
                    'SELECT id,violation_id,attempts,last_error,'
                    'failed_at,length(image) FROM failed_events ORDER BY '
                    'failed_at,id'
                ),
            ).fetchall()
        return [
            dict(
                zip(
                    (
                        'id',
                        'violation_id',
                        'attempts',
                        'reason',
                        'failed_at',
                        'image_bytes',
                    ),
                    row,
                ),
            )
            for row in rows
        ]

    def requeue(self, event_id: str) -> bool:
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute(
                (
                    'SELECT payload,image,violation_id FROM '
                    'failed_events WHERE id=?'
                ),
                (event_id,),
            ).fetchone()
            if row is None:
                return False
            db.execute('DELETE FROM failed_events WHERE id=?', (event_id,))
            self._check_capacity(db, len(row[0].encode()) + len(row[1]))
            db.execute(
                (
                    'INSERT INTO events(id,payload,image,violation_id) '
                    'VALUES (?,?,?,?)'
                ),
                (event_id, *row),
            )
            return True

    def purge_failures(self, before: float) -> int:
        with self._connect() as db:
            return db.execute(
                'DELETE FROM failed_events WHERE failed_at<?',
                (before,),
            ).rowcount

    def inspect(self, event_id: str) -> dict | None:
        with self._connect() as db:
            row = db.execute(
                (
                    'SELECT payload,violation_id,attempts,last_error,'
                    'length(image) FROM failed_events WHERE id=?'
                ),
                (event_id,),
            ).fetchone()
        return (
            dict(
                zip(
                    (
                        'payload',
                        'violation_id',
                        'attempts',
                        'reason',
                        'image_bytes',
                    ),
                    row,
                ),
            )
            if row
            else None
        )


class ViolationQueue:
    """A camera only persists events; its lifetime does not own delivery."""

    def __init__(self, camera_key: str) -> None:
        settings = DeliverySettings()
        name = hashlib.sha256(camera_key.encode()).hexdigest()
        self._writer: WriterLease | None = None
        self.outbox = ViolationOutbox(
            settings.directory / f'{name}.sqlite3',
            settings.max_events,
            settings.max_bytes,
            settings.archive,
        )

    async def submit(self, metadata: EvidenceMetadata, image: bytes) -> str:
        if not image:
            raise ValueError('Evidence image encoding produced an empty image')
        if self._writer is None:
            self._writer = await asyncio.to_thread(
                WriterLease,
                self.outbox.path,
            )
        event_id = metadata.event_id or str(uuid4())
        await asyncio.to_thread(
            self.outbox.put,
            ViolationEvent(
                event_id,
                replace(metadata, event_id=event_id),
                image,
            ),
        )
        return event_id

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None


class PermanentDeliveryError(RuntimeError):
    pass


class ViolationDelivery:

    def __init__(
        self,
        outbox: ViolationOutbox,
        violations: ViolationSender,
        notifications: FCMSender,
    ) -> None:
        self.outbox, self.violations, self.notifications = (
            outbox,
            violations,
            notifications,
        )

    async def deliver(self, event: ViolationEvent) -> None:
        violation_id = event.violation_id
        if violation_id is None:
            violation_id = await self.violations.send_violation(
                replace(event.metadata, event_id=event.id),
                event.image,
            )
            if violation_id is None:
                raise RuntimeError(
                    'Violation upload did not return an identifier',
                )
            int(violation_id)
            await asyncio.to_thread(
                self.outbox.uploaded,
                event,
                str(violation_id),
            )
        data = event.metadata
        message = json.loads(data.warnings_json) if data.warnings_json else {}
        result = await self.notifications.send_fcm_message_to_site(
            site=data.site,
            site_id=data.site_id,
            stream_name=data.stream_name,
            message=message,
            image_path=None,
            violation_id=int(violation_id),
        )
        if result.outcome == DeliveryOutcome.permanent_failure:
            raise PermanentDeliveryError(result.reason)
        if not result:
            raise RuntimeError(
                'Notification delivery was not completed: ' + result.reason,
            )
        await asyncio.to_thread(self.outbox.complete, event)


class OutboxDispatcher:
    """Drain all camera queues, including disabled or removed cameras."""

    def __init__(self) -> None:
        self.settings = DeliverySettings()
        self.task: asyncio.Task | None = None
        self.outboxes: OrderedDict[Path, ViolationOutbox] = OrderedDict()
        self.directory = OutboxDirectory(
            self.settings.directory,
            self.settings,
        )
        self._failures = 0
        self._last_error: str | None = None
        self._last_poll: float | None = None
        self._recovery_attempts = 0
        self._stop = asyncio.Event()
        self._closing: asyncio.Task | None = None

    @property
    def health(self) -> dict:
        return {
            'running': self.task is not None and not self.task.done(),
            'failures': self._failures,
            'last_error': self._last_error,
            'last_poll': self._last_poll,
        }

    def start(self) -> None:
        if self._closing is not None and not self._closing.done():
            raise RuntimeError('Outbox dispatcher is closing')
        if self.task is None or self.task.done():
            self._stop.clear()
            self._closing = None
            self.task = asyncio.create_task(self._supervise())

    async def _supervise(self):
        while not self._stop.is_set():
            try:
                await self.run()
                raise RuntimeError('Outbox dispatcher stopped unexpectedly')
            except asyncio.CancelledError:
                raise
            except Exception as error:
                if self._stop.is_set():
                    return
                self._failures += 1
                self._last_error = type(error).__name__
                self.directory.close()
                delay = min(
                    self.settings.max_recovery_seconds,
                    self.settings.recovery_seconds
                    * 2 ** min(self._recovery_attempts, 10),
                )
                self._recovery_attempts += 1
                logger.warning(
                    'Outbox dispatcher deferred error=%s '
                    'retry_seconds=%.2f',
                    self._last_error,
                    delay,
                )
                try:
                    await asyncio.wait_for(self._stop.wait(), delay)
                except TimeoutError:
                    pass

    async def close(self) -> None:
        if self._closing is None:
            self._closing = asyncio.create_task(self._shutdown())
        await finish_on_cancel(self._closing)

    async def _shutdown(self):
        self._stop.set()
        if self.task is not None:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
            self.task = None
        self.directory.close()

    async def run(self) -> None:
        await _OutboxRun(self).run()

    async def _deliver_one(
        self,
        path: Path,
        violations: ViolationSender,
        notifications: FCMSender,
    ) -> bool:
        box = self.outboxes.get(path)
        if box is None:
            box = self.outboxes[path] = ViolationOutbox(
                path,
                self.settings.max_events,
                self.settings.max_bytes,
                self.settings.archive,
            )
        self.outboxes.move_to_end(path)
        while len(self.outboxes) > self.settings.cache_size:
            self.outboxes.popitem(last=False)
        event = None
        try:
            claim = asyncio.create_task(asyncio.to_thread(box.claim))
            try:
                event = await asyncio.shield(claim)
            except asyncio.CancelledError:
                event = await claim
                raise
            if event is None:
                return False
            async with asyncio.timeout(self.settings.event_timeout_seconds):
                await ViolationDelivery(
                    box, violations, notifications,
                ).deliver(event)
            return True
        except asyncio.CancelledError:
            if event is not None:
                await asyncio.to_thread(box.retry, event, 0, 'interrupted')
            raise
        except Exception as error:
            if event is None:
                logger.exception('Evidence queue unavailable: %s', path.name)
                return False
            permanent = isinstance(
                error, (PermanentDeliveryError, ValueError, KeyError),
            )
            if isinstance(error, httpx.HTTPStatusError):
                permanent = not retryable_status(error.response.status_code)
            reason = (
                f'HTTP {error.response.status_code}'
                if isinstance(error, httpx.HTTPStatusError)
                else type(error).__name__
            )
            try:
                if (
                    permanent
                    or event.attempts + 1 >= self.settings.max_attempts
                ):
                    await asyncio.to_thread(box.fail, event, reason)
                    logger.error(
                        'Evidence event requires intervention: %s (%s)',
                        event.id,
                        reason,
                    )
                else:
                    delay = min(
                        self.settings.max_retry_seconds,
                        2 ** min(event.attempts, 10),
                    )
                    await asyncio.to_thread(box.retry, event, delay, reason)
            except Exception:
                logger.exception('Evidence queue unavailable: %s', path.name)
            return False


class _OutboxRun:
    def __init__(self, dispatcher: OutboxDispatcher):
        self.dispatcher = dispatcher

    async def run(self) -> None:
        async with AsyncExitStack() as resources:
            tokens = TokenManager()
            resources.push_async_callback(tokens.close)
            self.violations = ViolationSender(
                max_retries=1,
                token_manager=tokens,
            )
            resources.push_async_callback(self.violations.close)
            self.notifications = FCMSender(max_retries=0, token_manager=tokens)
            resources.push_async_callback(self.notifications.close)
            self.queue: asyncio.Queue[Path] = asyncio.Queue(
                maxsize=self.dispatcher.settings.scheduler_capacity,
            )
            self.scheduled: set[Path] = set()

            async with asyncio.TaskGroup() as workers:
                for _ in range(self.dispatcher.settings.concurrency):
                    workers.create_task(self._worker())
                while True:
                    discovering = asyncio.create_task(
                        asyncio.to_thread(self.dispatcher.directory.discover),
                    )
                    try:
                        await asyncio.shield(discovering)
                    except asyncio.CancelledError:
                        await discovering
                        raise
                    paths = await asyncio.to_thread(
                        self.dispatcher.directory.due,
                        self.scheduled.copy(),
                        self.queue.maxsize - self.queue.qsize(),
                    )
                    for path in paths:
                        if path not in self.scheduled:
                            try:
                                self.queue.put_nowait(path)
                            except asyncio.QueueFull:
                                break
                            self.scheduled.add(path)
                    await asyncio.to_thread(self.dispatcher.directory.reclaim)
                    self.dispatcher._last_poll = time.time()
                    self.dispatcher._last_error = None
                    self.dispatcher._recovery_attempts = 0
                    await asyncio.sleep(self.dispatcher.settings.poll_seconds)

    async def _worker(self) -> None:
        while True:
            path = await self.queue.get()
            try:
                if await self.dispatcher._deliver_one(
                    path,
                    self.violations,
                    self.notifications,
                ):
                    # Rejoin the tail so other cameras keep their turn.
                    try:
                        self.queue.put_nowait(path)
                    except asyncio.QueueFull:
                        self.scheduled.discard(path)
                else:
                    self.scheduled.discard(path)
            finally:
                self.queue.task_done()
