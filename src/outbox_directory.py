from __future__ import annotations

import fcntl
import json
import logging
import os
import shutil
import sqlite3
import time
from collections.abc import Iterator
from contextlib import closing
from contextlib import contextmanager
from contextlib import ExitStack
from pathlib import Path

from src.settings.delivery import DeliverySettings

logger = logging.getLogger(__name__)


class WriterLease:
    def __init__(self, path: Path):
        root = path.parent / '.writers'
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.file = (root / (path.name + '.lock')).open('a+b')
        fcntl.flock(self.file, fcntl.LOCK_SH)

    def close(self):
        stream = getattr(self, 'file', None)
        if stream is not None and not stream.closed:
            stream.close()

    def __del__(self):
        self.close()


class OutboxDirectory:
    """Own atomic directory accounting and bounded queue maintenance."""

    def __init__(self, root: Path, settings: DeliverySettings | None = None):
        self.root = root
        self.settings = settings or DeliverySettings(directory=root)
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.path = root / '.outbox-index.db'
        self._scanner: Iterator[os.DirEntry[str]] | None = None
        self._scan_resources = ExitStack()
        self._last_scan = 0.0

    def _initialize(self, db) -> None:
        if db.execute('PRAGMA user_version').fetchone()[0] >= 3:
            return
        db.execute(
            '''CREATE TABLE IF NOT EXISTS queues(name TEXT PRIMARY KEY,
            pending INTEGER NOT NULL, archived INTEGER NOT NULL,
            bytes INTEGER NOT NULL, due REAL, changed REAL NOT NULL)''',
        )
        db.execute(
            '''CREATE TABLE IF NOT EXISTS maintenance(
            name TEXT PRIMARY KEY, checked REAL NOT NULL)''',
        )
        db.execute(
            'CREATE TABLE IF NOT EXISTS quarantine(name TEXT PRIMARY KEY)',
        )
        db.execute(
            '''CREATE TABLE IF NOT EXISTS budget(
            id INTEGER PRIMARY KEY CHECK(id=1), bytes INTEGER NOT NULL)''',
        )
        columns = {r[1] for r in db.execute('PRAGMA table_info(queues)')}
        for name, definition in (
            ('pending_bytes', 'INTEGER NOT NULL DEFAULT 0'),
            ('archived_bytes', 'INTEGER NOT NULL DEFAULT 0'),
            ('oldest_failure', 'REAL'),
        ):
            if name not in columns:
                db.execute(
                    f'ALTER TABLE queues ADD COLUMN {name} {definition}',
                )
        if 'archive_bytes' not in {
            r[1] for r in db.execute('PRAGMA table_info(budget)')
        }:
            db.execute(
                'ALTER TABLE budget ADD COLUMN archive_bytes INTEGER NOT '
                'NULL DEFAULT 0',
            )
        db.execute(
            'INSERT OR IGNORE INTO budget(id,bytes,archive_bytes) '
            'VALUES(1,0,0)',
        )
        db.execute(
            'UPDATE budget SET bytes=(SELECT coalesce(sum(bytes),0) FROM '
            'queues), archive_bytes=(SELECT coalesce(sum(archived_bytes),'
            '0) FROM queues) WHERE id=1',
        )
        for operation, delta in (
            ('INSERT', '+NEW.bytes'),
            ('DELETE', '-OLD.bytes'),
            ('UPDATE', '+NEW.bytes-OLD.bytes'),
        ):
            db.execute(f'DROP TRIGGER IF EXISTS budget_{operation}')
            archive_delta = delta.replace('.bytes', '.archived_bytes')
            db.execute(
                f'''CREATE TRIGGER budget_{operation}
                AFTER {operation} ON queues BEGIN UPDATE budget SET
                bytes=bytes{delta},
                archive_bytes=archive_bytes{archive_delta} WHERE id=1; END''',
            )
        db.execute(
            '''CREATE INDEX IF NOT EXISTS queues_due
            ON queues(due,changed,name) WHERE pending>0''',
        )
        # Only bootstrap/legacy crash recovery requires a full reconciliation.
        self._reconcile_all(db)
        db.execute(
            'CREATE INDEX IF NOT EXISTS queue_archive_age ON '
            'queues(oldest_failure,name) WHERE archived>0',
        )
        db.execute('PRAGMA user_version=3')

    def _reconcile_all(self, db) -> None:
        for path in self.root.glob('*.sqlite3'):
            self._reconcile(db, path)
        for (name,) in db.execute(
            'SELECT name FROM queues UNION SELECT name FROM quarantine',
        ).fetchall():
            if not (self.root / name).exists():
                db.execute('DELETE FROM queues WHERE name=?', (name,))
                db.execute('DELETE FROM quarantine WHERE name=?', (name,))

    def _reconcile(self, index, path: Path) -> None:
        if not path.exists():
            index.execute('DELETE FROM queues WHERE name=?', (path.name,))
            index.execute('DELETE FROM quarantine WHERE name=?', (path.name,))
        elif not path.is_symlink():
            try:
                with closing(sqlite3.connect(path, timeout=0.1)) as box:
                    self.account(index, box, path.name)
                    index.execute(
                        'DELETE FROM quarantine WHERE name=?',
                        (path.name,),
                    )
            except sqlite3.DatabaseError:
                # Delivery of healthy queues continues, but new reservations
                # fail closed until all unknown payload sizes are reconciled.
                index.execute(
                    'INSERT OR IGNORE INTO quarantine VALUES(?)',
                    (path.name,),
                )
                logger.error(
                    'Evidence database needs accounting repair: %s',
                    path.name,
                )

    @contextmanager
    def locked(self):
        with (self.root / '.directory.lock').open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            dirty = self.root / '.accounting-dirty'
            with closing(sqlite3.connect(self.path, timeout=5)) as db, db:
                self._initialize(db)
                if dirty.exists():
                    names = self._dirty_names(dirty)
                    if names is None:
                        self._reconcile_all(db)
                    else:
                        for name in names:
                            self._reconcile(db, self.root / name)
                yield db
            dirty.unlink(missing_ok=True)

    @staticmethod
    def _dirty_names(marker):
        try:
            raw = marker.read_text()
            try:
                names = json.loads(raw)
            except ValueError:
                names = [raw]
            if (
                not isinstance(names, list)
                or not names
                or not all(
                    isinstance(name, str)
                    and Path(name).name == name
                    and name.endswith('.sqlite3')
                    for name in names
                )
            ):
                return None
            return set(names)
        except (OSError, UnicodeError):
            return None

    def mark_dirty(self, name: str = '') -> None:
        marker = self.root / '.accounting-dirty'
        names = {name} if name else None
        if marker.exists():
            previous = self._dirty_names(marker)
            if previous is None or (name and name in previous):
                return
            names = previous | names if names is not None else None
        descriptor = os.open(
            marker,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            0o600,
        )
        try:
            os.write(
                descriptor,
                (
                    json.dumps(sorted(names)).encode()
                    if names is not None
                    else b''
                ),
            )
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        directory = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    @staticmethod
    def account(index, box, name):
        oldest = None
        try:
            if box.execute('PRAGMA user_version').fetchone()[0] < 1:
                raise sqlite3.OperationalError(
                    'Legacy statistics need reconciliation',
                )
            pending, archived, pending_bytes, archived_bytes = box.execute(
                'SELECT pending,archived,pending_bytes,archived_bytes '
                'FROM outbox_stats WHERE id=1',
            ).fetchone()
            due = box.execute(
                'SELECT min(max(next_attempt,lease_until)) FROM events',
            ).fetchone()[0]
            size = pending_bytes + archived_bytes
        except sqlite3.OperationalError:
            tables = {
                row[0]
                for row in box.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'",
                )
            }
            if 'events' not in tables:
                return
            columns = {
                row[1] for row in box.execute('PRAGMA table_info(events)')
            }
            lease = 'lease_until' if 'lease_until' in columns else '0'
            pending, size, due = box.execute(
                'SELECT count(*),coalesce(sum(length(image)+'
                'length(cast(payload AS BLOB))),0),'
                f'min(max(next_attempt,{lease})) FROM events',
            ).fetchone()
            pending_bytes = size
            archived = 0
            archived_bytes = 0
            if 'failed_events' in tables:
                archived, archive_size = box.execute(
                    'SELECT count(*),'
                    'coalesce(sum(length(image)+length(cast(payload AS '
                    'BLOB))),0) FROM failed_events',
                ).fetchone()
                archived_bytes = archive_size
                size += archive_size
        if archived:
            oldest = box.execute(
                'SELECT min(failed_at) FROM failed_events',
            ).fetchone()[0]
        old = index.execute(
            'SELECT pending,archived,bytes,due,pending_bytes,'
            'archived_bytes,oldest_failure FROM queues WHERE name=?',
            (name,),
        ).fetchone()
        values = (
            pending,
            archived,
            size,
            due,
            pending_bytes,
            archived_bytes,
            oldest,
        )
        if old != values:
            index.execute(
                '''INSERT INTO queues(name,pending,archived,bytes,due,
                    pending_bytes,archived_bytes,oldest_failure,changed)
                VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(name) DO UPDATE SET
                pending=excluded.pending,archived=excluded.archived,
                bytes=excluded.bytes,
                due=excluded.due,pending_bytes=excluded.pending_bytes,
                archived_bytes=excluded.archived_bytes,
                oldest_failure=excluded.oldest_failure,
                changed=excluded.changed''',
                (name, *values, time.time()),
            )

    def trim_archives(self, index, incoming=0):
        deadline = time.monotonic() + self.settings.maintenance_budget_seconds
        for _ in range(self.settings.archive_cleanup_batch_size):
            total, archives = index.execute(
                'SELECT bytes,archive_bytes FROM budget WHERE id=1',
            ).fetchone()
            if (
                archives <= self.settings.archive_capacity
                and total + incoming <= self.settings.directory_max_bytes
            ):
                return
            row = index.execute(
                'SELECT name FROM queues WHERE archived>0 ORDER BY '
                'oldest_failure,name LIMIT 1',
            ).fetchone()
            if row is None:
                return
            name = row[0]
            path = self.root / name
            if not path.exists():
                index.execute('DELETE FROM queues WHERE name=?', (name,))
                continue
            self.mark_dirty(name)
            with closing(sqlite3.connect(path)) as box, box:
                box.execute(
                    'DELETE FROM failed_events WHERE id=(SELECT id FROM '
                    'failed_events ORDER BY failed_at,id LIMIT 1)',
                )
                self.account(index, box, name)
            if time.monotonic() >= deadline:
                return

    def _validate_admission(self, index, incoming: int) -> None:
        if index.execute('SELECT 1 FROM quarantine LIMIT 1').fetchone():
            raise RuntimeError(
                'Outbox capacity unknown; accounting repair required',
            )
        total, archives = index.execute(
            'SELECT bytes,archive_bytes FROM budget WHERE id=1',
        ).fetchone()
        if total - archives + incoming > self.settings.directory_max_bytes:
            raise RuntimeError('Outbox directory capacity exhausted')
        if (
            shutil.disk_usage(self.root).free
            < self.settings.min_free_bytes + incoming
        ):
            raise RuntimeError('Outbox disk reserve exhausted')

    def reserve(self, index, incoming: int) -> None:
        self._validate_admission(index, incoming)
        self.trim_archives(index, incoming)
        total = index.execute(
            'SELECT bytes FROM budget WHERE id=1',
        ).fetchone()[0]
        if total + incoming > self.settings.directory_max_bytes:
            raise RuntimeError('Outbox directory capacity exhausted')

    def discover(self) -> None:
        with self.locked() as db:
            for (name,) in db.execute(
                'SELECT name FROM quarantine LIMIT ?',
                (self.settings.maintenance_batch_size,),
            ).fetchall():
                if not (self.root / name).exists():
                    db.execute('DELETE FROM quarantine WHERE name=?', (name,))
        if self._scanner is None:
            if (
                time.monotonic() - self._last_scan
                < self.settings.discovery_seconds
            ):
                return
            self._scanner = self._scan_resources.enter_context(
                os.scandir(self.root),
            )
        deadline = time.monotonic() + self.settings.maintenance_budget_seconds
        try:
            for _ in range(self.settings.maintenance_batch_size):
                item = next(self._scanner, None)
                if item is None:
                    self._scan_resources.close()
                    self._scanner = None
                    self._last_scan = time.monotonic()
                    return
                if item.name.endswith('.sqlite3') and item.is_file(
                    follow_symlinks=False,
                ):
                    with self.locked() as db:
                        self._reconcile(db, Path(item.path))
                if time.monotonic() >= deadline:
                    return
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self._scanner is not None:
            self._scan_resources.close()
            self._scanner = None

    def due(self, excluded: set[Path], limit: int) -> list[Path]:
        if limit <= 0:
            return []
        with self.locked() as db:
            rows = db.execute(
                'SELECT name FROM queues WHERE pending>0 AND due<=? '
                'ORDER BY due,changed,name LIMIT ?',
                (time.time(), limit + len(excluded)),
            ).fetchall()
        return [
            self.root / name
            for (name,) in rows
            if self.root / name not in excluded
        ][:limit]

    def reclaim(self) -> int:
        with self.locked() as db:
            rows = db.execute(
                '''SELECT q.name FROM queues q LEFT JOIN
                maintenance m ON m.name=q.name WHERE q.pending=0
                AND q.changed<?
                ORDER BY coalesce(m.checked,0),q.changed LIMIT ?''',
                (
                    time.time() - self.settings.idle_seconds,
                    self.settings.maintenance_batch_size,
                ),
            ).fetchall()
        removed = 0
        deadline = time.monotonic() + self.settings.maintenance_budget_seconds
        for (name,) in rows:
            # Release the global lock between queues, so writers can proceed.
            with self.locked() as db:
                removed += self._reclaim_one(db, name)
            if time.monotonic() >= deadline:
                break
        return removed

    def _reclaim_one(self, index, name) -> int:
        path = self.root / name
        state = index.execute(
            'SELECT pending,changed FROM queues WHERE name=?',
            (name,),
        ).fetchone()
        if (
            path.name != name
            or state is None
            or state[0]
            or state[1] >= time.time() - self.settings.idle_seconds
        ):
            return 0
        writer_root = self.root / '.writers'
        writer_root.mkdir(mode=0o700, exist_ok=True)
        with (writer_root / (name + '.lock')).open('a+b') as writer:
            try:
                fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
                exclusive = True
            except BlockingIOError:
                exclusive = False
            index.execute(
                'INSERT OR REPLACE INTO maintenance VALUES(?,?)',
                (name, time.time()),
            )
            if not path.exists():
                index.execute('DELETE FROM queues WHERE name=?', (name,))
                return 0
            self.mark_dirty(name)
            with closing(sqlite3.connect(path)) as box, box:
                if box.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND "
                    "name='failed_events'",
                ).fetchone():
                    box.execute(
                        '''DELETE FROM failed_events WHERE id IN (
                        SELECT id FROM failed_events WHERE failed_at<?
                        LIMIT ?)''',
                        (
                            time.time() - self.settings.archive.seconds,
                            self.settings.archive_cleanup_batch_size,
                        ),
                    )
                box.execute('PRAGMA incremental_vacuum(128)')
                self.account(index, box, name)
                pending, archived = index.execute(
                    'SELECT pending,archived FROM queues WHERE name=?',
                    (name,),
                ).fetchone()
                if pending or archived or not exclusive:
                    return 0
            for suffix in ('', '-wal', '-shm'):
                path.with_name(path.name + suffix).unlink(missing_ok=True)
            index.execute('DELETE FROM queues WHERE name=?', (name,))
            index.execute('DELETE FROM maintenance WHERE name=?', (name,))
            return 1
