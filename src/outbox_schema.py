from __future__ import annotations


def install_outbox_statistics(db) -> None:
    """Maintain queue counts and payload bytes in the queue transaction."""
    db.execute(
        '''CREATE TABLE IF NOT EXISTS outbox_stats(
        id INTEGER PRIMARY KEY CHECK(id=1), pending INTEGER NOT NULL,
        archived INTEGER NOT NULL, pending_bytes INTEGER NOT NULL,
        archived_bytes INTEGER NOT NULL)''',
    )
    db.execute(
        '''INSERT OR IGNORE INTO outbox_stats SELECT 1,
        (SELECT count(*) FROM events),
        (SELECT count(*) FROM failed_events),
        (SELECT coalesce(sum(length(image)+length(cast(payload AS BLOB))),0)
         FROM events),
        (SELECT coalesce(sum(length(image)+length(cast(payload AS BLOB))),0)
         FROM failed_events)''',
    )
    for table, count, size in (
        ('events', 'pending', 'pending_bytes'),
        ('failed_events', 'archived', 'archived_bytes'),
    ):
        for operation, delta, byte_delta in (
            (
                'INSERT',
                '+1',
                '+length(NEW.image)+length(cast(NEW.payload AS BLOB))',
            ),
            (
                'DELETE',
                '-1',
                '-length(OLD.image)-length(cast(OLD.payload AS BLOB))',
            ),
            (
                'UPDATE',
                '+0',
                '+length(NEW.image)+length(cast(NEW.payload AS '
                'BLOB))-length(OLD.image)-length(cast(OLD.payload AS '
                'BLOB))',
            ),
        ):
            db.execute(
                f'''CREATE TRIGGER IF NOT EXISTS stats_{table}_{operation}
                AFTER {operation} ON {table} BEGIN UPDATE outbox_stats SET
                {count}={count}{delta}, {size}={size}{byte_delta}
                WHERE id=1; END''',
            )
    db.execute(
        '''CREATE INDEX IF NOT EXISTS events_due
        ON events(max(next_attempt,lease_until))''',
    )

    if db.execute('PRAGMA user_version').fetchone()[0] < 1:
        db.execute(
            '''UPDATE outbox_stats SET
            pending=(SELECT count(*) FROM events),
            archived=(SELECT count(*) FROM failed_events),
            pending_bytes=(SELECT
                coalesce(sum(length(image)+
                    length(cast(payload AS BLOB))),0) FROM events),
            archived_bytes=(SELECT
                coalesce(sum(length(image)+
                    length(cast(payload AS BLOB))),0) FROM failed_events)
            WHERE id=1''',
        )
        db.execute('PRAGMA user_version=1')
    db.execute(
        'CREATE INDEX IF NOT EXISTS failed_events_age ON '
        'failed_events(failed_at,id)',
    )
