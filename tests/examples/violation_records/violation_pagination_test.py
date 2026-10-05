from __future__ import annotations

import unittest
from datetime import datetime
from datetime import timedelta
from datetime import timezone

from sqlalchemy import and_
from sqlalchemy import Column
from sqlalchemy import create_engine
from sqlalchemy import DateTime
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy import or_
from sqlalchemy import select
from sqlalchemy import String
from sqlalchemy import Table

from examples.auth.models import Violation
from examples.violation_records.schemas import ViolationListItem
from examples.violation_records.violation_services import (
    _encode_violation_cursor,
)
from examples.violation_records.violation_services import (
    _violation_cursor_condition,
)


class TestViolationPagination(unittest.TestCase):
    """Execute cursor predicates with timestamp ties and site scopes."""

    def test_pages_preserve_ties_and_site_scope(self) -> None:
        """Multiple tied pages neither skip nor repeat authorised records."""
        engine = create_engine('sqlite://')
        metadata = MetaData()
        table = Table(
            'violations', metadata,
            Column('id', Integer, primary_key=True),
            Column('detection_time', DateTime, nullable=False),
            Column('site', String, nullable=False),
        )
        metadata.create_all(engine)
        timestamp = datetime(2026, 10, 3, tzinfo=timezone.utc)
        rows = [
            {
                'id': number,
                'detection_time': timestamp - timedelta(minutes=number // 4),
                'site': 'hidden' if number % 5 == 0 else 'visible',
            }
            for number in range(1, 25)
        ]
        try:
            with engine.begin() as connection:
                connection.execute(table.insert(), rows)
                base = select(Violation.id, Violation.detection_time).where(
                    Violation.site == 'visible',
                ).order_by(
                    Violation.detection_time.desc(), Violation.id.desc(),
                )
                expected = list(connection.execute(base).all())
                actual = []
                cursor = None
                while True:
                    statement = base.limit(3)
                    if cursor is not None:
                        statement = statement.where(
                            _violation_cursor_condition(cursor),
                        )
                    page = list(connection.execute(statement).all())
                    if not page:
                        break
                    actual.extend(page)
                    last = page[-1]
                    item = ViolationListItem(
                        id=last.id,
                        detection_time=last.detection_time,
                        site_name='visible',
                        stream_name='camera',
                        thumbnail_url='https://example.test/thumbnail',
                    )
                    cursor = _encode_violation_cursor(item)
                    # Verify equivalence to the previous predicate at every
                    # boundary, including ties and a deleted cursor row.
                    previous = or_(
                        Violation.detection_time < last.detection_time,
                        and_(
                            Violation.detection_time == last.detection_time,
                            Violation.id < last.id,
                        ),
                    )
                    self.assertEqual(
                        connection.execute(base.where(previous)).all(),
                        connection.execute(
                            base.where(
                                _violation_cursor_condition(cursor),
                            ),
                        ).all(),
                    )
                    connection.execute(
                        table.delete().where(table.c.id == last.id),
                    )
                self.assertEqual(actual, expected)
                self.assertEqual(len({row.id for row in actual}), len(actual))
        finally:
            engine.dispose()
