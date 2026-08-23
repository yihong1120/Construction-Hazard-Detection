from __future__ import annotations

import json
from datetime import datetime
from datetime import timezone
from typing import Any
from typing import Literal

from fastapi import HTTPException
from sqlalchemy import cast
from sqlalchemy import func
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from examples.auth.models import Violation
from examples.violation_records.schemas import ViolationAnalyticsResponse
from examples.violation_records.schemas import ViolationAnalyticsSummary
from examples.violation_records.violation_types import normalise_violation_type
from examples.violation_records.violation_types import (
    VIOLATION_TYPE_DEFINITIONS,
)


AnalyticsBucket = Literal['day', 'hour', 'week']
MAX_ANALYTICS_RANGE_YEARS = 5


def _empty_analytics_response() -> ViolationAnalyticsResponse:
    """Return the canonical empty analytics payload."""
    return ViolationAnalyticsResponse(
        summary=ViolationAnalyticsSummary(total=0, today=0),
        trend=[],
        by_type=[],
        by_site=[],
        by_hour=[],
    )


def _normalise_utc(value: datetime) -> datetime:
    """Treat naive values as UTC and convert aware values to UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _validate_analytics_range(start: datetime, end: datetime) -> tuple[
    datetime,
    datetime,
]:
    """Validate and normalise an analytics query range to UTC."""
    start_utc = _normalise_utc(start)
    end_utc = _normalise_utc(end)
    if start_utc >= end_utc:
        raise HTTPException(
            status_code=422,
            detail='start must be before end',
        )
    try:
        latest_end = start_utc.replace(
            year=start_utc.year + MAX_ANALYTICS_RANGE_YEARS,
        )
    except ValueError:
        latest_end = start_utc.replace(
            year=start_utc.year + MAX_ANALYTICS_RANGE_YEARS,
            day=28,
        )
    if end_utc > latest_end:
        raise HTTPException(
            status_code=422,
            detail='Query range must not exceed 5 years',
        )
    return start_utc, end_utc


def _analytics_dialect_name(db: AsyncSession) -> str:
    """Return the lower-case SQL dialect name for an analytics session."""
    bind = getattr(db, 'bind', None)
    dialect = getattr(bind, 'dialect', None)
    return str(getattr(dialect, 'name', '') or '')


def _analytics_bucket_expr(
    bucket: AnalyticsBucket,
    db: AsyncSession,
    detection_time: ColumnElement[Any] = Violation.detection_time,
) -> ColumnElement[Any]:
    """Build a dialect-aware UTC bucket expression for detection time."""
    dialect_name = _analytics_dialect_name(db)
    if dialect_name == 'postgresql':
        formats = {
            'hour': 'YYYY-MM-DD"T"HH24:00:00"Z"',
            'day': 'YYYY-MM-DD',
            'week': 'IYYY-"W"IW',
        }
        return func.to_char(detection_time, formats[bucket])
    if dialect_name in {'mysql', 'mariadb'}:
        formats = {
            'hour': '%Y-%m-%dT%H:00:00Z',
            'day': '%Y-%m-%d',
            'week': '%x-W%v',
        }
        return func.date_format(detection_time, formats[bucket])
    if dialect_name == 'sqlite':
        formats = {
            'hour': '%Y-%m-%dT%H:00:00Z',
            'day': '%Y-%m-%d',
            'week': '%Y-W%W',
        }
        return func.strftime(formats[bucket], detection_time)

    formats = {
        'hour': 'YYYY-MM-DD"T"HH24:00:00"Z"',
        'day': 'YYYY-MM-DD',
        'week': 'IYYY-"W"IW',
    }
    return func.to_char(detection_time, formats[bucket])


def _analytics_hour_expr(
    db: AsyncSession,
    detection_time: ColumnElement[Any] = Violation.detection_time,
) -> ColumnElement[Any]:
    """Build a dialect-aware UTC-hour expression for detection time."""
    dialect_name = _analytics_dialect_name(db)
    if dialect_name == 'postgresql':
        return cast(func.extract('hour', detection_time), Integer)
    if dialect_name in {'mysql', 'mariadb'}:
        return func.hour(detection_time)
    if dialect_name == 'sqlite':
        return cast(func.strftime('%H', detection_time), Integer)
    return cast(func.extract('hour', detection_time), Integer)


def _canonical_violation_type(violation_type: str) -> str:
    """Validate and normalise a supported violation-type code."""
    canonical = normalise_violation_type(violation_type)
    if canonical is not None:
        return canonical
    valid = ', '.join(
        definition.code for definition in VIOLATION_TYPE_DEFINITIONS
    )
    raise HTTPException(
        status_code=422,
        detail=f"Unsupported violation_type. Expected one of: {valid}",
    )


def _type_condition(
    violation_type: str,
    db: AsyncSession,
    type_codes: ColumnElement[Any] = Violation.violation_type_codes,
) -> ColumnElement[bool]:
    """Build a dialect-aware filter for a canonical violation type."""
    canonical = _canonical_violation_type(violation_type)
    dialect_name = _analytics_dialect_name(db)
    if dialect_name == 'postgresql':
        return cast(type_codes, JSONB).contains(
            [canonical],
        )
    if dialect_name in {'mysql', 'mariadb'}:
        return (
            func.json_contains(
                type_codes,
                json.dumps([canonical]),
            )
            == 1
        )
    return cast(type_codes, String).like(
        f'%"{canonical}"%',
    )
