from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from datetime import timezone
from typing import Literal

from fastapi import HTTPException
from sqlalchemy import and_
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.jwt_config import JwtAuthorizationCredentials
from examples.violation_records.schemas import ViolationAnalyticsResponse
from examples.violation_records.schemas import ViolationAnalyticsSummary
from examples.violation_records.violation_analytics import analytics_response
from examples.violation_records.violation_analytics import (
    build_analytics_query,
)
from examples.violation_records.violation_queries import (
    build_violation_conditions,
)
from examples.violation_records.violation_queries import load_violation_scope
from examples.violation_records.violation_queries import ViolationFilters
from examples.violation_records.violation_types import normalise_violation_type


async def get_violation_analytics(
    filters: ViolationFilters,
    db: AsyncSession,
    credentials: JwtAuthorizationCredentials,
    *,
    bucket: Literal['day', 'hour', 'week'] = 'day',
) -> ViolationAnalyticsResponse:
    """Return aggregated violation counts for charts and KPI widgets.

    The response intentionally excludes image paths, warning payloads, and
    individual violation records.
    """
    username: str | None = credentials.subject.get('username')
    if not username:
        raise HTTPException(status_code=401, detail='Invalid token')

    try:
        scope = await load_violation_scope(username, filters, db)
    except HTTPException as error:
        if error.status_code == 404 and error.detail == 'User not found':
            raise HTTPException(401, 'Invalid user') from error
        raise
    user = scope.user
    if user.role not in {'admin', 'super_admin'}:
        raise HTTPException(
            status_code=403,
            detail='violation_analytics_forbidden',
        )
    site_names = scope.site_names
    start, end = filters.start_time, filters.end_time
    if start is None or end is None:
        raise HTTPException(422, 'Analytics requires start and end times')
    start_utc = (
        start.replace(tzinfo=timezone.utc)
        if start.tzinfo is None
        else start.astimezone(timezone.utc)
    )
    end_utc = (
        end.replace(tzinfo=timezone.utc)
        if end.tzinfo is None
        else end.astimezone(timezone.utc)
    )
    if start_utc >= end_utc:
        raise HTTPException(
            status_code=422,
            detail='start must be before end',
        )
    try:
        latest_end = start_utc.replace(year=start_utc.year + 5)
    except ValueError:
        latest_end = start_utc.replace(
            year=start_utc.year + 5,
            day=28,
        )
    if end_utc > latest_end:
        raise HTTPException(
            status_code=422,
            detail='Query range must not exceed 5 years',
        )
    if not site_names:
        return ViolationAnalyticsResponse(
            summary=ViolationAnalyticsSummary(total=0, today=0),
        )

    filters = replace(filters, start_time=start_utc, end_time=end_utc)
    conditions = await build_violation_conditions(scope, filters, db)
    canonical_type = (
        normalise_violation_type(filters.violation_type)
        if filters.violation_type
        else None
    )

    where_clause = and_(*conditions)
    aggregate_rows = await db.execute(
        build_analytics_query(
            where_clause,
            canonical_type,
            bucket,
            datetime.now(timezone.utc),
        ),
    )
    return analytics_response(aggregate_rows.all())
