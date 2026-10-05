"""Build the single-trip analytics query and assemble its public response."""
from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from datetime import timedelta
from typing import cast as type_cast
from typing import Literal

from sqlalchemy import and_
from sqlalchemy import case
from sqlalchemy import cast
from sqlalchemy import func
from sqlalchemy import Integer
from sqlalchemy import literal
from sqlalchemy import select
from sqlalchemy import String
from sqlalchemy import union_all
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.sql.selectable import CompoundSelect

from examples.auth.models import Site
from examples.auth.models import Violation
from examples.violation_records.schemas import ViolationAnalyticsHourItem
from examples.violation_records.schemas import ViolationAnalyticsResponse
from examples.violation_records.schemas import ViolationAnalyticsSiteItem
from examples.violation_records.schemas import ViolationAnalyticsSummary
from examples.violation_records.schemas import ViolationAnalyticsTopSite
from examples.violation_records.schemas import ViolationAnalyticsTopType
from examples.violation_records.schemas import ViolationAnalyticsTrendItem
from examples.violation_records.schemas import ViolationAnalyticsTypeItem
from examples.violation_records.violation_types import VIOLATION_TYPE_BY_CODE
from examples.violation_records.violation_types import (
    VIOLATION_TYPE_DEFINITIONS,
)


def build_analytics_query(
    where_clause: ColumnElement[bool],
    canonical_type: str | None,
    bucket: Literal['day', 'hour', 'week'],
    now_utc: datetime,
) -> CompoundSelect:
    """Aggregate the same materialised, authorised record set in one query."""
    today_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    today_end = today_start + timedelta(days=1)
    type_names = (
        [canonical_type]
        if canonical_type is not None
        else [definition.code for definition in VIOLATION_TYPE_DEFINITIONS]
    )

    # One materialised CTE keeps dashboard aggregation to one database trip.
    filtered = (
        select(
            Violation.site.label('site'),
            Violation.detection_time.label('detection_time'),
            Violation.violation_type_codes.label('violation_type_codes'),
        )
        .where(where_clause)
        .cte('filtered_violations')
        .prefix_with('MATERIALIZED')
    )

    empty_text = cast(literal(None), String)
    zero = literal(0)
    bucket_format = {
        'hour': 'YYYY-MM-DD"T"HH24:00:00"Z"',
        'day': 'YYYY-MM-DD',
        'week': 'IYYY-"W"IW',
    }[bucket]
    bucket_expr = func.to_char(filtered.c.detection_time, bucket_format)
    hour_expr = cast(func.extract('hour', filtered.c.detection_time), Integer)
    aggregate_queries = [
        select(
            literal('summary').label('kind'),
            empty_text.label('value'),
            empty_text.label('label'),
            func.count().label('count'),
            cast(
                func.coalesce(
                    func.sum(
                        case(
                            (
                                and_(
                                    filtered.c.detection_time >= today_start,
                                    filtered.c.detection_time < today_end,
                                ),
                                1,
                            ),
                            else_=0,
                        ),
                    ),
                    0,
                ),
                Integer,
            ).label('today'),
        ).select_from(filtered),
        select(
            literal('trend').label('kind'),
            cast(bucket_expr, String).label('value'),
            empty_text.label('label'),
            func.count().label('count'),
            zero.label('today'),
        )
        .select_from(filtered)
        .group_by(bucket_expr),
        select(
            literal('site').label('kind'),
            cast(Site.id, String).label('value'),
            Site.name.label('label'),
            func.count().label('count'),
            zero.label('today'),
        )
        .select_from(filtered.join(Site, filtered.c.site == Site.name))
        .group_by(Site.id, Site.name),
        select(
            literal('hour').label('kind'),
            cast(hour_expr, String).label('value'),
            empty_text.label('label'),
            func.count().label('count'),
            zero.label('today'),
        )
        .select_from(filtered)
        .group_by(hour_expr),
    ]
    aggregate_queries.extend(
        select(
            literal('type').label('kind'),
            literal(type_name).label('value'),
            literal(VIOLATION_TYPE_BY_CODE[type_name].label).label('label'),
            cast(
                func.coalesce(
                    func.sum(
                        case(
                            (
                                cast(
                                    filtered.c.violation_type_codes,
                                    JSONB,
                                ).contains(
                                    [type_name],
                                ),
                                1,
                            ),
                            else_=0,
                        ),
                    ),
                    0,
                ),
                Integer,
            ).label('count'),
            zero.label('today'),
        ).select_from(filtered)
        for type_name in type_names
    )

    return union_all(*aggregate_queries)


def analytics_response(
    aggregate_rows: Iterable[
        tuple[str, str | None, str | None, int | None, int | None]
    ],
) -> ViolationAnalyticsResponse:
    """Assemble deterministic charts from database aggregate rows."""
    total = 0
    today = 0
    trend: list[ViolationAnalyticsTrendItem] = []
    by_site: list[ViolationAnalyticsSiteItem] = []
    by_hour: list[ViolationAnalyticsHourItem] = []
    by_type: list[ViolationAnalyticsTypeItem] = []
    for kind, value, label, count, row_today in aggregate_rows:
        count_value = int(count or 0)
        if kind == 'summary':
            total = count_value
            today = int(row_today or 0)
        elif kind == 'trend':
            trend.append(
                ViolationAnalyticsTrendItem(
                    bucket=str(value),
                    count=count_value,
                ),
            )
        elif kind == 'site':
            by_site.append(
                ViolationAnalyticsSiteItem(
                    site_id=int(type_cast(str, value)),
                    site_name=str(label),
                    count=count_value,
                ),
            )
        elif kind == 'hour':
            by_hour.append(
                ViolationAnalyticsHourItem(
                    hour=int(type_cast(str, value)),
                    count=count_value,
                ),
            )
        elif kind == 'type' and count_value:
            by_type.append(
                ViolationAnalyticsTypeItem(
                    type=str(value),
                    label=str(label),
                    count=count_value,
                ),
            )

    if total == 0:
        return ViolationAnalyticsResponse(
            summary=ViolationAnalyticsSummary(total=0, today=0),
        )

    trend.sort(key=lambda item: item.bucket)
    by_site.sort(key=lambda item: (-item.count, item.site_id))
    by_hour.sort(key=lambda item: item.hour)
    by_type.sort(key=lambda item: (-item.count, item.type))
    top_site = (
        ViolationAnalyticsTopSite(**by_site[0].model_dump())
        if by_site
        else None
    )
    top_type = (
        ViolationAnalyticsTopType(**by_type[0].model_dump())
        if by_type
        else None
    )

    return ViolationAnalyticsResponse(
        summary=ViolationAnalyticsSummary(
            total=total,
            today=today,
            top_site=top_site,
            top_type=top_type,
        ),
        trend=trend,
        by_type=by_type,
        by_site=by_site,
        by_hour=by_hour,
    )
