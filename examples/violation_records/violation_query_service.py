from __future__ import annotations

from datetime import datetime
from typing import cast as type_cast

from fastapi import HTTPException
from fastapi import Request
from sqlalchemy import and_
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth import user_service
from examples.auth.jwt_config import JwtAuthorizationCredentials
from examples.auth.models import Site
from examples.auth.models import StreamConfig
from examples.auth.models import Violation
from examples.violation_records.schemas import SiteOut
from examples.violation_records.schemas import ViolationFilterCamera
from examples.violation_records.schemas import ViolationFilterOptions
from examples.violation_records.schemas import ViolationList
from examples.violation_records.schemas import ViolationReviewStatus
from examples.violation_records.schemas import ViolationTypeOption
from examples.violation_records.violation_services import (
    _build_violation_conditions,
)
from examples.violation_records.violation_services import (
    _encode_violation_cursor,
)
from examples.violation_records.violation_services import (
    _violation_list_columns,
)
from examples.violation_records.violation_services import _violation_site_names
from examples.violation_records.violation_services import (
    _violation_to_list_item,
)
from examples.violation_records.violation_services import ViolationListRow
from examples.violation_records.violation_types import (
    VIOLATION_TYPE_DEFINITIONS,
)


async def get_my_sites(
    db: AsyncSession,
    credentials: JwtAuthorizationCredentials,
) -> list[SiteOut]:
    """Return the sites accessible to the authenticated user."""
    _, sites = await user_service.load_user_with_effective_sites(
        credentials.subject['username'],
        db,
    )
    return [
        SiteOut(
            id=site.id,
            name=site.name,
            created_at=site.created_at,
            updated_at=site.updated_at,
        )
        for site in sites
    ]


async def get_violation_filter_options(
    site_id: int,
    group_id: int | None,
    db: AsyncSession,
    credentials: JwtAuthorizationCredentials,
) -> ViolationFilterOptions:
    """Return cameras and violation types available for one authorised site."""
    user, site_names, _ = await user_service.load_user_access_context(
        db,
        credentials.subject['username'],
    )
    site = await db.scalar(
        select(Site).where(
            Site.id == site_id,
            Site.name.in_(site_names),
        ),
    )
    if site is None:
        raise HTTPException(status_code=403, detail='No access to site_id')
    if group_id is not None and user.role != 'super_admin':
        if group_id != user.group_id:
            raise HTTPException(
                status_code=403, detail='No access to group_id',
            )

    visible_group_id = (
        group_id if user.role == 'super_admin' else user.group_id
    )
    statement = (
        select(StreamConfig.id, StreamConfig.stream_name)
        .where(StreamConfig.site_id == site.id)
        .order_by(StreamConfig.stream_name, StreamConfig.id)
    )
    if visible_group_id is not None:
        statement = statement.where(StreamConfig.group_id == visible_group_id)
    rows = (await db.execute(statement)).all()
    return ViolationFilterOptions(
        cameras=[
            ViolationFilterCamera(stream_id=str(row[0]), name=str(row[1]))
            for row in rows
        ],
        violation_types=[
            ViolationTypeOption(code=item.code, label=item.label)
            for item in VIOLATION_TYPE_DEFINITIONS
        ],
    )


async def get_violations(
    request: Request,
    db: AsyncSession,
    credentials: JwtAuthorizationCredentials,
    site_id: int | None = None,
    stream_id: str | None = None,
    violation_type: str | None = None,
    keyword: str | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    limit: int = 20,
    flagged: bool | None = None,
    review_status: ViolationReviewStatus | None = None,
    cursor: str | None = None,
) -> ViolationList:
    """Return one compact, keyset-paginated violation list page."""
    username = credentials.subject['username']
    site_names = await _violation_site_names(
        username,
        flagged,
        review_status,
        db,
    )
    if not site_names:
        return ViolationList(items=[])
    conditions = await _build_violation_conditions(
        username,
        site_names,
        site_id,
        stream_id,
        violation_type,
        keyword,
        start_time,
        end_time,
        flagged,
        review_status,
        cursor,
        db,
    )
    statement = (
        select(*_violation_list_columns)
        .where(and_(*conditions))
        .order_by(Violation.detection_time.desc(), Violation.id.desc())
        .limit(limit + 1)
    )
    rows = [
        type_cast(ViolationListRow, tuple(row))
        for row in (await db.execute(statement)).all()
    ]
    items = [_violation_to_list_item(row, request) for row in rows[:limit]]
    has_more = len(rows) > limit
    return ViolationList(
        items=items,
        next_cursor=(
            _encode_violation_cursor(items[-1])
            if has_more and items
            else None
        ),
        has_more=has_more,
    )
