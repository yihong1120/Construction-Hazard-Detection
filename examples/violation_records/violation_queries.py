from __future__ import annotations

import base64
import binascii
import json
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Protocol

from fastapi import HTTPException
from sqlalchemy import cast
from sqlalchemy import or_
from sqlalchemy import select
from sqlalchemy import tuple_
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from examples.auth import user_service as _user_service
from examples.auth.models import Site
from examples.auth.models import StreamConfig
from examples.auth.models import User
from examples.auth.models import Violation
from examples.auth.models import ViolationFeedback
from examples.auth.models import ViolationReviewAuditLog
from examples.violation_records.schemas import ViolationFeedbackItem
from examples.violation_records.schemas import ViolationListItem
from examples.violation_records.schemas import ViolationReviewAuditItem
from examples.violation_records.schemas import ViolationReviewStatus
from examples.violation_records.search_utils import SearchUtils
from examples.violation_records.site_scope import violation_site_scope
from examples.violation_records.violation_responses import feedback_to_item
from examples.violation_records.violation_responses import review_audit_to_item
from examples.violation_records.violation_types import normalise_violation_type
from examples.violation_records.violation_types import (
    VIOLATION_TYPE_DEFINITIONS,
)


@dataclass(frozen=True, kw_only=True)
class ViolationFilters:
    site_id: int | None = None
    stream_id: str | None = None
    violation_type: str | None = None
    keyword: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    flagged: bool | None = None
    review_status: ViolationReviewStatus | None = None
    cursor: str | None = None


class StreamScopeUser(Protocol):
    """Define identity fields used to authorise a selected camera stream.

    Attributes:
        role: Role determining cross-group stream access.
        group_id: Optional group scope for an administrator.
    """

    role: str
    group_id: int | None


@lru_cache(maxsize=1)
def search_util() -> SearchUtils:
    """Create the lightweight multilingual keyword expander.

    Returns:
        Cached dictionary-backed multilingual search utility.
    """
    return SearchUtils()


async def load_review_audit_logs(
    db: AsyncSession,
    violation_id: int,
) -> list[ViolationReviewAuditItem]:
    """Return review audit rows in newest-first timeline order.

    Args:
        db: Database session used to load audit records.
        violation_id: Identifier of the violation.

    Returns:
        Public review audit items ordered newest first.
    """
    result = await db.execute(
        select(ViolationReviewAuditLog)
        .where(ViolationReviewAuditLog.violation_id == violation_id)
        .order_by(
            ViolationReviewAuditLog.reviewed_at.desc(),
            ViolationReviewAuditLog.id.desc(),
        ),
    )
    return [
        review_audit_to_item(audit_log) for audit_log in result.scalars().all()
    ]


async def load_violation_feedbacks(
    db: AsyncSession,
    violation_id: int,
) -> list[ViolationFeedbackItem]:
    """Return feedback rows for a violation in newest-first order.

    Args:
        db: Database session used to load feedback.
        violation_id: Identifier of the violation.

    Returns:
        Public feedback items ordered newest first.
    """
    result = await db.execute(
        select(ViolationFeedback)
        .where(ViolationFeedback.violation_id == violation_id)
        .order_by(
            ViolationFeedback.created_at.desc(),
            ViolationFeedback.id.desc(),
        ),
    )
    return [feedback_to_item(feedback) for feedback in result.scalars().all()]


def encode_violation_cursor(item: ViolationListItem) -> str:
    """Encode a response item's order key for cursor pagination.

    Args:
        item: Last violation item returned in a page.

    Returns:
        URL-safe opaque keyset-pagination cursor.
    """
    payload = json.dumps(
        {
            'detection_time': item.detection_time.isoformat(),
            'id': item.id,
        },
        separators=(',', ':'),
    ).encode('utf-8')
    return base64.urlsafe_b64encode(payload).decode('ascii').rstrip('=')


def decode_violation_cursor(cursor: str) -> tuple[datetime, int]:
    """Decode a cursor into the list ordering key.

    Args:
        cursor: URL-safe opaque keyset-pagination cursor.

    Returns:
        Detection time and violation identifier ordering key.

    Raises:
        HTTPException: If the cursor cannot be decoded or validated.
    """
    try:
        padded = cursor + '=' * (-len(cursor) % 4)
        raw_payload = base64.urlsafe_b64decode(padded.encode('ascii'))
        payload = json.loads(raw_payload.decode('utf-8'))
        detection_time = datetime.fromisoformat(payload['detection_time'])
        violation_id = int(payload['id'])
    except (
        binascii.Error,
        UnicodeDecodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise HTTPException(status_code=422, detail='Invalid cursor') from exc
    return detection_time, violation_id


async def resolve_stream_filter(
    stream_id: str,
    site_name: str | None,
    site_names: list[str],
    user: StreamScopeUser | User,
    db: AsyncSession,
) -> tuple[int, str]:
    """Resolve and authorise a stable camera before querying violations.

    Args:
        stream_id: Client-supplied stable stream identifier.
        site_name: Optional selected site name.
        site_names: Sites visible to the requesting user.
        user: User scope used for group access checks.
        db: Database session used to resolve the stream.

    Returns:
        Authorised stream configuration identifier and its site name.

    Raises:
        HTTPException: If the stream identifier or access scope is invalid.
    """
    stream_id_text = stream_id.strip()
    if not stream_id_text.isdigit() or int(stream_id_text) <= 0:
        raise HTTPException(
            status_code=422,
            detail='stream_id must be a positive stream configuration ID',
        )

    stream_result = await db.execute(
        select(
            StreamConfig.id,
            StreamConfig.group_id,
            Site.name,
        )
        .join(Site, StreamConfig.site_id == Site.id)
        .where(StreamConfig.id == int(stream_id_text)),
    )
    stream_row = stream_result.first()
    if (
        not stream_row
        or stream_row[2] not in site_names
        or (site_name is not None and stream_row[2] != site_name)
        or (
            user.role != 'super_admin'
            and (user.group_id is None or stream_row[1] != user.group_id)
        )
    ):
        raise HTTPException(status_code=403, detail='No access to stream_id')

    return int(stream_row[0]), str(stream_row[2])


async def load_review_scope(
    username: str,
    db: AsyncSession,
) -> tuple[User, list[str]]:
    """Load a reviewer and the sites they may review.

    Args:
        username: Authenticated username.
        db: Database session used to load effective site access.

    Returns:
        Reviewer user and accessible site names.

    Raises:
        HTTPException: If the user is not an administrator.
    """
    user, sites = await _user_service.load_user_with_effective_sites(
        username,
        db,
    )
    if user.role not in {'admin', 'super_admin'}:
        raise HTTPException(
            status_code=403,
            detail='Only admin or super_admin can review violations',
        )
    return user, [site.name for site in sites]


@dataclass(frozen=True)
class ViolationScope:
    user: User
    site_names: list[str]


async def load_violation_scope(
    username: str,
    filters: ViolationFilters,
    db: AsyncSession,
) -> ViolationScope:
    if filters.flagged is True or filters.review_status is not None:
        user, names = await load_review_scope(username, db)
        return ViolationScope(user, names)
    user, sites = await _user_service.load_user_with_effective_sites(
        username,
        db,
    )
    return ViolationScope(user, [site.name for site in sites])


async def build_violation_conditions(
    scope: ViolationScope,
    filters: ViolationFilters,
    db: AsyncSession,
) -> list:
    site_names = scope.site_names
    conditions: list = [violation_site_scope(site_names)]
    if filters.flagged is True or filters.review_status is not None:
        conditions.append(Violation.is_flagged.is_(True))
    if filters.review_status is not None:
        conditions.append(Violation.review_status == filters.review_status)
    site_name = await filtered_violation_site_name(
        filters.site_id,
        site_names,
        db,
    )
    if site_name is not None:
        conditions.append(violation_site_scope([site_name]))
    conditions.extend(
        await stream_violation_conditions(
            scope,
            filters.stream_id,
            site_name,
            db,
        ),
    )
    conditions.extend(optional_violation_conditions(filters))
    return conditions


async def filtered_violation_site_name(
    site_id: int | None,
    site_names: list[str],
    db: AsyncSession,
) -> str | None:
    """Resolve a site identifier and enforce caller scope.

    Args:
        site_id: Optional selected site identifier.
        site_names: Site names visible to the caller.
        db: Database session used to resolve the site.

    Returns:
        Authorised site name, or ``None`` when no site was selected.

    Raises:
        HTTPException: If the selected site is not accessible.
    """
    if site_id is None:
        return None
    site_stmt = select(Site.name).where(Site.id == site_id)
    site_name = (await db.execute(site_stmt)).scalar_one_or_none()
    if not site_name or site_name not in set(site_names):
        raise HTTPException(status_code=403, detail='No access to site_id')
    return site_name


async def stream_violation_conditions(
    scope: ViolationScope,
    stream_id: str | None,
    site_name: str | None,
    db: AsyncSession,
) -> list:
    if not stream_id:
        return []
    stream_config_id, stream_site_name = await resolve_stream_filter(
        stream_id,
        site_name,
        scope.site_names,
        scope.user,
        db,
    )
    return [
        Violation.stream_config_id == stream_config_id,
        violation_site_scope([stream_site_name]),
    ]


def optional_violation_conditions(
    filters: ViolationFilters,
) -> list[ColumnElement[bool]]:
    conditions: list[ColumnElement[bool]] = []
    if filters.violation_type:
        canonical_type = normalise_violation_type(filters.violation_type)
        if canonical_type is None:
            valid = ', '.join(
                definition.code for definition in VIOLATION_TYPE_DEFINITIONS
            )
            raise HTTPException(
                status_code=422,
                detail=(
                    f'Unsupported violation_type. Expected one of: {valid}'
                ),
            )
        conditions.append(
            cast(Violation.violation_type_codes, JSONB).contains(
                [canonical_type],
            ),
        )
    keyword_condition = violation_keyword_condition(filters.keyword)
    if keyword_condition is not None:
        conditions.append(keyword_condition)
    if filters.start_time:
        conditions.append(Violation.detection_time >= filters.start_time)
    if filters.end_time:
        conditions.append(Violation.detection_time <= filters.end_time)
    if filters.cursor:
        conditions.append(violation_cursor_condition(filters.cursor))
    return conditions


def violation_keyword_condition(
    keyword: str | None,
) -> ColumnElement[bool] | None:
    """Expand keyword synonyms into an OR search predicate.

    Args:
        keyword: Optional raw search keyword.

    Returns:
        SQL text-match predicate, or ``None`` for blank input.
    """
    keyword_text = keyword.strip() if keyword else ''
    if not keyword_text:
        return None
    synonyms: list[str] = search_util().expand_synonyms(keyword_text)
    matches = [
        *(
            Violation.stream_name.ilike(f'%{synonym}%'),
            Violation.warnings_json.ilike(f'%{synonym}%'),
        )
        for synonym in dict.fromkeys(synonyms)
    ]
    return or_(*matches) if matches else None


def violation_cursor_condition(cursor: str) -> ColumnElement[bool]:
    """Build a keyset-pagination condition for a decoded cursor.

    Args:
        cursor: Opaque cursor from the previous response page.

    Returns:
        SQL predicate for records after the cursor's ordering key.
    """
    cursor_time, cursor_id = decode_violation_cursor(cursor)
    # Both keys are NOT NULL. A row comparison preserves timestamp ties and
    # lets PostgreSQL seek directly into the composite pagination index.
    return tuple_(Violation.detection_time, Violation.id) < tuple_(
        cursor_time,
        cursor_id,
    )
