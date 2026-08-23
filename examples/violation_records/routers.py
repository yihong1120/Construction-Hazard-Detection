from __future__ import annotations

from datetime import datetime
from typing import Literal

from fastapi import APIRouter
from fastapi import Depends
from fastapi import File
from fastapi import Form
from fastapi import Query
from fastapi import Request
from fastapi import Security
from fastapi import UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.database import get_db
from examples.auth.jwt_config import jwt_access
from examples.auth.jwt_config import JwtAuthorizationCredentials
from examples.violation_records import violation_media_service
from examples.violation_records import violation_query_service
from examples.violation_records import violation_review_service
from examples.violation_records import violation_services
from examples.violation_records import violation_upload_service
from examples.violation_records.schemas import SiteOut
from examples.violation_records.schemas import UploadViolationResponse
from examples.violation_records.schemas import ViolationAnalyticsResponse
from examples.violation_records.schemas import ViolationFeedbackCreate
from examples.violation_records.schemas import ViolationFeedbackResponse
from examples.violation_records.schemas import ViolationFilterOptions
from examples.violation_records.schemas import ViolationItem
from examples.violation_records.schemas import ViolationList
from examples.violation_records.schemas import ViolationReviewAuditItem
from examples.violation_records.schemas import ViolationReviewStatus
from examples.violation_records.schemas import ViolationReviewUpdate

router = APIRouter()


@router.get(
    '/my_sites',
    response_model=list[SiteOut],
    summary='Get all accessible sites',
    description='Return a list of sites the user has access to.',
)
async def get_my_sites(
    db: AsyncSession = Depends(get_db),
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
) -> list[SiteOut]:
    """Return sites accessible to the authenticated user.

    Args:
        db: Database session used to load authorised sites.
        credentials: Validated JWT credentials for the requesting user.

    Returns:
        Accessible site records.
    """
    return await violation_query_service.get_my_sites(db, credentials)


@router.get(
    '/violations/filter-options',
    response_model=ViolationFilterOptions,
    summary='Get authorized camera and violation type filter options',
)
async def get_violation_filter_options(
    site_id: int = Query(..., gt=0),
    group_id: int | None = Query(None, gt=0),
    db: AsyncSession = Depends(get_db),
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
) -> ViolationFilterOptions:
    """Return violation filters available within an authorised site.

    Args:
        site_id: Selected site identifier.
        group_id: Optional group filter within the selected site.
        db: Database session used to load site cameras.
        credentials: Validated JWT credentials for the requesting user.

    Returns:
        Authorised camera and canonical type filter options.
    """
    return await violation_query_service.get_violation_filter_options(
        site_id,
        group_id,
        db,
        credentials,
    )


@router.get(
    '/violations',
    response_model=ViolationList,
    summary='Get paginated violation records',
    description='Supports filtering by site_id, keyword, and time range.',
)
async def get_violations(
    request: Request,
    site_id: int | None = None,
    stream_id: str | None = None,
    violation_type: str | None = None,
    keyword: str | None = None,
    start_time: datetime | None = Query(None),
    end_time: datetime | None = Query(None),
    limit: int = Query(
        20,
        gt=0,
        le=100,
        description='Records per page (1-100)',
    ),
    flagged: bool | None = Query(
        None,
        description='When true, return only flagged records for reviewers.',
    ),
    review_status: ViolationReviewStatus | None = Query(
        None,
        description='Filter flagged review records by review status.',
    ),
    cursor: str | None = Query(
        None,
        description='Cursor returned by the previous page.',
    ),
    db: AsyncSession = Depends(get_db),
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
) -> ViolationList:
    """Return an authorised filtered page of violation records.

    Args:
        request: HTTP request used to build protected media URLs.
        site_id: Optional site filter.
        stream_id: Optional stable stream identifier filter.
        violation_type: Optional canonical violation-type filter.
        keyword: Optional search keyword.
        start_time: Optional inclusive start time.
        end_time: Optional inclusive end time.
        limit: Maximum records returned in one page.
        flagged: Optional flag-state filter.
        review_status: Optional flagged-record review-status filter.
        cursor: Optional keyset-pagination cursor.
        db: Database session used to query records.
        credentials: Validated JWT credentials for the requesting user.

    Returns:
        Authorised page of violation records and an optional next cursor.
    """
    return await violation_query_service.get_violations(
        request,
        db,
        credentials,
        site_id=site_id,
        stream_id=stream_id,
        violation_type=violation_type,
        keyword=keyword,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
        flagged=flagged,
        review_status=review_status,
        cursor=cursor,
    )


@router.get(
    '/violations/analytics',
    response_model=ViolationAnalyticsResponse,
    summary='Get aggregated violation analytics',
    description=(
        'Return aggregates with the same authorized site, camera, violation '
        'type, time-range, and bucket filters applied to every result.'
    ),
)
async def get_violation_analytics(
    start: datetime = Query(..., description='Inclusive UTC start datetime'),
    end: datetime = Query(..., description='Inclusive UTC end datetime'),
    site_id: int | None = None,
    stream_id: str | None = None,
    violation_type: str | None = None,
    bucket: Literal['day', 'hour', 'week'] = Query('day'),
    db: AsyncSession = Depends(get_db),
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
) -> ViolationAnalyticsResponse:
    """Return authorised aggregate violation analytics.

    Args:
        start: Inclusive analytics range start.
        end: Inclusive analytics range end.
        site_id: Optional site filter.
        stream_id: Optional stable stream identifier filter.
        violation_type: Optional canonical type filter.
        bucket: Aggregate time bucket.
        db: Database session used to query analytics.
        credentials: Validated JWT credentials for the requesting user.

    Returns:
        Aggregate counts constrained by the user's accessible sites.
    """
    return await violation_services.get_violation_analytics(
        start,
        end,
        db,
        credentials,
        site_id=site_id,
        stream_id=stream_id,
        violation_type=violation_type,
        bucket=bucket,
    )


@router.get(
    '/violations/next',
    response_model=ViolationItem | None,
    summary='Get next violation review item',
    description='Return the next flagged record inside reviewer scope.',
)
async def get_next_review_violation(
    request: Request,
    review_status: ViolationReviewStatus = Query('pending'),
    site_id: int | None = None,
    current_id: int | None = Query(
        None,
        description='Optional record id to exclude from the next result.',
    ),
    db: AsyncSession = Depends(get_db),
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
) -> ViolationItem | None:
    """Return the next reviewable flagged violation in reviewer scope.

    Args:
        request: HTTP request used to build protected media URLs.
        review_status: Review state to select.
        site_id: Optional site restriction.
        current_id: Optional current record to exclude.
        db: Database session used to query violations.
        credentials: Validated JWT credentials for the reviewing user.

    Returns:
        Next review item, or ``None`` when none matches.
    """
    return await violation_review_service.get_next_review_violation(
        request,
        db,
        credentials,
        review_status=review_status,
        site_id=site_id,
        current_id=current_id,
    )


@router.get(
    '/violations/{violation_id}/audit-log',
    response_model=list[ViolationReviewAuditItem],
    summary='Get violation review audit log',
    description=(
        'Return review history for a flagged record in reviewer scope.'
    ),
)
async def get_violation_review_audit_log(
    violation_id: int,
    db: AsyncSession = Depends(get_db),
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
) -> list[ViolationReviewAuditItem]:
    """Return review history for an authorised flagged violation.

    Args:
        violation_id: Identifier of the reviewed violation.
        db: Database session used to load audit records.
        credentials: Validated JWT credentials for the reviewing user.

    Returns:
        Newest-first review audit entries.
    """
    return await violation_review_service.get_violation_review_audit_log(
        violation_id,
        db,
        credentials,
    )


@router.get(
    '/violations/{violation_id}',
    response_model=ViolationItem,
    summary='Get single violation details',
    description='Retrieve a single violation record by its ID.',
)
async def get_single_violation(
    violation_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
) -> ViolationItem:
    """Return one authorised violation record with protected media URLs.

    Args:
        violation_id: Identifier of the violation to retrieve.
        request: HTTP request used to build protected media URLs.
        db: Database session used to load the record.
        credentials: Validated JWT credentials for the requesting user.

    Returns:
        Authorised detailed violation record.
    """
    return await violation_review_service.get_single_violation(
        violation_id,
        request,
        db,
        credentials,
    )


@router.post(
    '/violations/{violation_id}/feedback',
    response_model=ViolationFeedbackResponse,
    status_code=201,
    summary='Submit feedback for a violation record',
    description='Store structured feedback in pending status for review.',
)
async def submit_violation_feedback(
    violation_id: int,
    payload: ViolationFeedbackCreate,
    db: AsyncSession = Depends(get_db),
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
) -> ViolationFeedbackResponse:
    """Store structured feedback against an authorised violation.

    Args:
        violation_id: Identifier of the violation receiving feedback.
        payload: Validated structured feedback payload.
        db: Database session used to persist feedback.
        credentials: Validated JWT credentials for the submitting user.

    Returns:
        Persisted feedback record in pending review status.
    """
    return await violation_review_service.submit_violation_feedback(
        violation_id,
        payload,
        db,
        credentials,
    )


@router.patch(
    '/violations/{violation_id}/review',
    response_model=ViolationItem,
    summary='Review a flagged violation record',
    description='Resolve or dismiss a flagged violation with audit logging.',
)
async def review_violation(
    violation_id: int,
    payload: ViolationReviewUpdate,
    request: Request,
    db: AsyncSession = Depends(get_db),
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
) -> ViolationItem:
    """Update review state for an authorised flagged violation.

    Args:
        violation_id: Identifier of the violation to review.
        payload: New review status and optional note.
        request: HTTP request used to build protected media URLs.
        db: Database session used to persist review state and audit data.
        credentials: Validated JWT credentials for the reviewing user.

    Returns:
        Updated detailed violation record.
    """
    return await violation_review_service.review_violation(
        violation_id,
        payload,
        request,
        db,
        credentials,
    )


@router.get(
    '/get_violation_image',
    summary='Get a violation image file',
    description="Retrieve an image file from the 'static' directory.",
)
async def get_violation_image(
    image_path: str,
    db: AsyncSession = Depends(get_db),
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
) -> FileResponse:
    """Return an authorised original violation image.

    Args:
        image_path: Stored relative image path.
        db: Database session used to verify media ownership.
        credentials: Validated JWT credentials for the requesting user.

    Returns:
        Protected original image response.
    """
    return await violation_media_service.get_violation_image(
        image_path,
        db,
        credentials,
    )


@router.get(
    '/get_violation_thumbnail',
    summary='Get a cached violation thumbnail',
    description='Return a small cached JPEG thumbnail for a violation image.',
)
async def get_violation_thumbnail(
    image_path: str,
    db: AsyncSession = Depends(get_db),
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
) -> FileResponse:
    """Return an authorised cached thumbnail for a violation image.

    Args:
        image_path: Stored relative image path.
        db: Database session used to verify media ownership.
        credentials: Validated JWT credentials for the requesting user.

    Returns:
        Protected generated thumbnail response.
    """
    return await violation_media_service.get_violation_thumbnail(
        image_path,
        db,
        credentials,
    )


@router.post(
    '/upload',
    response_model=UploadViolationResponse,
    summary='Upload a new violation record',
    description='Upload a violation image and associated metadata.',
)
async def upload_violation(
    site: str = Form(...),
    stream_name: str = Form(...),
    detection_time: datetime | None = Form(None),
    warnings_json: str | None = Form(None),
    detections_json: str | None = Form(None),
    cone_polygon_json: str | None = Form(None),
    pole_polygon_json: str | None = Form(None),
    image: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
) -> UploadViolationResponse:
    """Upload a violation image and its validated detector metadata.

    Args:
        site: Site name associated with the violation.
        stream_name: Camera or stream name associated with the violation.
        detection_time: Optional detector timestamp.
        warnings_json: Optional structured detector-warning JSON.
        detections_json: Optional tracked-detection JSON.
        cone_polygon_json: Optional safety-cone polygon JSON.
        pole_polygon_json: Optional utility-pole polygon JSON.
        image: Uploaded evidence image.
        db: Database session used to store the violation.
        credentials: Validated JWT credentials for the uploading service.

    Returns:
        Created violation identifier and upload result message.
    """
    return await violation_upload_service.upload_violation(
        site,
        stream_name,
        detection_time,
        warnings_json,
        detections_json,
        cone_polygon_json,
        pole_polygon_json,
        image,
        db,
        credentials,
    )
