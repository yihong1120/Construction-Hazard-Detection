from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi import Depends
from fastapi import File
from fastapi import Form
from fastapi import HTTPException
from fastapi import Query
from fastapi import Security
from fastapi import UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import and_
from sqlalchemy import func
from sqlalchemy import or_
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth import user_service as _user_service
from examples.auth.database import get_db
from examples.auth.jwt_config import jwt_access
from examples.auth.jwt_config import JwtAuthorizationCredentials
from examples.auth.models import Site
from examples.auth.models import Violation
from examples.shared.filename_utils import sanitize_filename
from examples.violation_records.path_utils import _determine_media_type
from examples.violation_records.path_utils import _normalize_safe_rel_path
from examples.violation_records.path_utils import _resolve_and_authorize
from examples.violation_records.schemas import SiteOut
from examples.violation_records.schemas import UploadViolationResponse
from examples.violation_records.schemas import ViolationItem
from examples.violation_records.schemas import ViolationList
from examples.violation_records.search_utils import SearchUtils
from examples.violation_records.settings import STATIC_DIR
from examples.violation_records.violation_manager import (
    EmptyViolationImageError,
)
from examples.violation_records.violation_manager import (
    ViolationImageReadError,
)
from examples.violation_records.violation_manager import ViolationManager

# Module-level aliases used by tests for patching
get_cached_effective_site_names = _user_service.get_cached_effective_site_names
load_user_with_effective_sites = _user_service.load_user_with_effective_sites
get_user_effective_sites = _user_service.load_user_with_effective_sites
get_user_sites_cached = _user_service.get_cached_effective_site_names

# Instantiate a global ViolationManager for handling image saving
# and record creation.
violation_manager: ViolationManager = ViolationManager(base_dir=STATIC_DIR)

# Create a global SearchUtils instance for expanding synonyms in query filters.
search_util: SearchUtils = SearchUtils(device=-1)

# Create a FastAPI router for violations-related endpoints.
router: APIRouter = APIRouter()

# Note: effective site access helpers are provided by
# examples.auth.user_service

_violation_columns = (
    Violation.id,
    Violation.site,
    Violation.stream_name,
    Violation.detection_time,
    Violation.image_path,
    Violation.created_at,
    Violation.detections_json,
    Violation.warnings_json,
    Violation.cone_polygon_json,
    Violation.pole_polygon_json,
)
_violation_column_count = len(_violation_columns)


def _violation_to_item(row: Any) -> ViolationItem:
    """Convert an ORM object or selected-column row into a response item."""
    if hasattr(row, 'site'):
        return ViolationItem(
            id=row.id,
            site_name=row.site,
            stream_name=row.stream_name,
            detection_time=row.detection_time.astimezone(),
            image_path=row.image_path,
            created_at=row.created_at.astimezone(),
            detection_items=row.detections_json,
            warnings=row.warnings_json,
            cone_polygons=row.cone_polygon_json,
            pole_polygons=row.pole_polygon_json,
        )

    (
        violation_id,
        site,
        stream_name,
        detection_time,
        image_path,
        created_at,
        detections_json,
        warnings_json,
        cone_polygon_json,
        pole_polygon_json,
    ) = row
    return ViolationItem(
        id=violation_id,
        site_name=site,
        stream_name=stream_name,
        detection_time=detection_time.astimezone(),
        image_path=image_path,
        created_at=created_at.astimezone(),
        detection_items=detections_json,
        warnings=warnings_json,
        cone_polygons=cone_polygon_json,
        pole_polygons=pole_polygon_json,
    )


def _split_violation_row_total(row: Any) -> tuple[Any, int | None]:
    """Split an optional window-count column from a selected violation row."""
    mapping = getattr(row, '_mapping', None)
    if mapping is not None and 'total_count' in mapping:
        try:
            row_length = len(row)
        except TypeError:
            return row, int(mapping['total_count'])
        if row_length == _violation_column_count + 1:
            return row[:_violation_column_count], int(mapping['total_count'])
        return row, int(mapping['total_count'])

    try:
        row_length = len(row)
    except TypeError:
        return row, None

    if row_length == _violation_column_count + 1:
        return row[:_violation_column_count], int(row[-1])
    return row, None


def _scalar_value(value: Any) -> Any:
    """Return a scalar result value."""
    if hasattr(value, 'name'):
        return value.name
    return value


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
    """
    Retrieve all sites accessible by the currently logged-in user.

    Args:
        db (AsyncSession): The SQLAlchemy async session.
        credentials (JwtAuthorizationCredentials): The JWT credentials from
            the request.

    Returns:
        list[dict]: A list of dictionaries containing the site's ID, name,
            creation timestamp, and update timestamp.

    Raises:
        HTTPException: If the token is invalid (401) or the user is not found
            (404).
    """
    username: str | None = credentials.subject.get('username')
    if not username:
        raise HTTPException(status_code=401, detail='Invalid token')

    _, sites = await get_user_effective_sites(username, db)

    return [
        SiteOut(
            id=s.id,
            name=s.name,
            created_at=s.created_at,
            updated_at=s.updated_at,
        )
        for s in sites
    ]


@router.get(
    '/violations',
    response_model=ViolationList,
    summary='Get paginated violation records',
    description='Supports filtering by site_id, keyword, and time range.',
)
async def get_violations(
    site_id: int | None = None,
    keyword: str | None = None,
    start_time: datetime | None = Query(None),
    end_time: datetime | None = Query(None),
    limit: int = Query(
        20, gt=0, le=100, description='Records per page (1-100)',
    ),
    offset: int = Query(0, ge=0, description='Starting record offset'),
    db: AsyncSession = Depends(get_db),
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
) -> ViolationList:
    """
    Retrieve a paginated list of violation records.

    Args:
        site_id (int | None): The ID of the site to filter violations by.
        keyword (str | None): A keyword to search for in violation records.
        start_time (datetime | None): The start of the detection time range.
        end_time (datetime | None): The end of the detection time range.
        limit (int): The maximum number of records to return (default is 20).
        offset (int): The starting record offset (default is 0).
        db (AsyncSession): The SQLAlchemy async session.
        credentials (JwtAuthorizationCredentials):
            The JWT credentials from the request.

    Returns:
        ViolationList: A dictionary with:
            - 'total': the total count of matching violations,
            - 'items': a list of violation records (paginated).

    Raises:
        HTTPException: If the token is invalid (401), if the user is not found
            (404), if the user lacks access to the site (403), or if any other
            error occurs.
    """
    username: str | None = credentials.subject.get('username')
    if not username:
        raise HTTPException(status_code=401, detail='Invalid token')

    # Retrieve user sites using the cache
    site_names: list[str] = await get_user_sites_cached(username, db)
    if not site_names:
        return ViolationList(total=0, items=[])

    conditions: list = [Violation.site.in_(site_names)]

    if site_id is not None:
        site_name_set = set(site_names)
        site_stmt = select(Site.name).where(Site.id == site_id)
        site_name = _scalar_value((await db.execute(site_stmt)).scalar())
        if not site_name or site_name not in site_name_set:
            raise HTTPException(status_code=403, detail='No access to site_id')
        conditions.append(Violation.site == site_name)

    keyword_text = keyword.strip() if keyword else None
    if keyword_text:
        synonyms: list[str] = search_util.expand_synonyms(keyword_text)
        or_list: list = []
        for syn in dict.fromkeys(synonyms):
            or_list.append(Violation.stream_name.ilike(f"%{syn}%"))
            or_list.append(Violation.warnings_json.ilike(f"%{syn}%"))
        if or_list:
            conditions.append(or_(*or_list))

    if start_time:
        conditions.append(Violation.detection_time >= start_time)
    if end_time:
        conditions.append(Violation.detection_time <= end_time)

    where_clause = and_(*conditions)

    rows_stmt = (
        select(*_violation_columns, func.count().over().label('total_count'))
        .where(where_clause)
        .order_by(Violation.detection_time.desc())
        .offset(offset)
        .limit(limit)
    )
    rows_result = await db.execute(rows_stmt)
    rows = rows_result.all()
    total = 0
    items: list[ViolationItem] = []
    for row in rows:
        item_row, row_total = _split_violation_row_total(row)
        if row_total is not None:
            total = row_total
        items.append(_violation_to_item(item_row))

    if not rows and offset:
        total_result = await db.execute(
            select(func.count())
            .select_from(Violation)
            .where(where_clause),
        )
        total = int(total_result.scalar() or 0)

    return ViolationList(total=total, items=items)


@router.get(
    '/violations/{violation_id}',
    response_model=ViolationItem,
    summary='Get single violation details',
    description='Retrieve a single violation record by its ID.',
)
async def get_single_violation(
    violation_id: int,
    db: AsyncSession = Depends(get_db),
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
) -> ViolationItem:
    """
    Retrieve detailed information for a specific violation record.

    Args:
        violation_id (int):
            The ID of the violation to retrieve.
        db (AsyncSession):
            The SQLAlchemy async session.
        credentials (JwtAuthorizationCredentials):
            The JWT credentials from the request.

    Returns:
        dict: A dictionary containing details of the violation, including:
            - 'id': The ID of the violation.
            - 'site_name': The name of the site.
            - 'stream_name': The name of the stream.
            - 'detection_time': The time of detection.
            - 'image_path': The path to the image.
            - 'created_at': The creation timestamp.
            - 'detection_items': JSON string with detection items.
            - 'warnings': JSON string with warnings.
            - 'cone_polygons': JSON string with cone polygons.
            - 'pole_polygons': JSON string with pole polygons.

    Raises:
        HTTPException: If the token is invalid (401), if the user is not found
            (404), if the user lacks access to the violation's site (403), or
            if the violation ID does not exist (404).
    """
    username: str | None = credentials.subject.get('username')
    if not username:
        raise HTTPException(status_code=401, detail='Invalid token')

    # Retrieve user sites using the cache
    site_names: list[str] = await get_user_sites_cached(username, db)
    if not site_names:
        raise HTTPException(
            status_code=403, detail='No access to this violation',
        )

    stmt_violation = select(*_violation_columns).where(
        Violation.id == violation_id,
        Violation.site.in_(site_names),
    )
    result = await db.execute(stmt_violation)
    row = result.first() if hasattr(result, 'first') else result.scalar()

    if not row:
        print(
            f"[get_single_violation] No access to violation_id {violation_id}",
        )
        raise HTTPException(
            status_code=403, detail='No access to this violation',
        )
    if hasattr(row, 'site') and row.site not in site_names:
        print(
            f"[get_single_violation] No access to violation_id {violation_id}",
        )
        raise HTTPException(
            status_code=403, detail='No access to this violation',
        )

    return _violation_to_item(row)


@router.get(
    '/get_violation_image',
    summary='Get a violation image file',
    description="Retrieve an image file from the 'static' directory.",
)
async def get_violation_image(
    image_path: str,
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
) -> FileResponse:
    """
    Retrieve a violation image file from the "static" directory.

    Args:
        image_path (str): The relative path of the image within the "static"
            directory.
        credentials (JwtAuthorizationCredentials): The JWT credentials from the
            request.

    Returns:
        FileResponse: The requested image file with inline Content-Disposition.

    Raises:
        HTTPException: If the token is invalid (401), if the path contains '..'
            (400), if the path is outside "static" (403), or if the file is not
            found (404).
    """
    username: str | None = credentials.subject.get('username')
    if not username:
        raise HTTPException(status_code=401, detail='Invalid token')

    # Normalize and authorize path
    safe_rel_path = _normalize_safe_rel_path(image_path, path_cls=Path)
    base_dir: Path = Path(STATIC_DIR).resolve()
    full_path: Path = _resolve_and_authorize(
        base_dir, safe_rel_path, username, path_cls=Path,
    )
    print(f'[DEBUG] full_path => {full_path}')

    if not full_path.exists():
        raise HTTPException(status_code=404, detail='Image not found')

    media_type: str = _determine_media_type(full_path)

    return FileResponse(
        path=full_path,
        media_type=media_type,
        headers={
            'Content-Disposition': (
                f'inline; filename="{sanitize_filename(full_path.name)}"'
            ),
        },
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
    """
    Upload a new violation record, including an image and associated metadata.

    Args:
        site (str):
            The name of the site where the violation occurred.
        stream_name (str):
            The name of the video stream or camera.
        detection_time (datetime | None):
            The detection time; defaults to local now.
        warnings_json (str | None):
            JSON string describing warnings.
        detections_json (str | None):
            JSON string describing detected items.
        cone_polygon_json (str | None):
            JSON string with cone polygon data.
        pole_polygon_json (str | None):
            JSON string with pole polygon data.
        image (UploadFile):
            The violation image file.
        db (AsyncSession):
            The SQLAlchemy async session.
        credentials (JwtAuthorizationCredentials):
            The JWT credentials.

    Returns:
        dict: A dictionary containing a success message and the violation ID.

    Raises:
        HTTPException: If the token is invalid (401), if the user has no access
            to the site (403), or if any error occurs during file reading or
            database operations (400, 500).
    """
    username: str | None = credentials.subject.get('username')
    if not username:
        raise HTTPException(status_code=401, detail='Invalid token')

    site_names: list[str] = await get_user_sites_cached(username, db)
    if site not in site_names:
        print(f"[upload_violation] No access to site {site}")
        raise HTTPException(status_code=403, detail='No access to this site')

    detection_time = (
        detection_time.astimezone()
        if detection_time is not None
        else datetime.now().astimezone()
    )

    try:
        violation_id: int | None = await violation_manager.save_violation(
            db=db,
            site=site,
            stream_name=stream_name,
            detection_time=detection_time,
            image_file=image,
            warnings_json=warnings_json,
            detections_json=detections_json,
            cone_polygon_json=cone_polygon_json,
            pole_polygon_json=pole_polygon_json,
        )
    except (EmptyViolationImageError, ViolationImageReadError) as exc:
        logging.error(f"[upload_violation] read error: {exc}")
        raise HTTPException(
            status_code=400, detail='Failed to read image file',
        )
    if not violation_id:
        raise HTTPException(
            status_code=500, detail='Failed to create violation record',
        )

    return UploadViolationResponse(
        message='Violation uploaded successfully.',
        violation_id=violation_id,
    )
