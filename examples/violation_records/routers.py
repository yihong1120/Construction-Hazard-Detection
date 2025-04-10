from __future__ import annotations

import logging
from datetime import datetime
from datetime import timezone
from pathlib import Path

from fastapi import APIRouter
from fastapi import Depends
from fastapi import File
from fastapi import Form
from fastapi import HTTPException
from fastapi import Query
from fastapi import UploadFile
from fastapi.responses import FileResponse
from fastapi_jwt import JwtAuthorizationCredentials
from sqlalchemy import and_
from sqlalchemy import func
from sqlalchemy import or_
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from examples.auth.database import get_db
from examples.auth.jwt_config import jwt_access
from examples.auth.models import Site
from examples.auth.models import User
from examples.auth.models import Violation
from examples.violation_records.schemas import SiteOut
from examples.violation_records.schemas import UploadViolationResponse
from examples.violation_records.schemas import ViolationItem
from examples.violation_records.schemas import ViolationList
from examples.violation_records.search_utils import SearchUtils
from examples.violation_records.violation_manager import ViolationManager

# Instantiate a global ViolationManager for handling image saving
# and record creation.
violation_manager: ViolationManager = ViolationManager(base_dir='static')

# Create a global SearchUtils instance for expanding synonyms in query filters.
search_util: SearchUtils = SearchUtils(device=0)

# Create a FastAPI router for violations-related endpoints.
router: APIRouter = APIRouter()


@router.get(
    '/my_sites',
    response_model=list[SiteOut],
    summary='Get all accessible sites',
    description='Return a list of sites the user has access to.',
)
async def get_my_sites(
    db: AsyncSession = Depends(get_db),
    credentials: JwtAuthorizationCredentials = Depends(jwt_access),
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

    stmt_user = (
        select(User)
        .where(User.username == username)
        .options(selectinload(User.sites))
    )
    result_user = await db.execute(stmt_user)
    user_obj: User | None = result_user.scalar()
    if not user_obj:
        raise HTTPException(status_code=404, detail='User not found')

    return [
        SiteOut(
            id=s.id,
            name=s.name,
            created_at=s.created_at,
            updated_at=s.updated_at,
        )
        for s in user_obj.sites
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
    credentials: JwtAuthorizationCredentials = Depends(jwt_access),
) -> dict:
    """
    Retrieve a paginated list of violation records.

    Args:
        site_id (int | None):
            The ID of the site to filter violations by.
        keyword (str | None):
            A keyword to search for in violation records.
        start_time (datetime | None):
            The start of the detection time range.
        end_time (datetime | None):
            The end of the detection time range.
        limit (int):
            The maximum number of records to return (default is 20).
        offset (int):
            The starting record offset (default is 0).
        db (AsyncSession):
            The SQLAlchemy async session.
        credentials (JwtAuthorizationCredentials):
            The JWT credentials from the request.

    Returns:
        dict: A dictionary with:
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

    stmt_user = (
        select(User)
        .where(User.username == username)
        .options(selectinload(User.sites))
    )
    user_obj: User | None = (await db.execute(stmt_user)).scalar()
    if not user_obj:
        raise HTTPException(status_code=404, detail='User not found')

    site_names: list[str] = [site.name for site in user_obj.sites]
    if not site_names:
        return ViolationList(total=0, items=[])

    conditions = [Violation.site.in_(site_names)]

    if site_id is not None:
        site_stmt = select(Site).where(Site.id == site_id)
        site_obj: Site | None = (await db.execute(site_stmt)).scalar()
        if not site_obj or site_obj.name not in site_names:
            print(f"[get_violations] No access to site_id {site_id}")
            raise HTTPException(status_code=403, detail='No access to site_id')
        conditions.append(Violation.site == site_obj.name)

    if keyword:
        synonyms = search_util.expand_synonyms(keyword)
        or_list = []
        for syn in synonyms:
            or_list.append(Violation.stream_name.ilike(f"%{syn}%"))
            or_list.append(Violation.warnings_json.ilike(f"%{syn}%"))
        if or_list:
            conditions.append(or_(*or_list))

    if start_time:
        conditions.append(Violation.detection_time >= start_time)
    if end_time:
        conditions.append(Violation.detection_time <= end_time)

    total_stmt = (
        select(func.count())
        .select_from(Violation)
        .where(and_(*conditions))
    )
    total: int = (await db.execute(total_stmt)).scalar()

    stmt = (
        select(Violation)
        .where(and_(*conditions))
        .order_by(Violation.detection_time.desc())
        .offset(offset)
        .limit(limit)
    )
    violations: list[Violation] = (await db.scalars(stmt)).all()

    items = []
    for v in violations:
        items.append(
            ViolationItem(
                id=v.id,
                site_name=v.site,
                stream_name=v.stream_name,
                detection_time=v.detection_time,
                image_path=v.image_path,
                created_at=v.created_at,
                detection_items=v.detections_json,
                warnings=v.warnings_json,
                cone_polygons=v.cone_polygon_json,
                pole_polygons=v.pole_polygon_json,
            ),
        )

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
    credentials: JwtAuthorizationCredentials = Depends(jwt_access),
) -> dict:
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

    stmt_user = (
        select(User)
        .where(User.username == username)
        .options(selectinload(User.sites))
    )
    user_obj: User | None = (await db.execute(stmt_user)).scalar()
    if not user_obj:
        raise HTTPException(status_code=404, detail='User not found')

    site_names: list[str] = [site.name for site in user_obj.sites]
    stmt_violation = select(Violation).where(Violation.id == violation_id)
    violation: Violation | None = (await db.execute(stmt_violation)).scalar()

    if not violation or violation.site not in site_names:
        print(
            f"[get_single_violation] No access to violation_id {violation_id}",
        )
        raise HTTPException(
            status_code=403, detail='No access to this violation',
        )

    return ViolationItem(
        id=violation.id,
        site_name=violation.site,
        stream_name=violation.stream_name,
        detection_time=violation.detection_time,
        image_path=violation.image_path,
        created_at=violation.created_at,
        detection_items=violation.detections_json,
        warnings=violation.warnings_json,
        cone_polygons=violation.cone_polygon_json,
        pole_polygons=violation.pole_polygon_json,
    )


@router.get(
    '/get_violation_image',
    summary='Get a violation image file',
    description="Retrieve an image file from the 'static' directory.",
)
async def get_violation_image(
    image_path: str,
    credentials: JwtAuthorizationCredentials = Depends(jwt_access),
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

    if '..' in Path(image_path).parts:
        raise HTTPException(status_code=400, detail='Invalid path')

    cleaned_path: str = image_path.lstrip('static/')
    base_dir: Path = Path('static').resolve()
    full_path: Path = base_dir / cleaned_path
    print(f'[DEBUG] full_path => {full_path}')

    try:
        full_path.relative_to(base_dir)
    except ValueError:
        print(
            f"[get_violation_image] User {username} tried to "
            'access outside of base_dir',
        )
        raise HTTPException(status_code=403, detail='Access denied')

    if not full_path.exists():
        raise HTTPException(status_code=404, detail='Image not found')

    media_type: str = (
        'image/png'
        if full_path.suffix.lower() == '.png'
        else 'image/jpeg'
    )
    return FileResponse(
        path=full_path,
        media_type=media_type,
        headers={
            'Content-Disposition': f'inline; filename="{full_path.name}"',
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
    credentials: JwtAuthorizationCredentials = Depends(jwt_access),
) -> dict:
    """
    Upload a new violation record, including an image and associated metadata.

    Args:
        site (str):
            The name of the site where the violation occurred.
        stream_name (str):
            The name of the video stream or camera.
        detection_time (datetime | None):
            The detection time; defaults to UTC now.
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

    site_stmt = await db.execute(
        select(Site)
        .join(User.sites)
        .where(User.username == username, Site.name == site),
    )
    site_obj: Site | None = site_stmt.scalar()
    if not site_obj:
        print(f"[upload_violation] No access to site {site}")
        raise HTTPException(status_code=403, detail='No access to this site')

    detection_time = detection_time or datetime.now(timezone.utc)

    try:
        image_bytes: bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail='Empty image file')
    except Exception as exc:
        logging.error(f"[upload_violation] read error: {exc}")
        raise HTTPException(
            status_code=400, detail='Failed to read image file',
        )

    violation_id: int | None = await violation_manager.save_violation(
        db=db,
        site=site,
        stream_name=stream_name,
        detection_time=detection_time,
        image_bytes=image_bytes,
        warnings_json=warnings_json,
        detections_json=detections_json,
        cone_polygon_json=cone_polygon_json,
        pole_polygon_json=pole_polygon_json,
    )
    if not violation_id:
        raise HTTPException(
            status_code=500, detail='Failed to create violation record',
        )

    return UploadViolationResponse(
        message='Violation uploaded successfully.',
        violation_id=violation_id,
    )
