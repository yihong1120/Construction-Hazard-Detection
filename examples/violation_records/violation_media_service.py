from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth import user_service
from examples.auth.jwt_config import JwtAuthorizationCredentials
from examples.auth.models import Violation
from examples.shared.filename_utils import sanitize_filename
from examples.violation_records.media_service import ensure_thumbnail
from examples.violation_records.path_utils import _determine_media_type
from examples.violation_records.path_utils import normalise_safe_relative_path
from examples.violation_records.path_utils import resolve_authorised_media_path
from examples.violation_records.settings import STATIC_DIR


async def _authorise_media_access(
    image_path: str,
    username: str,
    db: AsyncSession,
) -> tuple[Path, str]:
    """Resolve evidence media after verifying record and site access."""
    safe_path = normalise_safe_relative_path(image_path)
    full_path = resolve_authorised_media_path(
        Path(STATIC_DIR).resolve(),
        safe_path,
        username,
    )
    if not full_path.exists():
        raise HTTPException(status_code=404, detail='Image not found')

    site_names = await user_service.get_cached_effective_site_names(
        username,
        db,
    )
    if not site_names:
        raise HTTPException(status_code=403, detail='Access denied')

    record_id = await db.scalar(
        select(Violation.id)
        .where(
            Violation.image_path == safe_path.as_posix(),
            Violation.site.in_(site_names),
        )
        .limit(1),
    )
    if record_id is None:
        raise HTTPException(status_code=403, detail='Access denied')
    return full_path, _determine_media_type(full_path)


async def get_violation_image(
    image_path: str,
    db: AsyncSession,
    credentials: JwtAuthorizationCredentials,
) -> FileResponse:
    """Return an authorised original evidence image."""
    username = credentials.subject.get('username')
    if not username:
        raise HTTPException(status_code=401, detail='Invalid token')

    full_path, media_type = await _authorise_media_access(
        image_path,
        username,
        db,
    )
    return FileResponse(
        path=full_path,
        media_type=media_type,
        headers={
            'Content-Disposition': (
                f'inline; filename="{sanitize_filename(full_path.name)}"'
            ),
        },
    )


async def get_violation_thumbnail(
    image_path: str,
    db: AsyncSession,
    credentials: JwtAuthorizationCredentials,
) -> FileResponse:
    """Return an authorised cached thumbnail for one evidence image."""
    username = credentials.subject.get('username')
    if not username:
        raise HTTPException(status_code=401, detail='Invalid token')

    full_path, _ = await _authorise_media_access(image_path, username, db)
    thumbnail_path = await ensure_thumbnail(full_path)
    return FileResponse(
        path=thumbnail_path,
        media_type='image/jpeg',
        headers={
            'Content-Disposition': (
                f'inline; filename="{sanitize_filename(thumbnail_path.name)}"'
            ),
            'Cache-Control': 'private, max-age=86400',
        },
    )
