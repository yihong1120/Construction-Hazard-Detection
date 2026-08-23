from __future__ import annotations

import logging
from pathlib import Path

from fastapi import HTTPException

from examples.shared.filename_utils import sanitize_filename
from examples.violation_records.settings import STATIC_DIR

logger = logging.getLogger(__name__)


def normalise_safe_relative_path(image_path: str) -> Path:
    """Normalise and sanitise a user-provided relative image path.

    Args:
        image_path: Untrusted relative image path supplied by a client.

    Returns:
        Sanitised relative path rooted below the static media directory.

    Raises:
        HTTPException: If the path is absolute, traverses directories, or has
            an invalid component.
    """
    raw_path = Path(image_path)
    if raw_path.is_absolute() or '..' in raw_path.parts:
        raise HTTPException(status_code=400, detail='Invalid path')

    if raw_path.parts and raw_path.parts[0] == STATIC_DIR.name:
        raise HTTPException(status_code=400, detail='Invalid path')

    safe_parts: list[str] = []
    for part in raw_path.parts:
        cleaned = sanitize_filename(part)
        if not cleaned:
            raise HTTPException(status_code=400, detail='Invalid path segment')
        safe_parts.append(cleaned)
    return Path(*safe_parts) if safe_parts else Path()


def resolve_authorised_media_path(
    base_dir: Path,
    rel_path: Path,
    username: str,
) -> Path:
    """Resolve a media path below its authorised base directory.

    Args:
        base_dir: Root directory authorised for violation media.
        rel_path: Sanitised relative media path.
        username: Requesting username for security logging.

    Returns:
        Resolved absolute path contained by ``base_dir``.

    Raises:
        HTTPException: If resolution attempts to escape the media directory.
    """
    base_dir = base_dir.resolve()
    full_path = (base_dir / rel_path).resolve()
    try:
        full_path.relative_to(base_dir)
    except ValueError:
        logger.info(
            f"[get_violation_image] User {username} tried to "
            'access outside of base_dir',
        )
        raise HTTPException(status_code=403, detail='Access denied')
    return full_path


def _determine_media_type(full_path: Path) -> str:
    """Return the media type for a supported violation image.

    Args:
        full_path: Authorised image path whose suffix is inspected.

    Returns:
        PNG or JPEG media type for the image.

    Raises:
        HTTPException: If the file suffix is unsupported.
    """
    suffix = full_path.suffix.lower()
    allowed_ext = {'.png', '.jpg', '.jpeg'}
    if suffix not in allowed_ext:
        raise HTTPException(status_code=400, detail='Unsupported file type')
    return 'image/png' if suffix == '.png' else 'image/jpeg'
