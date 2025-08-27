from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException
from werkzeug.utils import secure_filename

from examples.violation_records.settings import STATIC_DIR


def _normalize_safe_rel_path(image_path: str, path_cls: type = Path) -> Path:
    """Normalize and sanitize a user-provided image path to a
    safe relative Path.

    - Reject absolute paths or traversal tokens ('..').
    - Strip leading static/ if present to avoid duplication.
    - Sanitize each segment with secure_filename; reject empty results.

    Raises HTTPException(400) on invalid inputs.
    """
    raw_path = path_cls(image_path)
    if raw_path.is_absolute() or '..' in raw_path.parts:
        raise HTTPException(status_code=400, detail='Invalid path')

    if raw_path.parts and raw_path.parts[0] == STATIC_DIR.name:
        raw_path = path_cls(*raw_path.parts[1:])

    safe_parts: list[str] = []
    for part in raw_path.parts:
        if part in {'', '.', '..'}:
            raise HTTPException(status_code=400, detail='Invalid path')
        cleaned = secure_filename(part)
        if not cleaned:
            raise HTTPException(status_code=400, detail='Invalid path segment')
        safe_parts.append(cleaned)
    return path_cls(*safe_parts) if safe_parts else path_cls()


def _resolve_and_authorize(
    base_dir: Path,
    rel_path: Path,
    username: str,
    path_cls: type = Path,
) -> Path:
    """Resolve rel_path under base_dir and ensure containment.

    Raises 403 on escape attempts. Returns the resolved absolute path.
    """
    base_dir = base_dir.resolve()
    full_path = (base_dir / rel_path).resolve()
    try:
        full_path.relative_to(base_dir)
    except ValueError:
        print(
            f"[get_violation_image] User {username} tried to "
            'access outside of base_dir',
        )
        raise HTTPException(status_code=403, detail='Access denied')
    return full_path


def _determine_media_type(full_path: Path) -> str:
    """Return appropriate media type for supported image suffix; else 400.
    """
    suffix = full_path.suffix.lower()
    allowed_ext = {'.png', '.jpg', '.jpeg'}
    if suffix not in allowed_ext:
        raise HTTPException(status_code=400, detail='Unsupported file type')
    return 'image/png' if suffix == '.png' else 'image/jpeg'
