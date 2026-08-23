from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import HTTPException
from PIL import Image
from PIL import ImageFile
from PIL import UnidentifiedImageError

from examples.violation_records.path_utils import normalise_safe_relative_path
from examples.violation_records.path_utils import resolve_authorised_media_path
from examples.violation_records.settings import STATIC_DIR

THUMBNAIL_DIR_NAME = '_thumbnails'
THUMBNAIL_MAX_EDGE = 360
THUMBNAIL_QUALITY = 78
THUMBNAIL_HEADER_SCAN_BYTES = 64 * 1024
_ISOBMFF_IMAGE_BRANDS = frozenset(
    {
        b'avif',
        b'avis',
        b'heic',
        b'heix',
        b'hevc',
        b'hevx',
        b'mif1',
        b'msf1',
    },
)


def _thumbnail_cache_path(
    full_path: Path,
    static_dir: str | Path = STATIC_DIR,
) -> Path:
    """Build the cached JPEG thumbnail path for an authorised image."""
    base_dir = Path(static_dir).resolve()
    relative_path = full_path.resolve().relative_to(base_dir)
    return base_dir / THUMBNAIL_DIR_NAME / relative_path.with_suffix('.jpg')


def _has_recognized_image_header(source_path: Path) -> bool:
    """Return whether Pillow recognises the file header as an image."""
    with source_path.open('rb') as source:
        header = source.read(THUMBNAIL_HEADER_SCAN_BYTES)
    if not header:
        return False
    parser = ImageFile.Parser()
    parser.feed(header)
    return parser.image is not None or (
        len(header) >= 12
        and header[4:8] == b'ftyp'
        and header[8:12] in _ISOBMFF_IMAGE_BRANDS
    )


def _generate_thumbnail_sync(source_path: Path, thumbnail_path: Path) -> None:
    """Generate or refresh a JPEG thumbnail, skipping an up-to-date cache."""
    if (
        thumbnail_path.exists()
        and thumbnail_path.stat().st_mtime >= source_path.stat().st_mtime
    ):
        return
    thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not _has_recognized_image_header(source_path):
            raise UnidentifiedImageError(str(source_path))
        with Image.open(source_path) as opened_image:
            image: Image.Image = opened_image
            image.thumbnail((THUMBNAIL_MAX_EDGE, THUMBNAIL_MAX_EDGE))
            if image.mode not in {'RGB', 'L'}:
                image = image.convert('RGB')
            image.save(
                thumbnail_path,
                format='JPEG',
                quality=THUMBNAIL_QUALITY,
                optimize=True,
            )
    except (ModuleNotFoundError, OSError) as exc:
        raise HTTPException(
            status_code=400,
            detail='Unsupported image content',
        ) from exc


async def ensure_thumbnail(
    source_path: Path,
    static_dir: str | Path = STATIC_DIR,
) -> Path:
    """Generate a cached thumbnail without blocking an API worker."""
    thumbnail_path = _thumbnail_cache_path(source_path, static_dir)
    await asyncio.to_thread(
        _generate_thumbnail_sync,
        source_path,
        thumbnail_path,
    )
    return thumbnail_path


def _image_size_sync(
    image_path: str,
    static_dir: str | Path,
) -> tuple[int, int] | None:
    """Read evidence dimensions after resolving the path safely."""
    try:
        safe_rel_path = normalise_safe_relative_path(image_path)
        base_dir = Path(static_dir).resolve()
        full_path = resolve_authorised_media_path(
            base_dir,
            safe_rel_path,
            '_internal',
        )
        with Image.open(full_path) as image:
            return image.size
    except (HTTPException, OSError, UnidentifiedImageError):
        return None


async def image_size_for_violation(
    image_path: str,
    static_dir: str | Path = STATIC_DIR,
) -> tuple[int, int] | None:
    """Read evidence dimensions in a worker thread."""
    return await asyncio.to_thread(_image_size_sync, image_path, static_dir)
