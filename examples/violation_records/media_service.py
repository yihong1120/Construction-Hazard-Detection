from __future__ import annotations

import asyncio
import os
from functools import partial
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException
from PIL import Image
from PIL import ImageFile
from PIL import UnidentifiedImageError

from examples.shared.image_validation import EvidenceImageSettings
from examples.shared.image_validation import open_image
from examples.shared.image_work import run_image_work
from examples.violation_records.path_utils import normalise_safe_relative_path
from examples.violation_records.path_utils import resolve_authorised_media_path
from examples.violation_records.settings import STATIC_DIR
from src.async_tasks import observe_task_exception

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
    return (
        base_dir
        / THUMBNAIL_DIR_NAME
        / relative_path.parent
        / (relative_path.name + '.jpg')
    )


def _has_recognized_image_header(source_path: Path) -> bool:
    """Return whether Pillow recognises the file header as an image."""
    with source_path.open('rb') as source:
        header = source.read(THUMBNAIL_HEADER_SCAN_BYTES)
    if not header:
        return False
    if (
        len(header) >= 12
        and header[4:8] == b'ftyp'
        and header[8:12] in _ISOBMFF_IMAGE_BRANDS
    ):
        return True
    parser = ImageFile.Parser()
    parser.feed(header)
    return parser.image is not None


def _thumbnail_is_current(source_path: Path, thumbnail_path: Path) -> bool:
    try:
        if thumbnail_path.stat().st_mtime_ns < source_path.stat().st_mtime_ns:
            return False
        with open_image(thumbnail_path) as image:
            if image.format != 'JPEG' or max(image.size) > THUMBNAIL_MAX_EDGE:
                return False
            image.load()
        return True
    except (OSError, ValueError, Image.DecompressionBombError):
        return False


def _generate_thumbnail_sync(source_path: Path, thumbnail_path: Path) -> None:
    if _thumbnail_is_current(source_path, thumbnail_path):
        return
    thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = thumbnail_path.with_name(
        '.' + thumbnail_path.name + '.' + uuid4().hex + '.tmp',
    )
    try:
        if not _has_recognized_image_header(source_path):
            raise UnidentifiedImageError(str(source_path))
        with open_image(source_path) as image:
            if image.width * image.height > EvidenceImageSettings().max_pixels:
                raise HTTPException(
                    413,
                    detail='Image dimensions exceed limit',
                )
            image.thumbnail((THUMBNAIL_MAX_EDGE, THUMBNAIL_MAX_EDGE))
            converted = (
                image.convert('RGB')
                if image.mode not in {'RGB', 'L'}
                else None
            )
            output = converted or image
            try:
                output.save(
                    temporary,
                    format='JPEG',
                    quality=THUMBNAIL_QUALITY,
                    optimize=True,
                )
                os.replace(temporary, thumbnail_path)
            finally:
                if converted is not None:
                    converted.close()
    except (
        ModuleNotFoundError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
    ) as exc:
        raise HTTPException(400, detail='Unsupported image content') from exc
    finally:
        temporary.unlink(missing_ok=True)


_thumbnail_jobs: dict[
    tuple[asyncio.AbstractEventLoop, Path],
    asyncio.Task[None],
] = {}


async def ensure_thumbnail(
    source_path: Path,
    static_dir: str | Path = STATIC_DIR,
) -> Path:
    thumbnail_path = _thumbnail_cache_path(source_path, static_dir)
    key = (asyncio.get_running_loop(), thumbnail_path)
    task = _thumbnail_jobs.get(key)
    if task is None:

        task = asyncio.create_task(
            _generate_thumbnail(source_path, thumbnail_path),
        )
        _thumbnail_jobs[key] = task

        task.add_done_callback(observe_task_exception)
    await asyncio.shield(task)
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


async def _generate_thumbnail(source_path: Path, thumbnail_path: Path):
    try:
        await run_image_work(
            partial(_generate_thumbnail_sync, source_path, thumbnail_path),
        )
    finally:
        _thumbnail_jobs.pop((asyncio.get_running_loop(), thumbnail_path), None)
