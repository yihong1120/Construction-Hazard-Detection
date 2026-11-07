from __future__ import annotations

import struct
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from PIL import UnidentifiedImageError
from pydantic import Field

from src.settings.base import EnvironmentSettings


class EvidenceImageSettings(EnvironmentSettings):
    model_config = {'env_prefix': 'VIOLATION_IMAGE_'}
    max_upload_bytes: int = Field(20 * 1024**2, ge=1)
    max_pixels: int = Field(64 * 1024**2, ge=1)


class ImageValidationError(ValueError):
    def __init__(self, code: str, status_code: int = 400):
        super().__init__(code)
        self.code, self.status_code = code, status_code


@dataclass(frozen=True)
class VerifiedImage:
    width: int
    height: int
    format: str
    suffix: str
    media_type: str


@contextmanager
def open_image(path: Path):
    # Use registered readers directly. ML packages can replace Image.open
    # with a wrapper that tries installing a package on malformed input.
    Image.init()
    with path.open('rb') as stream:
        prefix = stream.read(16)
        for factory, accept in tuple(Image.OPEN.values()):
            try:
                if accept is not None and accept(prefix) is not True:
                    continue
                stream.seek(0)
                image = factory(stream, '')
            except (SyntaxError, IndexError, TypeError, struct.error):
                continue
            try:
                yield image
            finally:
                image.close()
            return
    raise UnidentifiedImageError('Unsupported image header')


def verify_image(path: Path, limits: EvidenceImageSettings) -> VerifiedImage:
    if path.stat().st_size > limits.max_upload_bytes:
        raise ImageValidationError('IMAGE_UPLOAD_LIMIT_EXCEEDED', 413)
    try:
        with open_image(path) as image:
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > limits.max_pixels:
                raise ImageValidationError('IMAGE_DIMENSIONS_EXCEEDED', 413)
            image_format = image.format
            # Match the original-media endpoint's supported formats.
            suffixes = {
                'JPEG': '.jpg',
                'PNG': '.png',
            }
            if image_format not in suffixes:
                raise ImageValidationError('UNSUPPORTED_IMAGE_FORMAT')
            result = VerifiedImage(
                width,
                height,
                image_format,
                suffixes[image_format],
                Image.MIME.get(image_format, 'application/octet-stream'),
            )
            image.verify()
        with open_image(path) as decoded:
            decoded.load()
        return result
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        if isinstance(error, ImageValidationError):
            raise
        raise ImageValidationError('INVALID_IMAGE') from error
