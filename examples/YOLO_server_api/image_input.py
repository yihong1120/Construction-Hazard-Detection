from __future__ import annotations

import struct
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO

import cv2
import numpy as np
from fastapi import HTTPException
from PIL import Image
from PIL import UnidentifiedImageError


class DetectionInputError(HTTPException):
    def __init__(self, code: str = 'INVALID_IMAGE'):
        super().__init__(400, detail={'code': code})


def image_size(data: bytes) -> tuple[int, int]:
    # Read Pillow headers directly: ML libraries replace Image.open with a
    # wrapper that attempts package installation for unrecognized input.
    Image.init()
    prefix = data[:16]
    for factory, accept in tuple(Image.OPEN.values()):
        try:
            if accept is not None and accept(prefix) is not True:
                continue
            with BytesIO(data) as stream, factory(stream, '') as header:
                return header.size
        except (SyntaxError, IndexError, TypeError, struct.error):
            continue
    raise UnidentifiedImageError('Unsupported image header')


@dataclass(frozen=True)
class PreparedImage:
    data: bytes
    width: int
    height: int

    @property
    def decoded_bytes(self):
        return self.width * self.height * 3

    def decode(self):
        try:
            image = cv2.imdecode(
                np.frombuffer(self.data, np.uint8),
                cv2.IMREAD_COLOR,
            )
        except cv2.error as error:
            raise DetectionInputError() from error
        if image is None:
            raise DetectionInputError()
        if image.nbytes > self.decoded_bytes:
            raise DetectionInputError('IMAGE_DIMENSIONS_EXCEEDED')
        return image


def prepare_image(data: bytes, max_pixels: int) -> PreparedImage:
    try:
        width, height = image_size(data)
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
    ) as error:
        raise DetectionInputError() from error
    if width <= 0 or height <= 0 or width * height > max_pixels:
        raise DetectionInputError('IMAGE_DIMENSIONS_EXCEEDED')
    return PreparedImage(data, width, height)


def decode_image(data: bytes, max_pixels: int) -> np.ndarray:
    return prepare_image(data, max_pixels).decode()


class DecodedImageBudget:
    """Reserve active BGR input bytes before entering native decode."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.used = 0
        self._lock = threading.Lock()

    @contextmanager
    def reserve(self, size):
        with self._lock:
            if size > self.capacity:
                raise HTTPException(
                    413,
                    detail={'code': 'IMAGE_MEMORY_LIMIT_EXCEEDED'},
                )
            if self.used + size > self.capacity:
                raise HTTPException(
                    503,
                    detail={'code': 'INFERENCE_MEMORY_BUSY'},
                )
            self.used += size
        try:
            yield
        finally:
            with self._lock:
                self.used -= size
