from __future__ import annotations

import base64
import logging

import cv2
import numpy as np


def encode_text(value: str) -> str:
    """Encode text into a URL-safe Base64 string."""
    return base64.urlsafe_b64encode(value.encode('utf-8')).decode('utf-8')


def encode_frame(
    frame: np.ndarray | None,
    image_format: str = 'jpeg',
    quality: int = 85,
) -> bytes:
    """Encode a BGR image frame and return an empty payload on failure."""
    if frame is None:
        return b''
    try:
        if image_format.lower() == 'jpeg':
            success, buffer = cv2.imencode(
                '.jpg',
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, quality],
            )
        else:
            success, buffer = cv2.imencode(
                '.png',
                frame,
                [cv2.IMWRITE_PNG_COMPRESSION, min(quality // 10, 9)],
            )
        if not success:
            logging.getLogger(__name__).error(
                'OpenCV image encoding failed: format=%s',
                image_format,
            )
            return b''
        return buffer.tobytes()
    except Exception:
        logging.getLogger(__name__).exception(
            'Image encoding failed: format=%s',
            image_format,
        )
        return b''
