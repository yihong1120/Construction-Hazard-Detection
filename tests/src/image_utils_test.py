from __future__ import annotations

from unittest.mock import patch

import numpy as np

from src import image_utils


def test_encode_text_uses_url_safe_utf8_base64() -> None:
    """Text encoding preserves UTF-8 content using URL-safe Base64."""
    assert image_utils.encode_text('安全帽') == '5a6J5YWo5bi9'


def test_encode_frame_returns_jpeg_and_png_bytes() -> None:
    """OpenCV encoding returns the expected signature for both formats."""
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    jpeg = image_utils.encode_frame(frame, image_format='jpeg', quality=90)
    png = image_utils.encode_frame(frame, image_format='png', quality=50)

    assert jpeg.startswith(b'\xff\xd8')
    assert png.startswith(b'\x89PNG\r\n\x1a\n')


def test_encode_frame_handles_missing_and_failed_input() -> None:
    """Missing, rejected, and exceptional frames do not leak encoder errors."""
    frame = np.zeros((1, 1, 3), dtype=np.uint8)

    assert image_utils.encode_frame(None) == b''
    with patch(
        'src.image_utils.cv2.imencode',
        return_value=(False, np.array([], dtype=np.uint8)),
    ):
        assert image_utils.encode_frame(frame) == b''
    with patch(
        'src.image_utils.cv2.imencode',
        side_effect=RuntimeError('encoder unavailable'),
    ):
        assert image_utils.encode_frame(frame) == b''
