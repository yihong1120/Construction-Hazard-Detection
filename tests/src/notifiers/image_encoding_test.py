from __future__ import annotations

import numpy as np

from src.notifiers.image_encoding import encode_png


def test_encode_png_returns_a_rewound_png_buffer() -> None:
    """Notification providers receive a ready-to-upload PNG buffer."""
    image = np.zeros((2, 3, 3), dtype=np.uint8)

    buffer = encode_png(image)

    assert buffer.tell() == 0
    assert buffer.read(8) == b'\x89PNG\r\n\x1a\n'
