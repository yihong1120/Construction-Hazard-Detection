from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image


def encode_png(image: np.ndarray) -> BytesIO:
    """Convert an RGB NumPy image to a rewound in-memory PNG file."""
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format='PNG')
    buffer.seek(0)
    return buffer
