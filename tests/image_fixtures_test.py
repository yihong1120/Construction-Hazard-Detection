from __future__ import annotations

from io import BytesIO

from PIL import Image


def image_bytes(color='red', image_format='PNG') -> bytes:
    output = BytesIO()
    Image.new('RGB', (8, 6), color).save(output, format=image_format)
    return output.getvalue()
