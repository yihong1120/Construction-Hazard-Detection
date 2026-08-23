from __future__ import annotations

import os
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from examples.streaming_web.overlay_models import RenderedTextBitmap


_text_bitmap_cache_size = max(
    128,
    int(os.getenv('STREAMING_OVERLAY_TEXT_BITMAP_CACHE_SIZE', '4096')),
)


def _measure_label_text(
    label: str,
    frame: np.ndarray,
    font: int,
    scale: float,
    thickness: int,
) -> tuple[int, int, int]:
    """Measure label text using Pillow when OpenCV cannot render it.

    Args:
        label: Localised text to measure.
        frame: Target BGR image array.
        font: OpenCV font identifier for ASCII text.
        scale: OpenCV font scale for ASCII text.
        thickness: OpenCV text thickness.

    Returns:
        Text width, height, and baseline in pixels.
    """
    if _needs_pillow_text(label):
        rendered = _render_pillow_text_bitmap(
            label,
            _font_pixel_size(frame),
            (255, 255, 255),
        )
        if rendered is not None:
            return rendered.width, rendered.height, max(2, thickness)

    (width, height), baseline = cv2.getTextSize(
        label, font, scale, thickness,
    )
    return width, height, baseline


def _draw_label_text(
    frame: np.ndarray,
    label: str,
    text_origin: tuple[int, int],
    font: int,
    scale: float,
    text_color: tuple[int, int, int],
    thickness: int,
    text_area: tuple[int, int, int, int],
) -> None:
    """Draw label text using the appropriate text renderer.

    Args:
        frame: Mutable BGR image array to annotate.
        label: Localised text to draw.
        text_origin: Text baseline origin in pixels.
        font: OpenCV font identifier for ASCII text.
        scale: OpenCV font scale for ASCII text.
        text_color: Text colour in BGR order.
        thickness: OpenCV text thickness.
        text_area: Bounding area that clips the text bitmap.
    """
    if _needs_pillow_text(label):
        _draw_pillow_text(
            frame,
            label,
            text_origin,
            text_color,
            text_area,
        )
        return

    cv2.putText(
        frame,
        label,
        text_origin,
        font,
        scale,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        label,
        text_origin,
        font,
        scale,
        text_color,
        max(1, thickness),
        cv2.LINE_AA,
    )


def _needs_pillow_text(label: str) -> bool:
    """Determine whether OpenCV's Hershey font is unsafe for a label.

    Args:
        label: Localised text to inspect.

    Returns:
        ``True`` when non-ASCII glyph rendering requires Pillow.
    """
    return any(ord(char) > 127 for char in label)


def _font_pixel_size(frame: np.ndarray) -> int:
    """Calculate a readable Pillow font size for a frame.

    Args:
        frame: Target BGR image array.

    Returns:
        Bounded pixel font size scaled to the smaller frame dimension.
    """
    min_side = min(frame.shape[:2])
    return max(14, min(32, round(min_side / 32)))


@lru_cache(maxsize=32)
def _load_overlay_font(
    size: int,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont | None:
    """Load a font that supports multilingual overlay labels.

    Args:
        size: Requested font size in pixels.

    Returns:
        First usable configured or system font, or ``None`` when unavailable.
    """
    configured = os.getenv('STREAMING_OVERLAY_FONT_PATH', '').strip()
    candidates = [
        configured,
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc',
        '/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf',
        '/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    ]
    # Prefer the explicit deployment font, then common Noto fonts with broad
    # CJK and Thai coverage before accepting a system fallback.
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _draw_pillow_text(
    frame: np.ndarray,
    label: str,
    text_origin: tuple[int, int],
    text_color: tuple[int, int, int],
    text_area: tuple[int, int, int, int],
) -> None:
    """Draw non-ASCII text with cached Pillow bitmaps.

    Args:
        frame: Mutable BGR image array to annotate.
        label: Localised non-ASCII text to draw.
        text_origin: Requested text baseline origin in pixels.
        text_color: Text colour in BGR order.
        text_area: Bounding area that clips the rendered text.
    """
    rendered = _render_pillow_text_bitmap(
        label,
        _font_pixel_size(frame),
        text_color,
    )
    if rendered is None or rendered.width <= 0 or rendered.height <= 0:
        return

    x1, y1, x2, y2 = text_area
    text_x = max(x1, min(text_origin[0], max(x1, x2 - rendered.width)))
    text_y = max(
        y1,
        min(text_origin[1] - rendered.height, max(y1, y2 - rendered.height)),
    )
    _blend_bgra_roi(frame, rendered.bgra, text_x, text_y)


@lru_cache(maxsize=_text_bitmap_cache_size)
def _render_pillow_text_bitmap(
    label: str,
    font_size: int,
    text_color: tuple[int, int, int],
) -> RenderedTextBitmap | None:
    """Render one text string once for subsequent ROI alpha blending.

    Args:
        label: Localised text to render.
        font_size: Requested font size in pixels.
        text_color: Text colour in BGR order.

    Returns:
        Cached BGRA text bitmap, or ``None`` when no font is available.
    """
    font = _load_overlay_font(font_size)
    if font is None:
        return None

    stroke_width = 2
    measure_image = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
    measure_draw = ImageDraw.Draw(measure_image)
    bbox = measure_draw.textbbox(
        (0, 0),
        label,
        font=font,
        stroke_width=stroke_width,
    )
    width = max(1, int(round(bbox[2] - bbox[0])))
    height = max(1, int(round(bbox[3] - bbox[1])))
    image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    rgb_color = (text_color[2], text_color[1], text_color[0])
    draw.text(
        (-bbox[0], -bbox[1]),
        label,
        font=font,
        fill=(*rgb_color, 255),
        stroke_width=stroke_width,
        stroke_fill=(0, 0, 0, 255),
    )
    rgba = np.asarray(image, dtype=np.uint8)
    return RenderedTextBitmap(
        bgra=rgba[:, :, [2, 1, 0, 3]].copy(),
        width=width,
        height=height,
    )


def _blend_bgra_roi(
    frame: np.ndarray,
    bgra: np.ndarray,
    x: int,
    y: int,
) -> None:
    """Alpha-blend cached text into only the affected frame rectangle.

    Args:
        frame: Mutable BGR image array to annotate.
        bgra: Cached four-channel text bitmap in OpenCV channel order.
        x: Requested destination left coordinate.
        y: Requested destination top coordinate.
    """
    frame_height, frame_width = frame.shape[:2]
    text_height, text_width = bgra.shape[:2]
    dst_x1 = max(0, x)
    dst_y1 = max(0, y)
    dst_x2 = min(frame_width, x + text_width)
    dst_y2 = min(frame_height, y + text_height)
    if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
        return

    src_x1 = dst_x1 - x
    src_y1 = dst_y1 - y
    src_x2 = src_x1 + (dst_x2 - dst_x1)
    src_y2 = src_y1 + (dst_y2 - dst_y1)

    overlay = bgra[src_y1:src_y2, src_x1:src_x2]
    alpha = overlay[:, :, 3:4].astype(np.uint16)
    if not np.any(alpha):
        return
    inv_alpha = 255 - alpha
    roi = frame[dst_y1:dst_y2, dst_x1:dst_x2]
    blended = (
        overlay[:, :, :3].astype(np.uint16) * alpha
        + roi.astype(np.uint16) * inv_alpha
        + 127
    ) // 255
    roi[:, :] = blended.astype(np.uint8)
