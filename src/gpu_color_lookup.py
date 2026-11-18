"""Calibrate an experimental point-chroma color table outside the hot path."""
from __future__ import annotations

import av
import numpy as np


def build_nv12_lookup(colorspace: int = 1, color_range: int = 1) -> bytes:
    # Every U/V pair owns two identical rows and each Y owns two columns.
    # Sampling both pixels also detects an unsupported dithered conversion.
    width, block = 512, 16
    lut = np.empty((256, 256, 256, 3), dtype=np.uint8)
    for start in range(0, 256, block):
        height = block * 256 * 2
        nv12 = np.empty((height * 3 // 2, width), dtype=np.uint8)
        nv12[:height] = np.repeat(np.arange(256, dtype=np.uint8), 2)
        uv = nv12[height:]
        uv[:, ::2] = np.tile(np.arange(256, dtype=np.uint8), block)[:, None]
        uv[:, 1::2] = np.repeat(
            np.arange(start, start + block, dtype=np.uint8), 256,
        )[:, None]
        source = av.VideoFrame.from_ndarray(nv12, format='nv12')
        source.colorspace = colorspace
        source.color_range = color_range
        pixels = source.reformat(
            format='bgr24',
            interpolation='POINT',
        ).to_ndarray()
        if not (
            np.array_equal(pixels[::2, ::2], pixels[1::2, ::2])
            and np.array_equal(pixels[::2, ::2], pixels[::2, 1::2])
        ):
            raise RuntimeError(
                'swscale conversion is not a pointwise NV12 lookup',
            )
        lut[:, start:start + block] = pixels[
            ::2,
            ::2,
        ].reshape(block, 256, 256, 3).transpose(1, 0, 2, 3)
    return lut.tobytes()
