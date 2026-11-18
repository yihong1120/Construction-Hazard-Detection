"""Opt-in device-frame ownership, composition and actual NVENC validation."""
from __future__ import annotations

import io
import os
from pathlib import Path

import av
import numpy as np
import pytest
from av.codec.hwaccel import HWAccel

from src.experimental_gpu_media import ExperimentalGpuMedia
from src.gpu_color_lookup import build_nv12_lookup
from src.native_inference import load_inference_extension
from src.settings.inference import WorkerSettings

ENGINE = os.environ.get('CHD_NATIVE_TEST_ENGINE')
VIDEO = os.environ.get('CHD_NATIVE_TEST_VIDEO', 'tests/videos/test.mp4')
pytestmark = pytest.mark.skipif(
    not os.environ.get('CHD_NATIVE_TEST_GPU_MEDIA'),
    reason='Opt-in device media test',
)


@pytest.fixture(scope='module')
def device_source():
    native = load_inference_extension(WorkerSettings(backend='rust'))
    factory = native.GpuFrameFactory(build_nv12_lookup())
    source = av.open(
        VIDEO,
        hwaccel=HWAccel(
            'cuda', device='0', allow_software_fallback=False,
            options={'primary_ctx': '0'}, flags=2, is_hw_owned=True,
        ),
    )
    try:
        yield native, factory, source
    finally:
        source.close()


def read_gpu_frame(factory, source):
    decoded = next(source.decode(video=0))
    return factory.import_nv12(
        decoded.planes[0], decoded.planes[1], decoded.width, decoded.height,
    )


def test_device_inference_parity_and_pending_frame_ownership(
        device_source,
):
    native, factory, source = device_source
    frame = read_gpu_frame(factory, source)
    cpu = frame.download_bgr()
    engine = native.TensorRTEngine(ENGINE, asynchronous=True)
    try:
        expected = engine.predict([(frame.height, frame.width, cpu)] * 2)
        assert engine.predict_device_frames([frame, frame]) == expected
        ticket = engine.submit_device_frames([frame])
        with pytest.raises(ValueError, match='in use'):
            frame.blend_bgra(bytes(16), 2, 2, 0, 0)
        frame.close()
        assert engine.collect(ticket) == expected[:1]
        with pytest.raises(RuntimeError, match='closed'):
            frame.download_bgr()
        assert engine.async_stats[1] == 3
    finally:
        engine.close()
        frame.close()


def test_gpu_alpha_blend_is_pixel_exact_and_clone_is_independent(
        device_source,
):
    _, factory, source = device_source
    frame = read_gpu_frame(factory, source)
    copied = frame.copy()
    original = np.frombuffer(
        frame.download_bgr(),
        dtype=np.uint8,
    ).reshape(
        frame.height,
        frame.width,
        3,
    )
    rng = np.random.default_rng(32)
    overlay = rng.integers(0, 256, (16, 16, 4), dtype=np.uint8)
    overlay[:, :, 3] = np.arange(256, dtype=np.uint8).reshape(16, 16)
    expected = original.copy()
    roi = expected[4:20, 5:21]
    alpha = overlay[:, :, 3:4].astype(np.uint16)
    roi[:] = (
        (
            overlay[:, :, :3].astype(np.uint16) * alpha +
            roi.astype(np.uint16) * (255 - alpha) + 127
        ) // 255
    ).astype(np.uint8)
    try:
        copied.blend_bgra(overlay.tobytes(), 16, 16, 5, 4)
        actual = np.frombuffer(
            copied.download_bgr(),
            dtype=np.uint8,
        ).reshape(
            original.shape,
        )
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(
            np.frombuffer(
                frame.download_bgr(),
                dtype=np.uint8,
            ).reshape(
                original.shape,
            ),
            original,
        )
    finally:
        copied.close()
        frame.close()


def test_gpu_encoder_emits_decodable_h264_without_raw_frame_download(tmp_path):
    with ExperimentalGpuMedia(Path(ENGINE)) as pipeline:
        source = pipeline.open_source(VIDEO)
        try:
            decoded = next(source.decode(video=0))
            frame = pipeline.import_frame(decoded)
            try:
                factory = next(iter(pipeline.factories.values()))
                detections = pipeline.predict([frame])[0]
                rendered = pipeline.annotate(frame, detections)
                try:
                    reference = np.frombuffer(
                        rendered.download_bgr(), dtype=np.uint8,
                    ).reshape(frame.height, frame.width, 3)
                    before = list(factory.stats)
                    data = pipeline.encode(rendered) + pipeline.flush()
                finally:
                    rendered.close()
                assert factory.stats[1] == before[1]
                assert pipeline.model.engine.async_stats[0] == 0
                assert len(data) > 0
                container = av.open(io.BytesIO(data), format='h264')
                try:
                    output = next(container.decode(video=0))
                    assert (
                        output.width,
                        output.height,
                    ) == (
                        frame.width,
                        frame.height,
                    )
                    actual = output.to_ndarray(format='bgr24')
                    error = np.abs(
                        reference.astype(np.int16) - actual.astype(np.int16),
                    )
                    assert error.mean() < 12
                finally:
                    container.close()
            finally:
                frame.close()
        finally:
            source.close()
