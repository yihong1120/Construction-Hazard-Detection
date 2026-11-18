"""Parity and lifetime tests for the incremental native pipeline kernels."""
from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from examples.streaming_web.overlay_renderer import _blend_polygon_fill_roi
from examples.streaming_web.overlay_text import _blend_bgra_roi
from src.danger_detector import DangerDetector
from src.geometry import build_mst_pairs
from src.native_capture import RustCapture
from src.settings.capture import CaptureSettings

AVAILABLE = importlib.util.find_spec('construction_postprocess') is not None
VIDEO = Path(__file__).resolve().parents[1] / 'videos/test.mp4'


@pytest.fixture
def native():
    if not AVAILABLE:
        pytest.skip('Requires the native/postprocess wheel')
    import construction_postprocess
    if getattr(construction_postprocess, 'CAPTURE_API_VERSION', None) != 1:
        pytest.skip('Requires the updated native/postprocess wheel')
    return construction_postprocess


@pytest.mark.parametrize('offset', [(0, 0), (-11, -5), (58, 27), (-500, 9)])
def test_text_blend_preserves_integer_rounding_and_clipping(native, offset):
    rng = np.random.default_rng(32)
    source = rng.integers(0, 256, (32, 64, 3), dtype=np.uint8)
    text = rng.integers(0, 256, (16, 16, 4), dtype=np.uint8)
    text[:, :, 3] = np.arange(256, dtype=np.uint8).reshape(16, 16)
    expected, actual = source.copy(), source.copy()
    with patch('src.native_kernels.optional_kernel', return_value=None):
        _blend_bgra_roi(expected, text, *offset)
    native.blend_bgra(actual, text, *offset)
    np.testing.assert_array_equal(actual, expected)
    actual.setflags(write=False)
    with pytest.raises(ValueError, match='writable'):
        native.blend_bgra(actual, text, *offset)


@pytest.mark.parametrize('alpha', [.1, .4, .5, .95, 1.0])
def test_polygon_lut_preserves_opencv_pixels(native, alpha):
    source = np.repeat(
        np.arange(256, dtype=np.uint8)
        [None, :, None], 3, axis=2,
    )
    source = np.repeat(source, 16, axis=0)
    points = np.array([[-3, -1], [256, 0], [200, 15], [1, 12]], dtype=np.int32)
    for color in ((255, 64, 129), (68, 138, 255), (1, 0, 255)):
        expected, actual = source.copy(), source.copy()
        with patch('src.native_kernels.optional_kernel', return_value=None):
            _blend_polygon_fill_roi(expected, points, color, alpha)
        _blend_polygon_fill_roi(actual, points, color, alpha)
        np.testing.assert_array_equal(actual, expected)


def test_native_mst_preserves_python_candidate_order(native):
    rng = np.random.default_rng(32)
    for count in (0, 1, 2, 3, 4, 16, 80):
        poles = [
            tuple(map(float, row))
            for row in rng.integers(0, 100, (count, 3))
        ]
        with patch('src.geometry.optional_kernel', return_value=None):
            expected = build_mst_pairs(poles)
        assert build_mst_pairs(poles) == expected
    with pytest.raises(ValueError, match='index'):
        native.mst_edges(2, [(0, 2)])


def test_geometry_cache_rechecks_people_and_invalidates_configuration():
    flags = {
        'detect_no_safety_vest_or_helmet': False,
        'detect_near_machinery_or_vehicle': False,
        'detect_in_restricted_area': False,
        'detect_in_utility_pole_restricted_area': True,
        'detect_machinery_close_to_pole': False,
    }
    detector = DangerDetector(flags)
    pole = [100, 0, 110, 100, .9, 9]
    inside = [100, 90, 110, 110, .9, 5]
    outside = [1000, 900, 1010, 1000, .9, 5]
    first = detector.detect_danger([pole, inside])
    second = detector.detect_danger([pole, outside])
    warning = first[0]['warning_people_in_utility_pole_controlled_area']
    assert warning['count'] == 1
    assert not second[0]
    assert first[2] == second[2]
    assert detector.geometry_cache_hits == 1
    assert detector.geometry_cache_misses == 1
    detector.clusterer.set_params(min_samples=4)
    detector.detect_danger([pole, inside])
    assert detector.geometry_cache_misses == 2
    detector.detect_danger([[101, 0, 111, 100, .9, 9], inside])
    assert detector.geometry_cache_misses == 3
    assert len(detector._area_cache) == 1


def test_capture_owns_exported_pixels_after_close_and_drops_backlog(native):
    if shutil.which('ffmpeg') is None or shutil.which('ffprobe') is None:
        pytest.skip('Requires FFmpeg tools')
    capture = native.LatestCapture(str(VIDEO), timeout=5)
    try:
        owner = capture.read(5)
        assert owner is not None
        frame = np.frombuffer(owner, dtype=np.uint8).reshape(
            capture.height, capture.width, 3,
        )
        before = frame.copy()
        assert not frame.flags.writeable
        assert not frame.flags.owndata
        capture.close()
        capture.close()
        assert capture.read(1) is None
        np.testing.assert_array_equal(frame, before)
        with pytest.raises(ValueError):
            frame.setflags(write=True)
    finally:
        capture.close()


def test_rust_capture_adapter_and_invalid_parameters(native):
    if shutil.which('ffmpeg') is None or shutil.which('ffprobe') is None:
        pytest.skip('Requires FFmpeg tools')
    capture = RustCapture(str(VIDEO), CaptureSettings(backend='rust_ffmpeg'))
    try:
        ok, frame = capture.read()
        assert ok and frame.shape[2] == 3
        assert capture.isOpened()
    finally:
        capture.release()
    assert not capture.isOpened()
    assert capture.read() == (False, None)
    with pytest.raises(ValueError, match='timeout'):
        native.LatestCapture(str(VIDEO), timeout=0)
    with pytest.raises(RuntimeError, match='ffprobe'):
        native.LatestCapture('/nonexistent/native-test-video', timeout=1)


def test_native_decode_preserves_first_opencv_frame(native):
    reference = cv2.VideoCapture(str(VIDEO))
    ok, expected = reference.read()
    reference.release()
    assert ok
    capture = native.LatestCapture(str(VIDEO), realtime=True)
    try:
        owner = capture.read(5)
        actual = np.frombuffer(owner, dtype=np.uint8).reshape(expected.shape)
        assert capture.counters[0] >= 1
        np.testing.assert_array_equal(actual, expected)
    finally:
        capture.close()


@pytest.mark.skipif(
    not os.environ.get('CHD_NATIVE_TEST_CUDA_CAPTURE'),
    reason='Opt-in NVDEC test', )
def test_cuda_capture_reads_h264_with_host_output(native):
    capture = native.LatestCapture(str(VIDEO), cuda=True, timeout=5)
    try:
        assert capture.read(5) is not None
    finally:
        capture.close()
