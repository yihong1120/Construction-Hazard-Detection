"""Experimental media gates and native/lease ownership on failure."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from src.experimental_gpu_media import ExperimentalGpuMedia
from src.gpu_color_lookup import build_nv12_lookup
from src.gpu_resources import try_acquire_resource


def decoded(profile=(1, 1)):
    return SimpleNamespace(
        format=SimpleNamespace(name='cuda'), planes=[object(), object()],
        width=1280, height=720, colorspace=profile[0], color_range=profile[1],
    )


def native():
    return MagicMock(GPU_FRAME_API_VERSION=1, HAS_GPU_ENCODER=True)


def test_color_table_is_complete_and_deterministic_for_installed_swscale():
    first = build_nv12_lookup()
    assert len(first) == 256 ** 3 * 3
    assert first == build_nv12_lookup()


def test_missing_native_apis_stop_before_model_load():
    extension = native()
    with patch(
        'src.experimental_gpu_media.load_inference_extension',
        return_value=extension,
    ), patch('src.experimental_gpu_media.create_rust_model') as model:
        extension.GPU_FRAME_API_VERSION = 0
        with pytest.raises(RuntimeError, match='GPU-frame'):
            ExperimentalGpuMedia(Path('fixture.engine'))
        extension.GPU_FRAME_API_VERSION = 1
        extension.HAS_GPU_ENCODER = False
        with pytest.raises(RuntimeError, match='gpu-media'):
            ExperimentalGpuMedia(Path('fixture.engine'))
        model.assert_not_called()


def test_profile_and_frame_gates_and_cache_bound():
    extension = native()
    with patch(
        'src.experimental_gpu_media.load_inference_extension',
        return_value=extension,
    ), patch('src.experimental_gpu_media.create_rust_model') as model, patch(
        'src.experimental_gpu_media.build_nv12_lookup', return_value=b'lookup',
    ) as lookup:
        with ExperimentalGpuMedia(Path('fixture.engine')) as pipeline:
            pipeline.import_frame(decoded())
            pipeline.import_frame(decoded())
            assert pipeline.input_frames == 2
            lookup.assert_called_once_with(1, 1)
            pipeline.import_frame(decoded((5, 1)))
            with pytest.raises(RuntimeError, match='cache is full'):
                pipeline.import_frame(decoded((6, 1)))
            with pytest.raises(ValueError, match='metadata'):
                pipeline.import_frame(decoded((9, 1)))
            bad = decoded()
            bad.format.name = 'bgr24'
            with pytest.raises(ValueError, match='CUDA NV12'):
                pipeline.import_frame(bad)
            pipeline.predict([object()])
            model.return_value.predict_device_detections.assert_called_once()
            assert pipeline.flush() == b''
        model.return_value.close.assert_called_once()


def test_encoder_failure_releases_budget_and_render_failure_closes_copy(
        monkeypatch,
):
    monkeypatch.setenv('MEDIA_NVENC_MAX_SESSIONS', '1')
    extension = native()
    extension.GpuEncoder.side_effect = RuntimeError('encoder failed')
    with patch(
        'src.experimental_gpu_media.load_inference_extension',
        return_value=extension,
    ), patch('src.experimental_gpu_media.create_rust_model'):
        with ExperimentalGpuMedia(Path('fixture.engine')) as pipeline:
            frame = MagicMock(width=1280, height=720)
            frame.copy.return_value.blend_bgra.side_effect = RuntimeError(
                'drawing failed',
            )
            with pytest.raises(RuntimeError, match='drawing failed'):
                pipeline.annotate(frame, [[1, 2, 50, 60, .9, 5]])
            frame.copy.return_value.close.assert_called_once()
            with pytest.raises(RuntimeError, match='encoder failed'):
                pipeline.encode(frame)
            recovered = try_acquire_resource('nvenc', 1)
            assert recovered is not None
            recovered.close()
            extension.GpuEncoder.side_effect = None
            encoder = extension.GpuEncoder.return_value
            encoder.encode.return_value = b'packet'
            assert pipeline.encode(frame) == b'packet'
            assert try_acquire_resource('nvenc', 1) is None
            assert pipeline.flush() == b'packet'
            assert pipeline.encoded_bytes == 12
        encoder.close.assert_called_once()
        assert pipeline.reservation is None
