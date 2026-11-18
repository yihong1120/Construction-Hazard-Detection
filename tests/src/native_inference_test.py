from __future__ import annotations

import queue
from multiprocessing import shared_memory
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest
from pydantic import ValidationError

from src.native_inference import _preload_runtime
from src.native_inference import create_rust_model
from src.native_inference import load_inference_extension
from src.native_inference import RustYoloModel
from src.settings.inference import WorkerSettings
from src.yolo_shared_frames import _SharedFrameArray
from src.yolo_worker import YoloWorker
from src.yolo_worker_protocol import _WorkerRequest
from src.yolo_worker_protocol import YOLO_WORKER_STOP_MESSAGE


class UnsupportedEngine(ValueError):
    pass


class NativeDouble:
    def __init__(self, *args, **kwargs):
        self.max_batch = 4
        self.activation_bytes = 1024
        self.calls = []
        self.closed = 0

    def predict(self, frames):
        self.calls.append(frames)
        return [
            [[0, 0, width, height, 0.9, 5]]
            for height, width, _data in frames
        ]

    def close(self):
        self.closed += 1


class BufferDouble(NativeDouble):
    def predict_buffers(self, views):
        self.calls.append([(view.readonly, view.shape) for view in views])
        return [[] for _ in views]


def extension(engine=NativeDouble):
    return SimpleNamespace(
        ABI_VERSION=1, HAS_TENSORRT=True, TensorRTEngine=engine,
        UnsupportedEngineError=UnsupportedEngine,
    )


def make_model():
    return RustYoloModel(
        extension(), Path('verified.engine'), WorkerSettings(batch_size=8),
        device='cuda:0', rect=False,
    )


def test_settings_validate_backend_and_library_directories(monkeypatch):
    monkeypatch.setenv('YOLO_WORKER_BACKEND', 'unsupported')
    with pytest.raises(ValidationError):
        WorkerSettings()
    settings = WorkerSettings(
        backend='rust', native_library_dirs='/first:/second',
    )
    assert settings.native_library_dirs == ('/first', '/second')


def test_python_and_absent_extension_selection():
    with patch(
        'src.native_inference.importlib.util.find_spec', return_value=None,
    ):
        assert load_inference_extension(
            WorkerSettings(backend='python'),
        ) is None
        assert load_inference_extension(WorkerSettings(backend='auto')) is None
        with pytest.raises(RuntimeError, match='GPU wheel'):
            load_inference_extension(WorkerSettings(backend='rust'))


@pytest.mark.parametrize(
    ('module', 'message'),
    [
        (SimpleNamespace(ABI_VERSION=2), 'ABI'),
        (SimpleNamespace(ABI_VERSION=1, HAS_TENSORRT=True), 'Incomplete'),
        (
            SimpleNamespace(
                ABI_VERSION=1, HAS_TENSORRT=True,
                TensorRTEngine=NativeDouble,
            ), 'error interface',
        ),
    ],
)
def test_broken_extension_is_not_a_silent_fallback(module, message):
    with patch(
        'src.native_inference.importlib.util.find_spec', return_value=True,
    ), patch(
        'src.native_inference._preload_runtime',
    ), patch(
        'src.native_inference.importlib.import_module', return_value=module,
    ):
        with pytest.raises(RuntimeError, match=message):
            load_inference_extension(WorkerSettings(backend='auto'))


def test_cpu_wheel_is_optional_only_for_auto():
    module = extension()
    module.HAS_TENSORRT = False
    with patch(
        'src.native_inference.importlib.util.find_spec', return_value=True,
    ), patch(
        'src.native_inference._preload_runtime',
    ), patch(
        'src.native_inference.importlib.import_module', return_value=module,
    ):
        assert load_inference_extension(WorkerSettings(backend='auto')) is None
        with pytest.raises(RuntimeError, match='tensorrt feature'):
            load_inference_extension(WorkerSettings(backend='rust'))
        module.HAS_TENSORRT = True
        assert load_inference_extension(
            WorkerSettings(backend='rust'),
        ) is module


def test_loader_retries_packaged_runtime_and_respects_explicit_failures():
    module = extension()
    automatic = WorkerSettings(backend='auto')
    explicit = WorkerSettings(backend='rust', native_library_dirs=('/sdk',))
    with patch(
        'src.native_inference.importlib.util.find_spec', return_value=True,
    ), patch('src.native_inference._preload_runtime') as preload, patch(
        'src.native_inference.importlib.import_module',
        side_effect=[ImportError('ELF dependency'), module],
    ):
        assert load_inference_extension(automatic) is module
        preload.assert_called_once_with(())
    with patch(
        'src.native_inference.importlib.util.find_spec', return_value=True,
    ), patch('src.native_inference._preload_runtime') as preload, patch(
        'src.native_inference.importlib.import_module',
        side_effect=ImportError('missing driver'),
    ):
        with pytest.raises(ImportError, match='missing driver'):
            load_inference_extension(explicit)
        preload.assert_called_once_with(('/sdk',))


def test_runtime_preload_prefers_explicit_directory(tmp_path):
    first, second = tmp_path / 'one', tmp_path / 'two'
    first.mkdir()
    second.mkdir()
    (first / 'libnvinfer.so.11').touch()
    (second / 'libnvinfer.so.11').touch()
    with patch('src.native_inference.ctypes.CDLL') as load:
        _preload_runtime((str(first), str(second)))
    paths = [call.args[0] for call in load.call_args_list]
    assert str(first / 'libnvinfer.so.11') in paths
    assert str(second / 'libnvinfer.so.11') not in paths


def test_only_verified_cuda_engines_select_rust():
    path = Path('model.pt')
    assert create_rust_model(
        path, WorkerSettings(backend='auto'), device='cuda:0', rect=False,
    ) is None
    assert create_rust_model(
        path, WorkerSettings(backend='python'), device='cuda:0', rect=False,
    ) is None
    with pytest.raises(ValueError, match='verified CUDA'):
        create_rust_model(
            path, WorkerSettings(backend='rust'), device='cuda:0', rect=False,
        )
    with patch(
        'src.native_inference.load_inference_extension', return_value=None,
    ):
        assert create_rust_model(
            Path('model.engine'), WorkerSettings(backend='auto'),
            device='cuda:0', rect=False,
        ) is None


def test_unsupported_layout_falls_back_only_at_model_initialization():
    factory = MagicMock(side_effect=UnsupportedEngine('layout'))
    with patch(
        'src.native_inference.load_inference_extension',
        return_value=extension(factory),
    ):
        assert create_rust_model(
            Path('model.engine'), WorkerSettings(backend='auto'),
            device='cuda:0', rect=False,
        ) is None
        with pytest.raises(UnsupportedEngine):
            create_rust_model(
                Path('model.engine'), WorkerSettings(backend='rust'),
                device='cuda:0', rect=False,
            )
        factory.side_effect = RuntimeError('Runtime version mismatch')
        with pytest.raises(RuntimeError, match='version mismatch'):
            create_rust_model(
                Path('model.engine'), WorkerSettings(backend='auto'),
                device='cuda:0', rect=False,
            )
        factory.side_effect = None
        factory.return_value = NativeDouble()
        assert isinstance(
            create_rust_model(
                Path('model.engine'), WorkerSettings(backend='rust'),
                device='cuda:0', rect=False,
            ), RustYoloModel,
        )


def test_adapter_chunks_batches_and_owns_noncontiguous_frames():
    model = make_model()
    frame = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)[:, ::-1]
    result = model.predict_detections([frame] * 9)
    assert len(result) == 9
    assert [len(call) for call in model.engine.calls] == [4, 4, 1]
    assert model.engine.calls[0][0] == (3, 4, frame.tobytes())
    assert model.predict_detections([]) == []
    model.close()
    model.close()
    assert model.engine.closed == 1
    with pytest.raises(RuntimeError, match='closed'):
        model.predict_detections([frame])


def test_only_leased_contiguous_frames_use_optional_buffer_api():
    module = extension(BufferDouble)
    module.BUFFER_API_VERSION = 1
    model = RustYoloModel(
        module, Path('verified.engine'), WorkerSettings(),
        device='cuda:0', rect=False,
    )
    shm = shared_memory.SharedMemory(create=True, size=36)
    frame = np.ndarray((3, 4, 3), dtype=np.uint8, buffer=shm.buf).view(
        _SharedFrameArray,
    )
    frame.shared_memory_handle = shm
    try:
        assert model.predict_detections([frame]) == [[]]
        assert model.engine.calls == [[(True, (3, 4, 3))]]
        assert model.borrowed_frames == 1
        assert model.snapshot_bytes == 0
        model.predict_detections([np.zeros((3, 4, 3), dtype=np.uint8)])
        model.predict_detections([frame[:, ::-1]])
        assert model.snapshot_bytes == 72
        assert model.borrowed_frames == 1
        # Views exported to Rust are released before the worker unmaps them.
        frame.shared_memory_handle = None
        shm.close()
    finally:
        model.close()
        shm.close()
        shm.unlink()


def test_adapter_rejects_invalid_inputs_and_incomplete_results():
    with pytest.raises(ValueError, match='cuda:N'):
        RustYoloModel(
            extension(), Path('verified.engine'), WorkerSettings(),
            device='cpu', rect=False,
        )
    model = make_model()
    for frame in (
        np.zeros((2, 2), dtype=np.uint8),
        np.zeros((0, 2, 3), dtype=np.uint8),
        np.zeros((2, 2, 3), dtype=np.float32),
    ):
        with pytest.raises(ValueError, match='uint8 HWC BGR'):
            model.predict_detections([frame])
    model.engine.max_batch = 0
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match='capacity'):
        model.predict_detections([frame])
    model.engine.max_batch = 4
    model.engine.predict = MagicMock(return_value=[])
    with pytest.raises(RuntimeError, match='incomplete batch'):
        model.predict_detections([frame])


def test_worker_routes_native_batches_and_closes_engine_on_stop(monkeypatch):
    monkeypatch.setenv('YOLO_WORKER_BACKEND', 'python')
    requests = queue.Queue()
    worker = YoloWorker(requests)
    model = make_model()
    worker.model_cache['model'] = model
    assert len(
        worker._predict_batch(
            model, [np.zeros((2, 2, 3), dtype=np.uint8)] * 9,
        ),
    ) == 9
    requests.put(YOLO_WORKER_STOP_MESSAGE)
    worker.run()
    assert model.engine.closed == 1
    assert not worker.model_cache


def test_native_failure_returns_errors_to_every_camera(monkeypatch):
    monkeypatch.setenv('YOLO_WORKER_BACKEND', 'python')
    model = make_model()
    model.engine.predict = MagicMock(side_effect=RuntimeError('GPU failure'))
    replies = {f'camera-{index}': queue.Queue() for index in range(32)}
    worker = YoloWorker(None, result_queues=replies)
    worker.model_cache['model'] = model
    requests = [
        _WorkerRequest(
            str(index), camera, 'model',
            'fixture', (2, 2, 3), 'uint8',
        )
        for index, camera in enumerate(replies)
    ]
    with patch.object(
        worker, '_read_batch_frames',
        return_value=([np.zeros((2, 2, 3), dtype=np.uint8)] * 32, requests),
    ), patch.object(worker, '_close_shared_frames') as release:
        worker._handle_batch(requests)
    release.assert_called_once()
    for index, reply in enumerate(replies.values()):
        result = reply.get_nowait()
        assert result['id'] == str(index)
        assert result['ok'] is False
        assert 'GPU failure' in result['error']
    worker.close()
