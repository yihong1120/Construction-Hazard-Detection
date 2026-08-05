from __future__ import annotations

import asyncio
import queue
import sys
from collections.abc import Iterable
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from src import yolo_worker


class _QueueSpy:
    """Tests for _QueueSpy."""

    def __init__(self) -> None:
        """Support __init__."""
        self.requests: list[yolo_worker.WorkerRequestPayload] = []

    def put(
        self,
        request: yolo_worker.WorkerRequestPayload,
        block: bool = True,
        timeout: float | None = None,
    ) -> None:
        """Support put."""
        self.requests.append(request)


class _ResultQueue:
    """In-memory result queue used by worker IPC tests."""

    def __init__(self) -> None:
        """Initialise the result message buffer."""
        self.messages: list[yolo_worker.WorkerResult] = []

    def put(
        self,
        result: yolo_worker.WorkerResult,
        block: bool = True,
        timeout: float | None = None,
    ) -> None:
        """Store one worker result."""
        self.messages.append(result)

    def get(
        self,
        block: bool = True,
        timeout: float | None = None,
    ) -> yolo_worker.WorkerResult:
        """Return the next result or mirror multiprocessing queue.Empty."""
        if not self.messages:
            raise queue.Empty
        return self.messages.pop(0)


class _StopQueue:
    """Tests for _StopQueue."""

    def get(self, block: bool = True, timeout: float | None = None) -> object:
        """Support get."""
        return yolo_worker.YOLO_WORKER_STOP_MESSAGE

    def get_nowait(self) -> object:
        """Support get_nowait."""
        raise queue.Empty

    def put(
        self,
        obj: object,
        block: bool = True,
        timeout: float | None = None,
    ) -> None:
        """Support put."""


class _RunQueue:
    """Tests for _RunQueue."""

    def __init__(self, first_message: object) -> None:
        """Support __init__."""
        self.messages = [first_message, yolo_worker.YOLO_WORKER_STOP_MESSAGE]

    def get(self, block: bool = True, timeout: float | None = None) -> object:
        """Support get."""
        return self.messages.pop(0)

    def get_nowait(self) -> object:
        """Support get_nowait."""
        raise queue.Empty

    def put(
        self,
        obj: object,
        block: bool = True,
        timeout: float | None = None,
    ) -> None:
        """Support put."""
        self.messages.append(obj)


class _FullQueue:
    """Tests for _FullQueue."""

    def put(self, *_args, **_kwargs) -> None:
        """Support put."""
        raise queue.Full


class _DrainQueue:
    """Tests for _DrainQueue."""

    def __init__(self, messages: list[object]) -> None:
        """Support __init__."""
        self.messages = messages
        self.requeued: list[object] = []

    def get(self, block: bool = True, timeout: float | None = None) -> object:
        """Support get."""
        if not self.messages:
            raise queue.Empty
        return self.messages.pop(0)

    def get_nowait(self) -> object:
        """Support get_nowait."""
        return self.get(block=False)

    def put(
        self,
        obj: object,
        block: bool = True,
        timeout: float | None = None,
    ) -> None:
        """Support put."""
        self.requeued.append(obj)


class _Tensor:
    """Tests for _Tensor."""

    def __init__(self, data: np.ndarray) -> None:
        """Support __init__."""
        self._data = data

    def cpu(self) -> _Tensor:
        """Support cpu."""
        return self

    def numpy(self) -> np.ndarray:
        """Support numpy."""
        return self._data


class _Boxes:
    """Tests for _Boxes."""

    def __init__(self, data: np.ndarray) -> None:
        """Support __init__."""
        self.data = _Tensor(data)


class _Result:
    """Tests for _Result."""

    def __init__(self, data: np.ndarray) -> None:
        """Support __init__."""
        self.boxes = _Boxes(data)


class _Model:
    """Tests for _Model."""

    def __init__(self, data: np.ndarray) -> None:
        """Support __init__."""
        self.data = data
        self.calls: list[dict[str, object]] = []

    def predict(
        self,
        source: list[yolo_worker.FrameArray],
        verbose: bool,
        device: str,
        imgsz: int,
        batch: int,
        **kwargs: yolo_worker.PrecisionValue,
    ) -> Iterable[_Result]:
        """Support predict."""
        self.calls.append({
            'source': source,
            'verbose': verbose,
            'device': device,
            'imgsz': imgsz,
            'batch': batch,
            **kwargs,
        })
        return [_Result(self.data)]


class _StartupLock:
    """Tracks serialized first-inference access."""

    def __init__(self) -> None:
        """Support __init__."""
        self.enter_count = 0

    def __enter__(self) -> _StartupLock:
        """Support context entry."""
        self.enter_count += 1
        return self

    def __exit__(self, *_args: object) -> None:
        """Support context exit."""
        return None


def _request(
    request_id: str,
    camera_id: str,
    model_key: str,
    result_queue: yolo_worker.WorkerResultSender | None = None,
) -> yolo_worker._WorkerRequest:
    """Support _request."""
    return yolo_worker._WorkerRequest(
        id=request_id,
        camera_id=camera_id,
        model_key=model_key,
        shm_name='frame-shm',
        shape=(2, 2, 3),
        dtype='uint8',
        result_queue=result_queue or _ResultQueue(),
    )


def test_store_latest_request_replaces_same_camera() -> None:
    """Only the newest request for a camera remains pending."""
    result_queue = _ResultQueue()
    worker = yolo_worker.YoloWorker(None)

    worker.store_latest_request(
        _request('old', 'site|cam1', 'yolo26n', result_queue),
    )
    worker.store_latest_request(
        _request('new', 'site|cam1', 'yolo26n', result_queue),
    )

    assert worker.pending['site|cam1'].id == 'new'
    assert result_queue.messages == [
        {'id': 'old', 'ok': True, 'detections': [], 'skipped': True},
    ]


def test_pop_next_batch_groups_by_model_key() -> None:
    """Batch selection avoids mixing model keys."""
    worker = yolo_worker.YoloWorker(None)
    worker.batch_size = 8
    worker.pending = {
        'cam1': _request('1', 'cam1', 'yolo26n'),
        'cam2': _request('2', 'cam2', 'yolo26n'),
        'cam3': _request('3', 'cam3', 'yolo26s'),
    }

    batch = worker.pop_next_batch()

    assert [request.id for request in batch] == ['1', '2']
    assert set(worker.pending) == {'cam3'}


def test_pop_next_batch_respects_batch_size() -> None:
    """Exercise this test."""
    worker = yolo_worker.YoloWorker(None)
    worker.batch_size = 1
    worker.pending = {
        'cam1': _request('1', 'cam1', 'yolo26n'),
        'cam2': _request('2', 'cam2', 'yolo26n'),
    }

    batch = worker.pop_next_batch()

    assert [request.id for request in batch] == ['1']
    assert set(worker.pending) == {'cam2'}


def test_timeout_request_keeps_ring_slot_until_worker_reads_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Timed-out requests retain their ring slot until worker-side IO ends."""
    request_queue = _QueueSpy()
    result_queue = _ResultQueue()
    client = yolo_worker.YoloWorkerClient(
        request_queue,
        result_queue,
        camera_id='site|cam1',
        timeout_seconds=0.001,
    )
    with pytest.raises(TimeoutError):
        asyncio.run(
            client.detect(np.zeros((2, 2, 3), dtype=np.uint8), 'yolo26n'),
        )

    request = yolo_worker._WorkerRequest.from_mapping(
        request_queue.requests[0],
    )
    assert client._ring is not None
    assert client._ring.name == request.shm_name

    worker = yolo_worker.YoloWorker(None)
    frames, valid_requests = worker._read_batch_frames([request])

    assert len(frames) == 1
    assert valid_requests == [request]
    worker._close_shared_frames(frames)
    asyncio.run(client.close())


def test_client_detect_returns_worker_result(monkeypatch: Any) -> None:
    """Exercise this test."""
    async def fake_submit(
        request: yolo_worker.WorkerRequestPayload,
    ) -> None:
        """Support fake_submit.

        Args:
            request: Test helper value.
        """
        result_queue.put({
            'id': request['id'],
            'ok': True,
            'detections': [[1, 2, 3, 4, 0.9, 1]],
        })

    result_queue = _ResultQueue()
    client = yolo_worker.YoloWorkerClient(
        _QueueSpy(),
        result_queue,
        camera_id='cam1',
    )
    monkeypatch.setattr(client, '_submit_request', fake_submit)

    detections = asyncio.run(
        client.detect(np.zeros((2, 2, 3), dtype=np.uint8), 'yolo26n'),
    )

    assert detections == [[1, 2, 3, 4, 0.9, 1]]
    assert result_queue.messages == []


def test_shared_frame_ring_reuses_one_allocation() -> None:
    """Sequential frames reuse the same bounded shared-memory ring."""
    request_queue = _QueueSpy()
    result_queue = _ResultQueue()
    client = yolo_worker.YoloWorkerClient(
        request_queue,
        result_queue,
        camera_id='cam1',
    )

    async def submit_and_reply(
        request: yolo_worker.WorkerRequestPayload,
    ) -> None:
        """Echo a completed worker result without creating a new ring."""
        request_queue.put(request)
        result_queue.put({
            'id': request['id'],
            'ok': True,
            'detections': [],
        })

    async def run() -> None:
        """Submit two equal-sized frames through the same client."""
        setattr(client, '_submit_request', submit_and_reply)
        await client.detect(
            np.zeros((2, 2, 3), dtype=np.uint8),
            'yolo26n',
        )
        await client.detect(
            np.ones((2, 2, 3), dtype=np.uint8),
            'yolo26n',
        )
        await client.close()

    asyncio.run(run())

    assert len(request_queue.requests) == 2
    assert request_queue.requests[0]['shm_name'] == (
        request_queue.requests[1]['shm_name']
    )
    assert {request['slot'] for request in request_queue.requests} == {0, 1}


def test_worker_run_stops_on_stop_message() -> None:
    """Exercise this test."""
    worker = yolo_worker.YoloWorker(_StopQueue())

    worker.run()

    assert worker.pending == {}


def test_worker_run_handles_one_batch(monkeypatch: Any) -> None:
    """Exercise this test."""
    message = {
        'id': '1',
        'camera_id': 'cam1',
        'model_key': 'yolo26n',
        'shm_name': 'frame-shm',
        'slot': 0,
        'shape': (2, 2, 3),
        'dtype': 'uint8',
        'result_queue': _ResultQueue(),
    }
    handled: list[list[str]] = []
    worker = yolo_worker.YoloWorker(_RunQueue(message))
    monkeypatch.setattr(
        worker,
        '_handle_batch',
        lambda batch: handled.append([request.id for request in batch]),
    )

    worker.run()

    assert handled == [['1']]


def test_pop_next_batch_returns_empty_when_no_pending() -> None:
    """Exercise this test."""
    worker = yolo_worker.YoloWorker(None)

    assert worker.pop_next_batch() == []


def test_drain_queue_requeues_stop_message() -> None:
    """Exercise this test."""
    request_queue = _DrainQueue([yolo_worker.YOLO_WORKER_STOP_MESSAGE])
    worker = yolo_worker.YoloWorker(request_queue)

    worker._drain_queue(deadline=999999999.0)

    assert request_queue.requeued == [yolo_worker.YOLO_WORKER_STOP_MESSAGE]


def test_drain_queue_stores_requests_until_empty() -> None:
    """Exercise this test."""
    request = {
        'id': '1',
        'camera_id': 'cam1',
        'model_key': 'yolo26n',
        'shm_name': 'frame-shm',
        'slot': 0,
        'shape': (2, 2, 3),
        'dtype': 'uint8',
        'result_queue': _ResultQueue(),
    }
    request_queue = _DrainQueue([request])
    worker = yolo_worker.YoloWorker(request_queue)

    worker._drain_queue(deadline=999999999.0)

    assert worker.pending['cam1'].id == '1'


def test_submit_request_raises_when_queue_full() -> None:
    """Exercise this test."""
    client = yolo_worker.YoloWorkerClient(
        _FullQueue(),
        _ResultQueue(),
        camera_id='cam1',
        timeout_seconds=0.001,
    )

    with pytest.raises(TimeoutError, match='queue is full'):
        asyncio.run(
            client._submit_request({
                'id': '1',
                'camera_id': 'cam1',
                'model_key': 'yolo26n',
                'shm_name': 'frame-shm',
                'slot': 0,
                'shape': (2, 2, 3),
                'dtype': 'uint8',
                'result_queue': _ResultQueue(),
            }),
        )


def test_wait_for_result_returns_detections() -> None:
    """Exercise this test."""
    result_queue = _ResultQueue()
    client = yolo_worker.YoloWorkerClient(
        _QueueSpy(),
        result_queue,
        camera_id='cam1',
    )
    result_queue.put({
        'id': 'request',
        'ok': True,
        'detections': [[1, 2, 3, 4, 0.9, 1]],
    })

    detections = asyncio.run(client._wait_for_result('request'))

    assert detections == [[1, 2, 3, 4, 0.9, 1]]


def test_wait_for_result_discards_stale_camera_response() -> None:
    """One camera queue can harmlessly contain an older timed-out response."""
    result_queue = _ResultQueue()
    result_queue.put({'id': 'old', 'ok': True, 'detections': []})
    result_queue.put({
        'id': 'request',
        'ok': True,
        'detections': [[1, 2, 3, 4, 0.9, 1]],
    })
    client = yolo_worker.YoloWorkerClient(
        _QueueSpy(),
        result_queue,
        camera_id='cam1',
    )

    detections = asyncio.run(client._wait_for_result('request'))

    assert detections == [[1, 2, 3, 4, 0.9, 1]]


def test_wait_for_result_raises_worker_error() -> None:
    """Exercise this test."""
    result_queue = _ResultQueue()
    client = yolo_worker.YoloWorkerClient(
        _QueueSpy(),
        result_queue,
        camera_id='cam1',
    )
    result_queue.put({
        'id': 'request',
        'ok': False,
        'error': 'broken',
    })

    with pytest.raises(RuntimeError, match='broken'):
        asyncio.run(client._wait_for_result('request'))


def test_wait_for_result_times_out() -> None:
    """Exercise this test."""
    client = yolo_worker.YoloWorkerClient(
        _QueueSpy(),
        _ResultQueue(),
        camera_id='cam1',
        timeout_seconds=0.001,
    )

    with pytest.raises(TimeoutError, match='timed out'):
        asyncio.run(client._wait_for_result('missing'))


def test_handle_batch_returns_when_no_valid_requests(monkeypatch: Any) -> None:
    """Exercise this test."""
    worker = yolo_worker.YoloWorker(None)
    monkeypatch.setattr(
        worker,
        '_read_batch_frames',
        lambda _requests: ([], []),
    )

    worker._handle_batch([_request('1', 'cam1', 'yolo26n')])

    assert worker.model_cache == {}


def test_handle_batch_records_model_error(monkeypatch: Any) -> None:
    """Exercise this test."""
    worker = yolo_worker.YoloWorker(None)
    result_queue = _ResultQueue()
    request = _request('1', 'cam1', 'yolo26n', result_queue)
    monkeypatch.setattr(
        worker,
        '_read_batch_frames',
        lambda _requests: ([np.zeros((2, 2, 3), dtype=np.uint8)], [request]),
    )
    monkeypatch.setattr(
        worker,
        '_get_model',
        lambda _model_key: (_ for _ in ()).throw(RuntimeError('model bad')),
    )

    worker._handle_batch([request])

    assert result_queue.messages == [
        {'id': '1', 'ok': False, 'error': 'model bad'},
    ]


def test_handle_batch_converts_yolo_box_data(monkeypatch: Any) -> None:
    """Exercise this test."""
    worker = yolo_worker.YoloWorker(None)
    result_queue = _ResultQueue()
    request = _request('1', 'cam1', 'yolo26n', result_queue)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    data = np.array([[1, 2, 3, 4, 99, 0.8, 5]], dtype=float)
    model = _Model(data)
    monkeypatch.setattr(
        worker,
        '_read_batch_frames',
        lambda _requests: ([frame], [request]),
    )
    monkeypatch.setattr(worker, '_get_model', lambda _model_key: model)
    monkeypatch.setattr(worker, 'precision_args', {'quantize': 8})

    worker._handle_batch([request])

    assert result_queue.messages == [
        {'id': '1', 'ok': True, 'detections': [[1, 2, 3, 4, 0.8, 5]]},
    ]
    assert model.calls[0]['batch'] == 1
    assert model.calls[0]['quantize'] == 8


def test_handle_batch_returns_result_through_camera_queue(
    monkeypatch: Any,
) -> None:
    """The worker sends a completed inference to the request's own queue."""
    result_queue = _ResultQueue()
    worker = yolo_worker.YoloWorker(None)
    request = _request('1', 'cam1', 'yolo26n', result_queue)
    model = _Model(np.empty((0, 6), dtype=float))
    monkeypatch.setattr(
        worker,
        '_read_batch_frames',
        lambda _requests: ([np.zeros((2, 2, 3), dtype=np.uint8)], [request]),
    )
    monkeypatch.setattr(worker, '_get_model', lambda _model_key: model)

    worker._handle_batch([request])

    assert result_queue.messages == [
        {'id': '1', 'ok': True, 'detections': []},
    ]


def test_handle_batch_serializes_only_first_engine_inference(
    monkeypatch: Any,
) -> None:
    """The cross-process lock protects one engine's first CUDA allocation."""
    lock = _StartupLock()
    worker = yolo_worker.YoloWorker(None, startup_lock=lock)
    request = _request('1', 'cam1', 'yolo26n')
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    model = _Model(np.empty((0, 6), dtype=float))
    monkeypatch.setattr(
        worker,
        '_read_batch_frames',
        lambda _requests: ([frame], [request]),
    )

    def get_model(model_key: str) -> _Model:
        """Cache the test model like the real loader."""
        worker.model_cache[model_key] = model
        return model

    monkeypatch.setattr(worker, '_get_model', get_model)

    worker._handle_batch([request])
    worker._handle_batch([request])

    assert lock.enter_count == 1


def test_read_batch_frames_records_missing_shared_memory(
    monkeypatch: Any,
) -> None:
    """Exercise this test."""
    worker = yolo_worker.YoloWorker(None)
    result_queue = _ResultQueue()
    request = _request('1', 'cam1', 'yolo26n', result_queue)
    monkeypatch.setattr(
        worker,
        '_read_frame',
        lambda _request: (_ for _ in ()).throw(FileNotFoundError('missing')),
    )

    frames, valid_requests = worker._read_batch_frames([request])

    assert frames == []
    assert valid_requests == []
    assert result_queue.messages == [
        {'id': '1', 'ok': False, 'error': 'missing'},
    ]


def test_read_batch_frames_records_read_error(monkeypatch: Any) -> None:
    """Exercise this test."""
    worker = yolo_worker.YoloWorker(None)
    result_queue = _ResultQueue()
    request = _request('1', 'cam1', 'yolo26n', result_queue)
    monkeypatch.setattr(
        worker,
        '_read_frame',
        lambda _request: (_ for _ in ()).throw(ValueError('bad frame')),
    )

    frames, valid_requests = worker._read_batch_frames([request])

    assert frames == []
    assert valid_requests == []
    assert result_queue.messages == [
        {'id': '1', 'ok': False, 'error': 'bad frame'},
    ]


def test_read_batch_frames_returns_valid_frame(monkeypatch: Any) -> None:
    """Exercise this test."""
    frame = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    worker = yolo_worker.YoloWorker(None)
    request = _request('1', 'cam1', 'yolo26n')
    monkeypatch.setattr(
        worker,
        '_read_frame',
        lambda _request: frame,
    )

    frames, valid_requests = worker._read_batch_frames([request])

    assert frames == [frame]
    assert valid_requests == [request]


def test_read_frame_maps_shared_memory() -> None:
    """The mapped frame remains valid until the worker closes it."""
    frame = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    shm = yolo_worker.shared_memory.SharedMemory(
        create=True,
        size=frame.nbytes,
    )
    try:
        shared = np.ndarray(frame.shape, dtype=frame.dtype, buffer=shm.buf)
        shared[:] = frame
        request = yolo_worker._WorkerRequest(
            id='1',
            camera_id='cam1',
            model_key='yolo26n',
            shm_name=shm.name,
            shape=frame.shape,
            dtype=str(frame.dtype),
            result_queue=_ResultQueue(),
        )

        mapped = yolo_worker.YoloWorker._read_frame(request)
        try:
            assert np.array_equal(mapped, frame)
            shared[:] = 0
            assert np.array_equal(mapped, shared)
        finally:
            yolo_worker.YoloWorker._close_shared_frames([mapped])
    finally:
        shm.close()
        shm.unlink()


def test_get_model_returns_cached_model() -> None:
    """Exercise this test."""
    worker = yolo_worker.YoloWorker(None)
    model = _Model(np.empty((0, 6), dtype=float))
    worker.model_cache['yolo26n'] = model

    assert worker._get_model('yolo26n') is model


def test_get_model_raises_when_model_missing(tmp_path: Any) -> None:
    """Exercise this test."""
    worker = yolo_worker.YoloWorker(None)
    worker.model_dir = tmp_path

    with pytest.raises(FileNotFoundError, match='not found'):
        worker._get_model('missing')


def test_get_model_loads_and_caches_model(
    monkeypatch: Any, tmp_path: Any,
) -> None:
    """Exercise this test."""
    loaded_paths: list[str] = []
    module = ModuleType('ultralytics')

    class FakeYOLO:
        """Tests for FakeYOLO."""

        def __init__(self, path: str, task: str) -> None:
            """Support __init__."""
            loaded_paths.append(path)
            assert task == 'detect'

    setattr(module, 'YOLO', FakeYOLO)
    monkeypatch.setitem(sys.modules, 'ultralytics', module)
    worker = yolo_worker.YoloWorker(None)
    worker.model_dir = tmp_path
    model_path = tmp_path / f'best_yolo26n{worker.model_suffix}'
    model_path.write_text('model')

    model = worker._get_model('yolo26n')

    assert isinstance(model, FakeYOLO)
    assert loaded_paths == [str(model_path)]
    assert worker.model_cache['yolo26n'] is model
