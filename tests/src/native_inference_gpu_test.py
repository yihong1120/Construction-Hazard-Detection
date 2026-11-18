"""Opt-in CUDA integration tests; never read a production catalog or camera."""
from __future__ import annotations

import asyncio
import gc
import hashlib
import multiprocessing
import os
from functools import partial
from pathlib import Path

import numpy as np
import pytest

from src.native_inference import create_rust_model
from src.settings.inference import WorkerSettings
from src.yolo_worker import YoloWorker
from src.yolo_worker import YoloWorkerClient
from src.yolo_worker_protocol import YOLO_WORKER_STOP_MESSAGE

ENGINE = os.environ.get('CHD_NATIVE_TEST_ENGINE')
pytestmark = pytest.mark.skipif(not ENGINE, reason='Opt-in native CUDA test')


def _fixture_catalog(catalog):
    return catalog


def _gpu_worker(requests, replies, engine_path, digest, status):
    """An isolated test catalog still uses real artifact SHA256 validation."""
    from examples.YOLO_server_api.model_registry import ModelEntry
    from examples.YOLO_server_api.model_registry import ModelRegistry
    from src import yolo_worker

    entry = ModelEntry(
        id='native-test', display_name='Local CUDA test', version='fixture',
        artifact=engine_path, sha256=digest,
        capabilities=['stream'], classes=[],
    )
    yolo_worker.load_registry = partial(
        _fixture_catalog,
        ModelRegistry(revision='fixture', models=[entry]),
    )
    worker = YoloWorker(requests, result_queues=replies)
    try:
        worker.run()
        import sys
        status.put({
            'torch_loaded': 'torch' in sys.modules,
            'borrowed_frames': worker.native_borrowed_frames,
            'snapshot_bytes': worker.native_snapshot_bytes,
        })
    finally:
        worker.close()


def _image_frames():
    import cv2

    images = sorted(Path('tests/cv_dataset/images').glob('*'))
    frames = [cv2.imread(str(path)) for path in images]
    return [frame for frame in frames if frame is not None]


@pytest.mark.parametrize('batch', [1, 8])
def test_native_engine_matches_formal_square_worker_policy(batch):
    import torch
    from ultralytics import YOLO

    frames = _image_frames()
    settings = WorkerSettings(backend='rust', batch_size=batch)
    native = create_rust_model(
        Path(ENGINE), settings, device='cuda:0', rect=False,
    )
    reference = YOLO(ENGINE, task='detect')
    try:
        for start in range(0, len(frames), batch):
            current = frames[start:start + batch]
            expected = reference.predict(
                current, device='cuda:0', imgsz=640, batch=len(current),
                rect=False, verbose=False,
            )
            actual = native.predict_detections(current)
            for result, rows in zip(expected, actual, strict=True):
                left = result.boxes.data.cpu().numpy()
                right = np.asarray(rows, dtype=np.float32).reshape(-1, 6)
                assert left.shape == right.shape
                np.testing.assert_array_equal(left[:, 5], right[:, 5])
                np.testing.assert_allclose(
                    left[:, :4], right[:, :4], rtol=0, atol=.002,
                )
                np.testing.assert_array_equal(left[:, 4], right[:, 4])
        native.close()
        native.close()
        with pytest.raises(RuntimeError, match='closed'):
            native.predict_detections(frames[:1])
    finally:
        native.close()
        del reference
        gc.collect()
        torch.cuda.empty_cache()


async def _send_camera_frames(clients, frames, rounds):
    try:
        for _ in range(rounds):
            results = await asyncio.gather(
                *[
                    client.detect(frames[index % len(frames)], 'native-test')
                    for index, client in enumerate(clients)
                ],
            )
            assert len(results) == 32
            for client, rows in zip(clients, results, strict=True):
                assert client.last_model_metadata['model_id'] == 'native-test'
                assert all(len(row) == 6 for row in rows)
    finally:
        await asyncio.gather(*[client.close() for client in clients])


@pytest.mark.parametrize('asynchronous', [False, True])
def test_32_cameras_use_real_shared_memory_and_result_queues(
        monkeypatch, asynchronous,
):
    monkeypatch.setenv('YOLO_WORKER_BACKEND', 'rust')
    monkeypatch.setenv('YOLO_WORKER_BATCH_SIZE', '8')
    monkeypatch.setenv('YOLO_WORKER_NATIVE_ASYNC', str(asynchronous))
    monkeypatch.setenv('YOLO_WORKER_NATIVE_MICRO_BATCH_SIZE', '4')
    monkeypatch.setenv('GPU_RESOURCE_INFERENCE_SLOTS', '1')
    monkeypatch.setenv('YOLO_WORKER_PRECISION', 'int8')
    monkeypatch.setenv('YOLO_WORKER_ARTIFACT_MANIFEST', '')
    context = multiprocessing.get_context('spawn')
    requests = context.Queue(maxsize=64)
    replies = {
        f'camera-{index}': context.Queue(maxsize=4) for index in range(32)
    }
    status = context.Queue(maxsize=1)
    path = Path(ENGINE).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    process = context.Process(
        target=_gpu_worker, args=(
            requests, replies, str(path), digest, status,
        ),
    )
    process.start()
    clients = [
        YoloWorkerClient(
            requests, result_queue,
            camera_id=camera, timeout_seconds=60,
        )
        for camera, result_queue in replies.items()
    ]
    try:
        asyncio.run(_send_camera_frames(clients, _image_frames(), rounds=3))
        requests.put(YOLO_WORKER_STOP_MESSAGE)
        result = status.get(timeout=30)
        process.join(timeout=30)
        assert process.exitcode == 0
        assert result['torch_loaded'] is False
        assert result['borrowed_frames'] == 96
        assert result['snapshot_bytes'] == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
        for item in [requests, status, *replies.values()]:
            item.close()
            item.join_thread()


def test_profile_capacity_and_rejection_preserve_live_engine():
    native = create_rust_model(
        Path(ENGINE), WorkerSettings(backend='rust', batch_size=32),
        device='cuda:0', rect=False,
    )
    try:
        assert native.engine.max_batch <= 32
        frame = _image_frames()[0]
        assert len(native.predict_detections([frame] * 17)) == 17
        with pytest.raises(ValueError, match='HWC BGR'):
            native.engine.predict([(2, 2, b'invalid')])
        assert len(native.predict_detections([frame])) == 1
        native.engine.close()
        native.engine.close()
        with pytest.raises(RuntimeError, match='closed'):
            native.engine.predict([(2, 2, bytes(12))])
    finally:
        native.close()


def test_borrowed_buffers_match_snapshots_and_reject_invalid_exports():
    native = create_rust_model(
        Path(ENGINE), WorkerSettings(backend='rust', batch_size=8),
        device='cuda:0', rect=False,
    )
    frame = _image_frames()[0]
    try:
        expected = native.predict_detections([frame])
        with memoryview(frame).toreadonly() as view:
            assert native.engine.predict_buffers([view]) == expected
        with pytest.raises(ValueError, match='read-only contiguous'):
            native.engine.predict_buffers([memoryview(frame)])
        with pytest.raises(ValueError, match='read-only contiguous'):
            native.engine.predict_buffers([
                memoryview(frame[:, ::-1]).toreadonly(),
            ])
        with pytest.raises((ValueError, BufferError)):
            native.engine.predict_buffers([
                memoryview(np.zeros((2, 3, 3), dtype=np.float32)).toreadonly(),
            ])
        with pytest.raises(ValueError, match='HWC BGR'):
            native.engine.predict_buffers([
                memoryview(np.zeros((0, 3, 3), dtype=np.uint8)).toreadonly(),
            ])
        assert native.predict_detections([frame]) == expected
        native.close()
        with memoryview(frame).toreadonly() as view:
            with pytest.raises(RuntimeError, match='closed'):
                native.engine.predict_buffers([view])
    finally:
        native.close()


def test_steady_batch_reuses_host_workspace():
    native = create_rust_model(
        Path(ENGINE), WorkerSettings(backend='rust', batch_size=8),
        device='cuda:0', rect=False,
    )
    try:
        frames = _image_frames()[:8]
        expected = native.predict_detections(frames)
        growths = native.engine.workspace_growths
        reserved = native.engine.workspace_bytes
        assert growths == 2
        assert reserved >= 8 * 640 * 640 * 3
        for _ in range(5):
            assert native.predict_detections(frames) == expected
        assert native.engine.workspace_growths == growths
        assert native.engine.workspace_bytes == reserved
    finally:
        native.close()


def test_async_slots_preserve_snapshots_and_partial_batch_shapes():
    settings = WorkerSettings(
        backend='rust', batch_size=8, native_async=True,
        native_micro_batch_size=4,
    )
    native = create_rust_model(
        Path(ENGINE),
        settings,
        device='cuda:0',
        rect=False,
    )
    frames = _image_frames()
    try:
        reference = native.engine.predict([
            (frame.shape[0], frame.shape[1], frame.tobytes())
            for frame in frames[:4]
        ])
        batch = [(frame.shape[0], frame.shape[1], frame.tobytes())
                 for frame in frames[:4]]
        first = native.engine.submit(batch)
        second = native.engine.submit(batch)
        assert native.engine.pending_count == 2
        assert native.engine.pinned_bytes > 0
        with pytest.raises(ValueError, match='slots'):
            native.engine.submit(batch)
        with pytest.raises(ValueError, match='shape'):
            native.engine.submit(batch[:1])
        assert native.engine.collect(second) == reference
        assert native.engine.collect(first) == reference
        with pytest.raises(ValueError, match='ticket'):
            native.engine.collect(first)
        expected = native.predict_detections([frames[0]] * 4) * 4
        assert native.predict_detections(
            [frames[0]] * 17,
        ) == expected + expected[:1]
        assert native.engine.pending_count == 0
    finally:
        native.close()


def test_async_close_waits_for_pending_work():
    native = create_rust_model(
        Path(ENGINE), WorkerSettings(backend='rust', native_async=True),
        device='cuda:0', rect=False,
    )
    frame = _image_frames()[0]
    native.engine.submit([(frame.shape[0], frame.shape[1], frame.tobytes())])
    native.close()
    native.close()
    with pytest.raises(RuntimeError, match='closed'):
        native.engine.collect(1)
