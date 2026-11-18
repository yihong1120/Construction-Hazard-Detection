"""Exercise real shared-memory leases across asynchronous cancellation."""
from __future__ import annotations

import asyncio
import queue
import threading

import numpy as np
import pytest

from src.yolo_worker import YoloWorker
from src.yolo_worker import YoloWorkerClient
from src.yolo_worker_protocol import _WorkerRequest


class DelayedQueue:
    def __init__(self):
        self.started = threading.Event()
        self.finish = threading.Event()
        self.requests = []

    def put(self, request, block, timeout):
        self.started.set()
        if not self.finish.wait(5):
            raise queue.Full
        self.requests.append(request)


async def cancel_during_enqueue():
    requests = DelayedQueue()
    client = YoloWorkerClient(requests, queue.Queue(), 'camera', 1)
    client.ring_cleanup_delay_seconds = 0
    task = asyncio.create_task(
        client.detect(np.full((12, 16, 3), 7, dtype=np.uint8), 'fixture'),
    )
    try:
        async with asyncio.timeout(2):
            while not requests.started.is_set():
                await asyncio.sleep(.001)
        old_ring = client._ring
        request_id = next(iter(client._inflight_slots))
        old_slot = client._inflight_slots[request_id][1]
        frame = YoloWorker._read_frame(
            _WorkerRequest.from_mapping({
                'id': request_id, 'camera_id': 'camera',
                'model_key': 'fixture',
                'shm_name': old_ring.name, 'slot': old_slot,
                'shape': old_ring.shape, 'dtype': 'uint8',
            }),
        )
        try:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            new_ring, _, _ = await client._write_frame(
                'next', np.full((12, 16, 3), 99, dtype=np.uint8),
            )
            assert new_ring.name != old_ring.name
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert request_id not in client._inflight_slots
            np.testing.assert_array_equal(frame, 7)
            requests.finish.set()
            async with asyncio.timeout(2):
                while not requests.requests:
                    await asyncio.sleep(.001)
            assert requests.requests[0]['shm_name'] == old_ring.name
            np.testing.assert_array_equal(frame, 7)
        finally:
            YoloWorker._close_shared_frames([frame])
    finally:
        requests.finish.set()
        await client.close()


def test_cancellation_during_enqueue_never_reuses_reader_pixels():
    asyncio.run(cancel_during_enqueue())


async def acquire_from_retired_ring(close):
    client = YoloWorkerClient(queue.Queue(), queue.Queue(), 'camera', 1)
    client.ring_slots = 1
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    old_ring, _, _ = await client._write_frame('first', frame)
    waiter = asyncio.create_task(client._write_frame('second', frame + 1))
    try:
        await asyncio.sleep(0)
        if close:
            await client.close()
            old_ring.release_slot(0)
            with pytest.raises(RuntimeError, match='closed'):
                await waiter
        else:
            client._retire_ring(old_ring)
            client._release_slot('first')
            new_ring, _, _ = await waiter
            assert old_ring.name != new_ring.name
            assert client._inflight_slots['second'][0] is new_ring
    finally:
        await client.close()


@pytest.mark.parametrize('close', [False, True])
def test_waiting_writer_rechecks_ring_lease_after_acquisition(close):
    asyncio.run(acquire_from_retired_ring(close))


async def replace_idle_ring():
    client = YoloWorkerClient(queue.Queue(), queue.Queue(), 'camera', 1)
    ring, _, _ = await client._write_frame(
        'first', np.zeros((2, 3, 3), dtype=np.uint8),
    )
    client._release_slot('first')
    try:
        await client._write_frame(
            'second', np.zeros((3, 4, 3), dtype=np.uint8),
        )
        assert not client._retired_rings
        assert ring.shared_memory_handle.buf is None
    finally:
        await client.close()


def test_shape_changes_reclaim_idle_ring_immediately():
    asyncio.run(replace_idle_ring())
