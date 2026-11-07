from __future__ import annotations

import logging
import multiprocessing
import unittest
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

from src.process_startup_lock import ProcessStartupLock


class _KilledReaderGetsNewIpcAndMarksCallbacks:
    context: Any
    received: Any
    replacement: Any

    def _call_start(self, index, channel, device):

        self.replacement = self.context.Process(
            target=consume_replacement,
            args=(channel, self.received),
        )
        self.replacement.start()
        return self.replacement


def hold_startup(lock, entered):
    with lock:
        entered.set()
        import time

        time.sleep(10)


def use_startup(lock, entered):
    with lock:
        entered.set()


def abandon_reader(channel, entered):
    channel._rlock.acquire()
    entered.set()
    import time

    time.sleep(10)


def consume_replacement(channel, received):
    if channel.get(timeout=3) == 'new generation':
        received.set()


class WorkerGenerationTest(unittest.TestCase):
    def test_killed_loader_does_not_abandon_the_startup_lock(self):
        context = multiprocessing.get_context('spawn')
        lock = ProcessStartupLock()
        entered = context.Event()
        old = context.Process(target=hold_startup, args=(lock, entered))
        replacement = None
        try:
            old.start()
            self.assertTrue(entered.wait(5))
            old.terminate()
            old.join(3)
            self.assertFalse(old.is_alive())
            ready = context.Event()
            replacement = context.Process(
                target=use_startup,
                args=(lock, ready),
            )
            replacement.start()
            self.assertTrue(ready.wait(5))
            replacement.join(3)
            self.assertEqual(replacement.exitcode, 0)
        finally:
            for process in (old, replacement):
                if process is not None and process.is_alive():
                    process.kill()
                    process.join(3)
            lock.close()
        self.assertFalse(lock.path.exists())

    def test_killed_reader_gets_new_ipc_and_marks_only_affected_cameras(self):
        _callbacks = _KilledReaderGetsNewIpcAndMarksCallbacks()
        from src import worker_supervisor as module
        from src.settings.inference import WorkerPoolSettings

        _callbacks.context = multiprocessing.get_context('spawn')
        old_queue = _callbacks.context.Queue()
        old_results = _callbacks.context.Queue()
        entered = _callbacks.context.Event()
        _callbacks.received = _callbacks.context.Event()
        dead = _callbacks.context.Process(
            target=abandon_reader,
            args=(old_queue, entered),
        )
        _callbacks.replacement = None
        supervisor = module.WorkerSupervisor(logging.getLogger(__name__))
        supervisor.settings = WorkerPoolSettings()
        spec = module.ShardSpec(0, 'model', 'v1', ('id:1',), 1, 'cpu')
        other_spec = module.ShardSpec(1, 'other', 'v1', ('id:2',), 1, 'cpu')
        healthy = module.WorkerShard(
            other_spec,
            MagicMock(),
            {'id:2': MagicMock()},
            MagicMock(),
        )
        healthy.process.is_alive.return_value = True
        try:
            dead.start()
            self.assertTrue(entered.wait(5))
            dead.terminate()
            dead.join(3)
            supervisor.shards = {
                0: module.WorkerShard(
                    spec,
                    old_queue,
                    {'id:1': old_results},
                    dead,
                ),
                1: healthy,
            }

            start = _callbacks._call_start

            with patch.object(
                module.multiprocessing,
                'Queue',
                _callbacks.context.Queue,
            ), patch.object(supervisor, 'start_worker', start):
                supervisor.restart_dead()
            fresh = supervisor.shards[0]
            self.assertEqual(fresh.generation, 2)
            self.assertIsNot(fresh.request_queue, old_queue)
            self.assertIs(supervisor.shards[1], healthy)
            self.assertEqual(supervisor.restarted_cameras, {'id:1'})
            self.assertIs(
                supervisor.slot({'stream_id': 1})[0],
                fresh.request_queue,
            )
            fresh.request_queue.put('new generation')
            self.assertTrue(_callbacks.received.wait(5))
            _callbacks.replacement.join(3)
            self.assertEqual(_callbacks.replacement.exitcode, 0)
        finally:
            for process in (dead, _callbacks.replacement):
                if process is not None and process.is_alive():
                    process.kill()
                    process.join(3)
            for shard in supervisor.shards.values():
                supervisor._close_queues(shard)
            old_queue.cancel_join_thread()
            old_queue.close()
            old_results.cancel_join_thread()
            old_results.close()
