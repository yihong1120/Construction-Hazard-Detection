from __future__ import annotations

import logging
from functools import partial
from typing import Any
from unittest.mock import MagicMock

import pytest

from src import worker_supervisor as module
from src.stream_identity import stream_identity

pytestmark = pytest.mark.ml


def _make_magicmock(**kw):
    return MagicMock()


def _revision_change_replaces_only_affected_process(**kwargs):
    instance = MagicMock()
    instance.is_alive.return_value = True
    instance.terminate.side_effect = partial(
        _revision_change_replaces_only_affected_make_setattr,
        instance,
    )
    return instance


def _revision_change_replaces_only_affected_make_setattr(instance, /):
    return setattr(
        instance.is_alive,
        'return_value',
        False,
    )


def _make_magicmock2(*args):
    return MagicMock()


def _cancelled_replacement_preserves_resource_ownership_make_child(
    Child,
    /,
    *args,
):
    return Child()


class _CancelledReplacementPreservesResourceOwnershipCallbacks:
    asyncio: Any
    configs: Any
    entered: Any
    old: Any
    other: Any
    release: Any
    supervisor: Any

    async def _call_run(self):
        task = self.asyncio.create_task(
            self.supervisor.reconcile(self.configs),
        )
        try:
            assert await self.asyncio.to_thread(self.entered.wait, 1)
            task.cancel()
            await self.asyncio.sleep(0)
            assert not task.done()
            self.old.request_queue.close.assert_not_called()
            self.release.set()
            with pytest.raises(self.asyncio.CancelledError):
                await task
            self.old.request_queue.close.assert_called_once()
            assert self.supervisor.shards[1] is self.other
            assert not self.supervisor._retired
            await self.supervisor.reconcile(self.configs)
            assert self.supervisor.shards[1] is self.other
        finally:
            self.release.set()
            await self.supervisor.close_async()


class _CancelledReplacementPreservesResourceOwnershipChild:
    def __init__(self, *, _test_entered, _test_release):
        self._test_entered = _test_entered
        self._test_release = _test_release
        self.alive = True
        self.block = False

    def is_alive(self):
        return self.alive

    def terminate(self):
        pass

    def join(self, timeout):
        if self.block:
            self._test_entered.set()
            assert self._test_release.wait(2)
        self.alive = False

    def kill(self):
        self.alive = False


def cameras():
    return [
        {
            'stream_id': index,
            'site': 'site',
            'stream_name': f'camera-{index}',
            'video_url': 'rtsp://shared-source',
            'model_key': f'model-{index}',
            'model_revision': 'old',
        }
        for index in (
            1,
            2,
        )
    ]


def test_shared_url_and_renames_keep_distinct_durable_worker_slots(
    monkeypatch,
):
    monkeypatch.setenv('YOLO_WORKER_CAMERAS_PER_ENGINE', '1')
    supervisor = module.WorkerSupervisor(logging.getLogger(__name__))
    configs = cameras()
    count, slots, _ = supervisor.topology(configs)
    assert count == 2
    assert slots == {'id:1': 0, 'id:2': 1}
    configs[0]['site'] = 'renamed-site'
    configs[0]['stream_name'] = 'renamed-camera'
    assert stream_identity(configs[0]) == 'id:1'
    assert supervisor.topology(configs)[1] == slots


def test_duplicate_stream_identity_is_rejected():
    supervisor = module.WorkerSupervisor(logging.getLogger(__name__))
    first, second = cameras()
    second['stream_id'] = first['stream_id']
    with pytest.raises(ValueError, match='Duplicate stream identity'):
        supervisor.topology([first, second])


def test_revision_change_replaces_only_affected_shard_and_queues(monkeypatch):
    monkeypatch.setenv('YOLO_WORKER_CAMERAS_PER_ENGINE', '1')
    monkeypatch.setattr(module, 'YoloWorker', MagicMock())
    monkeypatch.setattr(module.multiprocessing, 'Lock', MagicMock())
    monkeypatch.setattr(
        module.multiprocessing,
        'Queue',
        MagicMock(side_effect=_make_magicmock),
    )

    process = _revision_change_replaces_only_affected_process

    monkeypatch.setattr(module, 'Process', process)
    supervisor = module.WorkerSupervisor(logging.getLogger(__name__))
    configs = cameras()
    assert supervisor.ensure(configs)
    original = list(supervisor.processes)
    requests = list(supervisor.request_queues)
    results = dict(supervisor.result_queues)
    configs[0]['model_revision'] = 'new'
    assert supervisor.ensure(configs)
    assert supervisor.restarted_cameras == {'id:1'}
    assert supervisor.processes[1] is original[1]
    original[1].terminate.assert_not_called()
    original[0].terminate.assert_called_once()
    assert supervisor.request_queues[0] is not requests[0]
    assert supervisor.request_queues[1] is requests[1]
    assert supervisor.result_queues['id:1'] is not results['id:1']
    assert supervisor.result_queues['id:2'] is results['id:2']
    assert supervisor.slot(configs[0]) == (
        supervisor.request_queues[0],
        supervisor.result_queues['id:1'],
    )
    assert not supervisor.ensure(configs)
    supervisor.close()


def test_membership_and_url_changes_preserve_other_shards(monkeypatch):
    monkeypatch.setenv('YOLO_WORKER_CAMERAS_PER_ENGINE', '1')
    monkeypatch.setattr(
        module.multiprocessing,
        'Queue',
        _make_magicmock,
    )
    monkeypatch.setattr(module.multiprocessing, 'Lock', MagicMock())
    supervisor = module.WorkerSupervisor(logging.getLogger(__name__))
    monkeypatch.setattr(supervisor, 'start_worker', _make_magicmock2)
    configs = cameras()
    assert supervisor.ensure(configs)
    before = dict(supervisor.shards)
    configs[0]['video_url'] = 'rtsp://replacement-source'
    assert not supervisor.ensure(configs)
    assert supervisor.shards == before
    third = {**configs[0], 'stream_id': 3, 'model_key': 'aaa-first-in-sort'}
    assert supervisor.ensure([third, *configs])
    assert supervisor.restarted_cameras == {'id:3'}
    for slot, shard in before.items():
        assert supervisor.shards[slot] is shard
    assert supervisor.ensure([configs[1], third])
    assert supervisor.restarted_cameras == {'id:1'}
    assert supervisor.result_queues['id:2'] is before[1].result_queues['id:2']
    supervisor.close()


def test_cancelled_replacement_preserves_resource_ownership(monkeypatch):
    _callbacks = _CancelledReplacementPreservesResourceOwnershipCallbacks()
    import asyncio

    _callbacks.asyncio = asyncio
    import threading

    monkeypatch.setenv('YOLO_WORKER_CAMERAS_PER_ENGINE', '1')
    monkeypatch.setattr(
        module.multiprocessing,
        'Queue',
        _make_magicmock,
    )
    monkeypatch.setattr(module.multiprocessing, 'Lock', MagicMock())
    _callbacks.entered, _callbacks.release = (
        threading.Event(),
        threading.Event(),
    )

    Child = partial(
        _CancelledReplacementPreservesResourceOwnershipChild,
        _test_entered=_callbacks.entered,
        _test_release=_callbacks.release,
    )

    _callbacks.supervisor = module.WorkerSupervisor(
        logging.getLogger(__name__),
    )
    monkeypatch.setattr(
        _callbacks.supervisor,
        'start_worker',
        partial(
            _cancelled_replacement_preserves_resource_ownership_make_child,
            Child,
        ),
    )
    _callbacks.configs = cameras()
    _callbacks.supervisor.ensure(_callbacks.configs)
    _callbacks.old = _callbacks.supervisor.shards[0]
    _callbacks.other = _callbacks.supervisor.shards[1]
    _callbacks.old.process.block = True
    _callbacks.configs[0]['model_revision'] = 'new'

    run = _callbacks._call_run

    _callbacks.asyncio.run(run())
