from __future__ import annotations

import hashlib
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock
from unittest.mock import patch

from fastapi import HTTPException

from examples.YOLO_server_api import models


def _manager_make_simplenamespace(value):
    return SimpleNamespace(name=value.id)


def _concurrent_requests_share_one_load_make_request(request, /, _):
    return request()


class _ConcurrentRequestsShareOneLoadAndModelCallbacks:
    active: Any
    barrier: Any
    guard: Any
    manager: Any
    peak: Any

    def _call_request(self):

        self.barrier.wait()
        with self.manager.lease(entry()) as model:
            with self.guard:
                self.active += 1
                self.peak = max(self.peak, self.active)
            threading.Event().wait(0.01)
            with self.guard:
                self.active -= 1
            return model


class _WaiterRechecksAModelEvictedBeforeItCallbacks:
    key: Any
    manager: Any
    original: Any

    def _call_evicted_before_wakeup(self, *args):
        self.original.loading = False
        self.original.users = 0
        self.manager._slots.pop(self.key)


class _PinnedCapacityWaitsWithoutOverallocatingCallbacks:
    acquired: Any
    manager: Any
    started: Any

    def _call_other(self):
        self.started.set()
        with self.manager.lease(entry('B')):
            self.acquired.set()


def entry(name='A'):
    return SimpleNamespace(id=name, version='v1', sha256='a' * 64, classes=[])


class ModelManagerTest(unittest.TestCase):
    def manager(self, capacity=2):
        manager = models.DetectionModelManager(capacity)
        manager._load_registry_model = Mock(
            side_effect=_manager_make_simplenamespace,
        )
        self.addCleanup(manager.close)
        return manager

    def test_hits_update_lru_and_only_idle_models_can_be_evicted(self):
        manager = self.manager()
        for name in ('A', 'B', 'A', 'C'):
            with manager.lease(entry(name)) as model:
                self.assertEqual(model.name, name)
        self.assertEqual([key[0] for key in manager._slots], ['A', 'C'])
        with manager.lease(entry('A')):
            with manager.lease(entry('B')):
                self.assertEqual(
                    [key[0] for key in manager._slots],
                    ['A', 'B'],
                )
        self.assertEqual(manager._load_registry_model.call_count, 4)

    def test_concurrent_requests_share_one_load_and_model_execution(self):
        _callbacks = _ConcurrentRequestsShareOneLoadAndModelCallbacks()
        _callbacks.manager = self.manager()
        _callbacks.active = 0
        _callbacks.peak = 0
        _callbacks.guard = threading.Lock()
        _callbacks.barrier = threading.Barrier(6)

        request = _callbacks._call_request

        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(
                pool.map(
                    partial(
                        _concurrent_requests_share_one_load_make_request,
                        request,
                    ),
                    range(6),
                ),
            )
        _callbacks.manager._load_registry_model.assert_called_once()
        self.assertTrue(all(model is results[0] for model in results))
        self.assertEqual(_callbacks.peak, 1)

    def test_waiter_rechecks_a_model_evicted_before_it_wakes(self):
        _callbacks = _WaiterRechecksAModelEvictedBeforeItCallbacks()
        _callbacks.manager = self.manager(1)
        _callbacks.key = ('A', 'v1', 'a' * 64)
        _callbacks.original = models.ModelSlot(users=1)
        _callbacks.manager._slots[_callbacks.key] = _callbacks.original

        evicted_before_wakeup = _callbacks._call_evicted_before_wakeup

        with patch.object(
            _callbacks.manager._condition,
            'wait',
            side_effect=evicted_before_wakeup,
        ):
            with _callbacks.manager.lease(entry()) as model:
                self.assertEqual(model.name, 'A')
        _callbacks.manager._load_registry_model.assert_called_once()

    def test_pinned_capacity_waits_without_overallocating(self):
        _callbacks = _PinnedCapacityWaitsWithoutOverallocatingCallbacks()
        _callbacks.manager = self.manager(1)
        _callbacks.started = threading.Event()
        _callbacks.acquired = threading.Event()

        other = _callbacks._call_other

        with ThreadPoolExecutor(max_workers=1) as pool:
            with _callbacks.manager.lease(entry('A')):
                task = pool.submit(other)
                self.assertTrue(_callbacks.started.wait(1))
                self.assertFalse(_callbacks.acquired.wait(0.03))
                self.assertEqual(len(_callbacks.manager._slots), 1)
            task.result(timeout=1)
        self.assertTrue(_callbacks.acquired.is_set())

    def test_failed_load_releases_reservation_for_retry(self):
        manager = self.manager(1)
        manager._load_registry_model.side_effect = [
            ValueError('bad'),
            object(),
        ]
        with self.assertRaises(ValueError):
            with manager.lease(entry()):
                pass
        self.assertFalse(manager._slots)
        with manager.lease(entry()):
            pass
        manager.close()
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            with manager.lease(entry()):
                pass
        with self.assertRaises(ValueError):
            models.DetectionModelManager(0)

    def test_cancelled_request_cannot_acquire_model(self):
        manager = self.manager()
        cancelled = threading.Event()
        cancelled.set()
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            with manager.lease(entry(), cancelled):
                pass
        manager._load_registry_model.assert_not_called()

    def test_artifact_copy_is_verified_and_released_on_close(self):
        with TemporaryDirectory() as root:
            path = Path(root) / 'model.pt'
            path.write_bytes(b'weights')
            selected = entry()
            selected.sha256 = hashlib.sha256(b'weights').hexdigest()
            selected.artifact = str(path)
            manager = models.DetectionModelManager()
            with patch.object(
                models,
                'artifact_path',
                return_value=path,
            ), patch.object(models, 'USE_SAHI', False), patch.object(
                models,
                'YOLO',
                return_value=Mock(),
            ):
                with manager.lease(selected) as model:
                    copied = Path(model._registry_artifact_directory.name)
                    self.assertEqual(
                        (copied / 'model.pt').read_bytes(),
                        b'weights',
                    )
                manager.close()
                self.assertFalse(copied.exists())
            selected.sha256 = '0' * 64
            manager = models.DetectionModelManager()
            with patch.object(models, 'artifact_path', return_value=path):
                with self.assertRaises(HTTPException):
                    with manager.lease(selected):
                        pass
            manager.close()

    def test_sahi_artifacts_use_the_same_registry_loader(self):
        with TemporaryDirectory() as root:
            path = Path(root) / 'model.pt'
            path.write_bytes(b'weights')
            selected = entry()
            selected.sha256 = hashlib.sha256(b'weights').hexdigest()
            manager = models.DetectionModelManager()
            with patch.object(
                models,
                'artifact_path',
                return_value=path,
            ), patch.object(models, 'USE_SAHI', True), patch.object(
                models,
                'AutoDetectionModel',
            ) as factory:
                with manager.lease(selected):
                    factory.from_pretrained.assert_called_once()
            manager.close()
