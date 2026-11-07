from __future__ import annotations

import asyncio
import threading
import unittest
from functools import partial
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock
from unittest.mock import patch

import cv2
import numpy as np
from fastapi import HTTPException

from examples.YOLO_server_api import inference_runtime as module
from examples.YOLO_server_api.detection import DetectionPolicy
from examples.YOLO_server_api.detection import process_labels
from examples.YOLO_server_api.models import DetectionModelManager
from src.settings.inference import ApiInferenceSettings


def _setUp_make_simplenamespace(data, limit):
    return SimpleNamespace(
        decoded_bytes=3,
        decode=partial(_setUp_make_simplenamespace2, data),
    )


def _setUp_make_simplenamespace2(data, /):
    return SimpleNamespace(shape=(1, 1, 3), tag=data)


def _real_image_decode_and_prediction_predict(
    _capture_testcase,
    worker,
    /,
    **kwargs,
):
    _capture_testcase.assertNotEqual(threading.get_ident(), worker)
    return [SimpleNamespace(object_prediction_list=[])]


def _identity(e):
    return e


def _shutdown_cancels_model_waiters_and_predict(
    entered,
    release,
    /,
    *args,
):
    entered.set()
    release.wait(2)
    return [], {}


def _cancelled_shutdown_finishes_native_cleanup_predict(
    entered,
    release,
    /,
    *args,
):
    entered.set()
    release.wait(2)
    return [], {}


class _CancelledNativeJobKeepsCapacityUntilItCallbacks:
    entered: Any
    loop_thread: Any
    release: Any
    testcase: InferenceRuntimeTest

    def _call_predict(self, *args):
        self.testcase.assertNotEqual(
            threading.get_ident(),
            self.loop_thread,
        )
        self.entered.set()
        self.testcase.assertTrue(self.release.wait(2))
        return [], {'inference': 0, 'post': 0}


class _QueuedCancellationSkipsInferenceAndReleasesBudgetCallbacks:
    calls: Any
    entered: Any
    release: Any

    def _call_predict(self, data, *args):
        self.calls.append(data.tag)
        self.entered.set()
        self.release.wait(2)
        return [], {'inference': 0, 'post': 0}


class _WaitersDoNotBlockOtherModelsAndCallbacks:
    calls: Any
    entered: Any
    other_entered: Any
    release: Any
    testcase: InferenceRuntimeTest

    def _call_predict(self, data, model, policy):
        self.calls.append(data.tag)
        if model.id == 'A':
            self.entered.set()
            self.testcase.assertTrue(self.release.wait(3))
        else:
            self.other_entered.set()
        return [], {}


def selection(name='A', classes=()):
    return SimpleNamespace(
        id=name,
        version='v1',
        sha256='a' * 64,
        classes=classes,
    )


class InferenceRuntimeTest(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.decode = patch.object(
            module,
            'prepare_image',
            side_effect=_setUp_make_simplenamespace,
        )
        self.decode.start()
        self.addCleanup(self.decode.stop)

    def runtime(self, capacity=2):
        models = DetectionModelManager(1)
        models._load_registry_model = Mock(return_value=Mock())
        runtime = module.InferenceRuntime(
            models,
            ApiInferenceSettings(concurrency=1, queue_capacity=capacity),
        )
        self.addAsyncCleanup(runtime.close)
        return runtime

    async def test_cancelled_native_job_keeps_capacity_until_it_finishes(self):
        _callbacks = _CancelledNativeJobKeepsCapacityUntilItCallbacks()
        _callbacks.testcase = self
        runtime = self.runtime(1)
        _callbacks.entered, _callbacks.release = (
            threading.Event(),
            threading.Event(),
        )
        _callbacks.loop_thread = threading.get_ident()

        predict = _callbacks._call_predict

        with patch.object(module, 'run_detection', predict):
            task = asyncio.create_task(runtime.infer(selection(), b'image'))
            self.assertTrue(
                await asyncio.to_thread(_callbacks.entered.wait, 1),
            )
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            with self.assertRaises(HTTPException) as caught:
                await runtime.infer(selection(), b'image')
            self.assertEqual(caught.exception.status_code, 503)
            _callbacks.release.set()
            for _ in range(100):
                if runtime._admitted == 0:
                    break
                await asyncio.sleep(0.005)
            self.assertEqual(runtime._admitted, 0)
            await runtime.infer(selection(), b'image')

    async def test_queued_cancellation_skips_inference_and_releases_budget(
        self,
    ):
        _callbacks = (
            _QueuedCancellationSkipsInferenceAndReleasesBudgetCallbacks()
        )
        runtime = self.runtime()
        _callbacks.entered, _callbacks.release = (
            threading.Event(),
            threading.Event(),
        )
        _callbacks.calls = []

        predict = _callbacks._call_predict

        with patch.object(module, 'run_detection', predict):
            running = asyncio.create_task(runtime.infer(selection(), b'first'))
            await asyncio.to_thread(_callbacks.entered.wait, 1)
            queued = asyncio.create_task(runtime.infer(selection(), b'queued'))
            await asyncio.sleep(0.01)
            queued.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await queued
            _callbacks.release.set()
            await running
        self.assertEqual(_callbacks.calls, [b'first'])
        self.assertEqual(runtime._admitted, 0)

    async def test_model_selection_does_not_pin_an_idle_websocket_model(self):
        runtime = self.runtime()
        first = await runtime.prepare(selection('A'))
        await runtime.prepare(selection('B'))
        self.assertEqual([key[0] for key in runtime.models._slots], ['B'])
        with patch.object(
            module,
            'run_detection',
            return_value=([], {}),
        ):
            await first.detect(b'image')
        self.assertEqual([key[0] for key in runtime.models._slots], ['A'])

    async def test_shared_http_and_websocket_admission_fails_before_reading(
        self,
    ):
        runtime = self.runtime(1)
        async with runtime.admission():
            with self.assertRaises(HTTPException):
                await runtime.prepare(selection())
        self.assertEqual(runtime._admitted, 0)
        await runtime.close()
        with self.assertRaises(HTTPException):
            await runtime.infer(selection(), b'image')

    async def test_real_image_decode_and_prediction_run_off_loop(self):
        self.decode.stop()
        runtime = self.runtime()
        worker = threading.get_ident()
        model = Mock()

        predict = partial(
            _real_image_decode_and_prediction_predict,
            self,
            worker,
        )

        model.predict.side_effect = predict
        runtime.models._load_registry_model.return_value = model
        image = cv2.imencode('.png', np.zeros((10, 10, 3), np.uint8))[
            1
        ].tobytes()
        with patch(
            'examples.YOLO_server_api.detection.USE_SAHI',
            False,
        ), patch('examples.YOLO_server_api.detection.USE_TENSORRT', True):
            result = await runtime.infer(selection(), image)
            data, timing = result.rows, result.timing
        self.assertEqual(data, [])
        self.assertEqual(set(timing), {'inference', 'post'})

    async def test_custom_class_ids_only_filter_semantic_conflicts(self):
        classes = [
            SimpleNamespace(id=0, code='person'),
            SimpleNamespace(id=2, code='helmet'),
        ]
        policy = DetectionPolicy.from_classes(classes)
        data = [[0, 0, 100, 200, 0.9, 0], [10, 5, 30, 30, 0.9, 2]]
        self.assertEqual(process_labels(data, policy), data)
        classes = [
            SimpleNamespace(id=42, code='helmet'),
            SimpleNamespace(id=99, code='no_helmet'),
        ]
        policy = DetectionPolicy.from_classes(classes)
        data = [[0, 0, 50, 50, 0.9, 42], [0, 0, 50, 50, 0.9, 99]]
        self.assertEqual(process_labels(data, policy), [data[0]])

    async def test_waiters_do_not_block_other_models_and_cancel_immediately(
        self,
    ):
        _callbacks = _WaitersDoNotBlockOtherModelsAndCallbacks()
        _callbacks.testcase = self
        models = DetectionModelManager(2)
        models._load_registry_model = Mock(side_effect=_identity)
        runtime = module.InferenceRuntime(
            models,
            ApiInferenceSettings(concurrency=2, queue_capacity=3),
        )
        _callbacks.entered, _callbacks.release, _callbacks.other_entered = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        tasks = []
        _callbacks.calls = []

        predict = _callbacks._call_predict

        try:
            with patch.object(module, 'run_detection', predict):
                first = asyncio.create_task(
                    runtime.infer(selection('A'), b'first'),
                )
                tasks.append(first)
                self.assertTrue(
                    await asyncio.to_thread(_callbacks.entered.wait, 1),
                )
                queued = asyncio.create_task(
                    runtime.infer(selection('A'), b'queued'),
                )
                tasks.append(queued)
                await asyncio.sleep(0.01)
                self.assertEqual(runtime._admitted, 2)
                queued.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await queued
                self.assertEqual(runtime._admitted, 1)
                other = asyncio.create_task(
                    runtime.infer(selection('B'), b'other'),
                )
                tasks.append(other)
                self.assertTrue(
                    await asyncio.to_thread(_callbacks.other_entered.wait, 1),
                )
                await other
                self.assertNotIn(b'queued', _callbacks.calls)
                _callbacks.release.set()
                await first
            self.assertFalse(runtime._gates)
            self.assertEqual(runtime._admitted, 0)
        finally:
            _callbacks.release.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            await runtime.close()

    async def test_shutdown_cancels_model_waiters_and_joins_native_work(self):
        runtime = self.runtime()
        entered, release = threading.Event(), threading.Event()

        predict = partial(
            _shutdown_cancels_model_waiters_and_predict,
            entered,
            release,
        )

        with patch.object(module, 'run_detection', predict):
            first = asyncio.create_task(runtime.infer(selection(), b'first'))
            await asyncio.to_thread(entered.wait, 1)
            waiter = asyncio.create_task(runtime.infer(selection(), b'waiter'))
            await asyncio.sleep(0.01)
            closing = asyncio.create_task(runtime.close())
            with self.assertRaises(asyncio.CancelledError):
                await waiter
            self.assertFalse(closing.done())
            release.set()
            await first
            await closing
        self.assertEqual(runtime._admitted, 0)
        self.assertFalse(runtime._gates)

    async def test_cancelled_shutdown_finishes_native_cleanup(
        self,
    ):
        runtime = self.runtime()
        entered, release = threading.Event(), threading.Event()

        predict = partial(
            _cancelled_shutdown_finishes_native_cleanup_predict,
            entered,
            release,
        )

        tasks = []
        try:
            with patch.object(module, 'run_detection', predict):
                first = asyncio.create_task(
                    runtime.infer(selection(), b'first'),
                )
                tasks.append(first)
                self.assertTrue(await asyncio.to_thread(entered.wait, 1))
                waiter = asyncio.create_task(
                    runtime.infer(selection(), b'waiter'),
                )
                tasks.append(waiter)
                await asyncio.sleep(0.01)
                closing = asyncio.create_task(runtime.close())
                tasks.append(closing)
                await asyncio.sleep(0)
                closing.cancel()
                await asyncio.sleep(0.01)
                self.assertFalse(closing.done())
                release.set()
                await first
                with self.assertRaises(asyncio.CancelledError):
                    await closing
            self.assertTrue(runtime.models._closed)
            self.assertEqual(runtime._admitted, 0)
            self.assertFalse(runtime._gates)
        finally:
            release.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            await runtime.close()
