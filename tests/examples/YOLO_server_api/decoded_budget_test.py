from __future__ import annotations

import asyncio
import threading
import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch

import cv2
import numpy as np
from fastapi import HTTPException

from examples.YOLO_server_api import inference_runtime as module
from examples.YOLO_server_api.models import DetectionModelManager
from src.settings.inference import ApiInferenceSettings


def _cancelled_native_work_keeps_its_predict(
    entered,
    release,
    /,
    *args,
):
    entered.set()
    release.wait(2)
    return [], {'inference': 0, 'post': 0}


def entry(name):
    return SimpleNamespace(id=name, version='v1', sha256='a' * 64, classes=())


class DecodedBudgetTest(unittest.IsolatedAsyncioTestCase):
    def runtime(self, budget):
        models = DetectionModelManager(2)
        models._load_registry_model = Mock(return_value=Mock())
        runtime = module.InferenceRuntime(
            models,
            ApiInferenceSettings(
                concurrency=2,
                queue_capacity=4,
                max_decoded_bytes=budget,
            ),
        )
        self.addAsyncCleanup(runtime.close)
        return runtime

    def image(self):
        return cv2.imencode('.png', np.zeros((4, 4, 3), dtype=np.uint8))[
            1
        ].tobytes()

    async def test_cancelled_native_work_keeps_its_decoded_reservation(
        self,
    ):
        runtime = self.runtime(48)
        entered, release = threading.Event(), threading.Event()

        predict = partial(
            _cancelled_native_work_keeps_its_predict,
            entered,
            release,
        )

        with patch.object(module, 'run_detection', predict):
            task = asyncio.create_task(runtime.infer(entry('A'), self.image()))
            self.assertTrue(await asyncio.to_thread(entered.wait, 1))
            self.assertEqual(runtime.decoded.used, 48)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            with self.assertRaises(HTTPException) as caught:
                await runtime.infer(entry('B'), self.image())
            self.assertEqual(caught.exception.status_code, 503)
            self.assertEqual(runtime.models._load_registry_model.call_count, 1)
            release.set()
            for _ in range(100):
                if runtime.decoded.used == 0:
                    break
                await asyncio.sleep(0.005)
            self.assertEqual(runtime.decoded.used, 0)
            await runtime.infer(entry('B'), self.image())

    async def test_impossible_single_image_is_rejected_before_native_decode(
        self,
    ):
        runtime = self.runtime(47)
        from examples.YOLO_server_api import image_input

        with patch.object(image_input.cv2, 'imdecode') as decode:
            with self.assertRaises(HTTPException) as caught:
                await runtime.infer(entry('A'), self.image())
        self.assertEqual(caught.exception.status_code, 413)
        decode.assert_not_called()
        runtime.models._load_registry_model.assert_not_called()
        self.assertEqual(runtime.decoded.used, 0)

    async def test_prediction_failure_releases_the_input_budget(self):
        runtime = self.runtime(48)
        with patch.object(
            module,
            'run_detection',
            side_effect=RuntimeError('native failure'),
        ):
            with self.assertRaises(RuntimeError):
                await runtime.infer(entry('A'), self.image())
        self.assertEqual(runtime.decoded.used, 0)
