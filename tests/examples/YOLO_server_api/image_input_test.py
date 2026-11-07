from __future__ import annotations

import unittest
from io import BytesIO
from unittest.mock import Mock
from unittest.mock import patch

from PIL import Image

from examples.YOLO_server_api import image_input
from examples.YOLO_server_api.inference_runtime import InferenceRuntime
from examples.YOLO_server_api.models import DetectionModelManager
from src.settings.inference import ApiInferenceSettings
from tests.examples.YOLO_server_api.inference_runtime_test import selection


def image_bytes(format='PNG'):
    output = BytesIO()
    Image.new('RGB', (4, 3)).save(output, format=format)
    return output.getvalue()


class ImageInputTest(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_input_does_not_load_models_or_install_plugins(
        self,
    ):
        manager = DetectionModelManager(1)
        manager._load_registry_model = Mock()
        runtime = InferenceRuntime(manager, ApiInferenceSettings())
        self.addAsyncCleanup(runtime.close)
        with patch.object(
            Image,
            'open',
            side_effect=AssertionError('ML fallback called'),
        ):
            with self.assertRaises(image_input.DetectionInputError) as caught:
                await runtime.infer(selection(), b'not-an-image')
        self.assertEqual(caught.exception.status_code, 400)
        self.assertEqual(caught.exception.detail, {'code': 'INVALID_IMAGE'})
        manager._load_registry_model.assert_not_called()

    def test_supported_images_keep_their_original_dimensions(self):
        for format in ('PNG', 'JPEG', 'BMP', 'WEBP', 'TIFF'):
            with self.subTest(format=format):
                frame = image_input.decode_image(image_bytes(format), 12)
                self.assertEqual(frame.shape, (3, 4, 3))

    def test_oversize_header_is_rejected_before_native_decode(self):
        with patch.object(image_input.cv2, 'imdecode') as decode:
            with self.assertRaises(image_input.DetectionInputError) as caught:
                image_input.decode_image(image_bytes(), 11)
        self.assertEqual(
            caught.exception.detail['code'],
            'IMAGE_DIMENSIONS_EXCEEDED',
        )
        decode.assert_not_called()

    def test_truncated_images_and_decode_failure_return_client_errors(
        self,
    ):
        data = image_bytes()
        for invalid in (b'\xff\xd8\xff', b'\x89PNG\r\n\x1a\n', data[:36]):
            with self.subTest(data=invalid):
                with self.assertRaises(image_input.DetectionInputError):
                    image_input.decode_image(invalid, 100)
        with patch.object(image_input.cv2, 'imdecode', return_value=None):
            with self.assertRaises(image_input.DetectionInputError):
                image_input.decode_image(data, 100)
