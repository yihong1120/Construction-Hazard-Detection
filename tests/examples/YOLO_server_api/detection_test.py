from __future__ import annotations

import unittest
from io import BytesIO
from unittest.mock import MagicMock

import cv2
import numpy as np
from flask import Flask
from flask_jwt_extended import create_access_token
from flask_jwt_extended import JWTManager
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from examples.YOLO_server_api.detection import calculate_overlap
from examples.YOLO_server_api.detection import check_containment
from examples.YOLO_server_api.detection import compile_detection_data
from examples.YOLO_server_api.detection import detection_blueprint
from examples.YOLO_server_api.detection import DetectionModelManager
from examples.YOLO_server_api.detection import is_contained
from examples.YOLO_server_api.detection import (
    remove_completely_contained_labels,
)
from examples.YOLO_server_api.detection import remove_overlapping_labels


class TestDetectionAPI(unittest.TestCase):
    def setUp(self):
        """
        Set up a Flask app and JWT for testing.
        """
        self.app = Flask(__name__)
        self.app.config['JWT_SECRET_KEY'] = 'super-secret'
        self.jwt = JWTManager(self.app)

        # Initialize blueprint, limiter, and model loader
        self.detection_blueprint = detection_blueprint
        self.limiter = Limiter(key_func=get_remote_address)
        self.model_loader = DetectionModelManager()

        self.app.register_blueprint(self.detection_blueprint)
        self.client = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()

    def tearDown(self):
        """
        Clean up after each test.
        """
        # Ensure to pop the app context to clean up
        self.app_context.pop()
        # Explicitly close the app (although usually not necessary)
        self.client = None
        # Clean up limiter and model loader
        self.limiter = None
        self.model_loader = None
        # Collect garbage to ensure all resources are released
        import gc
        gc.collect()
        # Close the Flask app
        self.app = None

    def test_detection_route(self):
        # Create JWT token
        access_token = create_access_token(identity='testuser')

        # Load test image
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', img)
        img_bytes = BytesIO(buffer.tobytes())

        # Test detection endpoint
        response = self.client.post(
            '/detect',
            headers={'Authorization': f'Bearer {access_token}'},
            content_type='multipart/form-data',
            data={'image': (img_bytes, 'test.jpg')},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json, list)


class TestDetectionFunctions(unittest.TestCase):
    def tearDown(self):
        """
        Clean up after each test.
        """
        import gc
        gc.collect()

    def test_remove_overlapping_labels(self):
        datas = [
            [10, 10, 50, 50, 0.9, 0],  # Hardhat
            [10, 10, 50, 50, 0.8, 2],  # NO-Hardhat
            [100, 100, 150, 150, 0.9, 7],  # Safety Vest
            [100, 100, 150, 150, 0.8, 4],  # NO-Safety Vest
        ]

        updated_datas = remove_overlapping_labels(datas)
        self.assertEqual(len(updated_datas), 2)
        self.assertTrue(all(d[5] in [0, 7] for d in updated_datas))

    def test_remove_completely_contained_labels(self):
        datas = [
            [10, 10, 50, 50, 0.9, 0],  # Hardhat
            [15, 15, 45, 45, 0.8, 2],  # NO-Hardhat contained within Hardhat
            [100, 100, 150, 150, 0.9, 7],  # Safety Vest
            # NO-Safety Vest contained within Safety Vest
            [105, 105, 145, 145, 0.8, 4],
        ]

        updated_datas = remove_completely_contained_labels(datas)
        self.assertEqual(len(updated_datas), 2)
        self.assertTrue(all(d[5] in [0, 7] for d in updated_datas))

    def test_calculate_overlap(self):
        bbox1 = [10, 10, 50, 50]
        bbox2 = [30, 30, 70, 70]
        overlap = calculate_overlap(bbox1, bbox2)
        self.assertGreater(overlap, 0)

    def test_is_contained(self):
        outer_bbox = [10, 10, 50, 50]
        inner_bbox = [20, 20, 30, 30]
        self.assertTrue(is_contained(inner_bbox, outer_bbox))
        self.assertFalse(is_contained(outer_bbox, inner_bbox))

    def test_compile_detection_data(self):
        # Mock the result object
        mock_result = MagicMock()
        mock_object_prediction = MagicMock()
        mock_object_prediction.category.id = 1
        mock_object_prediction.bbox.to_voc_bbox.return_value = [10, 20, 30, 40]
        mock_object_prediction.score.value = 0.95
        mock_result.object_prediction_list = [mock_object_prediction]

        # Call the function
        datas = compile_detection_data(mock_result)

        # Assertions
        self.assertEqual(len(datas), 1)
        self.assertEqual(datas[0], [10, 20, 30, 40, 0.95, 1])

    def test_check_containment(self):
        datas = [
            [10, 10, 50, 50, 0.9],  # 第一个检测框
            [12, 12, 48, 48, 0.8],  # 第二个检测框，完全包含在第一个检测框中
            [60, 60, 100, 100, 0.7],  # 第三个检测框，不与其他框重叠
        ]

        # 调用要测试的函数
        result = check_containment(0, 1, datas)

        # 断言 `result` 应包含 index2，因为 index1 的框包含 index2 的框
        self.assertIn(1, result)

        # 测试另一个组合，index2 完全包含在 index1 中
        result = check_containment(1, 0, datas)
        self.assertIn(1, result)

        # 测试不重叠的情况
        result = check_containment(0, 2, datas)
        self.assertNotIn(0, result)
        self.assertNotIn(2, result)


if __name__ == '__main__':
    unittest.main()
