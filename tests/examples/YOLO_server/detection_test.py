from __future__ import annotations

import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from examples.YOLO_server.detection import (
    detection_router, convert_to_image, get_prediction_result,
    compile_detection_data, process_labels, remove_overlapping_labels,
    remove_completely_contained_labels, get_category_indices, calculate_overlap,
    calculate_intersection, calculate_area, is_contained, find_overlaps,
    find_contained_labels, find_overlapping_indices, find_contained_indices,
    check_containment
)
from fastapi import FastAPI, Depends
from io import BytesIO
import numpy as np

app = FastAPI()

# Mock JWT 身份验证依赖
async def mock_jwt_dependency():
    return MagicMock(subject="test_user")

# 使用 mock 依赖替换实际的 jwt_access 依赖
app.include_router(detection_router, dependencies=[Depends(mock_jwt_dependency)])

class TestDetection(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.image_data = BytesIO(b"test_image_data").getvalue()

    @patch('examples.YOLO_server.detection.RateLimiter.__call__', new_callable=AsyncMock)
    @patch('examples.YOLO_server.detection.model_loader')
    @patch('examples.YOLO_server.detection.get_prediction_result', new_callable=AsyncMock)
    @patch('examples.YOLO_server.detection.jwt_access', new_callable=AsyncMock)
    def test_detect_endpoint(self, mock_jwt_access, mock_get_prediction_result, mock_model_loader, mock_rate_limiter):
        # Mock the rate limiter to bypass Redis requirements
        mock_rate_limiter.return_value = None

        # Mock the JWT access to provide valid credentials
        mock_jwt_access.verify.return_value = MagicMock(subject="test_user")

        # Mock the model loader and prediction result
        mock_model_loader.get_model.return_value = MagicMock()
        mock_get_prediction_result.return_value = MagicMock(object_prediction_list=[])

        # Send a test POST request to the /detect endpoint
        response = self.client.post(
            "/detect",
            files={"image": ("test.jpg", self.image_data, "image/jpeg")},
            data={"model": "yolo11n"},
            headers={"Authorization": "Bearer mocktoken"}
        )

        # Verify that the response status code is 200 and returns a JSON list
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json(), list)

    @patch('examples.YOLO_server.detection.np.frombuffer')
    @patch('examples.YOLO_server.detection.cv2.imdecode')
    async def test_convert_to_image(self, mock_imdecode, mock_frombuffer):
        mock_frombuffer.return_value = MagicMock()
        mock_imdecode.return_value = MagicMock()

        img = await convert_to_image(self.image_data)

        mock_frombuffer.assert_called_once_with(self.image_data, np.uint8)
        mock_imdecode.assert_called_once()
        self.assertIsNotNone(img)

    @patch('examples.YOLO_server.detection.get_sliced_prediction')
    async def test_get_prediction_result(self, mock_get_sliced_prediction):
        mock_get_sliced_prediction.return_value = MagicMock(object_prediction_list=[])
        img = MagicMock()
        model = MagicMock()

        result = await get_prediction_result(img, model)

        mock_get_sliced_prediction.assert_called_once_with(
            img,
            model,
            slice_height=370,
            slice_width=370,
            overlap_height_ratio=0.3,
            overlap_width_ratio=0.3,
        )
        self.assertIsNotNone(result)

    def test_compile_detection_data(self):
        mock_object_prediction = MagicMock()
        mock_object_prediction.category.id = 1
        mock_object_prediction.bbox.to_voc_bbox.return_value = [10, 20, 30, 40]
        mock_object_prediction.score.value = 0.9

        mock_result = MagicMock()
        mock_result.object_prediction_list = [mock_object_prediction]

        result = compile_detection_data(mock_result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], [10, 20, 30, 40, 0.9, 1])

    @patch('examples.YOLO_server.detection.remove_overlapping_labels', new_callable=AsyncMock)
    @patch('examples.YOLO_server.detection.remove_completely_contained_labels', new_callable=AsyncMock)
    async def test_process_labels(self, mock_remove_completely_contained_labels, mock_remove_overlapping_labels):
        mock_remove_overlapping_labels.side_effect = lambda datas: datas
        mock_remove_completely_contained_labels.side_effect = lambda datas: datas

        datas = [[10, 20, 30, 40, 0.9, 1]]
        result = await process_labels(datas)

        self.assertEqual(result, datas)
        mock_remove_overlapping_labels.assert_called()
        mock_remove_completely_contained_labels.assert_called()

    async def test_remove_overlapping_labels(self):
        datas = [[0, 0, 50, 50, 0.9, 1], [10, 10, 60, 60, 0.85, 2]]
        result = await remove_overlapping_labels(datas)
        self.assertIsInstance(result, list)

    async def test_remove_completely_contained_labels(self):
        datas = [[0, 0, 50, 50, 0.9, 1], [10, 10, 40, 40, 0.85, 2]]
        result = await remove_completely_contained_labels(datas)
        self.assertIsInstance(result, list)

    def test_get_category_indices(self):
        datas = [[10, 20, 30, 40, 0.9, 0], [50, 60, 70, 80, 0.85, 7]]
        indices = get_category_indices(datas)
        self.assertIn('hardhat', indices)
        self.assertIn('safety_vest', indices)
        self.assertIsInstance(indices, dict)

    def test_calculate_overlap(self):
        bbox1 = [0, 0, 50, 50]
        bbox2 = [25, 25, 75, 75]
        overlap = calculate_overlap(bbox1, bbox2)
        self.assertGreater(overlap, 0)

    def test_calculate_intersection(self):
        bbox1 = [0, 0, 50, 50]
        bbox2 = [25, 25, 75, 75]
        intersection = calculate_intersection(bbox1, bbox2)
        self.assertEqual(intersection, (25, 25, 50, 50))

    def test_calculate_area(self):
        x1, y1, x2, y2 = 0, 0, 50, 50
        area = calculate_area(x1, y1, x2, y2)
        self.assertEqual(area, 2601)

    def test_is_contained(self):
        inner_bbox = [10, 10, 30, 30]
        outer_bbox = [0, 0, 50, 50]
        result = is_contained(inner_bbox, outer_bbox)
        self.assertTrue(result)

    async def test_find_overlaps(self):
        indices1 = [0]
        indices2 = [1]
        datas = [[0, 0, 50, 50, 0.9, 1], [25, 25, 75, 75, 0.85, 2]]
        result = await find_overlaps(indices1, indices2, datas, 0.5)
        self.assertIsInstance(result, set)

    async def test_find_contained_labels(self):
        indices1 = [0]
        indices2 = [1]
        datas = [[0, 0, 50, 50, 0.9, 1], [10, 10, 40, 40, 0.85, 2]]
        result = await find_contained_labels(indices1, indices2, datas)
        self.assertIsInstance(result, set)

    async def test_find_overlapping_indices(self):
        index1 = 0
        indices2 = [1]
        datas = [[0, 0, 50, 50, 0.9, 1], [25, 25, 75, 75, 0.85, 2]]
        result = await find_overlapping_indices(index1, indices2, datas, 0.5)
        self.assertIsInstance(result, set)

    async def test_find_contained_indices(self):
        index1 = 0
        indices2 = [1]
        datas = [[0, 0, 50, 50, 0.9, 1], [10, 10, 40, 40, 0.85, 2]]
        result = await find_contained_indices(index1, indices2, datas)
        self.assertIsInstance(result, set)

    async def test_check_containment(self):
        index1 = 0
        index2 = 1
        datas = [[0, 0, 50, 50, 0.9, 1], [10, 10, 40, 40, 0.85, 2]]
        result = await check_containment(index1, index2, datas)
        self.assertIsInstance(result, set)


if __name__ == "__main__":
    unittest.main()
