from __future__ import annotations

import unittest
from io import BytesIO
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
from PIL import Image

from examples.YOLO_server_api.backend.detection import calculate_area
from examples.YOLO_server_api.backend.detection import calculate_intersection
from examples.YOLO_server_api.backend.detection import calculate_overlap
from examples.YOLO_server_api.backend.detection import check_containment
from examples.YOLO_server_api.backend.detection import compile_detection_data
from examples.YOLO_server_api.backend.detection import convert_to_image
from examples.YOLO_server_api.backend.detection import find_contained_indices
from examples.YOLO_server_api.backend.detection import find_contained_labels
from examples.YOLO_server_api.backend.detection import find_overlapping_indices
from examples.YOLO_server_api.backend.detection import find_overlaps
from examples.YOLO_server_api.backend.detection import get_category_indices
from examples.YOLO_server_api.backend.detection import get_prediction_result
from examples.YOLO_server_api.backend.detection import is_contained
from examples.YOLO_server_api.backend.detection import process_labels
from examples.YOLO_server_api.backend.detection import (
    remove_completely_contained_labels,
)
from examples.YOLO_server_api.backend.detection import (
    remove_overlapping_labels,
)


class TestDetection(unittest.IsolatedAsyncioTestCase):
    """
    Test cases for the detection functionalities.
    """

    def setUp(self) -> None:
        """
        Sets up test environment by creating image data.
        """
        # Create a simple JPEG image
        img = Image.new('RGB', (100, 100), color='white')
        buf = BytesIO()
        img.save(buf, format='JPEG')
        self.image_data = buf.getvalue()

    @patch('examples.YOLO_server_api.backend.detection.np.frombuffer')
    @patch('examples.YOLO_server_api.backend.detection.cv2.imdecode')
    async def test_convert_to_image(
        self,
        mock_imdecode: MagicMock,
        mock_frombuffer: MagicMock,
    ) -> None:
        """
        Tests the conversion of byte data to an image using mocked numpy
        and OpenCV methods.

        Args:
            mock_imdecode (MagicMock): Mocked image decoding method.
            mock_frombuffer (MagicMock): Mocked buffer conversion method.
        """
        mock_frombuffer.return_value = MagicMock()
        mock_imdecode.return_value = MagicMock()

        img = await convert_to_image(self.image_data)

        mock_frombuffer.assert_called_once_with(self.image_data, np.uint8)
        mock_imdecode.assert_called_once()
        self.assertIsNotNone(img)

    @patch('examples.YOLO_server_api.backend.detection.get_sliced_prediction')
    async def test_get_prediction_result(
        self, mock_get_sliced_prediction: MagicMock,
    ) -> None:
        """
        Tests obtaining a prediction result by mocking the prediction method.

        Args:
            mock_get_sliced_prediction (MagicMock): Mocked sliced
                prediction method.
        """
        mock_get_sliced_prediction.return_value = MagicMock(
            object_prediction_list=[],
        )
        img = MagicMock()
        model = MagicMock()

        result = await get_prediction_result(img, model)

        mock_get_sliced_prediction.assert_called_once_with(
            img, model, slice_height=370, slice_width=370,
            overlap_height_ratio=0.3, overlap_width_ratio=0.3,
        )
        self.assertIsNotNone(result)

    def test_compile_detection_data(self) -> None:
        """
        Tests compiling prediction data into structured format.
        """
        mock_object_prediction = MagicMock()
        mock_object_prediction.category.id = 1
        mock_object_prediction.bbox.to_voc_bbox.return_value = [10, 20, 30, 40]
        mock_object_prediction.score.value = 0.9

        mock_result = MagicMock()
        mock_result.object_prediction_list = [mock_object_prediction]

        result = compile_detection_data(mock_result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], [10, 20, 30, 40, 0.9, 1])

    @patch(
        'examples.YOLO_server_api.backend.detection.remove_overlapping_labels',
        new_callable=AsyncMock,
    )
    @patch(
        'examples.YOLO_server_api.backend.detection.'
        'remove_completely_contained_labels',
        new_callable=AsyncMock,
    )
    async def test_process_labels(
        self,
        mock_remove_completely_contained_labels: AsyncMock,
        mock_remove_overlapping_labels: AsyncMock,
    ) -> None:
        """
        Tests label processing by applying overlapping and containment checks.

        Args:
            mock_remove_completely_contained_labels (AsyncMock): Mocked
                containment removal method.
            mock_remove_overlapping_labels (AsyncMock): Mocked overlapping
                removal method.
        """
        mock_remove_overlapping_labels.side_effect = lambda datas: datas
        mock_remove_completely_contained_labels.side_effect = (
            lambda datas: datas
        )

        datas = [[10, 20, 30, 40, 0.9, 1]]
        result = await process_labels(datas)

        self.assertEqual(result, datas)
        mock_remove_overlapping_labels.assert_called()
        mock_remove_completely_contained_labels.assert_called()

    async def test_remove_overlapping_labels(self) -> None:
        """
        Tests removal of overlapping labels.
        """
        datas = [[0, 0, 50, 50, 0.9, 1], [10, 10, 60, 60, 0.85, 2]]
        result = await remove_overlapping_labels(datas)
        self.assertIsInstance(result, list)

    async def test_remove_completely_contained_labels(self) -> None:
        """
        Tests removal of completely contained labels.
        """
        datas = [[0, 0, 50, 50, 0.9, 1], [10, 10, 40, 40, 0.85, 2]]
        result = await remove_completely_contained_labels(datas)
        self.assertIsInstance(result, list)

    def test_get_category_indices(self) -> None:
        """
        Tests retrieval of category indices for different labels.
        """
        datas = [[10, 20, 30, 40, 0.9, 0], [50, 60, 70, 80, 0.85, 7]]
        indices = get_category_indices(datas)
        self.assertIsInstance(indices, dict)

    def test_calculate_overlap(self) -> None:
        """
        Tests calculation of overlapping area between two bounding boxes.
        """
        bbox1 = [0, 0, 50, 50]
        bbox2 = [25, 25, 75, 75]
        overlap = calculate_overlap(bbox1, bbox2)
        self.assertGreater(overlap, 0)

    def test_calculate_intersection(self) -> None:
        """
        Tests calculation of intersection area between two bounding boxes.
        """
        bbox1 = [0, 0, 50, 50]
        bbox2 = [25, 25, 75, 75]
        intersection = calculate_intersection(bbox1, bbox2)
        self.assertEqual(intersection, (25, 25, 50, 50))

    def test_calculate_area(self) -> None:
        """
        Tests calculation of area for a bounding box.
        """
        x1, y1, x2, y2 = 0, 0, 50, 50
        area = calculate_area(x1, y1, x2, y2)
        self.assertEqual(area, 2601)

    def test_is_contained(self) -> None:
        """
        Tests if one bounding box is contained within another.
        """
        inner_bbox = [10, 10, 30, 30]
        outer_bbox = [0, 0, 50, 50]
        result = is_contained(inner_bbox, outer_bbox)
        self.assertTrue(result)

    async def test_find_overlaps(self) -> None:
        """
        Tests identification of overlapping bounding boxes.
        """
        indices1 = [0]
        indices2 = [1]
        datas = [[0, 0, 50, 50, 0.9, 1], [25, 25, 75, 75, 0.85, 2]]
        result = await find_overlaps(indices1, indices2, datas, 0.5)
        self.assertIsInstance(result, set)

    async def test_find_contained_labels(self) -> None:
        """
        Tests identification of labels that are contained within others.
        """
        indices1 = [0]
        indices2 = [1]
        datas = [[0, 0, 50, 50, 0.9, 1], [10, 10, 40, 40, 0.85, 2]]
        result = await find_contained_labels(indices1, indices2, datas)
        self.assertIsInstance(result, set)

    async def test_find_overlapping_indices(self) -> None:
        """
        Tests finding indices of overlapping bounding boxes.
        """
        index1 = 0
        indices2 = [1]
        datas = [[0, 0, 50, 50, 0.9, 1], [25, 25, 75, 75, 0.85, 2]]
        result = await find_overlapping_indices(index1, indices2, datas, 0.5)
        self.assertIsInstance(result, set)

    async def test_find_contained_indices(self) -> None:
        """
        Tests finding indices of contained bounding boxes.
        """
        index1 = 0
        indices2 = [1]
        datas = [[0, 0, 50, 50, 0.9, 1], [10, 10, 40, 40, 0.85, 2]]
        result = await find_contained_indices(index1, indices2, datas)
        self.assertIsInstance(result, set)

    async def test_check_containment(self) -> None:
        """
        Tests checking if one bounding box is contained within another.
        """
        index1 = 0
        index2 = 1
        datas = [[0, 0, 50, 50, 0.9, 1], [10, 10, 40, 40, 0.85, 2]]
        result = await check_containment(index1, index2, datas)
        self.assertIsInstance(result, set)

    async def test_remove_completely_contained_labels_line_340(self) -> None:
        """
        Test the `remove_completely_contained_labels` function to ensure it
        removes labels that are completely contained within another label.
        """
        # Test data representing bounding boxes and their corresponding labels
        datas = [
            [50, 50, 150, 150, 0.9, 0],   # 'hardhat' label
            # 'no_hardhat' label, fully contained within 'hardhat'
            [70, 70, 130, 130, 0.8, 2],
            [200, 200, 300, 300, 0.85, 7],  # 'safety_vest' label
            # 'no_safety_vest' label, fully contained within 'safety_vest'
            [220, 220, 280, 280, 0.7, 4],
        ]

        # Call the function and pass a copy of the data
        result = await remove_completely_contained_labels(datas.copy())

        # Assert the length of the resulting list is as expected
        self.assertEqual(len(result), 2)

        # Extract remaining labels and verify they match the expected values
        remaining_labels = [d[5] for d in result]
        self.assertListEqual(remaining_labels, [0, 7])

    async def test_check_containment_elif_condition(self) -> None:
        """
        Test the `check_containment` function's `elif` condition to ensure
        that the appropriate lines of code are covered.

        This test simulates a scenario where one bounding box
        (`index1`) is completely contained within another bounding box
        (`index2`). The function should correctly identify that
        `index1` is contained and return the set containing `index1`.

        Args:
            None

        Returns:
            None
        """
        # Define the indices of the bounding boxes to test
        index1: int = 0
        index2: int = 1

        # Define the test data representing bounding boxes
        datas: list[list[float | int]] = [
            [70, 70, 130, 130, 0.9, 1],  # Bounding box for index1
            [50, 50, 150, 150, 0.85, 2],  # Bounding box for index2
        ]

        # Call the `check_containment` function with the test data
        result: set[int] = await check_containment(index1, index2, datas)

        # Assert that the result matches the expected output
        # In this case, `index1` is contained within `index2`
        self.assertSetEqual(
            result, {0},
            'The function did not correctly identify containment.',
        )


if __name__ == '__main__':
    unittest.main()
