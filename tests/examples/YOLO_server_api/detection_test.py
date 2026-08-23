from __future__ import annotations

import asyncio
import unittest
from collections.abc import Sequence
from io import BytesIO
from unittest.mock import ANY
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from PIL import Image

from examples.YOLO_server_api import detection
from examples.YOLO_server_api.detection import _calc_and_filter
from examples.YOLO_server_api.detection import area
from examples.YOLO_server_api.detection import compile_detection_data
from examples.YOLO_server_api.detection import contained
from examples.YOLO_server_api.detection import convert_to_image
from examples.YOLO_server_api.detection import find_contained
from examples.YOLO_server_api.detection import find_overlaps
from examples.YOLO_server_api.detection import get_category_indices
from examples.YOLO_server_api.detection import get_prediction_result
from examples.YOLO_server_api.detection import overlap_ratio
from examples.YOLO_server_api.detection import process_labels
from examples.YOLO_server_api.detection import (
    remove_completely_contained_labels,
)
from examples.YOLO_server_api.detection import (
    remove_overlapping_labels,
)


class _FakeTensor:
    """Tests for _FakeTensor."""

    def __init__(
        self,
        value: Sequence[Sequence[float]] | Sequence[float],
    ) -> None:
        """Support __init__."""
        self._value = np.asarray(value)

    def cpu(self) -> _FakeTensor:
        """Support cpu."""
        return self

    def numpy(self) -> np.ndarray:
        """Support numpy."""
        return self._value


class TestDetection(unittest.IsolatedAsyncioTestCase):
    """Test cases for the detection functionalities."""

    def setUp(self) -> None:
        """Sets up test environment by creating image data."""
        # Create a simple JPEG image
        img = Image.new('RGB', (100, 100), color='white')
        buf = BytesIO()
        img.save(buf, format='JPEG')
        self.image_data = buf.getvalue()

    @patch('examples.YOLO_server_api.detection.np.frombuffer')
    @patch('examples.YOLO_server_api.detection.cv2.imdecode')
    def test_convert_to_image(
        self,
        mock_imdecode: MagicMock,
        mock_frombuffer: MagicMock,
    ) -> None:
        """Tests the conversion of byte data to an image using mocked numpy and
        OpenCV methods."""
        mock_frombuffer.return_value = MagicMock()
        mock_imdecode.return_value = MagicMock()

        img = convert_to_image(self.image_data)

        mock_frombuffer.assert_called_once_with(self.image_data, np.uint8)
        mock_imdecode.assert_called_once()
        self.assertIsNotNone(img)

    @patch('examples.YOLO_server_api.detection.USE_TENSORRT', True)
    @patch('examples.YOLO_server_api.detection.USE_SAHI', False)
    async def test_get_prediction_result_tensorrt(self) -> None:
        """Tests obtaining a prediction result with TensorRT enabled."""
        img = MagicMock()
        model = MagicMock()
        model.predict.return_value = [MagicMock()]

        result = await get_prediction_result(img, model)

        model.predict.assert_called_once_with(source=img, verbose=False)
        self.assertIsNotNone(result)

    @patch('examples.YOLO_server_api.detection.USE_SAHI', True)
    @patch('examples.YOLO_server_api.detection.get_sliced_prediction')
    async def test_get_prediction_result_sahi(
        self,
        mock_get_sliced_prediction: MagicMock,
    ) -> None:
        """Tests obtaining a prediction result with SAHI (non-TensorRT)."""
        mock_get_sliced_prediction.return_value = MagicMock()
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

    def test_compile_detection_data_sahi(self) -> None:
        """Tests compiling SAHI prediction data into structured format."""
        mock_object_prediction = MagicMock()
        mock_object_prediction.category.id = 1
        mock_object_prediction.bbox.to_voc_bbox.return_value = [
            10.0,
            20.0,
            30.0,
            40.0,
        ]
        mock_object_prediction.score.value = 0.9

        mock_result = MagicMock()
        mock_result.object_prediction_list = [mock_object_prediction]

        result = compile_detection_data(mock_result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], [10, 20, 30, 40, 0.9, 1])

    def test_compile_detection_data_ultralytics(self) -> None:
        """Tests compiling Ultralytics prediction data into structured
        format."""
        # Mock ultralytics result
        mock_boxes = MagicMock()
        mock_boxes.xyxy = _FakeTensor([[10.0, 20.0, 30.0, 40.0]])
        mock_boxes.conf = _FakeTensor([0.9])
        mock_boxes.cls = _FakeTensor([1])

        mock_result = MagicMock()
        mock_result.boxes = mock_boxes
        # Ensure this doesn't have object_prediction_list attribute
        if hasattr(mock_result, 'object_prediction_list'):
            delattr(mock_result, 'object_prediction_list')

        result = compile_detection_data(mock_result)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], [10, 20, 30, 40, 0.9, 1])

    async def test_process_labels(self) -> None:
        """Tests label processing pipeline."""
        datas = [[10, 20, 30, 40, 0.9, 1]]
        result = await process_labels(datas)
        self.assertIsInstance(result, list)

    def test_get_category_indices(self) -> None:
        """Tests retrieval of category indices for different labels."""
        datas = [
            [10, 20, 30, 40, 0.9, 0],  # hardhat
            [50, 60, 70, 80, 0.85, 2],  # no_hardhat
            [90, 100, 110, 120, 0.8, 7],  # safety_vest
            [130, 140, 150, 160, 0.75, 4],  # no_safety_vest
        ]
        indices = get_category_indices(datas)

        self.assertEqual(indices['hardhat'], [0])
        self.assertEqual(indices['no_hardhat'], [1])
        self.assertEqual(indices['safety_vest'], [2])
        self.assertEqual(indices['no_safety_vest'], [3])

    def test_area(self) -> None:
        """Tests calculation of area for a bounding box."""
        result = area(0, 0, 50, 50)
        self.assertEqual(result, 51 * 51)  # (50-0+1) * (50-0+1)

    def test_area_zero(self) -> None:
        """Tests area calculation with zero or negative dimensions."""
        result = area(50, 50, 0, 0)
        self.assertEqual(result, 0)  # max(0, negative) = 0

    def test_overlap_ratio(self) -> None:
        """Tests calculation of overlap ratio between two bounding boxes."""
        bbox1: list[float] = [0, 0, 50, 50]
        bbox2: list[float] = [25, 25, 75, 75]
        overlap = overlap_ratio(bbox1, bbox2)
        self.assertGreater(overlap, 0)
        self.assertLessEqual(overlap, 1)

    def test_overlap_ratio_no_overlap(self) -> None:
        """Tests overlap ratio with no overlapping boxes."""
        bbox1: list[float] = [0, 0, 50, 50]
        bbox2: list[float] = [100, 100, 150, 150]
        overlap = overlap_ratio(bbox1, bbox2)
        self.assertEqual(overlap, 0)

    def test_contained_true(self) -> None:
        """Tests if one bounding box is contained within another."""
        inner_bbox: list[float] = [10, 10, 30, 30]
        outer_bbox: list[float] = [0, 0, 50, 50]
        result = contained(inner_bbox, outer_bbox)
        self.assertTrue(result)

    def test_contained_false(self) -> None:
        """Tests if one bounding box is NOT contained within another."""
        inner_bbox: list[float] = [10, 10, 60, 60]
        outer_bbox: list[float] = [0, 0, 50, 50]
        result = contained(inner_bbox, outer_bbox)
        self.assertFalse(result)

    async def test_find_overlaps(self) -> None:
        """Tests identification of overlapping bounding boxes."""
        i1 = 0
        idxs2 = [1, 2]
        datas = [
            [0, 0, 50, 50, 0.9, 1],
            [10, 10, 60, 60, 0.85, 2],  # significant overlap with i1
            [100, 100, 150, 150, 0.8, 3],  # no overlap with i1
        ]
        result = await find_overlaps(i1, idxs2, datas, 0.3)
        self.assertIsInstance(result, set)
        self.assertIn(1, result)  # should find overlap with index 1

    async def test_find_overlaps_no_overlap(self) -> None:
        """Tests find_overlaps when there are no overlaps."""
        i1 = 0
        idxs2 = [1]
        datas = [
            [0, 0, 50, 50, 0.9, 1],
            [100, 100, 150, 150, 0.85, 2],  # no overlap
        ]
        result = await find_overlaps(i1, idxs2, datas, 0.5)
        self.assertEqual(result, set())

    async def test_find_contained(self) -> None:
        """Tests identification of contained bounding boxes."""
        i1 = 0
        idxs2 = [1, 2]
        datas = [
            [0, 0, 50, 50, 0.9, 1],
            [10, 10, 40, 40, 0.85, 2],  # contained in i1
            [100, 100, 150, 150, 0.8, 3],  # not contained
        ]
        result = await find_contained(i1, idxs2, datas)
        self.assertIsInstance(result, set)
        self.assertIn(1, result)  # should find contained box at index 1

    async def test_find_contained_both_directions(self) -> None:
        """Tests find_contained when i1 is contained in one of idxs2."""
        i1 = 0
        idxs2 = [1]
        datas = [
            [10, 10, 40, 40, 0.9, 1],  # i1 - smaller box
            [0, 0, 50, 50, 0.85, 2],  # larger box that contains i1
        ]
        result = await find_contained(i1, idxs2, datas)
        self.assertIsInstance(result, set)
        self.assertIn(0, result)  # should find i1 is contained

    async def test_calc_and_filter(self) -> None:
        """Tests the _calc_and_filter helper function."""
        idxs1 = [0]
        idxs2 = [1, 2]
        datas = [
            [0, 0, 50, 50, 0.9, 1],
            [25, 25, 75, 75, 0.85, 2],
            [100, 100, 150, 150, 0.8, 3],
        ]

        result = await _calc_and_filter(idxs1, idxs2, datas, find_overlaps)
        self.assertIsInstance(result, set)

    async def test_remove_overlapping_labels_hardhat_pair(self) -> None:
        """Tests removal of overlapping hardhat vs no_hardhat labels."""
        datas = [
            [0, 0, 100, 100, 0.9, 0],  # hardhat
            [10, 10, 90, 90, 0.8, 2],  # no_hardhat (overlaps)
            [200, 200, 300, 300, 0.85, 7],  # safety_vest
            [400, 400, 500, 500, 0.7, 4],  # no_safety_vest (no overlap)
        ]
        result = await remove_overlapping_labels(datas.copy())
        self.assertEqual(len(result), 3)  # one should be removed

    async def test_remove_overlapping_labels_both_pairs(self) -> None:
        """Tests removal of overlapping labels for both hardhat and
        safety_vest."""
        datas = [
            [0, 0, 100, 100, 0.9, 0],  # hardhat
            [10, 10, 90, 90, 0.8, 2],  # no_hardhat (overlaps)
            [200, 200, 300, 300, 0.85, 7],  # safety_vest
            [210, 210, 290, 290, 0.7, 4],  # no_safety_vest (overlaps)
        ]
        result = await remove_overlapping_labels(datas.copy())
        self.assertEqual(len(result), 2)  # two should be removed

    async def test_remove_completely_contained_labels(self) -> None:
        """Tests removal of completely contained labels."""
        datas = [
            [0, 0, 100, 100, 0.9, 0],  # hardhat
            [10, 10, 90, 90, 0.8, 2],  # no_hardhat (contained)
            [200, 200, 300, 300, 0.85, 7],  # safety_vest
            [210, 210, 290, 290, 0.7, 4],  # no_safety_vest (contained)
        ]
        result = await remove_completely_contained_labels(datas.copy())
        # contained labels should be removed
        self.assertEqual(len(result), 2)

    async def test_remove_completely_contained_labels_no_containment(
        self,
    ) -> None:
        """Tests remove_completely_contained_labels when no labels are
        contained."""
        datas = [
            [0, 0, 50, 50, 0.9, 0],  # hardhat
            [100, 100, 150, 150, 0.8, 2],  # no_hardhat (not contained)
            [200, 200, 250, 250, 0.85, 7],  # safety_vest
            [300, 300, 350, 350, 0.7, 4],  # no_safety_vest (not contained)
        ]
        result = await remove_completely_contained_labels(datas.copy())
        self.assertEqual(len(result), 4)  # no labels should be removed

    async def test_empty_data_lists(self) -> None:
        """Tests functions with empty data lists."""
        empty_datas: list[list[float]] = []

        # Test process_labels with empty data
        result = await process_labels(empty_datas.copy())
        self.assertEqual(result, [])

        # Test remove_overlapping_labels with empty data
        result = await remove_overlapping_labels(empty_datas.copy())
        self.assertEqual(result, [])

        # Test remove_completely_contained_labels with empty data
        result = await remove_completely_contained_labels(empty_datas.copy())
        self.assertEqual(result, [])

    def test_get_category_indices_empty(self) -> None:
        """Tests get_category_indices with empty data."""
        datas: list[list[float]] = []
        indices = get_category_indices(datas)

        expected: dict[str, list[int]] = {
            'hardhat': [],
            'no_hardhat': [],
            'safety_vest': [],
            'no_safety_vest': [],
        }
        self.assertEqual(indices, expected)

    def test_overlap_ratio_identical_boxes(self) -> None:
        """Tests overlap ratio with identical bounding boxes."""
        bbox1: list[float] = [0, 0, 50, 50]
        bbox2: list[float] = [0, 0, 50, 50]
        overlap = overlap_ratio(bbox1, bbox2)
        self.assertEqual(overlap, 1.0)  # Should be 100% overlap

    def test_contained_edge_cases(self) -> None:
        """Tests contained function with edge cases."""
        # Test identical boxes
        bbox1: list[float] = [10, 10, 30, 30]
        bbox2: list[float] = [10, 10, 30, 30]
        result = contained(bbox1, bbox2)
        self.assertTrue(result)  # Identical boxes are contained

        # Test partially overlapping boxes (not contained)
        bbox1 = [10, 10, 30, 30]
        bbox2 = [20, 20, 40, 40]
        result = contained(bbox1, bbox2)
        self.assertFalse(result)

    async def test_find_contained_no_containment(self) -> None:
        """Tests find_contained when no boxes are contained."""
        i1 = 0
        idxs2 = [1, 2]
        datas = [
            [0, 0, 50, 50, 0.9, 1],
            [60, 60, 110, 110, 0.85, 2],  # not contained
            [120, 120, 170, 170, 0.8, 3],  # not contained
        ]
        result = await find_contained(i1, idxs2, datas)
        self.assertEqual(result, set())  # should find no contained boxes


if __name__ == '__main__':
    unittest.main()


class Tensor:
    """Minimal tensor adapter matching the Ultralytics box access contract."""

    def __init__(
        self,
        value: Sequence[Sequence[float]] | Sequence[float],
    ) -> None:
        """Perform init.

        Args:
            value: Value used by this callable.
        """
        self.value = np.asarray(value)

    def cpu(self) -> Tensor:
        """Perform cpu.

        Returns:
            The callable result.
        """
        return self

    def numpy(self) -> np.ndarray:
        """Perform numpy.

        Returns:
            The callable result.
        """
        return self.value


def test_conversion_and_standard_prediction_cover_invalid_and_cpu_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bad image bytes fail clearly and standard inference gets a device."""
    monkeypatch.setattr(detection.cv2, 'imdecode', lambda *_args: None)
    with pytest.raises(cv2.error, match='Unable to decode'):
        detection.convert_to_image(b'not-an-image')

    model = MagicMock()
    result = MagicMock()
    model.predict.return_value = [result]
    monkeypatch.setattr(detection, 'USE_SAHI', False)
    monkeypatch.setattr(detection, 'USE_TENSORRT', False)
    monkeypatch.setattr(detection, 'get_inference_device', lambda: 'cpu')
    image = np.zeros((1, 1, 3))

    assert asyncio.run(detection.get_prediction_result(image, model)) is result
    model.predict.assert_called_once_with(
        source=image,
        verbose=False,
        device='cpu',
    )


def test_ultralytics_empty_boxes_and_detection_timing_pipeline() -> None:
    """Empty boxes are valid and the byte pipeline returns timing phases."""
    boxes = MagicMock()
    boxes.xyxy = Tensor(np.empty((0, 4)))
    assert detection._compile_ultralytics_detection_data(boxes) == []

    image = np.zeros((2, 2, 3), dtype=np.uint8)
    prediction = MagicMock()
    compiled = [[1, 2, 3, 4, 0.9, 1]]
    processed = [[1, 2, 3, 4, 0.9, 1]]
    with (
        patch.object(detection, 'convert_to_image', return_value=image),
        patch.object(
            detection,
            'get_prediction_result',
            AsyncMock(return_value=prediction),
        ) as get_prediction,
        patch.object(
            detection,
            'compile_detection_data',
            return_value=compiled,
        ),
        patch.object(
            detection,
            'process_labels',
            AsyncMock(return_value=processed),
        ),
    ):
        labels, timing = asyncio.run(
            detection.run_detection_from_bytes(
                b'image',
                MagicMock(),
                semaphore=asyncio.Semaphore(1),
            ),
        )

    assert labels == processed
    assert set(timing) == {'inference', 'post'}
    assert timing['inference'] >= 0
    assert timing['post'] >= 0
    get_prediction.assert_awaited_once_with(image, ANY)
