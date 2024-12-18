from __future__ import annotations

import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

from shapely.geometry import Polygon

from src.danger_detector import DangerDetector
from src.danger_detector import main
from src.utils import Utils


class TestDangerDetector(unittest.TestCase):
    """
    Unit tests for the DangerDetector class methods.
    """

    def setUp(self) -> None:
        """
        Set up method to create an instance of DangerDetector for each test.
        """
        self.detector: DangerDetector = DangerDetector()

    def test_initialization(self) -> None:
        """
        Test case for checking if a person is driving based on bounding boxes.
        """
        # Valid detection items
        valid_detection_items = {
            'detect_no_safety_vest_or_helmet': True,
            'detect_near_machinery_or_vehicle': True,
            'detect_in_restricted_area': True,
        }
        detector = DangerDetector(valid_detection_items)
        self.assertEqual(detector.detection_items, valid_detection_items)

        # Invalid detection items (not a dictionary)
        invalid_items_not_dict = {'invalid', 'items'}
        detector = DangerDetector(invalid_items_not_dict)  # type: ignore
        self.assertEqual(detector.detection_items, {})

        # Invalid detection items (keys not strings or values not booleans)
        invalid_items_wrong_value = {
            'detect_no_safety_vest_or_helmet': 'yes',  # Invalid value
            'detect_near_machinery_or_vehicle': True,
            'detect_in_restricted_area': True,
        }
        detector = DangerDetector(invalid_items_wrong_value)  # type: ignore
        self.assertEqual(detector.detection_items, {})

    def test_detect_danger(self) -> None:
        """
        Test case for detecting danger based on a list of data points.
        """
        data: list[list[float]] = [
            [50, 50, 150, 150, 0.95, 0],    # Hardhat
            [200, 200, 300, 300, 0.85, 5],  # Person
            [400, 400, 500, 500, 0.75, 2],  # No-Safety Vest
            [0, 0, 10, 10, 0.88, 6],        # Safety cone
            [0, 1000, 10, 1010, 0.87, 6],   # Safety cone
            [1000, 0, 1010, 10, 0.89, 6],   # Safety cone
            [100, 100, 120, 120, 0.9, 6],   # Safety cone
            [150, 150, 170, 170, 0.85, 6],  # Safety cone
            [200, 200, 220, 220, 0.89, 6],  # Safety cone
            [250, 250, 270, 270, 0.85, 6],  # Safety cone
            [450, 450, 470, 470, 0.92, 6],  # Safety cone
            [500, 500, 520, 520, 0.88, 6],  # Safety cone
            [550, 550, 570, 570, 0.86, 6],  # Safety cone
            [600, 600, 620, 620, 0.84, 6],  # Safety cone
            [650, 650, 670, 670, 0.82, 6],  # Safety cone
            [700, 700, 720, 720, 0.80, 6],  # Safety cone
            [750, 750, 770, 770, 0.78, 6],  # Safety cone
            [800, 800, 820, 820, 0.76, 6],  # Safety cone
            [850, 850, 870, 870, 0.74, 6],  # Safety cone
        ]
        data = Utils.normalise_data(data)
        warnings, polygons = self.detector.detect_danger(data)
        self.assertIn(
            'Warning: 1 people have entered the controlled area!', warnings,
        )
        self.assertIn('Warning: Someone is not wearing a hardhat!', warnings)

        data = [
            [706.87, 445.07, 976.32, 1073.6, 0.91, 5.0],  # Person
            [0.45513, 471.77, 662.03, 1071.4, 0.75853, 5.0],  # Person
            [1042.7, 638.5, 1077.5, 731.98, 0.56060, 4.0],  # No safety vest
            [500, 500, 700, 700, 0.95, 8],  # Machinery
            [50, 50, 70, 70, 0.89, 6],
            [250, 250, 270, 270, 0.85, 6],
            [450, 450, 470, 470, 0.92, 6],
        ]
        data = Utils.normalise_data(data)
        warnings, polygons = self.detector.detect_danger(data)
        self.assertIn(
            'Warning: Someone is not wearing a safety vest!', warnings,
        )

        data = [
            [706.87, 445.07, 976.32, 1073.6, 0.91, 2.0],  # No hardhat
            [0.45513, 471.77, 662.03, 1071.4, 0.75853, 4.0],  # No safety vest
            [500, 500, 700, 700, 0.95, 8],  # Machinery
            [50, 50, 70, 70, 0.89, 6],
            [250, 250, 270, 270, 0.85, 6],
            [450, 450, 470, 470, 0.92, 6],
        ]
        data = Utils.normalise_data(data)
        warnings, polygons = self.detector.detect_danger(data)
        self.assertIn('Warning: Someone is not wearing a hardhat!', warnings)
        self.assertIn(
            'Warning: Someone is not wearing a safety vest!', warnings,
        )
        self.assertNotIn(
            'Warning: Someone is too close to machinery!', warnings,
        )

    def test_no_data(self) -> None:
        """
        Test case for checking behavior when no data is provided.
        """
        data: list[list[float]] = []
        warnings, polygons = self.detector.detect_danger(data)
        self.assertEqual(len(warnings), 0)
        self.assertEqual(len(polygons), 0)

    def test_vehicle_too_close(self) -> None:
        """
        Test case for checking behaviour
        when a person is too close to a vehicle.
        """
        data: list[list[float]] = [
            [100, 100, 120, 120, 0.95, 5],  # Person
            [110, 110, 200, 200, 0.85, 8],  # Machinery
        ]
        normalised_data = Utils.normalise_data(data)
        print(f"Normalized data: {normalised_data}")
        warnings, polygons = self.detector.detect_danger(normalised_data)
        self.assertIn('Warning: Someone is too close to machinery!', warnings)

    @patch('builtins.print')
    def test_main(self, mock_print: MagicMock) -> None:
        """
        Test case for the main function.
        """
        with patch.object(
            DangerDetector, 'detect_danger', return_value=(
                [
                    'Warning: Someone is not wearing a hardhat!',
                    'Warning: 1 people have entered the controlled area!',
                ],
                [Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])],
            ),
        ) as mock_detect_danger:
            main()
            mock_detect_danger.assert_called_once()
            mock_print.assert_any_call(
                "Warnings: ['Warning: Someone is not wearing a hardhat!', "
                "'Warning: 1 people have entered the controlled area!']",
            )
            mock_print.assert_any_call(
                'Polygons: [<POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))>]',
            )


if __name__ == '__main__':
    unittest.main()
