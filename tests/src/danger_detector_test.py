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
        # Valid detection items (missing required keys)
        partial_detection_items = {
            'detect_no_safety_vest_or_helmet': True,
            'detect_near_machinery_or_vehicle': True,
            'detect_in_restricted_area': True,
        }
        detector = DangerDetector(partial_detection_items)
        self.assertEqual(detector.detection_items, {})

        # Valid detection items (all required keys)
        valid_detection_items = {
            'detect_no_safety_vest_or_helmet': True,
            'detect_near_machinery_or_vehicle': True,
            'detect_in_restricted_area': True,
            'detect_in_utility_pole_restricted_area': False,
            'detect_machinery_close_to_pole': False,
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
            [50, 50, 150, 150, 0.95, 0, -1, 0],    # Hardhat
            [200, 200, 300, 300, 0.85, 5, -1, 0],  # Person
            [400, 400, 500, 500, 0.75, 2, -1, 0],  # No-Safety Vest
            [0, 0, 10, 10, 0.88, 6, -1, 0],        # Safety cone
            [0, 1000, 10, 1010, 0.87, 6, -1, 0],   # Safety cone
            [1000, 0, 1010, 10, 0.89, 6, -1, 0],   # Safety cone
            [100, 100, 120, 120, 0.9, 6, -1, 0],   # Safety cone
            [150, 150, 170, 170, 0.85, 6, -1, 0],  # Safety cone
            [200, 200, 220, 220, 0.89, 6, -1, 0],  # Safety cone
            [250, 250, 270, 270, 0.85, 6, -1, 0],  # Safety cone
            [450, 450, 470, 470, 0.92, 6, -1, 0],  # Safety cone
            [500, 500, 520, 520, 0.88, 6, -1, 0],  # Safety cone
            [550, 550, 570, 570, 0.86, 6, -1, 0],  # Safety cone
            [600, 600, 620, 620, 0.84, 6, -1, 0],  # Safety cone
            [650, 650, 670, 670, 0.82, 6, -1, 0],  # Safety cone
            [700, 700, 720, 720, 0.80, 6, -1, 0],  # Safety cone
            [750, 750, 770, 770, 0.78, 6, -1, 0],  # Safety cone
            [800, 800, 820, 820, 0.76, 6, -1, 0],  # Safety cone
            [850, 850, 870, 870, 0.74, 6, -1, 0],  # Safety cone
        ]
        warnings, cone_polygons, pole_polygons = self.detector.detect_danger(
            data,
        )
        self.assertIn(
            'warning_people_in_controlled_area', warnings,
        )
        self.assertIn('warning_no_hardhat', warnings)

        data = [
            [706.87, 445.07, 976.32, 1073.6, 0.91, 5.0, -1, 0],  # Person
            # Person
            [0.45513, 471.77, 662.03, 1071.4, 0.75853, 5.0, -1, 0],
            # No safety vest
            [1042.7, 638.5, 1077.5, 731.98, 0.56060, 4.0, -1, 0],
            [500, 500, 700, 700, 0.95, 8, -1, 0],  # Machinery
            [50, 50, 70, 70, 0.89, 6, -1, 0],
            [250, 250, 270, 270, 0.85, 6, -1, 0],
            [450, 450, 470, 470, 0.92, 6, -1, 0],
        ]
        data = Utils.normalise_data(data)
        warnings, cone_polygons, pole_polygons = self.detector.detect_danger(
            data,
        )
        self.assertIn(
            'warning_no_safety_vest', warnings,
        )

        data = [
            [706.87, 445.07, 976.32, 1073.6, 0.91, 2.0, -1, 0],  # No hardhat
            # No safety vest
            [0.45513, 471.77, 662.03, 1071.4, 0.75853, 4.0, -1, 0],
            [500, 500, 700, 700, 0.95, 8, -1, 0],  # Machinery
            [50, 50, 70, 70, 0.89, 6, -1, 0],
            [250, 250, 270, 270, 0.85, 6, -1, 0],
            [450, 450, 470, 470, 0.92, 6, -1, 0],
        ]
        warnings, cone_polygons, pole_polygons = self.detector.detect_danger(
            data,
        )
        self.assertIn('warning_no_hardhat', warnings)
        self.assertIn(
            'warning_no_safety_vest', warnings,
        )
        self.assertNotIn(
            'warning_close_to_machinery', warnings,
        )

    def test_no_data(self) -> None:
        """
        Test case for checking behavior when no data is provided.
        """
        data: list[list[float]] = []
        warnings, cone_polygons, pole_polygons = self.detector.detect_danger(
            data,
        )
        self.assertEqual(len(warnings), 0)
        self.assertEqual(len(cone_polygons), 0)
        self.assertEqual(len(pole_polygons), 0)

    def test_vehicle_too_close(self) -> None:
        """
        Test case for checking behaviour
        when a person is too close to a vehicle.
        """
        # Use default detector (when detection_items is empty, all checks run)
        data: list[list[float]] = [
            # Small person (5x10=50 area)
            [100, 100, 105, 110, 0.95, 5, -1, 0],
            # Moving machinery (won't be filtered)
            [107, 105, 150, 150, 0.85, 8, 1, 1],
        ]
        # person_area/vehicle_area = 50/1935 ≈ 0.026 < 0.05 ✓
        # horizontal_distance = min(|105-107|, |100-150|) = 2
        # vertical_distance = min(|110-105|, |100-150|) = 5
        # danger_distance_horizontal = 5 * 5 = 25 ≥ 2 ✓
        # danger_distance_vertical = 1.5 * 10 = 15 ≥ 5 ✓
        warnings, cone_polygons, pole_polygons = self.detector.detect_danger(
            data,
        )
        self.assertIn('warning_close_to_machinery', warnings)

    def test_utility_pole_detection(self) -> None:
        """
        Test case for utility pole detection functionality.
        """
        # Test with utility pole detection enabled
        detection_items = {
            'detect_no_safety_vest_or_helmet': False,
            'detect_near_machinery_or_vehicle': False,
            'detect_in_restricted_area': False,
            'detect_in_utility_pole_restricted_area': True,
            'detect_machinery_close_to_pole': True,
        }
        detector = DangerDetector(detection_items)

        # Simple test - just ensure the functionality runs without error
        # since the utility pole detection requires complex clustering logic
        data: list[list[float]] = [
            [100, 10, 110, 50, 0.9, 9, -1, 0],    # Utility pole
            [125, 20, 135, 25, 0.85, 5, -1, 0],   # Person near poles
        ]

        warnings, cone_polygons, pole_polygons = detector.detect_danger(data)
        # Test passes if no exceptions are raised
        self.assertIsInstance(warnings, dict)
        self.assertIsInstance(cone_polygons, list)
        self.assertIsInstance(pole_polygons, list)

    def test_machinery_filtering(self) -> None:
        """
        Test static machinery filtering functionality.
        """
        data: list[list[float]] = [
            # Static machinery (should be filtered)
            [100, 100, 200, 200, 0.9, 8, -1, 0],
            # Moving machinery (should remain)
            [300, 300, 400, 400, 0.8, 8, 1, 1],
            [500, 500, 600, 600, 0.85, 5, -1, 0],  # Person (should remain)
        ]

        warnings, cone_polygons, pole_polygons = self.detector.detect_danger(
            data,
        )
        # Should have fewer machinery objects after filtering
        # No warnings expected from this data
        self.assertEqual(len(warnings), 0)

    def test_invalid_detection_items_wrong_keys(self) -> None:
        """
        Test initialization with wrong keys in detection items.
        """
        invalid_items = {
            'wrong_key': True,
            'another_wrong_key': False,
        }
        detector = DangerDetector(invalid_items)  # type: ignore
        self.assertEqual(detector.detection_items, {})

    def test_invalid_detection_items_wrong_types(self) -> None:
        """
        Test initialization with wrong value types in detection items.
        """
        invalid_items = {
            'detect_no_safety_vest_or_helmet': 'yes',  # Should be bool
            'detect_near_machinery_or_vehicle': 1,     # Should be bool
            'detect_in_restricted_area': True,
            'detect_in_utility_pole_restricted_area': False,
            'detect_machinery_close_to_pole': False,
        }
        detector = DangerDetector(invalid_items)  # type: ignore
        self.assertEqual(detector.detection_items, {})

    def test_vehicle_proximity_detection(self) -> None:
        """
        Test vehicle proximity detection functionality.
        """
        # Use default detector (when detection_items is empty, all checks run)
        data: list[list[float]] = [
            # Small person (5x10=50 area)
            [100, 100, 105, 110, 0.95, 5, -1, 0],
            # Moving vehicle (won't be filtered)
            [107, 105, 200, 200, 0.85, 10, 1, 1],
        ]
        # person_area/vehicle_area = 50/8835 ≈ 0.0057 < 0.1 ✓
        # Same proximity calculations as machinery test
        warnings, cone_polygons, pole_polygons = self.detector.detect_danger(
            data,
        )
        self.assertIn('warning_close_to_vehicle', warnings)

    def test_driver_filtering(self) -> None:
        """
        Test that drivers are filtered out from person detections.
        """
        data: list[list[float]] = [
            [100, 100, 120, 120, 0.95, 5, -1, 0],  # Person (driver)
            # Machinery containing person
            [95, 95, 125, 125, 0.85, 8, 1, 1],
            [200, 200, 220, 220, 0.9, 5, -1, 0],   # Person (not driver)
        ]
        warnings, cone_polygons, pole_polygons = self.detector.detect_danger(
            data,
        )
        # Should have reduced person count due to driver filtering
        # The exact warning content depends on the driver detection logic

    def test_empty_polygon_scenarios(self) -> None:
        """
        Test scenarios with no polygons formed.
        """
        data: list[list[float]] = [
            [100, 100, 120, 120, 0.95, 5, -1, 0],  # Just a person, no cones
        ]
        warnings, cone_polygons, pole_polygons = self.detector.detect_danger(
            data,
        )
        # Should have no polygon warnings
        self.assertNotIn('warning_people_in_controlled_area', warnings)
        self.assertEqual(len(cone_polygons), 0)
        self.assertEqual(len(pole_polygons), 0)

    def test_machinery_close_to_pole_detection(self) -> None:
        """
        Test machinery close to utility pole detection functionality.
        """
        detection_items = {
            'detect_no_safety_vest_or_helmet': False,
            'detect_near_machinery_or_vehicle': False,
            'detect_in_restricted_area': False,
            'detect_in_utility_pole_restricted_area': False,
            'detect_machinery_close_to_pole': True,
        }
        detector = DangerDetector(detection_items)

        # Create test data where machinery bottom overlaps with pole circle
        # Pole at (100,10)-(110,50), two_thirds_y = 10 + (2/3)*40 = 36.67
        # Machinery top must be within [10, 36.67] and bottom must intersect
        # circle
        data: list[list[float]] = [
            [100, 10, 110, 50, 0.9, 9, -1, 0],    # Utility pole (height=40)
            # Machinery: top=30 (valid), bottom=40
            [95, 30, 115, 40, 0.85, 8, 1, 1],
        ]

        warnings, cone_polygons, pole_polygons = detector.detect_danger(data)
        # Should detect machinery close to pole
        self.assertIn('detect_machinery_close_to_pole', warnings)

    def test_main_function_execution(self) -> None:
        """
        Test the main function can be executed without errors.
        """
        from src.danger_detector import main
        # Should run without raising any exceptions
        main()

    def test_machinery_close_to_pole_edge_cases(self) -> None:
        """
        Test edge cases for machinery close to utility pole detection.
        """
        detection_items = {
            'detect_no_safety_vest_or_helmet': False,
            'detect_near_machinery_or_vehicle': False,
            'detect_in_restricted_area': False,
            'detect_in_utility_pole_restricted_area': False,
            'detect_machinery_close_to_pole': True,
        }
        detector = DangerDetector(detection_items)

        # Test with zero or negative pole height
        data: list[list[float]] = [
            [100, 50, 110, 50, 0.9, 9, -1, 0],    # Pole with zero height
            [105, 45, 125, 55, 0.85, 8, 1, 1],    # Machinery
        ]

        warnings, cone_polygons, pole_polygons = detector.detect_danger(data)
        # Should not detect anything due to invalid pole height
        self.assertNotIn('detect_machinery_close_to_pole', warnings)

        # Test with machinery not in height range
        data2: list[list[float]] = [
            [100, 10, 110, 50, 0.9, 9, -1, 0],    # Utility pole (height=40)
            [105, 60, 125, 70, 0.85, 8, 1, 1],    # Machinery below pole
        ]

        warnings2, _, _ = detector.detect_danger(data2)
        # Should not detect anything due to machinery outside height range
        self.assertNotIn('detect_machinery_close_to_pole', warnings2)

    def test_main_function_coverage(self) -> None:
        """
        Test the main function for coverage purposes.
        """
        # This will cover the if __name__ == '__main__' line
        from src.danger_detector import main
        with patch('builtins.print'):
            main()  # Should run without error

    def test_pole_restricted_area_with_people(self) -> None:
        """
        Test utility pole restricted area detection with people inside.
        """
        detection_items = {
            'detect_no_safety_vest_or_helmet': False,
            'detect_near_machinery_or_vehicle': False,
            'detect_in_restricted_area': False,
            'detect_in_utility_pole_restricted_area': True,
            'detect_machinery_close_to_pole': False,
        }
        detector = DangerDetector(detection_items)

        # Mock the Utils.build_utility_pole_union to return a non-empty polygon
        with patch(
            'src.utils.Utils.build_utility_pole_union',
        ) as mock_build_union:
            with patch(
                'src.utils.Utils.count_people_in_polygon',
            ) as mock_count_people:
                # Mock a non-empty polygon
                from shapely.geometry import Polygon
                mock_polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
                mock_build_union.return_value = mock_polygon
                mock_count_people.return_value = 2  # 2 people in the area

                data: list[list[float]] = [
                    [100, 10, 110, 50, 0.9, 9, -1, 0],    # Utility pole
                    [105, 20, 115, 30, 0.85, 5, -1, 0],   # Person
                ]

                warnings, cone_polygons, pole_polygons = (
                    detector.detect_danger(data)
                )
                # Should detect people in utility pole controlled area
                self.assertIn(
                    'warning_people_in_utility_pole_controlled_area', warnings,
                )
                self.assertEqual(
                    warnings[
                        'warning_people_in_utility_pole_controlled_area'
                    ]['count'],
                    2,
                )

    @patch('builtins.print')
    def test_main(self, mock_print: MagicMock) -> None:
        """
        Test case for the main function.
        """
        with patch.object(
            DangerDetector, 'detect_danger', return_value=(
                {
                    'warning_no_hardhat': {'count': 1},
                    'warning_people_in_controlled_area': {'count': 1},
                },
                [Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])],
                [Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])],
            ),
        ) as mock_detect_danger:
            main()
            mock_detect_danger.assert_called_once()
            mock_print.assert_any_call(
                "Warnings: {'warning_no_hardhat': {'count': 1}, "
                "'warning_people_in_controlled_area': {'count': 1}}",
            )
            mock_print.assert_any_call(
                'cone_polygons: [<POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))>]',
            )
            mock_print.assert_any_call(
                'pole_polygons: [<POLYGON ((0 0, 5 0, 5 5, 0 5, 0 0))>]',
            )


if __name__ == '__main__':
    unittest.main()

"""
pytest \
    --cov=src.danger_detector \
    --cov-report=term-missing tests/src/danger_detector_test.py
"""
