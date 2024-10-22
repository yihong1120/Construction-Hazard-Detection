from __future__ import annotations

import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

from shapely.geometry import Polygon

from src.danger_detector import DangerDetector
from src.danger_detector import main


class TestDangerDetector(unittest.TestCase):
    """
    Unit tests for the DangerDetector class methods.
    """

    def setUp(self) -> None:
        """
        Set up method to create an instance of DangerDetector for each test.
        """
        self.detector: DangerDetector = DangerDetector()

    def test_is_driver(self) -> None:
        """
        Test case for checking if a person is driving based on bounding boxes.
        """
        person_bbox: list[float] = [100, 200, 150, 250]
        vehicle_bbox: list[float] = [50, 100, 200, 300]
        self.assertTrue(self.detector.is_driver(person_bbox, vehicle_bbox))

        person_bbox = [100, 200, 200, 400]
        self.assertFalse(self.detector.is_driver(person_bbox, vehicle_bbox))

    def test_overlap_percentage(self) -> None:
        """
        Test calculating overlap percentage between two bounding boxes.
        """
        bbox1: list[float] = [100, 100, 200, 200]
        bbox2: list[float] = [150, 150, 250, 250]
        self.assertAlmostEqual(
            self.detector.overlap_percentage(bbox1, bbox2), 0.142857, places=6,
        )

        bbox1 = [100, 100, 200, 200]
        bbox2 = [300, 300, 400, 400]
        self.assertEqual(
            self.detector.overlap_percentage(
                self.detector.normalise_bbox(
                    bbox1,
                ), self.detector.normalise_bbox(bbox2),
            ), 0.0,
        )

    def test_is_dangerously_close(self) -> None:
        """
        Test case for checking if a person is dangerously close to a vehicle.
        """
        person_bbox: list[float] = [100, 100, 120, 120]
        vehicle_bbox: list[float] = [100, 100, 200, 200]
        self.assertTrue(
            self.detector.is_dangerously_close(
                self.detector.normalise_bbox(
                    person_bbox,
                ), self.detector.normalise_bbox(vehicle_bbox), 'Vehicle',
            ),
        )

        person_bbox = [0, 0, 10, 10]
        vehicle_bbox = [100, 100, 200, 200]
        self.assertFalse(
            self.detector.is_dangerously_close(
                self.detector.normalise_bbox(
                    person_bbox,
                ), self.detector.normalise_bbox(vehicle_bbox), 'Vehicle',
            ),
        )

    def test_calculate_people_in_controlled_area(self) -> None:
        """
        Test case for calculating the number of people in the controlled area.
        """
        datas: list[list[float]] = [
            [50, 50, 150, 150, 0.95, 0],    # 安全帽
            [200, 200, 300, 300, 0.85, 5],  # 人員
            [400, 400, 500, 500, 0.75, 9],  # 車輛
        ]
        normalised_datas = self.detector.normalise_data(datas)
        polygons = self.detector.detect_polygon_from_cones(normalised_datas)
        people_count = self.detector.calculate_people_in_controlled_area(
            polygons, normalised_datas,
        )
        self.assertEqual(people_count, 0)

        datas = [
            [100, 100, 120, 120, 0.9, 6],  # Safety cone
            [150, 150, 170, 170, 0.85, 6],  # Safety cone
            [130, 130, 140, 140, 0.95, 5],  # Person inside the area
            [300, 300, 320, 320, 0.85, 5],  # Person outside the area
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

        normalised_datas = self.detector.normalise_data(datas)
        polygons = self.detector.detect_polygon_from_cones(normalised_datas)
        people_count = self.detector.calculate_people_in_controlled_area(
            polygons, normalised_datas,
        )
        self.assertEqual(people_count, 1)

    def test_detect_danger(self) -> None:
        """
        Test case for detecting danger based on a list of data points.
        """
        data: list[list[float]] = [
            [50, 50, 150, 150, 0.95, 0],    # Hardhat
            [200, 200, 300, 300, 0.85, 5],  # Person
            [400, 400, 500, 500, 0.75, 2],  # No-Safety Vest
            [0, 0, 10, 10, 0.88, 6],
            [0, 1000, 10, 1010, 0.87, 6],
            [1000, 0, 1010, 10, 0.89, 6],
            [100, 100, 120, 120, 0.9, 6],  # Safety cone
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
        data = self.detector.normalise_data(data)
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
        data = self.detector.normalise_data(data)
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
        data = self.detector.normalise_data(data)
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

    def test_no_cones(self) -> None:
        """
        Test case for checking behavior when no cones are detected.
        """
        data: list[list[float]] = [
            [50, 50, 150, 150, 0.95, 0],  # Hardhat
            [200, 200, 300, 300, 0.85, 5],  # Person
            [400, 400, 500, 500, 0.75, 2],  # No-Safety Vest
        ]
        normalised_data = self.detector.normalise_data(data)
        polygons = self.detector.detect_polygon_from_cones(normalised_data)
        self.assertEqual(len(polygons), 0)

    def test_person_inside_polygon(self) -> None:
        """
        Test case for checking behavior when a person is inside a polygon.
        """
        polygons = [
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        ]
        data: list[list[float]] = [
            [2, 2, 8, 8, 0.95, 5],  # Person inside the polygon
        ]
        normalised_data = self.detector.normalise_data(data)
        people_count = self.detector.calculate_people_in_controlled_area(
            polygons, normalised_data,
        )
        self.assertEqual(people_count, 1)

    def test_vehicle_too_close(self) -> None:
        """
        Test case for checking behaviour
        when a person is too close to a vehicle.
        """
        data: list[list[float]] = [
            [100, 100, 120, 120, 0.95, 5],  # Person
            [110, 110, 200, 200, 0.85, 8],  # Machinery
        ]
        normalised_data = self.detector.normalise_data(data)
        print(f"Normalized data: {normalised_data}")
        warnings, polygons = self.detector.detect_danger(normalised_data)
        self.assertIn('Warning: Someone is too close to machinery!', warnings)

    def test_driver_detection_coverage(self) -> None:
        """
        Test case to ensure driver detection code coverage.
        """
        # Case where person is likely the driver
        person_bbox = self.detector.normalise_bbox([150, 250, 170, 350])
        vehicle_bbox = self.detector.normalise_bbox([100, 200, 300, 400])
        self.assertTrue(self.detector.is_driver(person_bbox, vehicle_bbox))

        # Case where person is not the driver
        # due to horizontal position (left outside bounds)
        person_bbox = self.detector.normalise_bbox([50, 250, 90, 300])
        self.assertFalse(self.detector.is_driver(person_bbox, vehicle_bbox))

        # Case where person is not the driver
        # due to horizontal position (right outside bounds)
        person_bbox = self.detector.normalise_bbox([310, 250, 350, 300])
        self.assertFalse(self.detector.is_driver(person_bbox, vehicle_bbox))

        # Case where person is not the driver
        # due to vertical position (above vehicle)
        person_bbox = self.detector.normalise_bbox([100, 50, 150, 100])
        self.assertFalse(self.detector.is_driver(person_bbox, vehicle_bbox))

        # Case where person is the driver
        # due to person's top being below vehicle's top
        person_bbox = self.detector.normalise_bbox([150, 210, 180, 300])
        self.assertTrue(self.detector.is_driver(person_bbox, vehicle_bbox))

        # Case where person is not the driver
        # due to person's height being more than half vehicle's height
        person_bbox = self.detector.normalise_bbox([150, 300, 180, 450])
        self.assertFalse(self.detector.is_driver(person_bbox, vehicle_bbox))

        person_bbox = self.detector.normalise_bbox([80, 250, 110, 300])
        print(
            f"Testing with person_bbox: "
            f"{person_bbox} and vehicle_bbox: {vehicle_bbox}",
        )
        self.assertFalse(self.detector.is_driver(person_bbox, vehicle_bbox))

    def test_horizontal_position_check(self) -> None:
        """
        Test case to ensure coverage for horizontal position check.
        """
        # Case where person is within the acceptable horizontal bounds
        person_bbox = self.detector.normalise_bbox([200, 200, 240, 240])
        vehicle_bbox = self.detector.normalise_bbox([190, 150, 250, 300])
        self.assertTrue(self.detector.is_driver(person_bbox, vehicle_bbox))

        # Case where person is to the left of the acceptable horizontal bounds
        person_bbox = self.detector.normalise_bbox([50, 200, 90, 240])
        vehicle_bbox = self.detector.normalise_bbox([100, 150, 200, 300])
        self.assertFalse(self.detector.is_driver(person_bbox, vehicle_bbox))

        # Case where person is to the right of the acceptable horizontal bounds
        person_bbox = self.detector.normalise_bbox([210, 200, 250, 240])
        vehicle_bbox = self.detector.normalise_bbox([100, 150, 200, 300])
        self.assertFalse(self.detector.is_driver(person_bbox, vehicle_bbox))

        # Case where person is exactly on the left boundary
        person_bbox = self.detector.normalise_bbox([150, 200, 190, 240])
        vehicle_bbox = self.detector.normalise_bbox([190, 150, 250, 300])
        self.assertFalse(self.detector.is_driver(person_bbox, vehicle_bbox))

        # Case where person is exactly on the right boundary
        person_bbox = self.detector.normalise_bbox([250, 200, 290, 240])
        vehicle_bbox = self.detector.normalise_bbox([190, 150, 250, 300])
        self.assertFalse(self.detector.is_driver(person_bbox, vehicle_bbox))

        # Case where person's height is exactly half the vehicle's height
        person_bbox = self.detector.normalise_bbox([100, 200, 150, 300])
        vehicle_bbox = self.detector.normalise_bbox([50, 100, 200, 400])
        self.assertTrue(self.detector.is_driver(person_bbox, vehicle_bbox))

        # Case where person's height is more than half the vehicle's height
        person_bbox = self.detector.normalise_bbox([100, 200, 150, 350])
        self.assertFalse(self.detector.is_driver(person_bbox, vehicle_bbox))

    def test_height_check(self) -> None:
        """
        Test case to ensure coverage for height check.
        """
        person_bbox = self.detector.normalise_bbox([150, 250, 180, 400])
        vehicle_bbox = self.detector.normalise_bbox([100, 200, 300, 300])
        self.assertFalse(self.detector.is_driver(person_bbox, vehicle_bbox))

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
