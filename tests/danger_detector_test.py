from __future__ import annotations

import unittest

from shapely.geometry import MultiPoint
from shapely.geometry import Polygon

from src.danger_detector import DangerDetector


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
        self.assertEqual(self.detector.overlap_percentage(bbox1, bbox2), 0.0)

    def test_is_dangerously_close(self) -> None:
        """
        Test case for checking if a person is dangerously close to a vehicle.
        """
        person_bbox: list[float] = [100, 100, 120, 120]
        vehicle_bbox: list[float] = [100, 100, 200, 200]
        self.assertTrue(
            self.detector.is_dangerously_close(
                person_bbox, vehicle_bbox, '車輛',
            ),
        )

        person_bbox = [0, 0, 10, 10]
        vehicle_bbox = [100, 100, 200, 200]
        self.assertFalse(
            self.detector.is_dangerously_close(
                person_bbox, vehicle_bbox, '車輛',
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
        polygon: Polygon = MultiPoint(
            [(100, 100), (250, 250), (450, 450), (500, 200), (150, 400)],
        ).convex_hull
        people_count: int = self.detector.calculate_people_in_controlled_area(
            datas, polygon,
        )
        self.assertEqual(people_count, 1)

        datas = [
            [100, 100, 120, 120, 0.9, 6],  # Safety cone
            [150, 150, 170, 170, 0.85, 6],  # Safety cone
            [130, 130, 140, 140, 0.95, 5],  # Person inside the area
            [300, 300, 320, 320, 0.85, 5],  # Person outside the area
        ]
        # Not enough cones to form a polygon
        polygon = MultiPoint([(100, 100), (150, 150)]).convex_hull
        people_count = self.detector.calculate_people_in_controlled_area(
            datas, None,
        )  # should pass None
        self.assertEqual(people_count, 0)

    def test_detect_danger(self) -> None:
        """
        Test case for detecting danger based on a list of data points.
        """
        data: list[list[float]] = [
            [50, 50, 150, 150, 0.95, 0],    # 安全帽
            [200, 200, 300, 300, 0.85, 5],  # 人員
            [400, 400, 500, 500, 0.75, 2],  # 無安背心
        ]
        polygon: Polygon = MultiPoint(
            [(100, 100), (250, 250), (450, 450), (500, 200), (150, 400)],
        ).convex_hull
        warnings: set[str] = self.detector.detect_danger(data, polygon)
        print(f"warnings: {warnings}")
        self.assertIn('警告: 有1個人進入受控區域!', warnings)
        self.assertIn('警告: 有人無配戴安全帽!', warnings)

        data = [
            [706.87, 445.07, 976.32, 1073.6, 0.91, 5.0],  # Person
            [0.45513, 471.77, 662.03, 1071.4, 0.75853, 5.0],  # Person
            [1042.7, 638.5, 1077.5, 731.98, 0.56060, 4.0],  # No safety vest
            [500, 500, 700, 700, 0.95, 8],  # Machinery
        ]
        polygon = MultiPoint([(100, 100), (150, 150), (200, 200)]).convex_hull
        warnings = self.detector.detect_danger(data, polygon)
        self.assertNotIn('警告: 有人無配戴安全帽!', warnings)
        self.assertIn('警告: 有人無穿著安全背心!', warnings)

        data = [
            [706.87, 445.07, 976.32, 1073.6, 0.91, 2.0],  # No hardhat
            [0.45513, 471.77, 662.03, 1071.4, 0.75853, 4.0],  # No safety vest
            [500, 500, 700, 700, 0.95, 8],  # Machinery
        ]
        polygon = MultiPoint([(100, 100), (150, 150), (200, 200)]).convex_hull
        warnings = self.detector.detect_danger(data, polygon)
        self.assertIn('警告: 有人無配戴安全帽!', warnings)
        self.assertIn('警告: 有人無穿著安全背心!', warnings)
        self.assertNotIn('警告: 有人過於靠近機具!', warnings)


if __name__ == '__main__':
    unittest.main()
