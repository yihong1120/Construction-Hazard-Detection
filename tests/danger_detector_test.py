from __future__ import annotations

import unittest

from examples.user_management.models import User
from src.danger_detector import DangerDetector


class TestDangerDetector(unittest.TestCase):
    def setUp(self):
        self.detector = DangerDetector()

    def test_set_password(self):
        user = User(username='test_user')
        user.set_password('password')
        self.assertTrue(user.check_password('password'))
        self.assertFalse(user.check_password('wrong_password'))

    def test_is_driver(self):
        person_bbox = [100, 200, 150, 250]
        vehicle_bbox = [50, 100, 200, 300]
        self.assertTrue(self.detector.is_driver(person_bbox, vehicle_bbox))

        person_bbox = [100, 200, 200, 400]
        self.assertFalse(self.detector.is_driver(person_bbox, vehicle_bbox))

    def test_overlap_percentage(self):
        bbox1 = [100, 100, 200, 200]
        bbox2 = [150, 150, 250, 250]
        self.assertAlmostEqual(
            self.detector.overlap_percentage(
                bbox1, bbox2,
            ), 0.142857, places=6,
        )

        bbox1 = [100, 100, 200, 200]
        bbox2 = [300, 300, 400, 400]
        self.assertEqual(self.detector.overlap_percentage(bbox1, bbox2), 0.0)

    def test_is_dangerously_close(self):
        person_bbox = [100, 100, 120, 120]
        vehicle_bbox = [100, 100, 200, 200]
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

    def test_detect_danger(self):
        data = [
            [706.87, 445.07, 976.32, 1073.6, 3, 5.0],
            [0.45513, 471.77, 662.03, 1071.4, 12, 5.0],
            [1042.7, 638.5, 1077.5, 731.98, 18, 2.0],
        ]
        warnings = self.detector.detect_danger(data)
        self.assertIn('警告: 有人無配戴安全帽!', warnings)
        self.assertNotIn('警告: 有人無穿著安全背心!', warnings)


if __name__ == '__main__':
    unittest.main()
