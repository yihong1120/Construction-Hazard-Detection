import unittest
from src.danger_detector import DangerDetector

class TestDangerDetector(unittest.TestCase):
    def setUp(self):
        self.detector = DangerDetector()

    def test_is_driver(self):
        # Define test cases for the is_driver method
        person_bbox = (100, 100, 200, 200)  # x1, y1, x2, y2
        vehicle_bbox = (150, 150, 300, 300)
        self.assertTrue(self.detector.is_driver(person_bbox, vehicle_bbox))

        # Add more test cases as needed

    def test_overlap_percentage(self):
        # Define test cases for the overlap_percentage method
        bbox1 = (100, 100, 200, 200)
        bbox2 = (150, 150, 250, 250)
        overlap = self.detector.overlap_percentage(bbox1, bbox2)
        self.assertEqual(overlap, 0.14285714285714285)

        # Add more test cases as needed

    def test_is_dangerously_close(self):
        # Define test cases for the is_dangerously_close method
        person_bbox = (100, 100, 200, 200)
        vehicle_bbox = (150, 150, 300, 300)
        label = 'vehicle'
        self.assertTrue(self.detector.is_dangerously_close(person_bbox, vehicle_bbox, label))

        # Add more test cases as needed

    def test_detect_danger(self):
        # Define test cases for the detect_danger method
        datas = [
            [706.87, 445.07, 976.32, 1073.6, 0.91, 5],  # Person
            [0.45513, 471.77, 662.03, 1071.4, 0.75853, 2],  # No hardhat
            [1042.7, 638.5, 1077.5, 731.98, 0.56060, 8]  # Machinery
        ]
        warnings = self.detector.detect_danger(datas)
        expected_warnings = [
            "Warning: Someone is not wearing a hardhat!",
            "Warning: Someone is dangerously close to machinery!"
        ]
        self.assertEqual(warnings, expected_warnings)

        # Add more test cases as needed

if __name__ == '__main__':
    unittest.main()
