import unittest
from src.danger_detector import DangerDetector

class TestDangerDetector(unittest.TestCase):
    """
    A class to test the DangerDetector functionalities.
    """

    def setUp(self):
        """
        Set up the DangerDetector instance before each test.
        """
        self.detector = DangerDetector()

    def test_no_violations(self):
        """
        Test that no warnings are generated when there are no violations.
        """
        datas = [
            [100, 200, 300, 400, 0.95, 5.0]  # Person with no violations
        ]
        warnings = self.detector.detect_danger(datas)
        self.assertEqual(len(warnings), 0, "Expected no warnings when there are no violations.")

    def test_hardhat_violations(self):
        """
        Test detection of workers not wearing hardhats.
        """
        datas = [
            [100, 200, 300, 400, 0.95, 5.0],  # Person
            [100, 200, 300, 400, 0.90, 2.0]  # No hardhat
        ]
        warnings = self.detector.detect_danger(datas)
        self.assertIn("Warning: Someone is not wearing a hardhat!", warnings, "Expected a hardhat violation warning.")

    def test_safety_vest_violations(self):
        """
        Test detection of workers not wearing safety vests.
        """
        datas = [
            [100, 200, 300, 400, 0.95, 5.0],  # Person
            [100, 200, 300, 400, 0.90, 4.0]  # No safety vest
        ]
        warnings = self.detector.detect_danger(datas)
        self.assertIn("Warning: Someone is not wearing a safety vest!", warnings, "Expected a safety vest violation warning.")

    def test_proximity_to_machinery(self):
        """
        Test detection of workers dangerously close to machinery.
        """
        datas = [
            [100, 200, 300, 400, 0.95, 5.0],  # Person
            [290, 350, 410, 450, 0.88, 8.0]  # Machinery
        ]
        warnings = self.detector.detect_danger(datas)
        self.assertIn("Warning: Someone is dangerously close to machinery!", warnings, "Expected a proximity warning for machinery.")

    def test_proximity_to_vehicles(self):
        """
        Test detection of workers dangerously close to vehicles.
        """
        datas = [
            [100, 200, 300, 400, 0.95, 5.0],  # Person
            [290, 350, 410, 450, 0.88, 9.0]  # Vehicle
        ]
        warnings = self.detector.detect_danger(datas)
        self.assertIn("Warning: Someone is dangerously close to a vehicle!", warnings, "Expected a proximity warning for vehicles.")

if __name__ == '__main__':
    unittest.main()
