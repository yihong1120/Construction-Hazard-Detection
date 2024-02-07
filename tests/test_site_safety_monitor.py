import unittest
from src.site_safety_monitor import detect_danger, is_too_close

class TestSiteSafetyMonitor(unittest.TestCase):
    """
    Test cases for site_safety_monitor.py.
    """

    def test_detect_danger(self):
        """
        Test the detect_danger function to ensure it generates correct warnings based on detections.
        """
        # Define a test case with known detections
        test_detections = [
            {"label": "NO-Hardhat", "bbox": [100, 100, 200, 200]},
            {"label": "Person", "bbox": [150, 150, 250, 250]},
            {"label": "machinery", "bbox": [300, 300, 400, 400]},
        ]

        # Call the detect_danger function with the test detections
        warnings = detect_danger(test_detections)

        # Define the expected warnings
        expected_warnings = [
            "Warning: Someone is not wearing a helmet! Location: [100, 100, 200, 200]",
        ]

        # Assert that the warnings match the expected warnings
        self.assertEqual(warnings, expected_warnings, "The warnings should match the expected warnings")

    def test_is_too_close(self):
        """
        Test the is_too_close function to ensure it correctly determines if bounding boxes are too close.
        """
        # Define two bounding boxes that are too close
        bbox1 = (100, 100, 200, 200)
        bbox2 = (150, 150, 250, 250)

        # Call the is_too_close function with the bounding boxes
        result = is_too_close(bbox1, bbox2)

        # Assert that the result is True, indicating the bounding boxes are too close
        self.assertTrue(result, "The bounding boxes should be too close")

        # Define two bounding boxes that are not too close
        bbox1 = (100, 100, 200, 200)
        bbox2 = (300, 300, 400, 400)

        # Call the is_too_close function with the bounding boxes
        result = is_too_close(bbox1, bbox2)

        # Assert that the result is False, indicating the bounding boxes are not too close
        self.assertFalse(result, "The bounding boxes should not be too close")

if __name__ == '__main__':
    unittest.main()