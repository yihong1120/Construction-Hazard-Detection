import unittest

from site_safety_monitor import detect_danger, is_too_close
import site_safety_monitor


class TestSiteSafetyMonitor(unittest.TestCase):
    def test_detect_danger_no_helmet(self):
        detections = [
            {"label": "NO-Hardhat", "bbox": [100, 100, 200, 200]},
            {"label": "Person", "bbox": [150, 150, 250, 250]},
        ]
        warnings = detect_danger(detections)
        self.assertEqual(len(warnings), 1)
        self.assertIn("Someone is not wearing a helmet", warnings[0])

    def test_detect_danger_no_safety_vest(self):
        detections = [
            {"label": "NO-Safety Vest", "bbox": [100, 100, 200, 200]},
            {"label": "Person", "bbox": [150, 150, 250, 250]},
        ]
        warnings = detect_danger(detections)
        self.assertEqual(len(warnings), 1)
        self.assertIn("Someone is not wearing a safety vest", warnings[0])

    def test_detect_danger_person_too_close_to_machinery(self):
        detections = [
            {"label": "Person", "bbox": [100, 100, 200, 200]},
            {"label": "machinery", "bbox": [150, 150, 250, 250]},
        ]
        warnings = detect_danger(detections)
        self.assertEqual(len(warnings), 1)
        self.assertIn("There is a person approaching the machinery or vehicle", warnings[0])

    def test_is_too_close_distance_less_than_threshold(self):
        bbox1 = (100, 100, 200, 200)
        bbox2 = (150, 150, 250, 250)
        self.assertTrue(is_too_close(bbox1, bbox2))
        self.assertTrue(is_too_close(bbox1, bbox2))

    def test_is_too_close_distance_greater_than_threshold(self):
        bbox1 = (100, 100, 200, 200)
        bbox2 = (300, 300, 400, 400)
        self.assertFalse(is_too_close(bbox1, bbox2))

    def test_is_too_close_distance_equal_to_threshold(self):
        bbox1 = (100, 100, 200, 200)
        bbox2 = (250, 250, 350, 350)
        self.assertFalse(is_too_close(bbox1, bbox2))

if __name__ == "__main__":
    unittest.main()
