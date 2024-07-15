import unittest
from src.stream_capture import StreamCapture

class TestStreamCapture(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Setup
        url = "https://cctv4.kctmc.nat.gov.tw/50204bfc/"
        capture_interval = 15
        cls.stream_capture = StreamCapture(url, capture_interval)
    
    def test_initialisation(self):
        self.assertIsInstance(self.stream_capture, StreamCapture)
    
    def test_capture_interval_update(self):
        new_interval = 10
        self.stream_capture.update_capture_interval(new_interval)
        self.assertEqual(self.stream_capture.capture_interval, new_interval)
    
    def test_select_quality_based_on_speed(self):
        stream_url = self.stream_capture.select_quality_based_on_speed()
        self.assertIsInstance(stream_url, (str, type(None)))
    
    def test_check_internet_speed(self):
        download_speed, upload_speed = self.stream_capture.check_internet_speed()
        self.assertIsInstance(download_speed, float)
        self.assertIsInstance(upload_speed, float)
        self.assertGreaterEqual(download_speed, 0)
        self.assertGreaterEqual(upload_speed, 0)
    
    def test_execute_capture(self):
        generator = self.stream_capture.execute_capture()
        self.assertTrue(hasattr(generator, '__iter__'))
        # Assuming we want to test the first frame and timestamp
        frame, timestamp = next(generator)
        self.assertIsNotNone(frame)
        self.assertIsInstance(timestamp, float)
        # Release resources
        del frame
        self.stream_capture.release_resources()
    
    @classmethod
    def tearDownClass(cls):
        # Teardown or release resources if needed
        cls.stream_capture.release_resources()

if __name__ == '__main__':
    unittest.main()
