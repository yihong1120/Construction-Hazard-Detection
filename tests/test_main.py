import unittest
from unittest.mock import patch, MagicMock
from main import main, process_stream

class TestMain(unittest.TestCase):
    """
    Test cases for main.py.
    """

    @patch('src.live_stream_detection.LiveStreamDetector')
    @patch('src.line_notifier.LineNotifier')
    @patch('src.danger_detector.DangerDetector')
    @patch('src.monitor_logger.LoggerConfig')
    def test_main(self, mock_logger_config, mock_danger_detector, mock_line_notifier, mock_live_stream_detector):
        """
        Test the main function to ensure it detects hazards, sends notifications, and logs warnings.
        """
        # Mock configurations
        video_url = "http://example.com/live"
        model_path = "path/to/model.pt"
        image_path = "path/to/image.png"
        line_token = "dummy_token"

        # Mock the logger to avoid creating actual log files
        mock_logger = MagicMock()
        mock_logger_config.return_value.get_logger.return_value = mock_logger

        # Mock LiveStreamDetector's method
        mock_live_stream_detector_instance = MagicMock()
        mock_live_stream_detector.return_value = mock_live_stream_detector_instance
        mock_live_stream_detector_instance.generate_detections.return_value = iter([
            (['dummy_data'], 'dummy_frame', 1609459200),  # Sample detection data
        ])

        # Mock DangerDetector's method
        mock_danger_detector_instance = MagicMock()
        mock_danger_detector.return_value = mock_danger_detector_instance
        mock_danger_detector_instance.detect_danger.return_value = ["Warning: Hazard detected"]

        # Mock LineNotifier's method
        mock_line_notifier_instance = MagicMock()
        mock_line_notifier.return_value = mock_line_notifier_instance
        mock_line_notifier_instance.send_notification.return_value = 200

        # Call the main function with mock parameters
        main(mock_logger, video_url, model_path, image_path, line_token)

        # Assert that the DangerDetector's detect_danger method was called once
        mock_danger_detector_instance.detect_danger.assert_called_once()

        # Assert that the LineNotifier's send_notification method was called with the expected message
        mock_line_notifier_instance.send_notification.assert_called()

        # Assert that the logger's warning method was called with the expected message
        mock_logger.warning.assert_called_with("Notification sent successfully: [2021-01-01 00:00:00] Warning: Hazard detected")

if __name__ == '__main__':
    unittest.main()
