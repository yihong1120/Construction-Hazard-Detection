import unittest
from unittest.mock import patch, MagicMock
from src.demo import main

class TestDemo(unittest.TestCase):
    """
    Test cases for demo.py.
    """

    @patch('src.demo.detect_danger')
    @patch('src.demo.send_line_notification')
    @patch('src.demo.setup_logging')
    def test_main(self, mock_setup_logging, mock_send_line_notification, mock_detect_danger):
        """
        Test the main function to ensure it detects hazards, sends notifications, and logs warnings.
        """
        # Mock the logger to avoid creating actual log files
        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger

        # Mock the detect_danger function to return a list of warnings
        mock_detect_danger.return_value = ["Warning: Someone is not wearing a helmet! Location: [100, 100, 200, 200]"]

        # Call the main function with the mocked logger
        main(mock_logger)

        # Assert that detect_danger was called once
        mock_detect_danger.assert_called_once()

        # Assert that send_line_notification was called with the expected token and message
        mock_send_line_notification.assert_called_once_with(
            'YOUR_LINE_NOTIFY_TOKEN',
            '[2024-02-07 15:35:06] Warning: Someone is not wearing a helmet! Location: [100, 100, 200, 200]'
        )

        # Assert that the logger's warning method was called with the expected message
        mock_logger.warning.assert_called_once_with(
            '[2024-02-07 15:35:06] Warning: Someone is not wearing a helmet! Location: [100, 100, 200, 200]'
        )

if __name__ == '__main__':
    unittest.main()