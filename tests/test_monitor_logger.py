import unittest
from src.monitor_logger import setup_logging
import logging
import os

class TestMonitorLogger(unittest.TestCase):
    """
    Test cases for monitor_logger.py.
    """

    def test_setup_logging(self):
        """
        Test the logging setup to ensure it creates log files and logs messages correctly.
        """
        # Call the setup_logging function from monitor_logger.py
        logger = setup_logging()

        # Check if the logger is an instance of logging.Logger
        self.assertIsInstance(logger, logging.Logger, "The logger should be an instance of logging.Logger")

        # Check if the log file is created
        log_file = 'site_safety.log'
        self.assertTrue(os.path.isfile(log_file), "Log file should exist")

        # Check if the logger can log a message
        test_message = "This is a test log message."
        logger.info(test_message)

        # Read the last line from the log file
        with open(log_file, 'r') as file:
            logs = file.readlines()
            last_log = logs[-1] if logs else ''

        # Check if the last log contains the test message
        self.assertIn(test_message, last_log, "The log file should contain the test log message")

        # Clean up the log file after the test
        os.remove(log_file)

if __name__ == '__main__':
    unittest.main()