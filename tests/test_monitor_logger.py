import unittest
from src.monitor_logger import LoggerConfig
import logging
import os

class TestMonitorLogger(unittest.TestCase):
    """
    Test cases for monitor_logger.py.
    """

    def test_logger_configuration(self):
        """
        Test the logger configuration to ensure it creates log files and logs messages correctly.
        """
        # Initialise the logger configuration using the LoggerConfig class
        logger_config = LoggerConfig(log_file='site_safety.log')
        logger = logger_config.get_logger()

        # Check if the logger is an instance of logging.Logger
        self.assertIsInstance(logger, logging.Logger, "The logger should be an instance of logging.Logger")

        # Check if the log file is created in the logs directory
        log_file_path = 'logs/site_safety.log'
        self.assertTrue(os.path.isfile(log_file_path), "Log file should exist")

        # Check if the logger can log a message
        test_message = "This is a test log message."
        logger.info(test_message)

        # Read the last line from the log file
        with open(log_file_path, 'r') as file:
            logs = file.readlines()
            last_log = logs[-1] if logs else ''

        # Check if the last log contains the test message
        self.assertIn(test_message, last_log, "The log file should contain the test log message")

        # Clean up the log file after the test
        os.remove(log_file_path)

if __name__ == '__main__':
    unittest.main()
