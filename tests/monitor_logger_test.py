from __future__ import annotations

import logging
import unittest
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

from src.monitor_logger import LoggerConfig  # Adjust the import path as needed


class TestLoggerConfig(unittest.TestCase):
    def setUp(self):
        self.log_file = 'test.log'
        self.log_dir = 'test_logs'
        self.level = logging.DEBUG
        self.formatter = logging.Formatter('%(message)s')

        # Ensure log directory exists
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        self.logger_config = LoggerConfig(
            log_file=self.log_file,
            log_dir=self.log_dir,
            level=self.level,
            formatter=self.formatter,
        )

    @patch('src.monitor_logger.Path.mkdir')
    @patch('src.monitor_logger.RotatingFileHandler')
    @patch('src.monitor_logger.logging.StreamHandler')
    def test_setup_logger(
        self, mock_stream_handler, mock_rotating_file_handler, mock_mkdir,
    ):
        # Create mock handlers
        mock_file_handler = MagicMock()
        mock_console_handler = MagicMock()
        mock_rotating_file_handler.return_value = mock_file_handler
        mock_stream_handler.return_value = mock_console_handler

        # Set level attribute for the handlers
        mock_file_handler.level = self.level
        mock_console_handler.level = self.level

        # Run setup_logger
        self.logger_config.setup_logger()

        # Verify handlers were added to the logger
        logger = self.logger_config.get_logger()
        handlers = logger.handlers

        print(f"Handlers: {handlers}")  # Debug information to see the handlers

        self.assertIn(mock_file_handler, handlers)
        self.assertIn(mock_console_handler, handlers)

        # Verify the log directory was created
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

        # Verify file handler configuration
        mock_rotating_file_handler.assert_called_once_with(
            filename=Path(self.log_dir) / self.log_file,
            maxBytes=1_000_000,
            backupCount=5,
        )
        mock_file_handler.setLevel.assert_called_once_with(self.level)
        mock_file_handler.setFormatter.assert_called_once_with(self.formatter)

        # Verify console handler configuration
        mock_console_handler.setLevel.assert_called_once_with(self.level)
        mock_console_handler.setFormatter.assert_called_once_with(
            self.formatter,
        )

    def test_get_file_handler(self):
        file_handler = self.logger_config.get_file_handler()
        self.assertIsInstance(file_handler, logging.Handler)
        self.assertEqual(file_handler.level, self.level)
        self.assertEqual(file_handler.formatter, self.formatter)

    def test_get_console_handler(self):
        console_handler = self.logger_config.get_console_handler()
        self.assertIsInstance(console_handler, logging.Handler)
        self.assertEqual(console_handler.level, self.level)
        self.assertEqual(console_handler.formatter, self.formatter)

    @patch('src.monitor_logger.Path.mkdir')
    @patch('src.monitor_logger.RotatingFileHandler')
    @patch('src.monitor_logger.logging.StreamHandler')
    def test_logger_output(
        self, mock_stream_handler, mock_rotating_file_handler, mock_mkdir,
    ):
        mock_file_handler = MagicMock()
        mock_console_handler = MagicMock()
        mock_rotating_file_handler.return_value = mock_file_handler
        mock_stream_handler.return_value = mock_console_handler

        # Initialize logger configuration
        from src.monitor_logger import LoggerConfig
        self.logger_config = LoggerConfig(
            log_file='test.log', log_dir='logs_test',
        )

        # Mock the mkdir function to avoid actual directory creation
        mock_mkdir.return_value = None

        # Test logger setup
        self.logger_config.setup_logger()
        logger = self.logger_config.get_logger()

        # Set level attribute for the handlers in case they are checked
        mock_file_handler.level = logging.DEBUG
        mock_console_handler.level = logging.DEBUG

        # Test logging output
        with self.assertLogs(logger, level='INFO') as log:
            logger.info('Test log message')
            expected_message = 'INFO:SiteSafetyMonitor_test.log:Test log message'.upper()
            log_messages = [msg.upper() for msg in log.output]
            self.assertIn(expected_message, log_messages)

    def tearDown(self):
        # Clean up any files or directories created during tests
        if Path(self.log_dir).exists():
            for file in Path(self.log_dir).glob('*'):
                file.unlink()
            Path(self.log_dir).rmdir()


if __name__ == '__main__':
    unittest.main()
