from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


class LoggerConfig:
    """
    Sets up app logger with console and file handlers.
    """

    def __init__(
        self,
        log_file='monitor.log',
        log_dir='logs',
        level=logging.INFO,
        formatter=None,
    ):
        """
        Initialise logger with file name, level, and formatter.

        Args:
            log_file (str): Log file name, defaults to 'monitor.log'.
            log_dir (str): Log storage directory, defaults to 'logs'.
            level (logging.Level): The logging level. Defaults to logging.INFO.
            formatter (logging.Formatter): Log formatter, defaults to standard.
        """
        self.log_file = log_file
        self.log_dir = log_dir
        self.level = level
        self.formatter = formatter or logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        )

        # Ensure that we get a unique logger instance by using a unique name
        self.logger = logging.getLogger(f"SiteSafetyMonitor_{log_file}")
        self.setup_logger()

    def setup_logger(self):
        """
        Configures the logger with rotating file handler and console handler.
        """
        # Prevent adding handlers multiple times
        if self.logger.hasHandlers():
            return

        # Create log directory if it doesn't exist
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        # Configure the file handler
        file_handler = self.get_file_handler()
        # Configure the console handler
        console_handler = self.get_console_handler()

        # Add the handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.setLevel(self.level)

        # Debug: Log to verify the handlers have been added
        self.logger.debug('Logger handlers set up complete.')

    def get_file_handler(self):
        """
        Creates and returns a rotating file handler.

        Returns:
            logging.Handler: A configured rotating file handler.
        """
        file_handler = RotatingFileHandler(
            filename=Path(self.log_dir) / self.log_file,
            maxBytes=1_000_000,
            backupCount=5,
        )
        file_handler.setLevel(self.level)
        file_handler.setFormatter(self.formatter)
        return file_handler

    def get_console_handler(self):
        """
        Creates and returns a console handler.

        Returns:
            logging.Handler: A configured console handler.
        """
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.level)
        console_handler.setFormatter(self.formatter)
        return console_handler

    def get_logger(self):
        """
        Returns the configured logger instance.

        Returns:
            logging.Logger: A configured logger instance.
        """
        return self.logger


# This block is executed when the script is run directly, not when imported
if __name__ == '__main__':
    # Example usage of the LoggerConfig class:

    # Initialise the logger configuration
    logger_config = LoggerConfig()
    logger = logger_config.get_logger()

    # Log a message indicating that the logging setup is complete
    logger.info('Logging setup complete.')
