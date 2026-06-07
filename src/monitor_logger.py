from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


class LoggerConfig:
    """Configure the application logger with console and file handlers."""

    def __init__(
        self,
        log_file: str = 'monitor.log',
        log_dir: str = 'logs',
        level: int = logging.INFO,
        formatter: logging.Formatter | None = None,
    ) -> None:
        """Initialise the logger configuration.

        Args:
            log_file: File name for the rotating log file.
            log_dir: Directory where log files are stored.
            level: Logging level passed to handlers and the logger.
            formatter: Optional formatter. A standard timestamp formatter is
                used when omitted.
        """
        self.log_file = log_file
        self.log_dir = log_dir
        self.level = level
        self.formatter = formatter or logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        )

        self.logger = logging.getLogger(f"SiteSafetyMonitor_{log_file}")
        self.setup_logger()

    def setup_logger(self) -> None:
        """Configure rotating file and console handlers."""
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        # Reconfiguration must be idempotent because tests and reloads may
        # construct more than one LoggerConfig in a process.
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        file_handler = self.get_file_handler()
        console_handler = self.get_console_handler()

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.setLevel(self.level)

        self.logger.propagate = False
        self.logger.debug('Logger handlers set up complete.')

    def get_file_handler(self) -> RotatingFileHandler:
        """Create a rotating file handler.

        Returns:
            A configured rotating file handler.
        """
        file_handler = RotatingFileHandler(
            filename=Path(self.log_dir) / self.log_file,
            maxBytes=1_000_000,
            backupCount=5,
        )
        file_handler.setLevel(self.level)
        file_handler.setFormatter(self.formatter)
        return file_handler

    def get_console_handler(self) -> logging.StreamHandler:
        """Create a console stream handler.

        Returns:
            A configured console stream handler.
        """
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.level)
        console_handler.setFormatter(self.formatter)
        return console_handler

    def get_logger(self) -> logging.Logger:
        """Return the configured logger instance.

        Returns:
            A configured logger.
        """
        return self.logger


def main() -> None:
    """Initialise logging for direct script execution."""
    logger_config = LoggerConfig()
    logger = logger_config.get_logger()
    logger.info('Logging setup complete.')


# This block is executed when the script is run directly, not when imported
if __name__ == '__main__':
    main()
