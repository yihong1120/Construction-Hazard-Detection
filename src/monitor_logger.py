import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

class LoggerConfig:
    """
    A class to set up a logger for the application with both console and file handlers.
    
    This class creates a logger with a rotating file handler, which ensures that the log
    files do not grow indefinitely. The log files rotate when they reach a specified size.
    """

    def __init__(self, log_file='monitor.log', level=logging.INFO):
        """
        Initialises the logger configuration with a log file name and logging level.
        
        Args:
            log_file (str): The name of the log file. Defaults to 'monitor.log'.
            level (logging.Level): The logging level. Defaults to logging.INFO.
        """
        self.log_file = log_file
        self.level = level
        self.logger = logging.getLogger('SiteSafetyMonitor')
        self.setup_logger()

    def setup_logger(self):
        """
        Configures the logger with rotating file handler and console handler.
        """
        # Check if the logs directory exists, create it if necessary
        Path('logs').mkdir(parents=True, exist_ok=True)

        # Create a rotating file handler that logs to a file and rotates when it reaches a certain size
        file_handler = RotatingFileHandler(f'logs/{self.log_file}', maxBytes=1_000_000, backupCount=5)
        file_handler.setLevel(self.level)

        # Create a console handler that logs to the standard output (console)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.level)

        # Create a formatter that specifies the log message format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add the file and console handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.setLevel(self.level)

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