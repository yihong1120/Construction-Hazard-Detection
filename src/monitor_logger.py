import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logging(log_file='monitor.log', level=logging.INFO):
    """
    Set up a logger for the application with both console and file handlers.

    This function creates a logger with a rotating file handler, which ensures that the log
    files do not grow indefinitely. The log files rotate when they reach a specified size.

    Args:
        log_file (str): The name of the log file. Defaults to 'monitor.log'.
        level (logging.Level): The logging level. Defaults to logging.INFO.

    Returns:
        logging.Logger: A configured logger instance.

    Example:
        >>> logger = setup_logging()
        >>> logger.info('Logging is configured.')
    """
    # Create a logger with the specified name
    logger = logging.getLogger('SiteSafetyMonitor')
    logger.setLevel(level)

    # Check if the logs directory exists, create it if necessary
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # Create a rotating file handler that logs to a file and rotates when it reaches a certain size
    handler = RotatingFileHandler(f'logs/{log_file}', maxBytes=1000000, backupCount=5)
    handler.setLevel(level)

    # Create a console handler that logs to the standard output (console)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Create a formatter that specifies the log message format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the file and console handlers to the logger
    logger.addHandler(handler)
    logger.addHandler(console_handler)

    # Return the configured logger
    return logger

# This block is executed when the script is run directly, not when imported
if __name__ == '__main__':
    # Example usage of the setup_logging function:
    
    # Set up the logger
    logger = setup_logging()
    
    # Log a message indicating that the logging setup is complete
    logger.info('Logging setup complete.')