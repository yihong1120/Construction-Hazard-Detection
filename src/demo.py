from site_safety_monitor import detect_danger
from line_notifier import send_line_notification
from monitor_logger import setup_logging
from datetime import datetime

def main(logger):
    """
    Main execution function that detects hazards, sends notifications, and logs warnings.

    This function utilises the `detect_danger` function from the `site_safety_monitor` module
    to identify potential safety hazards. If any hazards are detected, it sends out notifications
    through LINE Notify and logs the warnings.

    Args:
        logger (logging.Logger): A logger instance for logging messages.

    Returns:
        None
    """
    # Assume these are the detection results obtained from your detection system
    detections = [
        # ...Your detection results
    ]

    # Utilise the detection function from site_safety_monitor.py
    warnings = detect_danger(detections)
    
    # If there are any warnings, send them via LINE Chatbot and log them
    if warnings:
        line_token = 'YOUR_LINE_NOTIFY_TOKEN'  # Your LINE Notify token
        for warning in warnings:
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            message = f'[{current_time}] {warning}'
            send_line_notification(line_token, message)
            logger.warning(message)  # Log the warning

# Set up logging and call the main function
if __name__ == '__main__':
    logger = setup_logging()  # Set up logging
    main(logger)