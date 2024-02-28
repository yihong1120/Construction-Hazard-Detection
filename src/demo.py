import argparse
import os
from pathlib import Path
from dotenv import load_dotenv
from line_notifier import LineNotifier
from monitor_logger import setup_logging
from live_stream_detection import LiveStreamDetector
from danger_detector import DangerDetector
from datetime import datetime, timedelta
import time

def main(logger, youtube_url: str, model_path: str):
    """
    Main execution function that detects hazards, sends notifications, and logs warnings.

    Args:
        logger (logging.Logger): A logger instance for logging messages.
        youtube_url (str): The URL of the YouTube live stream to monitor.
        model_path (str): The file path of the YOLOv8 model to use for detection.
    """
    # Initialise the live stream detector
    live_stream_detector = LiveStreamDetector(youtube_url, model_path)

    # Initialise the LINE notifier
    line_notifier = LineNotifier()

    # Initialise the DangerDetector
    danger_detector = DangerDetector()

    # Initialise the last_notification_time variable (set to 300 seconds ago, without microseconds)
    last_notification_time = int(time.time()) - 300

    # Use the generator function to process detections
    for datas, frame, timestamp in live_stream_detector.generate_detections():
        # Convert UNIX timestamp to datetime object and format it as string
        detection_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        warnings = danger_detector.detect_danger(timestamp, datas)

        # If there are any new warnings and sufficient time has passed since the last notification
        if warnings and (timestamp - last_notification_time) > 300:
            unique_warnings = set(warnings)  # Remove duplicates
            for warning in unique_warnings:
                message = f'[{detection_time}] {warning}'
                status = line_notifier.send_notification(message)
                if status == 200:
                    logger.warning(f"Notification sent successfully: {message}")
                else:
                    logger.error(f"Failed to send notification: {message}")

            # Update the last_notification_time to the current time
            last_notification_time = timestamp #int(time.time())

    # Release resources after processing
    live_stream_detector.release_resources()

if __name__ == '__main__':
    # Load environment variables from the specified .env file
    env_path = Path('../.env')  # Adjust if your .env file is located elsewhere
    load_dotenv(dotenv_path=env_path)

    parser = argparse.ArgumentParser(description='Monitor a live stream for safety hazards using YOLOv8.')
    parser.add_argument('--url', type=str, required=True, help='YouTube video URL for monitoring')
    parser.add_argument('--model', type=str, default='../models/yolov8n.pt', help='Path to the YOLOv8 model')
    args = parser.parse_args()

    logger = setup_logging()  # Set up logging
    main(logger, args.url, args.model)