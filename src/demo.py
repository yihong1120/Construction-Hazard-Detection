import argparse
import os
from pathlib import Path
from dotenv import load_dotenv
from line_notifier import LineNotifier
from monitor_logger import setup_logging
from live_stream_detection_SAHI_YOLOv8 import LiveStreamDetector
from danger_detector import DangerDetector
from datetime import datetime, timedelta

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

    # Initialise the last_notification_time variable
    last_notification_time = datetime.now() - timedelta(seconds=60)  # Set to 60 seconds ago

    # Use the generator function to process detections
    for datas, frame, timestamp in live_stream_detector.generate_detections():
        print("Timestamp:", timestamp)
        print("Data (xyxy format):")
        print(datas)

        # Utilise the detection method from the DangerDetector instance
        warnings = danger_detector.detect_danger(timestamp, datas)  # Use it here

        # If there are any warnings, check if a minute has passed since the last notification
        if warnings and (datetime.now() - last_notification_time).total_seconds() >= 60:
            for warning in warnings:
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                message = f'[{current_time}] {warning}'
                status = line_notifier.send_notification(message)
                if status == 200:
                    logger.warning(f"Notification sent successfully: {message}")
                else:
                    logger.error(f"Failed to send notification: {message}")
            
            # Update the last notification time
            last_notification_time = datetime.now()

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