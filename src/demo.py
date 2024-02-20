import argparse
import os
from pathlib import Path
from dotenv import load_dotenv
from line_notifier import LineNotifier
from monitor_logger import setup_logging
from live_stream_tracker import LiveStreamDetector
from danger_detector import DangerDetector
from datetime import datetime

def main(logger, youtube_url: str, model_path: str):
    """
    Main execution function that detects hazards, sends notifications, and logs warnings.

    Args:
        logger (logging.Logger): A logger instance for logging messages.
        youtube_url (str): The URL of the YouTube live stream to monitor.
        model_path (str): The file path of the YOLOv8 model to use for detection.
    """
    # Initialise the live stream detector
    detector = LiveStreamDetector(youtube_url, model_path)

    # Initialise the LINE notifier
    line_notifier = LineNotifier()

    # Initialise the DangerDetector
    danger_detector = DangerDetector()  # Create an instance of DangerDetector once and use it properly

    # Use the generator function to process detections
    for ids, datas, frame, timestamp in detector.generate_detections():
        print("Timestamp:", timestamp)
        print("IDs:", ids)
        print("Data (xyxy format):")
        for data in datas:
            print(data)

        # Utilize the detection method from the DangerDetector instance
        warnings = danger_detector.detect_danger(timestamp, ids, datas)  # Use it here

        # If there are any warnings, send them via LINE Chatbot and log them
        if warnings:
            for warning in warnings:
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                message = f'[{current_time}] {warning}'
                status = line_notifier.send_notification(message)
                if status == 200:
                    logger.warning(f"Notification sent successfully: {message}")
                else:
                    logger.error(f"Failed to send notification: {message}")
    
    # Release resources after processing
    detector.release_resources()

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