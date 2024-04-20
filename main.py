import argparse
from logging import Logger
import numpy
import json
import multiprocessing
from datetime import datetime
from multiprocessing import Pool
import time
import gc
from typing import List, NoReturn, Dict

from src.line_notifier import LineNotifier
from src.monitor_logger import LoggerConfig
from src.live_stream_detection import LiveStreamDetector
from src.danger_detector import DangerDetector

def main(logger, video_url: str, api_url: str = 'http://localhost:5000/detect', model_key: str = 'yolov8x',image_path: str = 'prediction_visual.png', line_token: str = None, output_path: str = None) -> NoReturn:
    """
    Main execution function that detects hazards, sends notifications, logs warnings, and optionally saves output images.

    Args:
        logger (logging.Logger): A logger instance for logging messages.
        video_url (str): The URL of the live stream to monitor.
        image_path (str, optional): The file path of the image to send with notifications. Defaults to 'demo_data/prediction_visual.png'.
        output_path (str, optional): The file path where output images should be saved. If not specified, images are not saved.
    """
    # Initialise the live stream detector
    live_stream_detector = LiveStreamDetector(stream_url = video_url, api_url=api_url, model_key=model_key, output_filename = image_path)

    # Initialise the LINE notifier
    line_notifier = LineNotifier(line_token)

    # Initialise the DangerDetector
    danger_detector = DangerDetector()

    # Initialise the last_notification_time variable (set to 300 seconds ago, without microseconds)
    last_notification_time = int(time.time()) - 300

    # Use the generator function to process detections
    for datas, frame, timestamp in live_stream_detector.generate_detections():
        # Convert UNIX timestamp to datetime object and format it as string
        detection_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        print(detection_time)

        # Draw the detections on the frame
        live_stream_detector.draw_detections_on_frame(frame, datas)
        
        # Optionally save the frame with detections to the specified output path
        if output_path:
            # Here you could modify or extend the output filename based on timestamp or other criteria
            output_file = output_path.format(timestamp=timestamp)
            live_stream_detector.save_frame(frame, output_file)

        # Check for warnings and send notifications if necessary
        warnings = danger_detector.detect_danger(datas)

        # Get the current hour
        current_hour = datetime.now().hour
        
        if (warnings and # Check if there are warnings
            (timestamp - last_notification_time) > 300 and # Check if the last notification was more than 5 minutes ago
            (7 <= current_hour < 18)): # Check if the current hour is between 7 AM and 6 PM

            # Remove duplicates
            unique_warnings = set(warnings)  # Remove duplicates

            # Combine all warnings into one message
            message = '\n'.join([f'[{detection_time}] {warning}' for warning in unique_warnings])

            # Send notification with or without image based on image_path value
            status = line_notifier.send_notification(message, image_path if image_path != 'None' else None)
            if status == 200:
                logger.warning(f"Notification sent successfully: {message}")
            else:
                logger.error(f"Failed to send notification: {message}")

            # Update the last_notification_time to the current time
            last_notification_time = timestamp

            del unique_warnings, message, status, warnings

        # Clear variables to free up memory
        del datas, frame, timestamp, detection_time
        gc.collect()

    # Release resources after processing
    live_stream_detector.release_resources()

    gc.collect()

def process_stream(config: Dict[str, str], output_path: str = None) -> NoReturn:
    """
    Process a single video stream with configuration.

    Args:
        config (dict): The configuration dictionary.
        output_path (str): The path to save the output images with timestamp in the filename.

    Returns:
        None
    """
    # Load configurations
    logger_config = LoggerConfig()

    # Get logger
    logger = logger_config.get_logger()

    # Run hazard detection on a single video stream
    main(logger, **config, output_path=output_path)

def run_multiple_streams(config_file: str, output_path: str = None):
    """
    Run hazard detection on multiple video streams from a configuration file.

    Args:
        config_file (str): The path to the configuration file.
        output_path (str): The path to save the output images with timestamp in the filename.

    Returns:
        None
    """
    # Load configurations from file
    with open(config_file, 'r') as file:
        configurations = json.load(file)

    # Process streams in parallel
    cpu_count = multiprocessing.cpu_count()
    num_processes = max(len(configurations), cpu_count)

    # Process streams in parallel
    with Pool(processes=num_processes) as pool:
        pool.starmap(process_stream, [(config, output_path) for config in configurations])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run hazard detection on multiple video streams.')
    parser.add_argument('--config', type=str, default='config/configuration.json', help='Configuration file path')
    parser.add_argument('--output', type=str, help='Path to save output images with timestamp in filename', required=False)
    args = parser.parse_args()

    run_multiple_streams(args.config, args.output)