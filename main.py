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

def main(logger, video_url: str, image_path: str = 'prediction_visual.png', line_token: str = None, output_path: str = None) -> NoReturn:
    """
    Orchestrates the detection, logging, and notification processes for a single video stream.

    Args:
        logger (Logger): A logger object for logging messages.
        video_url (str): The URL of the video stream.
        image_path (str): The path to save the output image with detections.
        line_token (str): The LINE notification token.
        output_path (str): The path to save the output images with timestamp in the filename.

    Returns:
        None
    """
    # Initialise objects
    live_stream_detector = LiveStreamDetector(stream_url=video_url, output_filename=image_path)
    line_notifier = LineNotifier(line_token)
    danger_detector = DangerDetector()

    last_notification_time = int(time.time()) - 300  # Initialising last_notification_time

    # Start detection process
    for datas, frame, timestamp in live_stream_detector.generate_detections():
        # Convert timestamp to human-readable format
        detection_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        
        # Process frame and notifications
        process_frame(live_stream_detector, frame, datas, detection_time, output_path)

        # Process notifications
        process_notifications(logger, line_notifier, danger_detector, datas, timestamp, last_notification_time, detection_time, image_path)
    
    # Release resources 
    live_stream_detector.release_resources()
    gc.collect()

def process_frame(detector: LiveStreamDetector, frame: numpy.ndarray, datas: list, detection_time: str, output_path: str) -> None:
    """
    Processes each frame by drawing detections and optionally saving the frame.

    Args:
        detector (LiveStreamDetector): The detector object.
        frame (numpy.ndarray): The frame to process.
        datas (list): The list of detection data.
        detection_time (str): The timestamp of the detection.
        output_path (str): The path to save the output images with timestamp in the filename.

    Returns:
        None
    """
    # Draw detections on frame
    detector.draw_detections_on_frame(frame, datas)

    # Save frame with detections
    if output_path:
        output_file = output_path.format(timestamp=detection_time)
        detector.save_frame(frame, output_file)

def process_notifications(logger: Logger, notifier: LineNotifier, detector: DangerDetector, datas: List, timestamp: int, last_notification_time: int, detection_time: str, image_path: str) -> None:
    """
    Processes notifications if new warnings are detected and the conditions are met.

    Args:
        logger (Logger): A logger object for logging messages.
        notifier (LineNotifier): A notifier object for sending LINE notifications.
        detector (DangerDetector): A detector object for detecting dangers.
        datas (list): The list of detection data.
        timestamp (int): The timestamp of the detection.
        last_notification_time (int): The timestamp of the last notification.
        detection_time (str): The timestamp of the detection.
        image_path (str): The path to save the output image with detections.

    Returns:
        None
    """
    # Detect dangers
    warnings = detector.detect_danger(datas)
    current_hour = datetime.now().hour

    # Send notifications if new warnings are detected and the conditions are met
    if warnings and (timestamp - last_notification_time > 300) and 8 <= current_hour < 18:
        unique_warnings = set(warnings)

        # Send notifications for each unique warning
        for warning in unique_warnings:
            send_notification(logger, notifier, warning, detection_time, image_path)

        # Update last notification time    
        last_notification_time = timestamp

    gc.collect()

def send_notification(logger: Logger, notifier: LineNotifier, warning: str, detection_time: str, image_path: str) -> None:
    """
    Sends notifications with the detected warnings.

    Args:
        logger (Logger): A logger object for logging messages.
        notifier (LineNotifier): A notifier object for sending LINE notifications.
        warning (str): The warning message.
        detection_time (str): The timestamp of the detection.
        image_path (str): The path to save the output image with detections.

    Returns:
        None
    """
    # Send notification
    message = f'[{detection_time}] {warning}'
    status = notifier.send_notification(message, image_path if image_path != 'None' else None)

    # Log notification success
    if status == 200:
        logger.warning(f"Notification sent successfully: {message}")
    # Log notification failure
    else:
        logger.error(f"Failed to send notification: {message}")

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