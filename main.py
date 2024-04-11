import argparse
from datetime import datetime
import json
import multiprocessing
from multiprocessing import Pool
import gc
import time
from typing import NoReturn, Dict
from src.line_notifier import LineNotifier
from src.monitor_logger import LoggerConfig
from src.live_stream_detection import LiveStreamDetector
from src.danger_detector import DangerDetector

def main(logger, video_url: str, model_path: str, image_path: str = 'prediction_visual.png', line_token: str = None, output_path: str = None) -> NoReturn:
    """
    Main execution function that detects hazards, sends notifications, logs warnings, and optionally saves output images.

    Args:
        logger (logging.Logger): A logger instance for logging messages.
        video_url (str): The URL of the live stream to monitor.
        model_path (str): The file path of the YOLOv8 model to use for detection.
        image_path (str, optional): The file path of the image to send with notifications. Defaults to 'demo_data/prediction_visual.png'.
        output_path (str, optional): The file path where output images should be saved. If not specified, images are not saved.
    """
    # Initialise the live stream detector
    live_stream_detector = LiveStreamDetector(video_url, model_path, image_path)

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

        # If there are any new warnings and sufficient time has passed since the last notification
        if warnings and (timestamp - last_notification_time) > 300:
            unique_warnings = set(warnings)  # Remove duplicates
            for warning in unique_warnings:
                message = f'[{detection_time}] {warning}'
                # Send notification with or without image based on image_path value
                status = line_notifier.send_notification(message, image_path if image_path != 'None' else None)
                if status == 200:
                    logger.warning(f"Notification sent successfully: {message}")
                else:
                    logger.error(f"Failed to send notification: {message}")

            # Update the last_notification_time to the current time
            last_notification_time = timestamp

            del unique_warnings, message, status, warning

        # Clear variables to free up memory
        del datas, frame, timestamp, detection_time
        gc.collect()

    # Release resources after processing
    live_stream_detector.release_resources()

    gc.collect()

def process_stream(config: Dict[str, str], output_path: str = None) -> NoReturn:
    '''
    Process a single video stream with the given configuration.

    Args:
        config (Dict[str, str]): Configuration dictionary containing the following keys:
            - video_url (str): The URL of the live stream to monitor.
            - model_path (str): The file path of the YOLOv8 model to use for detection.
            - image_path (str, optional): The file path of the image to send with notifications. Defaults to 'demo_data/prediction_visual.png'.
            - line_token (str, optional): The LINE notification token. If not provided, notifications are not sent.
        output_path (str, optional): The file path where output images should be saved. If not specified, images are not saved.
    '''
    logger_config = LoggerConfig()
    logger = logger_config.get_logger()
    main(logger, **config, output_path=output_path)


def run_multiple_streams(config_file: str, output_path: str = None):
    '''
    Run hazard detection on multiple video streams with the given configuration file.

    Args:
        config_file (str): Path to the configuration file.
        output_path (str): Path to save output images. Use {timestamp} to include the detection timestamp in the filename.
    '''
    with open(config_file, 'r') as f:
        configurations = json.load(f)

    # For example, use the number of CPU cores available but leave some for other tasks if necessary
    num_processes = min(len(configurations), multiprocessing.cpu_count() - 1)

    # Create a multiprocessing pool
    with Pool(processes=num_processes) as pool:
        # Map each configuration to the process_stream function
        # This will automatically distribute the configurations across the pool of processes
        pool.starmap(process_stream, [(config, output_path) for config in configurations])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run hazard detection on multiple video streams.')
    parser.add_argument('--config', type=str, default='config/configuration.json', help='Path to the configuration file')
    parser.add_argument('--output', type=str, help='Path to save output images. Use {timestamp} to include the detection timestamp in the filename.', required=False)
    args = parser.parse_args()
    
    run_multiple_streams(args.config, args.output)