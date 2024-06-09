import argparse
import yaml
from datetime import datetime
from multiprocessing import Process
import time
import gc
from typing import NoReturn, Dict
import os
from dotenv import load_dotenv
import threading
import redis
import cv2

from src.stream_capture import StreamCapture
from src.line_notifier import LineNotifier
from src.monitor_logger import LoggerConfig
from src.live_stream_detection import LiveStreamDetector
from src.danger_detector import DangerDetector

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0)

def main(logger, video_url: str, model_key: str = 'yolov8x', label: str = None, image_name: str = 'prediction_visual', line_token: str = None) -> NoReturn:
    """
    Main execution function that detects hazards, sends notifications, logs warnings, and optionally saves output images.

    Args:
        logger (logging.Logger): A logger instance for logging messages.
        video_url (str): The URL of the live stream to monitor.
        label (str): The label of image_name.
        image_name (str, optional): The file name of the image to send with notifications. Defaults to 'demo_data/{label}/prediction_visual.png'.
    """
    # Load environment variables
    load_dotenv()
    api_url = os.getenv('API_URL', 'http://localhost:5000') 

    streaming_capture = StreamCapture(stream_url = video_url)

    # Initialise the live stream detector
    live_stream_detector = LiveStreamDetector(api_url=api_url, model_key=model_key, output_folder = label)

    # Initialise the LINE notifier
    line_notifier = LineNotifier(line_token)

    # Initialise the DangerDetector
    danger_detector = DangerDetector()

    # Initialise the last_notification_time variable (set to 300 seconds ago, without microseconds)
    last_notification_time = int(time.time()) - 300

    # Use the generator function to process detections
    for frame, timestamp in streaming_capture.execute_capture():
        start_time = time.time()
        # Convert UNIX timestamp to datetime object and format it as string
        detection_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

        # Detect hazards in the frame
        datas, _ = live_stream_detector.generate_detections(frame)

        # Draw the detections on the frame
        frame_with_detections = live_stream_detector.draw_detections_on_frame(frame, datas)

        # Convert the frame to a byte array
        _, buffer = cv2.imencode('.png', frame_with_detections)
        frame_bytes = buffer.tobytes()

        # Save the frame with detections
        # save_file_name = f'{label}_{image_name}_{detection_time}'
        # live_stream_detector.save_frame(frame_with_detections, save_file_name)        

        # Check for warnings and send notifications if necessary
        warnings = danger_detector.detect_danger(datas)

        # Log the detection results
        logger.info(f"{label} - {image_name}")
        logger.info(f"Detection time: {detection_time}")

        # Get the current hour
        current_hour = datetime.now().hour
        
        if (warnings and # Check if there are warnings
            (timestamp - last_notification_time) > 300 and # Check if the last notification was more than 5 minutes ago
            (7 <= current_hour < 18)): # Check if the current hour is between 7 AM and 6 PM

            # Combine all warnings into one message
            # message = '\n'.join([f'[{detection_time}] {warning}' for warning in warnings])
            message = f'{image_name}\n[{detection_time}]\n' + '\n'.join([f'{warning}' for warning in warnings])

            # Send notification with or without image based on image_name value
            notification_status = line_notifier.send_notification(message, label, image=frame_bytes if frame_bytes is not None else None)
            if notification_status == 200:
                logger.warning(f"Notification sent successfully: {message}")
            else:
                logger.error(f"Failed to send notification: {message}")

            # Update the last_notification_time to the current time
            last_notification_time = timestamp

        else:
            logger.info("No warnings detected or not within the notification time range")

        # Use a unique key for each thread or process
        key = f'{label}_{image_name}'
        
        # Store the frame in Redis
        r.set(key, frame_bytes)

        end_time = time.time()

        # Log the detection results
        logger.info(f"Processing time: {end_time - start_time:.2f} seconds")

        # Clear variables to free up memory
        del datas, frame, timestamp, detection_time
        gc.collect()

    # Release resources after processing
    live_stream_detector.release_resources()
    gc.collect()

def process_stream(config: Dict[str, str]) -> NoReturn:
    """
    Process a single video stream with configuration.

    Args:
        config (dict): The configuration dictionary.

    Returns:
        None
    """
    # Load configurations
    logger_config = LoggerConfig()

    # Get logger
    logger = logger_config.get_logger()

    try:
        # Run hazard detection on a single video stream
        main(logger, **config)
    finally:
        # Clean up Redis keys when the process ends
        label = config.get('label')
        image_name = config.get('image_name', 'prediction_visual')
        key = f'{label}_{image_name}'
        r.delete(key)
        logger.info(f"Deleted Redis key: {key}")

def start_process(config: Dict[str, str]) -> Process:
    """
    Start a process for a single video stream with configuration.

    Args:
        config (dict): The configuration dictionary.

    Returns:
        Process: The started process.
    """
    p = Process(target=process_stream, args=(config,))
    p.start()
    return p

def stop_process(process: Process) -> None:
    """
    Stop a running process.

    Args:
        process (Process): The process to stop.

    Returns:
        None
    """
    process.terminate()
    process.join()

def run_multiple_streams(config_file: str) -> NoReturn:
    """
    Run hazard detection on multiple video streams from a configuration file.

    Args:
        config_file (str): The path to the configuration file.

    Returns:
        None
    """
    running_processes = {}
    lock = threading.Lock()

    while True:
        with open(config_file, 'r') as file:
            configurations = yaml.safe_load(file)

        current_configs = {config['video_url']: config for config in configurations}

        with lock:
            # Stop processes for removed configurations
            for video_url in list(running_processes.keys()):
                if video_url not in current_configs:
                    print(f"Stop workflow: {video_url}")
                    stop_process(running_processes[video_url])
                    del running_processes[video_url]

            # Start processes for new configurations
            for video_url, config in current_configs.items():
                if video_url not in running_processes:
                    print(f"Launch new workflow: {video_url}")
                    running_processes[video_url] = start_process(config)

        time.sleep(3600)  # Check every hour for configuration changes

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run hazard detection on multiple video streams.')
    parser.add_argument('--config', type=str, default='config/configuration.yaml', help='Configuration file path')
    args = parser.parse_args()

    run_multiple_streams(args.config)
