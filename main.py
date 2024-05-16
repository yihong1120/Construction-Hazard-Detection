import argparse
from logging import Logger
import yaml
from datetime import datetime
from multiprocessing import Pool
import time
import gc
from typing import NoReturn, Dict
import os
from dotenv import load_dotenv

from src.stream_capture import StreamCapture
from src.line_notifier import LineNotifier
from src.monitor_logger import LoggerConfig
from src.live_stream_detection import LiveStreamDetector
from src.danger_detector import DangerDetector

def main(logger, video_url: str, model_key: str = 'yolov8x', label: str = None, image_name: str = 'prediction_visual.png', line_token: str = None) -> NoReturn:
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
    live_stream_detector = LiveStreamDetector(api_url=api_url, model_key=model_key, output_folder = label, output_filename = image_name)

    # Initialise the LINE notifier
    line_notifier = LineNotifier(line_token)

    # Initialise the DangerDetector
    danger_detector = DangerDetector()

    # Initialise the last_notification_time variable (set to 300 seconds ago, without microseconds)
    last_notification_time = int(time.time()) - 300

    # Use the generator function to process detections
    for frame, timestamp in streaming_capture.execute_capture():
        # Convert UNIX timestamp to datetime object and format it as string
        detection_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        print(detection_time)

        # Detect hazards in the frame
        datas, _ = live_stream_detector.generate_detections(frame)

        # Draw the detections on the frame
        live_stream_detector.draw_detections_on_frame(frame, datas)
        
        # Optionally save the frame with detections to the specified output path
        if output_path:
            # Here you could modify or extend the output filename based on timestamp or other criteria
            output_file = output_path.format(timestamp=timestamp)
            live_stream_detector.save_frame(frame)

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

            # Send notification with or without image based on image_name value
            status = line_notifier.send_notification(message, label, image_name if image_name != 'None' else None)
            if status == 200:
                logger.warning(f"Notification sent successfully: {message}")
            else:
                logger.error(f"Failed to send notification: {message}")

            # Update the last_notification_time to the current time
            last_notification_time = timestamp

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

    # Run hazard detection on a single video stream
    main(logger, **config)

def run_multiple_streams(config_file: str):
    """
    Run hazard detection on multiple video streams from a configuration file.

    Args:
        config_file (str): The path to the configuration file.

    Returns:
        None
    """
    running_processes = {}
    lock = threading.Lock()
    
    def start_process(config):
        p = Process(target=process_stream, args=(config,))
        p.start()
        return p

    def stop_process(process):
        process.terminate()
        process.join()

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
