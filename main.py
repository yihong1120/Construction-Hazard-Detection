from __future__ import annotations

import argparse
import gc
import logging
import os
import threading
import time
from datetime import datetime
from multiprocessing import Process
from typing import TypedDict

import cv2
import yaml
from dotenv import load_dotenv

from src.danger_detector import DangerDetector
from src.drawing_manager import DrawingManager
from src.line_notifier import LineNotifier
from src.live_stream_detection import LiveStreamDetector
from src.monitor_logger import LoggerConfig
from src.stream_capture import StreamCapture

# Load environment variables
load_dotenv()

is_windows = os.name == 'nt'

if not is_windows:
    import redis

    # Redis configuration
    redis_host: str = os.getenv('redis_host', 'localhost')
    redis_port: int = int(os.getenv('redis_port', '6379'))
    redis_password: str | None = os.getenv('redis_password', None)

    # Connect to Redis
    r = redis.StrictRedis(
        host=redis_host,
        port=redis_port,
        password=redis_password,
        decode_responses=True,
    )


class StreamConfig(TypedDict):
    video_url: str
    model_key: str
    label: str
    image_name: str
    line_token: str
    run_local: bool


def main(
    logger: logging.Logger,
    video_url: str,
    model_key: str = 'yolov8x',
    label: str | None = None,
    image_name: str = 'prediction_visual',
    line_token: str | None = None,
    run_local: bool = True,
) -> None:
    """
    Main function to detect hazards, notify, log, save images (optional).

    Args:
        logger (logging.Logger): A logger instance for logging messages.
        video_url (str): The URL of the live stream to monitor.
        label (Optional[str]): The label of image_name.
        image_name (str, optional): Image file name for notifications.
            Defaults to 'demo_data/{label}/prediction_visual.png'.
        line_token (Optional[str]): The LINE token for sending notifications.
            Defaults to None.
        run_local (bool): Whether to run detection using a local model.
            Defaults to True.
    """
    # Initialise the stream capture object
    streaming_capture = StreamCapture(stream_url=video_url)

    # Get the API URL from environment variables
    api_url = os.getenv('API_URL', 'http://localhost:5000')

    # Initialise the live stream detector
    live_stream_detector = LiveStreamDetector(
        api_url=api_url,
        model_key=model_key,
        output_folder=label,
        run_local=run_local,
    )

    # Initialise the drawing manager
    drawing_manager = DrawingManager()

    # Initialise the LINE notifier
    line_notifier = LineNotifier(line_token)

    # Initialise the DangerDetector
    danger_detector = DangerDetector()

    # Init last_notification_time to 300s ago, no microseconds
    last_notification_time = int(time.time()) - 300

    # Use the generator function to process detections
    for frame, timestamp in streaming_capture.execute_capture():
        start_time = time.time()
        # Convert UNIX timestamp to datetime object and format it as string
        detection_time = datetime.fromtimestamp(
            timestamp,
        ).strftime('%Y-%m-%d %H:%M:%S')

        # Detect hazards in the frame
        datas, _ = live_stream_detector.generate_detections(frame)

        # Draw the detections on the frame
        frame_with_detections, controlled_zone_polygon = (
            drawing_manager.draw_detections_on_frame(frame, datas)
        )

        # Convert the frame to a byte array
        _, buffer = cv2.imencode('.png', frame_with_detections)
        frame_bytes = buffer.tobytes()

        # Save the frame with detections
        # save_file_name = f'{label}_{image_name}_{detection_time}'
        # drawing_manager.save_frame(
        #   frame_with_detections,
        #   save_file_name
        # )

        # Log the detection results
        logger.info(f"{label} - {image_name}")
        logger.info(f"Detection time: {detection_time}")

        # Get the current hour
        current_hour = datetime.now().hour

        if (
            # >5 mins since last notification
            (timestamp - last_notification_time) > 300 and
            # Between 7 AM and 6 PM
            (7 <= current_hour < 18)
        ):
            # Check for warnings and send notifications if necessary
            warnings = danger_detector.detect_danger(
                datas, controlled_zone_polygon,
            )

            # Check if there are any warnings
            if warnings:
                # Combine all warnings into one message
                message = f"{image_name}\n[{detection_time}]\n" + '\n'.join(
                    [f"{warning}" for warning in warnings],
                )

                # Send notification, with image if image_name set
                notification_status = line_notifier.send_notification(
                    message,
                    image=frame_bytes if frame_bytes is not None else None,
                )
                if notification_status == 200:
                    logger.warning(
                        f"Notification sent successfully: {message}",
                    )
                else:
                    logger.error(f"Failed to send notification: {message}")

                # Update the last_notification_time to the current time
                last_notification_time = int(timestamp)

        else:
            logger.info(
                'No warnings or outside notification time.',
            )

        if not is_windows:
            # Use a unique key for each thread or process
            key = f"{label}_{image_name}".encode()

            # Store the frame in Redis
            r.set(key, frame_bytes)

        end_time = time.time()

        # Calculate the processing time
        processing_time = end_time - start_time

        # Update the capture interval based on the processing time
        new_interval = int(processing_time) + 5
        streaming_capture.update_capture_interval(new_interval)

        # Log the processing time
        logger.info(f"Processing time: {processing_time:.2f} seconds")

        # Clear variables to free up memory
        del datas, frame, timestamp, detection_time
        del frame_with_detections, buffer, frame_bytes
        gc.collect()

    # Release resources after processing
    streaming_capture.release_resources()
    gc.collect()


def process_stream(config: StreamConfig) -> None:
    """
    Process a video stream based on the given configuration.

    Args:
        config (StreamConfig): The configuration for the stream processing.

    Returns:
        None
    """
    # Load the logger configuration
    logger_config = LoggerConfig()

    # Initialise the logger
    logger = logger_config.get_logger()

    try:
        # Run hazard detection on a single video stream
        main(logger, **config)
    finally:
        if not is_windows:
            label = config.get('label')
            image_name = config.get('image_name', 'prediction_visual')
            key = f"{label}_{image_name}"
            r.delete(key)
            logger.info(f"Deleted Redis key: {key}")


def start_process(config: StreamConfig) -> Process:
    """
    Start a new process for processing a video stream.

    Args:
        config (StreamConfig): The configuration for the stream processing.

    Returns:
        Process: The newly started process.
    """
    p = Process(target=process_stream, args=(config,))
    p.start()
    return p


def stop_process(process: Process) -> None:
    """
    Stop a running process.

    Args:
        process (Process): The process to be terminated.

    Returns:
        None
    """
    process.terminate()
    process.join()


def run_multiple_streams(config_file: str) -> None:
    """
    Manage multiple video streams based on a config file.


    Args:
        config_file (str): The path to the YAML configuration file.

    Returns:
        None
    """
    running_processes: dict[str, Process] = {}
    lock = threading.Lock()

    while True:
        with open(config_file, encoding='utf-8') as file:
            configurations = yaml.safe_load(file)

        current_configs = {
            config['video_url']: config for config in configurations
        }

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

        time.sleep(3600)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run hazard detection on multiple video streams.',
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/configuration.yaml',
        help='Configuration file path',
    )
    args = parser.parse_args()

    run_multiple_streams(args.config)
