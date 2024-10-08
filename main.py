from __future__ import annotations

import argparse
import gc
import logging
import os
import re
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
from src.live_stream_detection import LiveStreamDetector
from src.monitor_logger import LoggerConfig
from src.notifiers.line_notifier import LineNotifier
from src.stream_capture import StreamCapture

# Load environment variables
load_dotenv()

is_windows = os.name == 'nt'

if not is_windows:
    import redis
    from redis import Redis

    # Redis configuration
    redis_host: str = os.getenv('redis_host', 'localhost')
    redis_port: int = int(os.getenv('redis_port', '6379'))
    redis_password: str | None = os.getenv('redis_password', None)

    # Connect to Redis
    r: Redis = redis.StrictRedis(
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


def process_single_stream(
    logger: logging.Logger,
    video_url: str,
    model_key: str = 'yolov8x',
    label: str | None = None,
    image_name: str = 'prediction_visual',
    line_token: str | None = None,
    run_local: bool = True,
    language: str = 'en',
) -> None:
    """
    Function to detect hazards, notify, log, save images (optional).

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
        language (str): The language for the notifications.
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
    drawing_manager = DrawingManager(language=language)

    # Initialise the LINE notifier
    line_notifier = LineNotifier(line_token=line_token)

    # Initialise the DangerDetector
    danger_detector = DangerDetector(language=language)

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

        # Check for warnings and send notifications if necessary
        warnings, controlled_zone_polygon = danger_detector.detect_danger(
            datas,
        )

        # Draw the detections on the frame
        frame_with_detections = (
            drawing_manager.draw_detections_on_frame(
                frame, controlled_zone_polygon, datas,
            )
        )

        # Save the frame with detections
        # save_file_name = f'{label}_{image_name}_{detection_time}'
        # drawing_manager.save_frame(
        #   frame_with_detections,
        #   save_file_name
        # )

        # Convert the frame to a byte array
        _, buffer = cv2.imencode('.png', frame_with_detections)
        frame_bytes = buffer.tobytes()

        # Log the detection results
        logger.info(f"{label} - {image_name}")
        logger.info(f"Detection time: {detection_time}")

        # Get the current hour
        current_hour = datetime.now().hour

        # Check if it is outside the specified time range
        # and if warnings contaim a warning for people in the controlled zone
        if (timestamp - last_notification_time) > 300:
            # Check if there is a warning for people in the controlled zone
            controlled_zone_warning_template = danger_detector.get_text(
                'warning_people_in_controlled_area', count='',
            )

            # Create a regex pattern to match the warning
            pattern = re.escape(controlled_zone_warning_template).replace(
                re.escape('{count}'), r'\d+',
            )

            controlled_zone_warning = next(
                (
                    warning for warning in warnings
                    if re.match(pattern, warning)
                ),
                None,
            )

            # If it is outside working hours and there is
            # a warning for people in the controlled zone
            if controlled_zone_warning and not (7 <= current_hour < 18):
                message = (
                    f"{image_name}\n[{detection_time}]\n"
                    f"{controlled_zone_warning}"
                )

            elif warnings and (7 <= current_hour < 18):
                # During working hours, combine all warnings
                message = (
                    f"{image_name}\n[{detection_time}]\n"
                    + '\n'.join(warnings)
                )

            else:
                message = None

            # If a notification needs to be sent
            if message:
                notification_status = line_notifier.send_notification(
                    message, image=frame_bytes
                    if frame_bytes is not None
                    else None,
                )

                # If you want to connect to the broadcast system, do it here:
                # broadcast_status = (
                #   broadcast_notifier.broadcast_message(message)
                # )
                # logger.info(f"Broadcast status: {broadcast_status}")

                if notification_status == 200:
                    logger.info(
                        f"Notification sent successfully: {message}",
                    )
                else:
                    logger.error(f"Failed to send notification: {message}")

                # Update the last notification time
                last_notification_time = int(timestamp)
        else:
            logger.info('No warnings or outside notification time.')

        if not is_windows:
            try:
                # Use a unique key for each thread or process
                key = f"{label}_{image_name}".encode()

                # Store the frame in Redis
                r.set(key, frame_bytes)

            except Exception as e:
                logger.error(f"Failed to store frame in Redis: {e}")

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


def process_streams(config: StreamConfig) -> None:
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
        process_single_stream(logger, **config)
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
    p = Process(target=process_streams, args=(config,))
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


def process_single_image(
    image_path: str,
    model_key: str = 'yolov8x',
    output_folder: str = 'output_images',
    image_name: str = None,
    language: str = 'en',
) -> None:
    """
    Process a single image for hazard detection and save the result.
    """
    try:
        # Check if the image path exists
        if not os.path.exists(image_path):
            print(f"Error: Image path {image_path} does not exist.")
            return

        # Load the image using OpenCV
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error: Failed to load image {image_path}")
            return

        # Initialise the live stream detector (but here used for a single image)
        api_url = os.getenv('API_URL', 'http://localhost:5000')
        live_stream_detector = LiveStreamDetector(
            api_url=api_url, model_key=model_key, output_folder=output_folder,
        )

        # Initialise the drawing manager
        drawing_manager = DrawingManager(language=language)

        # Detect hazards in the image
        detections, _ = live_stream_detector.generate_detections(image)

        # For this example, no polygons are needed, so pass an empty list
        frame_with_detections = drawing_manager.draw_detections_on_frame(
            image, [], detections,
        )

        # Ensure the output folder exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Generate the output file name if not provided
        if not image_name:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            image_name = f"detection_{timestamp}.png"
        output_path = os.path.join(output_folder, image_name)

        # Save the image with detections
        cv2.imwrite(output_path, frame_with_detections)
        print(f"Processed image saved to {output_path}")

    except Exception as e:
        print(f"Error processing the image: {str(e)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            'Run hazard detection on multiple video streams or a single image.'
        ),
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/configuration.yaml',
        help='Configuration file path for stream processing',
    )
    parser.add_argument(
        '--image',
        type=str,
        help='Path to a single image for detection',
    )
    parser.add_argument(
        '--model_key',
        type=str,
        default='yolov8x',
        help='Model key to use for detection',
    )
    parser.add_argument(
        '--output_folder',
        type=str,
        default='output_images',
        help='Folder to save the output image',
    )
    parser.add_argument(
        '--language',
        type=str,
        default='en',
        help='Language for labels on the output image',
    )
    args = parser.parse_args()

    # If an image path is provided, process the single image
    if args.image:
        process_single_image(
            image_path=args.image,
            model_key=args.model_key,
            output_folder=args.output_folder,
            language=args.language,
        )
    else:
        # Otherwise, run hazard detection on multiple video streams
        run_multiple_streams(args.config)
