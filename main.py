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
from watchdog.observers import Observer

from src.danger_detector import DangerDetector
from src.drawing_manager import DrawingManager
from src.lang_config import Translator
from src.live_stream_detection import LiveStreamDetector
from src.monitor_logger import LoggerConfig
from src.notifiers.line_notifier import LineNotifier
from src.stream_capture import StreamCapture
from src.utils import FileEventHandler
from src.utils import RedisManager
from src.utils import Utils

# Load environment variables
load_dotenv()

is_windows = os.name == 'nt'

# Initialise Redis manager
if not is_windows:
    redis_manager = RedisManager()


class StreamConfig(TypedDict, total=False):
    video_url: str
    model_key: str
    site: str | None
    stream_name: str
    notifications: dict[str, str] | None
    detect_with_server: bool
    expire_date: str | None
    line_token: str | None
    language: str | None


def run_multiple_streams(config_file: str) -> None:
    """
    Manage multiple video streams based on a config file.

    Args:
        config_file (str): The path to the YAML configuration file.

    Returns:
        None
    """
    running_processes: dict[str, Process] = {}
    current_config_hashes: dict[str, str] = {}
    lock = threading.Lock()

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    def compute_config_hash(config: dict) -> str:
        """
        Compute a hash based on relevant configuration parameters.

        Args:
            config (dict): The configuration dictionary for a video stream.

        Returns:
            str: A hash representing the configuration.
        """
        relevant_config = {
            'video_url': config['video_url'],
            'model_key': config['model_key'],
            'site': config['site'],
            'stream_name': config.get('stream_name', 'prediction_visual'),
            'notifications': config['notifications'],
            'detect_with_server': config['detect_with_server'],
        }
        return str(relevant_config)  # Convert to string for hashing

    def reload_configurations():
        with open(config_file, encoding='utf-8') as file:
            configurations = yaml.safe_load(file)

        current_configs = {
            config['video_url']: config for config in configurations
        }

        with lock:
            # Stop processes for removed or updated configurations
            for video_url in list(running_processes.keys()):
                config = current_configs.get(video_url)

                # Stop the process if the configuration is removed
                if not config or Utils.is_expired(config.get('expire_date')):
                    logger.info(f"Stop workflow: {video_url}")
                    stop_process(running_processes[video_url])
                    del running_processes[video_url]
                    del current_config_hashes[video_url]

                # Restart the process if the configuration is updated
                elif compute_config_hash(config) != current_config_hashes.get(
                    video_url,
                ):
                    logger.info(
                        f"Config changed for {video_url}. "
                        'Restarting workflow.',
                    )
                    stop_process(running_processes[video_url])
                    running_processes[video_url] = start_process(config)
                    current_config_hashes[video_url] = compute_config_hash(
                        config,
                    )

            # Start processes for new configurations
            for video_url, config in current_configs.items():
                if Utils.is_expired(config.get('expire_date')):
                    logger.info(f"Skip expired configuration: {video_url}")
                    continue

                if video_url not in running_processes:
                    logger.info(f"Launch new workflow: {video_url}")
                    running_processes[video_url] = start_process(config)
                    current_config_hashes[video_url] = compute_config_hash(
                        config,
                    )

    # Initial load of configurations
    reload_configurations()

    # Set up watchdog observer
    event_handler = FileEventHandler(config_file, reload_configurations)
    observer = Observer()
    observer.schedule(
        event_handler, path=os.path.dirname(
            config_file,
        ), recursive=False,
    )
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


def process_single_stream(
    logger: logging.Logger,
    video_url: str,
    model_key: str = 'yolo11n',
    site: str | None = None,
    stream_name: str = 'prediction_visual',
    notifications: dict[str, str] | None = None,
    detect_with_server: bool = False,
) -> None:
    """
    Function to detect hazards, notify, log, save images (optional).

    Args:
        logger (logging.Logger): A logger instance for logging messages.
        video_url (str): The URL of the live stream to monitor.
        site (Optional[str]): The site for stream processing.
        stream_name (str, optional): Image file name for notifications.
            Defaults to 'demo_data/{site}/prediction_visual.png'.
        notifications (Optional[dict]): Line tokens with their languages.
        detect_with_server (bool): Whether to run detection using a server api.
    """
    # Initialise the stream capture object
    streaming_capture = StreamCapture(stream_url=video_url)

    # Get the API URL from environment variables
    api_url = os.getenv('API_URL', 'http://localhost:5000')

    # Initialise the live stream detector
    live_stream_detector = LiveStreamDetector(
        api_url=api_url,
        model_key=model_key,
        output_folder=site,
        detect_with_server=detect_with_server,
    )

    # Initialise the drawing manager
    drawing_manager = DrawingManager()

    # Initialise the LINE notifier
    line_notifier = LineNotifier()

    # Initialise the DangerDetector
    danger_detector = DangerDetector()

    # Init last_notification_time to 300s ago, no microseconds
    last_notification_time = int(time.time()) - 300

    # Use the generator function to process detections
    for frame, timestamp in streaming_capture.execute_capture():
        start_time = time.time()
        # Convert UNIX timestamp to datetime object and format it as string
        detection_time = datetime.fromtimestamp(timestamp)
        current_hour = detection_time.hour

        # Detect hazards in the frame
        datas, _ = live_stream_detector.generate_detections(frame)

        # Check for warnings and send notifications if necessary
        warnings, controlled_zone_polygon = danger_detector.detect_danger(
            datas,
        )

        # Check if there is a warning for people in the controlled zone
        controlled_zone_warning_str = next(
            # Find the first warning containing 'controlled area'
            (warning for warning in warnings if 'controlled area' in warning),
            None,
        )

        # Convert the warning to a list for translation
        controlled_zone_warning: list[str] = [
            controlled_zone_warning_str,
        ] if controlled_zone_warning_str else []

        # Track whether we sent any notification
        notification_sent = False
        last_line_token = None
        last_language = None
        frame_with_detections = None

        if not notifications:
            logger.info('No notifications provided.')

            # Draw the detections on the frame
            frame_with_detections = (
                drawing_manager.draw_detections_on_frame(
                    frame, controlled_zone_polygon, datas,
                )
            )

            # Convert the frame to a byte array
            _, buffer = cv2.imencode('.png', frame_with_detections)
            frame_bytes = buffer.tobytes()
            continue

        # Check if notifications are provided
        for line_token, language in notifications.items():
            # Check if notification should be skipped
            # (sent within last 300 seconds)
            if (timestamp - last_notification_time) < 300:
                # Store the last token and language,
                # but don't send notifications
                last_line_token = line_token
                last_language = language
                # Skip the current notification but remember the last one
                continue

            # Translate the warnings
            translated_warnings = Translator.translate_warning(
                warnings, language,
            )

            # Draw the detections on the frame
            frame_with_detections = (
                drawing_manager.draw_detections_on_frame(
                    frame, controlled_zone_polygon, datas,
                    language=language,
                )
            )

            # Save the frame with detections
            # save_file_name = f'{site}_{stream_name}_{detection_time}'
            # drawing_manager.save_frame(
            #   frame_with_detections,
            #   save_file_name
            # )

            # Convert the frame to a byte array
            _, buffer = cv2.imencode('.png', frame_with_detections)
            frame_bytes = buffer.tobytes()

            # Log the detection results
            logger.info(f"{site} - {stream_name}")
            logger.info(f"Detection time: {detection_time}")

            # If it is outside working hours and there is
            # a warning for people in the controlled zone
            if controlled_zone_warning and not (7 <= current_hour < 18):
                translated_controlled_zone_warning: list[str] = (
                    Translator.translate_warning(
                        controlled_zone_warning, language,
                    )
                )
                message = (
                    f"{stream_name}\n[{detection_time}]\n"
                    f"{translated_controlled_zone_warning}"
                )

            elif translated_warnings and (7 <= current_hour < 18):
                # During working hours, combine all warnings
                message = (
                    f"{stream_name}\n[{detection_time}]\n"
                    + '\n'.join(translated_warnings)
                )

            else:
                message = None

            # If a notification needs to be sent
            if not message:
                logger.info('No warnings or outside notification time.')
                continue

            notification_status = line_notifier.send_notification(
                message,
                image=frame_bytes
                if frame_bytes is not None
                else None,
                line_token=line_token,
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
                notification_sent = True  # Mark that a notification was sent
            else:
                logger.error(f"Failed to send notification: {message}")

            # Log the notification token and language
            logger.info(f"Notification sent to {line_token} in {language}.")

        # If no notification was sent and the time condition was met,
        # only draw the image
        if last_line_token and last_language:

            # Draw the detections on the frame for the last token/language
            # (if not already drawn)
            if frame_with_detections is None:
                frame_with_detections = (
                    drawing_manager.draw_detections_on_frame(
                        frame, controlled_zone_polygon,
                        datas,
                        language=last_language,
                    )
                )

            # Convert the frame to a byte array
            _, buffer = cv2.imencode('.png', frame_with_detections)
            frame_bytes = buffer.tobytes()

            # Optionally save the frame with detections
            # drawing_manager.save_frame(frame_with_detections, save_file_name)

        # Update last_notification_time only
        # if at least one notification was sent
        if notification_sent:
            last_notification_time = int(timestamp)

        # Store the frame in Redis if not running on Windows
        if not is_windows:
            try:
                # Use a unique key for each thread or process
                key = f"{site}_{stream_name}"

                # Store the frame in Redis
                redis_manager.set(key, frame_bytes)
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
        # Check if 'notifications' field exists (new format)
        if 'notifications' in config and config['notifications'] is not None:
            notifications = config['notifications']
        # Otherwise, handle the old format
        elif 'line_token' in config and 'language' in config:
            line_token = config.get('line_token')
            language = config.get('language')
            if line_token is not None and language is not None:
                notifications = {line_token: language}
            else:
                notifications = None
        else:
            notifications = None

        # Continue processing the remaining configuration
        video_url = config.get('video_url', '')
        model_key = config.get('model_key', 'yolo11n')
        site = config.get('site')
        stream_name = config.get('stream_name', 'prediction_visual')
        detect_with_server = config.get('detect_with_server', False)

        # Run hazard detection on a single video stream
        process_single_stream(
            logger,
            video_url=video_url,
            model_key=model_key,
            site=site,
            stream_name=stream_name,
            notifications=notifications,
            detect_with_server=detect_with_server,
        )
    finally:
        if not is_windows:
            site = config.get('site')
            stream_name = config.get('stream_name', 'prediction_visual')
            key = f"{site}_{stream_name}"
            redis_manager.delete(key)
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


def process_single_image(
    image_path: str,
    model_key: str = 'yolo11n',
    output_folder: str = 'output_images',
    stream_name: str | None = None,
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

        # Initialise the live stream detector,
        # but here used for a single image
        api_url = os.getenv('API_URL', 'http://localhost:5000')
        live_stream_detector = LiveStreamDetector(
            api_url=api_url, model_key=model_key, output_folder=output_folder,
        )

        # Initialise the drawing manager
        drawing_manager = DrawingManager()

        # Detect hazards in the image
        detections, _ = live_stream_detector.generate_detections(image)

        # For this example, no polygons are needed, so pass an empty list
        frame_with_detections = drawing_manager.draw_detections_on_frame(
            image, [], detections,
            language=language,
        )

        # Ensure the output folder exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Generate the output file name if not provided
        if not stream_name:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            stream_name = f"detection_{timestamp}.png"
        output_path = os.path.join(output_folder, stream_name)

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
        default='yolo11n',
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
