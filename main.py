from __future__ import annotations

import argparse
import asyncio
import gc
import json
import logging
import os
import time
from datetime import datetime
from multiprocessing import Process
from typing import TypedDict

import cv2
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


class AppConfig(TypedDict, total=False):
    """
    Typed dictionary for the configuration of a video stream.
    """
    video_url: str
    model_key: str
    site: str | None
    stream_name: str
    notifications: dict[str, str] | None
    detect_with_server: bool
    expire_date: str | None
    line_token: str | None
    language: str | None
    detection_items: dict[str, bool] | None


class MainApp:
    """
    Main application class for managing multiple video streams.
    """

    def __init__(self, config_file: str):
        """
        Initialise the MainApp class.

        Args:
            config_file (str): The path to the JSON configuration file.
        """
        self.config_file = config_file
        self.running_processes: dict[str, dict] = {}
        self.current_config_hashes: dict[str, str] = {}
        self.lock = asyncio.Lock()
        self.logger = LoggerConfig().get_logger()

    def compute_config_hash(self, config: dict) -> str:
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

    async def reload_configurations(self):
        async with self.lock:
            if not is_windows:
                redis_manager = RedisManager()

            self.logger.info('Reloading configurations...')
            with open(self.config_file, encoding='utf-8') as file:
                configurations = json.load(file)

            current_configs = {
                config['video_url']: config for config in configurations
            }

            # Track keys that exist in the current config
            current_keys = {
                (
                    f"{config['site']}_"
                    f"{config.get('stream_name', 'prediction_visual')}"
                )
                for config in configurations
            }

            # Stop processes for removed or updated configurations
            for video_url in list(self.running_processes.keys()):
                config_data = self.running_processes[video_url]
                config = current_configs.get(video_url)

                # Get the key to be deleted
                # if config has been removed or modified
                site = config_data['config']['site']
                stream_name = config_data['config'].get(
                    'stream_name', 'prediction_visual',
                )

                site = Utils.encode(site)
                stream_name = Utils.encode(stream_name)

                key_to_delete = f"stream_frame:{site}|{stream_name}"

                # Stop the process if the configuration is removed
                if not config or Utils.is_expired(config.get('expire_date')):
                    self.logger.info(f"Stop workflow: {video_url}")
                    self.stop_process(config_data['process'])
                    del self.running_processes[video_url]
                    del self.current_config_hashes[video_url]

                    # Delete old key in Redis
                    # if it no longer exists in the config
                    if key_to_delete not in current_keys:
                        await redis_manager.delete(key_to_delete)
                        self.logger.info(f"Deleted Redis key: {key_to_delete}")

                # Restart the process if the configuration is updated
                elif self.compute_config_hash(config) != (
                    self.current_config_hashes.get(
                        video_url,
                    )
                ):
                    self.logger.info(
                        f"Config changed for {video_url}. "
                        'Restarting workflow.',
                    )
                    self.stop_process(config_data['process'])

                    # Delete old key in Redis
                    # if it no longer exists in the config
                    if key_to_delete not in current_keys:
                        await redis_manager.delete(key_to_delete)
                        self.logger.info(f"Deleted Redis key: {key_to_delete}")

                    # Start the new process
                    self.running_processes[video_url] = {
                        'process': self.start_process(config),
                        'config': config,
                    }
                    self.current_config_hashes[video_url] = (
                        self.compute_config_hash(
                            config,
                        )
                    )

            # Start processes for new configurations
            for video_url, config in current_configs.items():
                if Utils.is_expired(config.get('expire_date')):
                    self.logger.info(
                        f"Skip expired configuration: {video_url}",
                    )
                    continue

                if video_url not in self.running_processes:
                    self.logger.info(f"Launch new workflow: {video_url}")
                    self.running_processes[video_url] = {
                        'process': self.start_process(config),
                        'config': config,
                    }
                    self.current_config_hashes[video_url] = (
                        self.compute_config_hash(
                            config,
                        )
                    )

            # Close Redis connection
            await redis_manager.close_connection()

    async def run_multiple_streams(self) -> None:
        """
        Manage multiple video streams based on a config file.
        """
        # Initial load of configurations
        await self.reload_configurations()

        # Get the absolute path of the configuration file
        config_file_path = os.path.abspath(self.config_file)
        # Get the directory of the configuration file
        config_dir = os.path.dirname(config_file_path)
        # Get the current event loop
        loop = asyncio.get_running_loop()

        # Set up watchdog observer
        event_handler = FileEventHandler(
            config_file_path, self.reload_configurations, loop,
        )
        observer = Observer()
        observer.schedule(event_handler, path=config_dir, recursive=False)
        observer.start()

        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            self.logger.info(
                '\n[INFO] Received KeyboardInterrupt. Stopping observer...',
            )
            observer.stop()
        finally:
            observer.join()  # Wait for the observer to finish
            self.logger.info('[INFO] Observer stopped.')
            # Stop all running processes
            for video_url, data in self.running_processes.items():
                self.stop_process(data['process'])
            self.running_processes.clear()
            self.logger.info('[INFO] All processes stopped.')

    async def process_single_stream(
        self,
        logger: logging.Logger,
        video_url: str,
        model_key: str = 'yolo11n',
        site: str | None = None,
        stream_name: str = 'prediction_visual',
        notifications: dict[str, str] | None = None,
        detect_with_server: bool = False,
        detection_items: dict[str, bool] | None = {},
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
            detect_with_server (bool): If run detection with server api or not.
            detection_items (dict): The detection items to check for.
        """
        if not is_windows:
            redis_manager = RedisManager()

        # Initialise the stream capture object
        streaming_capture = StreamCapture(stream_url=video_url)

        # Initialise the live stream detector
        live_stream_detector = LiveStreamDetector(
            api_url=os.getenv('API_URL', 'http://localhost:5000'),
            model_key=model_key,
            output_folder=site,
            detect_with_server=detect_with_server,
        )

        # Initialise the drawing manager
        drawing_manager = DrawingManager()

        # Initialise the LINE notifier
        line_notifier = LineNotifier()

        # Initialise the DangerDetector
        danger_detector = DangerDetector(detection_items=detection_items or {})

        # Dictionary to store last notification time for each language
        if notifications is None:
            notifications = {}

        last_notification_times = {
            line_token: int(
                time.time(),
            ) - 300 for line_token in notifications
        }

        # Use the generator function to process detections
        async for frame, timestamp in streaming_capture.execute_capture():
            start_time = time.time()
            # Convert UNIX timestamp to datetime object and format it as string
            detection_time = datetime.fromtimestamp(timestamp)
            current_hour = detection_time.hour

            # Detect hazards in the frame
            datas, _ = await live_stream_detector.generate_detections(frame)

            # Check for warnings and send notifications if necessary
            warnings, controlled_zone_polygon = danger_detector.detect_danger(
                datas,
            )

            # Check if there is a warning for people in the controlled zone
            controlled_zone_warning_str = next(
                # Find the first warning containing 'controlled area'
                (
                    warning
                    for warning in warnings
                    if 'controlled area' in warning
                ),
                None,
            )

            # Convert the warning to a list for translation
            controlled_zone_warning: list[str] = [
                controlled_zone_warning_str,
            ] if controlled_zone_warning_str else []

            # Track whether we sent any notification
            last_line_token = None
            last_language = None
            frame_with_detections = None

            if not notifications:
                logger.info('No notifications provided.')

            else:
                # Check if notifications are provided
                for line_token, language in notifications.items():
                    # Check if notification should be skipped
                    # (sent within last 300 seconds)
                    if (timestamp - last_notification_times[line_token]) < 300:
                        # Store the last token and language,
                        # but don't send notifications
                        last_line_token = line_token
                        last_language = language
                        # Skip the current notification
                        # but remember the last one
                        continue

                    # Translate the warnings
                    translated_warnings = Translator.translate_warning(
                        tuple(warnings), language,
                    )

                    # Draw the detections on the frame
                    frame_with_detections = (
                        drawing_manager.draw_detections_on_frame(
                            frame, controlled_zone_polygon, datas,
                            language=language,
                        )
                    )

                    # Convert the frame to a byte array
                    _, buffer = cv2.imencode('.png', frame_with_detections)
                    frame_bytes = buffer.tobytes()

                    # If it is outside working hours and there is
                    # a warning for people in the controlled zone
                    if (
                        controlled_zone_warning
                        and not (7 <= current_hour < 18)
                    ):
                        translated_controlled_zone_warning: list[str] = (
                            Translator.translate_warning(
                                tuple(controlled_zone_warning), language,
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
                        logger.info(
                            'No warnings or outside notification time.',
                        )
                        continue

                    notification_status = line_notifier.send_notification(
                        message,
                        image=frame_bytes
                        if frame_bytes is not None
                        else None,
                        line_token=line_token,
                    )

                    # To connect to the broadcast system, do it here:
                    # broadcast_status = (
                    #   broadcast_notifier.broadcast_message(message)
                    # )
                    # logger.info(f"Broadcast status: {broadcast_status}")

                    if notification_status == 200:
                        logger.info(
                            f"Notification sent successfully: {message}",
                        )
                        last_notification_times[line_token] = int(timestamp)
                    else:
                        logger.error(f"Failed to send notification: {message}")

                    # Log the notification token and language
                    logger.info(
                        f"Notification sent to {line_token} in {language}.",
                    )

                # If no notification was sent and the time condition was met,
                # only draw the image
                if last_line_token and last_language:
                    language = last_language

            # Draw the detections on the frame for the last token/language
            # (if not already drawn)
            if frame_with_detections is None:
                frame_with_detections = (
                    drawing_manager.draw_detections_on_frame(
                        frame, controlled_zone_polygon,
                        datas,
                        language=last_language or 'en',
                    )
                )

            # Convert the frame to a byte array
            _, buffer = cv2.imencode('.png', frame_with_detections)
            frame_bytes = buffer.tobytes()

            # Save the frame with detections
            # save_file_name = f'{site}_{stream_name}_{detection_time}'
            # drawing_manager.save_frame(
            #   frame_with_detections,
            #   save_file_name
            # )

            # Store the frame and warnings in Redis if not running on Windows
            if not is_windows:
                try:
                    # Encode site and stream_name to avoid issues
                    # with special characters
                    encoded_site = Utils.encode(site or 'default site')
                    encoded_stream_name = Utils.encode(
                        stream_name,
                    ) or 'default stream name'

                    # Use a unique key for each thread or process
                    key = f"stream_frame:{encoded_site}|{encoded_stream_name}"

                    if not warnings:
                        warnings = ['No warning']

                    # Translate the warnings
                    translated_warnings = Translator.translate_warning(
                        warnings=tuple(warnings), language='zh-TW',
                    )

                    # Combine warnings into a single string for storage
                    warnings_str = '\n'.join(translated_warnings)

                    # Store the frame and warnings in Redis Stream
                    # with a maximum length of 10
                    await redis_manager.add_to_stream(
                        key,
                        {'frame': frame_bytes, 'warnings': warnings_str},
                        maxlen=10,
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to store frame and warnings in Redis: {e}",
                    )

            # Update the capture interval based on processing time
            processing_time = time.time() - start_time
            streaming_capture.update_capture_interval(int(processing_time) + 1)

            # Log the detection results
            logger.info(f"{site} - {stream_name}")
            logger.info(f"Detection time: {detection_time}")
            logger.info(f"Processing time: {processing_time:.2f} seconds")

            # Clear variables to free up memory
            gc.collect()

        # Release resources after processing
        await streaming_capture.release_resources()

        # Close the Redis connection
        if not is_windows:
            await redis_manager.close_connection()

        gc.collect()

    async def process_streams(self, config: AppConfig) -> None:
        """
        Process a video stream based on the given configuration.

        Args:
            config (StreamConfig): The configuration for the stream processing.

        Returns:
            None
        """
        try:
            # Check if 'notifications' field exists (new format)
            if (
                'notifications' in config and
                config['notifications'] is not None
            ):
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
            detection_items = config.get('detection_items', None)

            # Run hazard detection on a single video stream
            await self.process_single_stream(
                self.logger,
                video_url=video_url,
                model_key=model_key,
                site=site,
                stream_name=stream_name,
                notifications=notifications,
                detect_with_server=detect_with_server,
                detection_items=detection_items,
            )
        finally:
            if not is_windows:
                redis_manager = RedisManager()
                site = config.get('site') or 'default site'
                stream_name = config.get(
                    'stream_name',
                ) or 'default stream name'

                site = Utils.encode(site)
                stream_name = Utils.encode(
                    stream_name,
                )

                key = f"stream_frame:{site}|{stream_name}"
                await redis_manager.delete(key)
                self.logger.info(f"Deleted Redis key: {key}")

                # Close the Redis connection
                await redis_manager.close_connection()

    def start_process(self, config: AppConfig) -> Process:
        """
        Start a new process for processing a video stream.

        Args:
            config (StreamConfig): The configuration for the stream processing.

        Returns:
            Process: The newly started process.
        """
        p = Process(target=lambda: asyncio.run(self.process_streams(config)))
        p.start()
        return p

    def stop_process(self, process: Process) -> None:
        """
        Stop a running process.

        Args:
            process (Process): The process to be terminated.

        Returns:
            None
        """
        process.terminate()
        process.join()


async def process_single_image(
    image_path: str,
    model_key: str = 'yolo11n',
    output_folder: str = 'output_images',
    stream_name: str | None = None,
    language: str = 'en',
) -> None:
    """
    Process a single image for hazard detection and save the result.

    Args:
        image_path (str): The path to the image file.
        model_key (str): The model key to use for detection.
        output_folder (str): The folder to save the output image.
        stream_name (str): The name of the output image file.
        language (str): The language for labels on the output image.

    Returns:
        None
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
        live_stream_detector = LiveStreamDetector(
            api_url=os.getenv('API_URL', 'http://localhost:5000'),
            model_key=model_key,
            output_folder=output_folder,
        )

        # Initialise the drawing manager
        drawing_manager = DrawingManager()

        # Detect hazards in the image
        detections, _ = await live_stream_detector.generate_detections(image)

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


async def main():
    parser = argparse.ArgumentParser(
        description=(
            'Run hazard detection on multiple video streams or a single image.'
        ),
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/configuration.json',
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

    try:
        if args.image:
            # If an image path is provided, process the single image
            await process_single_image(
                image_path=args.image,
                model_key=args.model_key,
                output_folder=args.output_folder,
                language=args.language,
            )
        else:
            # Otherwise, run hazard detection on multiple video streams
            app = MainApp(args.config)
            await app.run_multiple_streams()
    except KeyboardInterrupt:
        print('\n[INFO] Received KeyboardInterrupt. Shutting down...')
    finally:
        # Perform necessary cleanup if needed
        # if not is_windows:
        #     await redis_manager.close_connection()
        #     print('[INFO] Redis connection closed.')
        print('[INFO] Application stopped.')
        # Clear the asyncio event loop
        await asyncio.sleep(0)


if __name__ == '__main__':
    asyncio.run(main())
