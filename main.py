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
from src.live_stream_detection import LiveStreamDetector
from src.monitor_logger import LoggerConfig
from src.notifiers.line_notifier import LineNotifier
from src.stream_capture import StreamCapture
from src.utils import FileEventHandler
from src.utils import RedisManager
from src.utils import Utils

# Load environment variables
load_dotenv()


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
    work_start_hour: int | None
    work_end_hour: int | None
    store_in_redis: bool


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
            'expire_date': config.get('expire_date'),
            'language': config.get('language'),
            'detection_items': config.get('detection_items'),
            'work_start_hour': config.get('work_start_hour'),
            'work_end_hour': config.get('work_end_hour'),
            'store_in_redis': config.get('store_in_redis', False),
        }
        return str(relevant_config)  # Convert to string for hashing

    async def reload_configurations(self):
        async with self.lock:
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
        work_start_hour: int = 7,
        work_end_hour: int = 18,
        store_in_redis: bool = False,
    ) -> None:
        """
        Process a single video stream with hazard detection, notifications,
        and Redis storage.

        Args:
            logger (logging.Logger): Logger instance for logging.
            video_url (str): Video stream URL.
            model_key (str): Detection model key.
            site (str): Site name for the stream.
            stream_name (str): Stream name for notifications.
            notifications (dict): Line tokens with their languages.
            detect_with_server (bool): Whether to use server for detection.
            detection_items (dict): Items to detect.
        """
        if store_in_redis:
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
        danger_detector = DangerDetector(detection_items or {})

        # Notifications setup
        last_notification_times = {
            token: int(
                time.time(),
            ) - 300 for token in (notifications or {})
        }

        # Get the language for Redis storage
        redis_storage_language = (
            list(notifications.values())[-1] if notifications else 'en'
        )

        # Use the generator function to process detections
        async for frame, timestamp in streaming_capture.execute_capture():
            timestamp = int(timestamp)
            start_time = time.time()

            # Convert UNIX timestamp to datetime object and format it as string
            detection_time = datetime.fromtimestamp(timestamp)

            # Check if the current time is within working hours
            is_working_hour = (
                work_start_hour <= detection_time.hour < work_end_hour
            )

            # Detection step
            datas, _ = await live_stream_detector.generate_detections(frame)
            warnings, controlled_zone_polygon = danger_detector.detect_danger(
                datas,
            )
            controlled_zone_warning = [
                w for w in warnings if 'controlled area' in w
            ]

            # Notification step
            for token, lang in (notifications or {}).items():
                # Check if it is time to send a notification
                if not Utils.should_notify(
                    timestamp,
                    last_notification_times[token],
                ):
                    continue

                # Generate the notification message
                message = Utils.generate_message(
                    stream_name,
                    detection_time,
                    warnings,
                    controlled_zone_warning,
                    lang,
                    is_working_hour,
                )
                if not message:
                    continue

                # Draw detections for LINE notification
                frame_with_detections = (
                    drawing_manager.draw_detections_on_frame(
                        frame, controlled_zone_polygon, datas, language=lang,
                    )
                )
                frame_bytes = Utils.encode_frame(frame_with_detections)

                # Send the notification
                status = await line_notifier.send_notification(
                    message,
                    image=frame_bytes
                    if frame_bytes is not None
                    else None,
                    line_token=token,
                )
                if status == 200:
                    logger.info(f"Notification sent: {message}")
                    last_notification_times[token] = timestamp

            # Draw detections for Redis storage
            frame_with_detections = drawing_manager.draw_detections_on_frame(
                frame,
                controlled_zone_polygon,
                datas,
                language=redis_storage_language or 'en',
            )
            frame_bytes = Utils.encode_frame(frame_with_detections)

            # Store the frame bytes to Redis
            if store_in_redis:
                await redis_manager.store_to_redis(
                    site=site or 'default',
                    stream_name=stream_name,
                    frame_bytes=frame_bytes,
                    warnings=warnings,
                    language=redis_storage_language,
                )

            # Update the capture interval based on processing time
            processing_time = time.time() - start_time
            streaming_capture.update_capture_interval(
                max(1, int(processing_time) + 1),
            )

            # Log the detection results
            logger.info(
                f"Processed {site}-{stream_name} in {processing_time:.2f}s",
            )

        # Release resources after processing
        await streaming_capture.release_resources()

        # Close the Redis connection
        if store_in_redis:
            await redis_manager.close_connection()

        gc.collect()

    async def process_streams(self, config: AppConfig) -> None:
        """
        Process a video stream based on the given configuration.

        Args:
            config (AppConfig): The configuration for the stream processing.

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
            store_in_redis = config.get('store_in_redis', False)
            work_start_hour = config.get('work_start_hour', 7)
            work_end_hour = config.get('work_end_hour', 18)

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
                work_start_hour=work_start_hour or 7,
                work_end_hour=work_end_hour or 18,
                store_in_redis=store_in_redis,
            )
        finally:
            if config.get('store_in_redis', False):
                redis_manager = RedisManager()
                site = config.get('site') or 'default site'
                stream_name = config.get(
                    'stream_name',
                ) or 'default stream name'

                site = Utils.encode(site)
                stream_name = Utils.encode(stream_name)

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
        # if store_in_redis:
        #     await redis_manager.close_connection()
        #     print('[INFO] Redis connection closed.')
        print('[INFO] Application stopped.')
        # Clear the asyncio event loop
        await asyncio.sleep(0)


if __name__ == '__main__':
    asyncio.run(main())
