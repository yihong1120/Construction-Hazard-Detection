from __future__ import annotations

import argparse
import asyncio
import gc
import json
import math
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from multiprocessing import Process
from typing import TypedDict

from asyncmy import create_pool
from dotenv import load_dotenv
from sqlalchemy.engine.url import make_url

from src.danger_detector import DangerDetector
from src.frame_sender import BackendFrameSender
from src.live_stream_detection import LiveStreamDetector
from src.monitor_logger import LoggerConfig
from src.notifiers.fcm_notifier import FCMSender
from src.stream_capture import StreamCapture
from src.utils import RedisManager
from src.utils import Utils
from src.violation_sender import ViolationSender

load_dotenv()


class StreamConfig(TypedDict, total=False):
    """
    Represents the configuration structure for a video stream as retrieved
    from the database.
    """
    video_url: str
    updated_at: str
    model_key: str
    site: str
    stream_name: str
    detect_with_server: bool
    expire_date: str | None
    detection_items: dict[str, bool]
    work_start_hour: int
    work_end_hour: int
    store_in_redis: bool


class MainApp:
    """
    Core application responsible for:
        - Polling stream configuration from database periodically
        - Dynamically spawning/stopping child processes for each video stream
        - Cleaning up expired or modified configurations
    """

    def __init__(self, poll_interval: int = 10) -> None:
        """
        Initialise the application.

        Args:
            poll_interval (int): Interval in seconds to poll the database for
                stream configuration updates.
        """
        self.poll_interval = poll_interval
        self.logger = LoggerConfig().get_logger()
        # video_url → process info dict
        self.running_processes: dict[str, dict] = {}
        self.lock = asyncio.Lock()  # Prevent overlapping reloads
        self.db_pool = None  # Will hold MySQL connection pool

        # Process pool management to improve performance
        self.max_workers = min(multiprocessing.cpu_count(), 8)
        self.process_executor: ProcessPoolExecutor | None = None

    async def _ensure_db_pool(self) -> None:
        """
        Ensure a connection pool to the database is
        established before querying.
        """
        if self.db_pool is None:
            database_url = os.getenv('DATABASE_URL')
            if database_url is None:
                raise RuntimeError(
                    'DATABASE_URL environment variable is required',
                )
            url = make_url(database_url)
            self.db_pool = await create_pool(
                host=url.host,
                port=url.port or 3306,
                user=url.username,
                password=url.password,
                db=url.database,
                minsize=2,  # Minimum connections
                maxsize=10,  # Maximum connections
                pool_recycle=3600,  # 1 hour connection recycling
                autocommit=True,
                echo=False,  # Disable SQL logging for performance
            )

    async def fetch_stream_configs(self) -> list[StreamConfig]:
        """
        Query the database for current stream configurations.

        Returns:
            list[StreamConfig]: All active stream configuration records.
        """
        await self._ensure_db_pool()
        if self.db_pool is None:
            raise RuntimeError(
                'Database pool is not initialized. Check DATABASE_URL and '
                'DB connectivity.',
            )
        sql = """
        SELECT sc.video_url,
               sc.updated_at,
               sc.model_key,
               s.name              AS site,
               sc.stream_name,
               sc.detect_with_server,
               sc.expire_date,
               sc.work_start_hour,
               sc.work_end_hour,
               sc.store_in_redis,
               sc.detect_no_safety_vest_or_helmet,
               sc.detect_near_machinery_or_vehicle,
               sc.detect_in_restricted_area,
               sc.detect_in_utility_pole_restricted_area,
               sc.detect_machinery_close_to_pole
        FROM stream_configs sc
        JOIN sites s ON sc.site_id = s.id
        """
        async with self.db_pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql)
                rows = await cur.fetchall()

        configs: list[StreamConfig] = []

        for row in rows:
            (
                video_url, updated_at, model_key, site, stream_name,
                detect_with_server, expire_date, work_start, work_end,
                store_in_redis, vest_helmet, near_vehicle, in_area,
                in_pole_area, machine_close_pole,
            ) = row

            # Organise detection flags into a dictionary
            detection_items = {
                'detect_no_safety_vest_or_helmet': bool(vest_helmet),
                'detect_near_machinery_or_vehicle': bool(near_vehicle),
                'detect_in_restricted_area': bool(in_area),
                'detect_in_utility_pole_restricted_area': bool(in_pole_area),
                'detect_machinery_close_to_pole': bool(machine_close_pole),
            }

            configs.append(
                StreamConfig(
                    video_url=video_url,
                    updated_at=updated_at.isoformat(),
                    model_key=model_key,
                    site=site,
                    stream_name=stream_name,
                    detect_with_server=bool(detect_with_server),
                    expire_date=(
                        expire_date.isoformat() if expire_date else None
                    ),
                    detection_items=detection_items,
                    work_start_hour=int(work_start or 7),
                    work_end_hour=int(work_end or 18),
                    store_in_redis=bool(store_in_redis),
                ),
            )

        return configs

    async def poll_and_reload(self) -> None:
        """
        Periodically poll the database and trigger reload logic.
        This function will run indefinitely unless interrupted.
        """
        while True:
            try:
                await self.reload_configurations()
            except Exception as e:
                self.logger.exception(f"[poll] Reload error: {e}")
            await asyncio.sleep(self.poll_interval)

    async def reload_configurations(self) -> None:
        """
        Main configuration reload logic:
            - Stops expired or deleted stream processes
            - Restarts modified streams (based on updated_at)
            - Starts newly added streams not yet tracked
        """
        async with self.lock:
            configs = await self.fetch_stream_configs()
            cfg_map = {c['video_url']: c for c in configs}

            # 1. Stop removed or expired streams
            for video_url in list(self.running_processes.keys()):
                proc_info = self.running_processes[video_url]
                cfg = cfg_map.get(video_url)

                if not cfg or Utils.is_expired(cfg.get('expire_date')):
                    self.logger.info(f"Stop stream {video_url}")
                    self.stop_process(proc_info['process'])

                    if proc_info['cfg'].get('store_in_redis'):
                        redis_key = (
                            f"stream_frame:"
                            f"{Utils.encode(proc_info['cfg']['site'])}"
                            f"|{Utils.encode(proc_info['cfg']['stream_name'])}"
                        )
                        redis_manager = RedisManager()
                        await redis_manager.delete(redis_key)

                    del self.running_processes[video_url]
                    continue

                # 2. Restart if config updated
                if cfg['updated_at'] != proc_info['updated_at']:
                    self.logger.info(
                        f"Restart stream {video_url} (updated_at changed)",
                    )
                    self.stop_process(proc_info['process'])

                    if proc_info['cfg'].get('store_in_redis'):
                        redis_key = (
                            f"stream_frame:"
                            f"{Utils.encode(proc_info['cfg']['site'])}"
                            f"|{Utils.encode(proc_info['cfg']['stream_name'])}"
                        )
                        redis_manager = RedisManager()
                        await redis_manager.delete(redis_key)

                    new_proc = self.start_process(cfg)
                    self.running_processes[video_url] = {
                        'process': new_proc,
                        'updated_at': cfg['updated_at'],
                        'cfg': cfg,
                    }

            # 3. Start any new streams
            for video_url, cfg in cfg_map.items():
                if Utils.is_expired(cfg.get('expire_date')):
                    continue
                if video_url not in self.running_processes:
                    self.logger.info(
                        f"Launch new stream {video_url}",
                    )
                    proc = self.start_process(cfg)
                    self.running_processes[video_url] = {
                        'process': proc,
                        'updated_at': cfg['updated_at'],
                        'cfg': cfg,
                    }

    def start_process(self, cfg: StreamConfig) -> Process:
        """
        Launch a new child process to handle stream detection.

        Args:
            cfg (StreamConfig): Configuration for the stream.

        Returns:
            Process: The new multiprocessing.Process object.
        """
        p = Process(target=process_single_stream, args=(cfg,))
        p.start()
        return p

    def stop_process(self, proc: Process) -> None:
        """
        Gracefully terminate a child process.

        Args:
            proc (Process): The process to be terminated.
        """
        try:
            # Attempt graceful termination
            proc.terminate()
            proc.join(timeout=10)  # Wait for up to 10 seconds

            if proc.is_alive():
                # If still alive, force kill
                proc.kill()
                proc.join()
        except Exception as e:
            self.logger.error(f"Error stopping process: {e}")

    async def cleanup_resources(self) -> None:
        """
        Clean up all resources
        """
        # Stop all processes
        for info in self.running_processes.values():
            self.stop_process(info['process'])
        self.running_processes.clear()

        # Close process pool
        if self.process_executor:
            self.process_executor.shutdown(wait=True)

        # Close database connection pool
        if self.db_pool:
            self.db_pool.close()
            await self.db_pool.wait_closed()

    async def run(self) -> None:
        """
        Start the application loop that continuously checks the stream configs.
        """
        try:
            await self.poll_and_reload()
        except KeyboardInterrupt:
            self.logger.info('Received keyboard interrupt, shutting down...')
        except Exception as e:
            self.logger.error(f"Unexpected error in main loop: {e}")
        finally:
            await self.cleanup_resources()


async def main() -> None:
    """
    Parse command-line arguments and run the MainApp.
    """
    parser = argparse.ArgumentParser(
        description='Hazard detection from DB configs or JSON file',
    )
    parser.add_argument(
        '--poll', type=int, default=10,
        help='DB polling interval in seconds',
    )
    parser.add_argument(
        '--config', type=str,
        help='Path to JSON config file for stream configs',
    )
    args = parser.parse_args()

    if args.config:
        # Load configs from JSON file
        with open(args.config, encoding='utf-8') as f:
            configs = json.load(f)
        # Start a process for each config
        procs = []
        for cfg in configs:
            proc = Process(target=process_single_stream, args=(cfg,))
            proc.start()
            procs.append(proc)
        try:
            while any(p.is_alive() for p in procs):
                for p in procs:
                    p.join(timeout=1)
        except KeyboardInterrupt:
            print('\n[INFO] KeyboardInterrupt, shutting down...')
        finally:
            for p in procs:
                if p.is_alive():
                    p.terminate()
                    p.join()
    else:
        app = MainApp(poll_interval=args.poll)
        try:
            await app.run()
        except KeyboardInterrupt:
            print('\n[INFO] KeyboardInterrupt, shutting down...')
        finally:
            # Ensure all child processes are stopped and DB closed
            for info in app.running_processes.values():
                app.stop_process(info['process'])
            if app.db_pool:
                app.db_pool.close()
                await app.db_pool.wait_closed()


def process_single_stream(cfg: StreamConfig) -> None:
    """
    Logic executed by each child process to capture frames, detect hazards,
    and send results to backend or Redis.

    Args:
        cfg (StreamConfig): The configuration dict for this stream.
    """

    video_url = cfg['video_url']
    model_key = cfg['model_key']
    site = cfg['site']
    stream_name = cfg['stream_name']
    detect_with_server = cfg['detect_with_server']
    detection_items = cfg['detection_items']
    work_start_hour = cfg['work_start_hour']
    work_end_hour = cfg['work_end_hour']
    store_in_redis = cfg['store_in_redis']

    async def _main() -> None:
        # Initialise detection components
        streaming_capture = StreamCapture(stream_url=video_url)
        live_stream_detector = LiveStreamDetector(
            api_url=os.getenv('DETECT_API_URL') or '',
            model_key=model_key,
            output_folder=site,
            detect_with_server=detect_with_server,
        )
        danger_detector = DangerDetector(detection_items)
        fcm_sender = FCMSender(api_url=os.getenv('FCM_API_URL') or '')
        violation_sender = ViolationSender(
            api_url=os.getenv('VIOLATION_RECORD_API_URL') or '',
        )
        frame_sender = BackendFrameSender(
            api_url=os.getenv('STREAMING_API_URL') or '',
            max_retries=3,
            timeout=30,  # Increase timeout
            reconnect_backoff=2.0,  # Moderate backoff time
        )

        last_notification_time: int = 0
        redis_key = (
            f"stream_frame:{Utils.encode(site)}|{Utils.encode(stream_name)}"
        )
        redis_manager = RedisManager()

        try:
            # Process each frame
            async for frame, ts in streaming_capture.execute_capture():
                start = time.time()
                detection_time = datetime.fromtimestamp(int(ts))
                is_working = (
                    work_start_hour <= detection_time.hour < work_end_hour
                )

                datas, track_data = (
                    await live_stream_detector.generate_detections(
                        frame,
                    )
                )
                warnings, cone_polys, pole_polys = (
                    danger_detector.detect_danger(
                        track_data,
                    )
                )
                warnings = Utils.filter_warnings_by_working_hour(
                    warnings, is_working,
                )

                # Use optimized frame encoding (JPEG for better compression)
                frame_bytes = Utils.encode_frame(frame, 'jpeg', 85)

                # Optionally stream result to backend
                # using optimised transmission
                if store_in_redis:
                    try:
                        result = await frame_sender.send_optimized_frame(
                            frame=frame,
                            site=site,
                            stream_name=stream_name,
                            encoding_format='jpeg',
                            jpeg_quality=85,
                            use_websocket=True,
                            warnings_json=json.dumps(warnings),
                            cone_polygons_json=json.dumps(cone_polys),
                            pole_polygons_json=json.dumps(pole_polys),
                            detection_items_json=json.dumps(datas),
                        )

                        # Check send result
                        if result.get('status') != 'ok':
                            print(
                                f"[{site}:{stream_name}] Frame send failed: "
                                f"{result}",
                            )

                    except Exception as e:
                        # Handle frame send error
                        print(f"[{site}:{stream_name}] Frame send error: {e}")

                # Send violation record + FCM push if needed
                if (
                    warnings
                    and Utils.should_notify(int(ts), last_notification_time)
                ):
                    violation_id_str = await violation_sender.send_violation(
                        site=site,
                        stream_name=stream_name,
                        warnings_json=json.dumps(warnings),
                        detection_time=detection_time,
                        image_bytes=frame_bytes,
                        detections_json=json.dumps(datas),
                        cone_polygon_json=json.dumps(cone_polys),
                        pole_polygon_json=json.dumps(pole_polys),
                    )
                    # Try to convert violation_id to int, else None
                    try:
                        violation_id: int | None = (
                            int(violation_id_str)
                            if violation_id_str is not None else None
                        )
                    except Exception:
                        violation_id = None
                    await fcm_sender.send_fcm_message_to_site(
                        site=site,
                        stream_name=stream_name,
                        message=warnings,
                        image_path=None,
                        violation_id=violation_id,
                    )
                    last_notification_time = int(ts)

                # Dynamically adjust processing interval
                proc_time = time.time() - start
                streaming_capture.update_capture_interval(
                    int((math.floor(proc_time * 2) + 1) / 2),
                )

            await streaming_capture.release_resources()
            gc.collect()

        finally:
            # Ensure cleanup
            await live_stream_detector.close()
            await streaming_capture.release_resources()
            await frame_sender.close()  # Ensure WebSocket connection is closed
            if store_in_redis:
                try:
                    await redis_manager.delete(redis_key)
                except Exception as e:
                    print(
                        f"[WARN] Failed to delete redis key {redis_key}: {e}",
                    )

    asyncio.run(_main())


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    asyncio.run(main())
