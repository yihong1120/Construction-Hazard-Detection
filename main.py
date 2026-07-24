from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from contextlib import suppress
from datetime import datetime
from multiprocessing import Process
from multiprocessing.managers import SyncManager
from typing import Any
from typing import cast

from asyncpg import create_pool  # type: ignore[import-untyped]
from asyncpg.pool import Pool  # type: ignore[import-untyped]
from dotenv import load_dotenv
from sqlalchemy.engine.url import make_url

from src.monitor_logger import LoggerConfig
from src.stream_processor import delete_stream_live_metadata
from src.stream_processor import process_single_stream
from src.stream_processor import StreamConfig
from src.utils import Utils
from src.yolo_worker import ResultStore
from src.yolo_worker import YOLO_WORKER_STOP_MESSAGE
from src.yolo_worker import YoloWorker

load_dotenv(override=True)


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
        self.db_pool: Pool | None = None  # PostgreSQL async connection pool
        self._config_listener_connection: Any | None = None
        self._config_reload_task: asyncio.Task[None] | None = None

        # Process pool management to improve performance
        self.max_workers = min(multiprocessing.cpu_count(), 8)
        self.process_executor: ProcessPoolExecutor | None = None
        self.yolo_manager: SyncManager | None = None
        self.yolo_request_queues: list[Any] = []
        self.yolo_result_stores: list[Any] = []
        self.yolo_worker_processes: list[Process] = []

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
            port = url.port
            if url.drivername.startswith('mysql') and port == 3306:
                port = 5432
            self.db_pool = await create_pool(
                host=url.host,
                port=port or 5432,
                user=url.username,
                password=url.password,
                database=url.database,
                min_size=2,
                max_size=10,
                max_inactive_connection_lifetime=300,
                command_timeout=30,
                server_settings={
                    'application_name': 'construction-hazard-detection',
                },
            )

    async def _ensure_config_listener(self) -> None:
        """Listen for committed stream-config changes between poll cycles."""
        if self._config_listener_connection is not None:
            return

        await self._ensure_db_pool()
        if self.db_pool is None:
            return

        connection = await self.db_pool.acquire()
        try:
            await connection.add_listener(
                'stream_config_changed',
                self._on_stream_config_changed,
            )
        except Exception:
            await self.db_pool.release(connection)
            raise
        self._config_listener_connection = connection

    def _on_stream_config_changed(
        self,
        _connection: object,
        _pid: int,
        _channel: str,
        _payload: str,
    ) -> None:
        """Schedule a reload when PostgreSQL signals a config change."""
        if (
            self._config_reload_task is None
            or self._config_reload_task.done()
        ):
            reload_task = asyncio.create_task(
                self.reload_configurations(),
            )
            reload_task.add_done_callback(
                self._log_config_reload_failure,
            )
            self._config_reload_task = reload_task

    def _log_config_reload_failure(self, task: asyncio.Task[None]) -> None:
        """Log callback reload failures instead of losing the task error."""
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            self.logger.error(f"[config] Immediate reload failed: {error}")

    async def _cancel_config_reload_task(self) -> None:
        """Cancel an in-flight callback reload before process shutdown."""
        task = self._config_reload_task
        self._config_reload_task = None
        if task is None or task.done() or task is asyncio.current_task():
            return
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    async def _close_config_listener(self, pool: Pool | None = None) -> None:
        """Release the reserved LISTEN connection before closing its pool."""
        connection = self._config_listener_connection
        self._config_listener_connection = None
        if connection is None:
            return

        target_pool = pool or self.db_pool
        if target_pool is None:
            return
        with suppress(Exception):
            await connection.remove_listener(
                'stream_config_changed',
                self._on_stream_config_changed,
            )
        with suppress(Exception):
            await target_pool.release(connection)

    async def fetch_stream_configs(self) -> list[StreamConfig]:
        """
        Query the database for current stream configurations.

        Returns:
            list[StreamConfig]: All configured stream records.
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
               sc.recognition_enabled,
               sc.expire_date,
               sc.work_start_hour,
               sc.work_end_hour,
               sc.detect_no_safety_vest_or_helmet,
               sc.detect_near_machinery_or_vehicle,
               sc.detect_in_restricted_area,
               sc.detect_in_utility_pole_restricted_area,
               sc.detect_machinery_close_to_pole
        FROM stream_configs sc
        JOIN sites s ON sc.site_id = s.id
        """
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(sql)

        configs: list[StreamConfig] = []

        for row in rows:
            (
                video_url, updated_at, model_key, site, stream_name,
                recognition_enabled, expire_date,
                work_start, work_end,
                vest_helmet, near_vehicle, in_area,
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
                    recognition_enabled=bool(recognition_enabled),
                    expire_date=(
                        expire_date.isoformat() if expire_date else None
                    ),
                    detection_items=detection_items,
                    work_start_hour=(
                        int(work_start) if work_start is not None else 7
                    ),
                    work_end_hour=(
                        int(work_end) if work_end is not None else 18
                    ),
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
            except TimeoutError as e:
                self.logger.exception(f"[poll] Reload timeout: {e}")
                await self._reset_db_pool()
            except Exception as e:
                self.logger.exception(f"[poll] Reload error: {e}")
            await asyncio.sleep(self.poll_interval)

    async def reload_configurations(self) -> None:
        """
        Main configuration reload logic:
            - Stops disabled, unscheduled, expired, or deleted stream processes
            - Restarts modified streams (based on updated_at)
            - Starts newly added streams not yet tracked
        """
        async with self.lock:
            configs = await self.fetch_stream_configs()
            cfg_map = {c['video_url']: c for c in configs}
            now = datetime.now()
            runnable_configs = {
                video_url: cfg
                for video_url, cfg in cfg_map.items()
                if self._can_run_recognition(cfg, now)
            }
            if runnable_configs:
                workers_restarted = (
                    self._ensure_yolo_worker()
                    if self.yolo_worker_processes
                    else False
                )
            else:
                workers_restarted = False
                if self.yolo_worker_processes:
                    self._stop_yolo_worker()

            # 1. Stop streams that are removed or cannot currently run.
            for video_url in list(self.running_processes.keys()):
                proc_info = self.running_processes[video_url]
                cfg = cfg_map.get(video_url)

                if video_url not in runnable_configs:
                    self.logger.info(f"Stop stream {video_url}")
                    self.stop_process(proc_info['process'])

                    await self._delete_stream_redis_keys(proc_info['cfg'])

                    del self.running_processes[video_url]
                    continue

                if self._stream_needs_restart(
                    proc_info,
                    cfg,
                    workers_restarted,
                ):
                    await self._restart_stream_process(
                        video_url,
                        proc_info,
                        cfg,
                    )

            # 2. Start any newly runnable streams.
            for video_url, cfg in runnable_configs.items():
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

    @staticmethod
    def _can_run_recognition(
        cfg: StreamConfig,
        now: datetime | None = None,
    ) -> bool:
        """Return whether this stream is currently eligible for recognition."""
        if not cfg.get('recognition_enabled', True):
            return False
        if Utils.is_expired(cfg.get('expire_date')):
            return False

        current_hour = (now or datetime.now()).hour
        work_start_hour = int(cfg.get('work_start_hour', 7))
        work_end_hour = int(cfg.get('work_end_hour', 18))
        return work_start_hour <= current_hour < work_end_hour

    def start_process(self, cfg: StreamConfig) -> Process:
        """
        Launch a new child process to handle stream detection.

        Args:
            cfg (StreamConfig): Configuration for the stream.

        Returns:
            Process: The new multiprocessing.Process object.
        """
        self._ensure_yolo_worker()
        yolo_request_queue, yolo_result_store = self._yolo_worker_slot(cfg)
        p = Process(
            target=process_single_stream,
            args=(cfg, yolo_request_queue, yolo_result_store),
        )
        p.start()
        return p

    def _ensure_yolo_worker(self) -> bool:
        """Start the shared YOLO worker pool when enabled."""
        if os.getenv(
            'YOLO_WORKER_ENABLED',
            'true',
        ).strip().lower() not in {'1', 'true', 'yes', 'on'}:
            if self.yolo_worker_processes:
                self._stop_yolo_worker()
                return True
            return False
        worker_count = max(1, int(os.getenv('YOLO_WORKER_COUNT', '2')))
        if (
            len(self.yolo_worker_processes) == worker_count
            and all(p.is_alive() for p in self.yolo_worker_processes)
        ):
            return False
        self._stop_yolo_worker()
        if self.yolo_manager is None:
            self.yolo_manager = multiprocessing.Manager()
        devices = _csv_env('YOLO_WORKER_DEVICES', 'cuda:0')
        for worker_index in range(worker_count):
            request_queue = self.yolo_manager.Queue(
                maxsize=int(os.getenv('YOLO_WORKER_QUEUE_SIZE', '64')),
            )
            result_store = self.yolo_manager.dict()
            device = devices[worker_index % len(devices)]
            worker = YoloWorker(
                request_queue,
                cast(ResultStore, result_store),
                device,
            )
            process = Process(
                target=worker.run,
                daemon=True,
            )
            process.start()
            self.yolo_request_queues.append(request_queue)
            self.yolo_result_stores.append(result_store)
            self.yolo_worker_processes.append(process)
            self.logger.info(
                '[YOLO-Worker] process %s started on %s',
                worker_index,
                device,
            )
        return True

    async def _restart_stream_process(
        self,
        video_url: str,
        proc_info: dict[str, Any],
        cfg: StreamConfig,
    ) -> None:
        """Restart one stream process and refresh its process metadata."""
        reason = self._restart_reason(proc_info, cfg)
        self.logger.info(f"Restart stream {video_url} ({reason})")
        self.stop_process(proc_info['process'])
        await self._delete_stream_redis_keys(proc_info['cfg'])
        new_proc = self.start_process(cfg)
        self.running_processes[video_url] = {
            'process': new_proc,
            'updated_at': cfg['updated_at'],
            'cfg': cfg,
        }

    @staticmethod
    def _stream_needs_restart(
        proc_info: dict[str, Any],
        cfg: StreamConfig,
        workers_restarted: bool,
    ) -> bool:
        """Return True when an existing stream should be relaunched."""
        proc = proc_info['process']
        return (
            workers_restarted
            or cfg['updated_at'] != proc_info['updated_at']
            or not proc.is_alive()
        )

    @staticmethod
    def _restart_reason(
        proc_info: dict[str, Any],
        cfg: StreamConfig,
    ) -> str:
        """Build a compact restart reason for logs."""
        if cfg['updated_at'] != proc_info['updated_at']:
            return 'updated_at changed'
        if not proc_info['process'].is_alive():
            return 'process exited'
        return 'YOLO worker restarted'

    def _yolo_worker_slot(
        self,
        cfg: StreamConfig,
    ) -> tuple[object | None, object | None]:
        """Return the worker queue/result store assigned to this camera."""
        if not self.yolo_request_queues or not self.yolo_result_stores:
            return None, None
        camera_key = f"{cfg.get('site', '')}|{cfg.get('stream_name', '')}"
        digest = hashlib.blake2b(camera_key.encode(), digest_size=4).digest()
        index = int.from_bytes(digest, 'big') % len(self.yolo_request_queues)
        return self.yolo_request_queues[index], self.yolo_result_stores[index]

    async def _delete_stream_redis_keys(self, cfg: StreamConfig) -> None:
        """Delete compact live metadata for one configured camera."""
        await delete_stream_live_metadata(cfg)

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

        self._stop_yolo_worker()
        await self._cancel_config_reload_task()

        # Close database connection pool
        if self.db_pool:
            await self._close_config_listener(self.db_pool)
            await self.db_pool.close()
            self.db_pool = None

    async def _reset_db_pool(self) -> None:
        """Close the current database pool so the next poll reconnects."""
        pool = self.db_pool
        self.db_pool = None
        await self._cancel_config_reload_task()
        if pool is not None:
            await self._close_config_listener(pool)
            with suppress(Exception):
                await pool.close()

    def _stop_yolo_worker(self) -> None:
        """Stop the shared YOLO worker pool and manager."""
        for request_queue in self.yolo_request_queues:
            try:
                request_queue.put(YOLO_WORKER_STOP_MESSAGE)
            except Exception as e:
                self.logger.error(f"Error signalling YOLO worker: {e}")
        for process in self.yolo_worker_processes:
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join()
        if self.yolo_manager is not None:
            self.yolo_manager.shutdown()
            self.yolo_manager = None
        self.yolo_request_queues.clear()
        self.yolo_result_stores.clear()
        self.yolo_worker_processes.clear()

    async def run(self) -> None:
        """
        Start the application loop that continuously checks the stream configs.
        """
        try:
            await self._ensure_config_listener()
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
        app = MainApp(poll_interval=args.poll)
        runnable_configs = [
            cfg for cfg in configs if app._can_run_recognition(cfg)
        ]
        if runnable_configs:
            app._ensure_yolo_worker()
        # Start a process for each currently runnable config.
        procs = []
        for cfg in runnable_configs:
            yolo_request_queue, yolo_result_store = app._yolo_worker_slot(cfg)
            proc = Process(
                target=process_single_stream,
                args=(cfg, yolo_request_queue, yolo_result_store),
            )
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
            await app.cleanup_resources()
    else:
        app = MainApp(poll_interval=args.poll)
        try:
            await app.run()
        except KeyboardInterrupt:
            print('\n[INFO] KeyboardInterrupt, shutting down...')
        finally:
            await app.cleanup_resources()


def _csv_env(name: str, default: str) -> list[str]:
    """Read a comma-separated environment setting.

    Args:
        name: Environment variable name.
        default: Default comma-separated value.

    Returns:
        Normalised non-empty entries.
    """
    value = os.getenv(name, default)
    items = [item.strip() for item in value.split(',') if item.strip()]
    return items or [default]


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    asyncio.run(main())

"""
python main.py --poll 15

uv run python main.py --poll 15
"""
