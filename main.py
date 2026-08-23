from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing
import os
import signal
from contextlib import suppress
from datetime import datetime
from datetime import timezone
from multiprocessing import Process
from types import FrameType
from typing import Any

from asyncpg import create_pool  # type: ignore[import-untyped]
from asyncpg.pool import Pool  # type: ignore[import-untyped]
from dotenv import load_dotenv
from sqlalchemy.engine.url import make_url

from src.monitor_logger import LoggerConfig
from src.runtime_utils import is_expired
from src.stream_processor import delete_stream_live_metadata
from src.stream_processor import process_single_stream
from src.stream_processor import StreamConfig
from src.yolo_worker import YOLO_WORKER_STOP_MESSAGE
from src.yolo_worker import YoloWorker

load_dotenv(override=True)


class MainApp:
    """
    Core application responsible for:
        - Reacting to PostgreSQL configuration notifications with a bounded
          polling fallback
        - Dynamically spawning/stopping child processes for each video stream
        - Cleaning up expired or modified configurations
    """

    def __init__(self, poll_interval: int = 300) -> None:
        """Initialise the application.

        Args:
            poll_interval (int): Maximum seconds between fallback database
                refreshes when PostgreSQL notifications are unavailable.
        """
        self.poll_interval = max(1, poll_interval)
        self.logger = LoggerConfig().get_logger()
        # Stable stream identity → process info.  RTSP URLs are sources, not
        # identifiers: multiple configured cameras may deliberately share one.
        self.running_processes: dict[str, dict] = {}
        self.lock = asyncio.Lock()  # Prevent overlapping reloads
        self.db_pool: Pool | None = None  # PostgreSQL async connection pool
        self._config_listener_connection: Any | None = None
        self._config_reload_task: asyncio.Task[None] | None = None
        self._config_reload_requested = False
        self._last_config_summary: tuple[int, int, int] | None = None

        self.yolo_request_queues: list[Any] = []
        self.yolo_result_queues: dict[str, Any] = {}
        self.yolo_worker_processes: list[Process] = []
        self.yolo_worker_slots: dict[str, int] = {}
        self.yolo_worker_camera_slots: dict[str, int] = {}
        self.yolo_worker_topology_signature: str | None = None
        self.yolo_worker_startup_lock: Any | None = None

    async def _ensure_db_pool(self) -> None:
        """Ensure a connection pool to the database is established before
        querying."""
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
            self.logger.info(
                '[database] Connecting to stream configuration database '
                '%s:%s/%s',
                url.host or 'localhost',
                port or 5432,
                url.database or '',
            )
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
            self.logger.info(
                '[database] Stream configuration database connection ready',
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
        self.logger.info(
            '[database] Listening for stream configuration changes',
        )

    def _on_stream_config_changed(
        self,
        _connection: object,
        _pid: int,
        _channel: str,
        _payload: str,
    ) -> None:
        """Schedule a reload when PostgreSQL signals a config change."""
        self._config_reload_requested = True
        if self._config_reload_task is None or self._config_reload_task.done():
            reload_task = asyncio.create_task(
                self._reload_configurations_from_notification(),
            )
            reload_task.add_done_callback(
                self._log_config_reload_failure,
            )
            self._config_reload_task = reload_task

    async def _reload_configurations_from_notification(self) -> None:
        """Coalesce notifications that arrive during one configuration read."""
        while self._config_reload_requested:
            self._config_reload_requested = False
            await self.reload_configurations()

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
        """Query the database for current stream configurations.

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
        SELECT sc.id               AS stream_id,
               sc.video_url,
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
                stream_id,
                video_url,
                updated_at,
                model_key,
                site,
                stream_name,
                recognition_enabled,
                expire_date,
                work_start,
                work_end,
                vest_helmet,
                near_vehicle,
                in_area,
                in_pole_area,
                machine_close_pole,
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
                    stream_id=int(stream_id),
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
        """Run the low-frequency health fallback and stream-expiry timer.

        PostgreSQL ``LISTEN`` drives ordinary changes.  The fallback recovers
        from a lost listener and wakes at the nearest configured expiry even
        when no database row changes.
        """
        while True:
            try:
                await self.reload_configurations()
            except TimeoutError as e:
                self.logger.exception(f"[poll] Reload timeout: {e}")
                await self._reset_db_pool()
                await self._ensure_config_listener()
            except Exception as e:
                self.logger.exception(f"[poll] Reload error: {e}")
            await asyncio.sleep(self._next_configuration_reload_delay())

    def _next_configuration_reload_delay(self) -> float:
        """Return the earlier of the health interval and next stream expiry."""
        delay = float(self.poll_interval)
        now = datetime.now(timezone.utc)
        for process_info in self.running_processes.values():
            raw_expiry = process_info['cfg'].get('expire_date')
            if not raw_expiry:
                continue
            try:
                expiry = datetime.fromisoformat(str(raw_expiry))
            except ValueError:
                continue
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=timezone.utc)
            seconds_until_expiry = (expiry - now).total_seconds()
            if seconds_until_expiry > 0:
                delay = min(delay, seconds_until_expiry)
        return max(0.1, delay)

    async def reload_configurations(self) -> None:
        """
        Main configuration reload logic:
            - Stops disabled, expired, or deleted stream processes
            - Restarts modified streams (based on updated_at)
            - Starts newly added streams not yet tracked
        """

        async with self.lock:
            configs = await self.fetch_stream_configs()
            cfg_map = {
                self._stream_process_key(config): config for config in configs
            }
            active_configs = {
                stream_key: cfg
                for stream_key, cfg in cfg_map.items()
                if self._can_run_stream(cfg)
            }
            if active_configs:
                worker_broker_replaced = self._ensure_yolo_worker(
                    list(active_configs.values()),
                )
            else:
                worker_broker_replaced = False
                if self.yolo_worker_processes:
                    self._stop_yolo_worker()

            # 1. Stop streams that are removed, disabled, or expired.
            for stream_key in list(self.running_processes.keys()):
                proc_info = self.running_processes[stream_key]
                if stream_key not in active_configs:
                    self.logger.info(
                        'Stop stream %s',
                        self._stream_log_label(proc_info['cfg']),
                    )
                    self.stop_process(proc_info['process'])

                    await self._delete_stream_redis_keys(proc_info['cfg'])

                    del self.running_processes[stream_key]
                    continue

                cfg = cfg_map[stream_key]
                if self._stream_needs_restart(
                    proc_info,
                    cfg,
                    worker_broker_replaced,
                ):
                    await self._restart_stream_process(
                        stream_key,
                        proc_info,
                        cfg,
                    )

            # 2. Start every enabled, non-expired recognition stream.
            for stream_key, cfg in active_configs.items():
                if stream_key not in self.running_processes:
                    self.logger.info(
                        'Launch new stream %s',
                        self._stream_log_label(cfg),
                    )
                    proc = self.start_process(cfg)
                    self.running_processes[stream_key] = {
                        'process': proc,
                        'updated_at': cfg['updated_at'],
                        'cfg': cfg,
                    }
            self._log_config_summary(len(configs), len(active_configs))

    @staticmethod
    def _stream_process_key(cfg: StreamConfig) -> str:
        """Return the durable process key for a configured camera.

        The fallback is only for programmatic callers predating ``stream_id``;
        records loaded from PostgreSQL always use the immutable primary key.
        """
        stream_id = cfg.get('stream_id')
        if stream_id is not None:
            return f"id:{stream_id}"
        return str(cfg['video_url'])

    @staticmethod
    def _stream_log_label(cfg: StreamConfig) -> str:
        """Return a useful, non-secret stream label for lifecycle logs."""
        return (
            f"{cfg['site']}/{cfg['stream_name']}"
            f" ({MainApp._stream_process_key(cfg)})"
        )

    def _log_config_summary(
        self,
        configured_count: int,
        active_count: int,
        active_process_count: int | None = None,
    ) -> None:
        """Log configuration state changes without flooding each poll cycle."""
        summary = (
            configured_count,
            active_count,
            (
                len(self.running_processes)
                if active_process_count is None
                else active_process_count
            ),
        )
        if summary == self._last_config_summary:
            return
        self._last_config_summary = summary
        self.logger.info(
            '[config] Loaded %s configs; %s recognition streams enabled; '
            '%s stream processes active',
            *summary,
        )

    @staticmethod
    def _can_run_stream(
        cfg: StreamConfig,
    ) -> bool:
        """Return whether capture and MediaMTX publishing should stay active.

        Working hours are enforced inside the stream processor only for
        violation records and notifications. Live capture and playback remain
        available whenever recognition is enabled and the stream is valid.
        """
        if not cfg.get('recognition_enabled', True):
            return False
        return not is_expired(cfg.get('expire_date'))

    def start_process(self, cfg: StreamConfig) -> Process:
        """Launch a new child process to handle stream detection.

        Args:
            cfg (StreamConfig): Configuration for the stream.

        Returns:
            Process: The new multiprocessing.Process object.
        """
        if not self.yolo_worker_processes:
            self._ensure_yolo_worker([cfg])
        yolo_request_queue, yolo_result_queue = self._yolo_worker_slot(cfg)
        p = Process(
            target=process_single_stream,
            args=(cfg, yolo_request_queue, yolo_result_queue),
        )
        p.start()
        return p

    def _ensure_yolo_worker(
        self,
        configs: list[StreamConfig] | None = None,
    ) -> bool:
        """Maintain native shared-memory YOLO workers for camera shards.

        A dead worker is replaced with its existing queues. A topology change
        rebuilds the queue set before restarting its associated stream clients.
        """
        (
            worker_count,
            worker_slots,
            topology_signature,
        ) = self._yolo_worker_topology(configs)
        if worker_count == 0:
            return False
        devices = _csv_env('YOLO_WORKER_DEVICES', 'cuda:0')
        if self._has_yolo_worker_broker(
            worker_count,
            topology_signature,
        ):
            self._ensure_yolo_result_queues(configs)
            self._restart_dead_yolo_workers(devices)
            return False
        self._stop_yolo_worker()
        self._ensure_yolo_result_queues(configs)
        self.yolo_worker_slots = worker_slots
        self.yolo_worker_camera_slots = {
            f"{cfg['site']}|{cfg['stream_name']}": worker_slots[
                str(cfg['video_url'])
            ]
            for cfg in configs or []
        }
        self.yolo_worker_startup_lock = multiprocessing.Lock()
        for worker_index in range(worker_count):
            request_queue: Any = multiprocessing.Queue(
                maxsize=int(os.getenv('YOLO_WORKER_QUEUE_SIZE', '64')),
            )
            self.yolo_request_queues.append(request_queue)
            self.yolo_worker_processes.append(
                self._start_yolo_worker(
                    worker_index,
                    request_queue,
                    devices[worker_index % len(devices)],
                ),
            )
        self.yolo_worker_topology_signature = topology_signature
        if topology_signature is not None:
            self.logger.info(
                '[YOLO-Worker] camera-sharded topology started: %s workers',
                worker_count,
            )
        return True

    def _yolo_worker_topology(
        self,
        configs: list[StreamConfig] | None,
    ) -> tuple[int, dict[str, int], str | None]:
        """Build stable same-model camera shards for TensorRT workers."""
        if not configs:
            return 0, {}, None
        default_cameras_per_engine = max(
            1,
            _positive_int_env(
                'YOLO_WORKER_CAMERAS_PER_ENGINE',
                default=3,
            ),
        )
        cameras_per_engine_by_model = _model_camera_limits_env()

        grouped_configs: dict[str, list[StreamConfig]] = {}
        for cfg in configs:
            grouped_configs.setdefault(str(cfg['model_key']), []).append(cfg)

        slots: dict[str, int] = {}
        signature_shards: list[dict[str, object]] = []
        worker_index = 0
        for model_key, model_configs in sorted(grouped_configs.items()):
            cameras_per_engine = cameras_per_engine_by_model.get(
                model_key.lower(),
                default_cameras_per_engine,
            )
            sorted_configs = sorted(
                model_configs,
                key=lambda cfg: str(cfg['video_url']),
            )
            for start in range(0, len(sorted_configs), cameras_per_engine):
                shard = sorted_configs[start: start + cameras_per_engine]
                source_urls = [str(cfg['video_url']) for cfg in shard]
                camera_ids = [
                    f"{cfg['site']}|{cfg['stream_name']}" for cfg in shard
                ]
                for source_url in source_urls:
                    slots[source_url] = worker_index
                signature_shards.append(
                    {
                        'model_key': model_key,
                        'cameras_per_engine': cameras_per_engine,
                        'sources': source_urls,
                        'cameras': camera_ids,
                    },
                )
                worker_index += 1

        signature = json.dumps(
            {
                'default_cameras_per_engine': default_cameras_per_engine,
                'cameras_per_engine_by_model': {
                    model_key: cameras_per_engine_by_model.get(
                        model_key.lower(),
                        default_cameras_per_engine,
                    )
                    for model_key in sorted(grouped_configs)
                },
                'shards': signature_shards,
            },
            sort_keys=True,
            separators=(',', ':'),
        )
        return worker_index, slots, signature

    def _has_yolo_worker_broker(
        self,
        worker_count: int,
        topology_signature: str | None = None,
    ) -> bool:
        """Return whether existing worker IPC can be reused safely."""
        return (
            len(self.yolo_request_queues) == worker_count
            and len(self.yolo_worker_processes) == worker_count
            and self.yolo_worker_topology_signature == topology_signature
        )

    def _restart_dead_yolo_workers(self, devices: list[str]) -> None:
        """Replace dead workers while preserving fixed native IPC queues."""
        for worker_index, process in enumerate(self.yolo_worker_processes):
            if process.is_alive():
                continue
            with suppress(Exception):
                process.join()
            request_queue = self.yolo_request_queues[worker_index]
            device = devices[worker_index % len(devices)]
            self.yolo_worker_processes[worker_index] = self._start_yolo_worker(
                worker_index,
                request_queue,
                device,
            )
            self.logger.warning(
                '[YOLO-Worker] process %s exited; restarted with existing '
                'IPC broker on %s',
                worker_index,
                device,
            )

    def _start_yolo_worker(
        self,
        worker_index: int,
        request_queue: Any,
        device: str,
    ) -> Process:
        """Start one worker against fixed native request/result queues."""
        result_queues = {
            camera_id: result_queue
            for camera_id, result_queue in self.yolo_result_queues.items()
            if self.yolo_worker_camera_slots.get(camera_id) == worker_index
        }
        worker = YoloWorker(
            request_queue,
            device,
            result_queues=result_queues,
            startup_lock=self.yolo_worker_startup_lock,
        )
        process = Process(target=worker.run, daemon=True)
        process.start()
        self.logger.info(
            '[YOLO-Worker] process %s started on %s',
            worker_index,
            device,
        )
        return process

    async def _restart_stream_process(
        self,
        stream_key: str,
        proc_info: dict[str, Any],
        cfg: StreamConfig,
    ) -> None:
        """Restart one stream process and refresh its process metadata."""
        reason = self._restart_reason(proc_info, cfg)
        self.logger.info(
            'Restart stream %s (%s)',
            self._stream_log_label(cfg),
            reason,
        )
        self.stop_process(proc_info['process'])
        await self._delete_stream_redis_keys(proc_info['cfg'])
        new_proc = self.start_process(cfg)
        self.running_processes[stream_key] = {
            'process': new_proc,
            'updated_at': cfg['updated_at'],
            'cfg': cfg,
        }

    @staticmethod
    def _stream_needs_restart(
        proc_info: dict[str, Any],
        cfg: StreamConfig,
        worker_broker_replaced: bool,
    ) -> bool:
        """Return True when an existing stream should be relaunched."""
        proc = proc_info['process']
        return (
            worker_broker_replaced
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
        """Return the worker and dedicated result queue assigned to a camera.

        Sharded mode pins up to the configured number of same-model cameras to
        one engine worker.
        """
        if not self.yolo_request_queues:
            return None, None
        result_queue = self._yolo_worker_result_queue(cfg)
        if result_queue is None:
            return None, None
        source_url = str(cfg['video_url'])
        slot_index = self.yolo_worker_slots.get(source_url)
        if slot_index is not None:
            return (
                self.yolo_request_queues[slot_index],
                result_queue,
            )
        return None, None

    def _ensure_yolo_result_queues(
        self,
        configs: list[StreamConfig] | None,
    ) -> None:
        """Pre-create one bounded result queue for each active camera."""
        if configs is None:
            return
        for cfg in configs:
            self._yolo_worker_result_queue(cfg)

    def _yolo_worker_result_queue(
        self,
        cfg: StreamConfig,
    ) -> Any | None:
        """Return a stable response queue that only one camera consumes."""
        camera_key = f"{cfg['site']}|{cfg['stream_name']}"
        result_queue = self.yolo_result_queues.get(camera_key)
        if result_queue is None:
            result_queue = multiprocessing.Queue(
                maxsize=max(
                    1,
                    int(os.getenv('YOLO_WORKER_RESULT_QUEUE_SIZE', '8')),
                ),
            )
            self.yolo_result_queues[camera_key] = result_queue
        return result_queue

    async def _delete_stream_redis_keys(self, cfg: StreamConfig) -> None:
        """Delete compact live metadata for one configured camera."""
        await delete_stream_live_metadata(cfg)

    def stop_process(self, proc: Process) -> None:
        """Gracefully terminate a child process.

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
        """Clean up all resources."""
        # Stop all processes
        for info in self.running_processes.values():
            self.stop_process(info['process'])
        self.running_processes.clear()

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
        """Stop the shared YOLO worker pool and close native IPC queues."""
        for request_queue in self.yolo_request_queues:
            try:
                request_queue.put(YOLO_WORKER_STOP_MESSAGE, block=False)
            except Exception as e:
                self.logger.error(f"Error signalling YOLO worker: {e}")
        for process in self.yolo_worker_processes:
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join()
        for worker_queue in [
            *self.yolo_request_queues,
            *self.yolo_result_queues.values(),
        ]:
            with suppress(Exception):
                worker_queue.close()
            with suppress(Exception):
                worker_queue.join_thread()
        self.yolo_request_queues.clear()
        self.yolo_result_queues.clear()
        self.yolo_worker_processes.clear()
        self.yolo_worker_slots.clear()
        self.yolo_worker_camera_slots.clear()
        self.yolo_worker_topology_signature = None
        self.yolo_worker_startup_lock = None

    async def run(self) -> None:
        """Start the application loop that continuously checks the stream
        configs."""
        try:
            self.logger.info(
                '[startup] Stream supervisor started; PostgreSQL LISTEN '
                'enabled with %s-second fallback',
                self.poll_interval,
            )
            await self._ensure_config_listener()
            await self.poll_and_reload()
        except KeyboardInterrupt:
            self.logger.info('Received keyboard interrupt, shutting down...')
        except Exception as e:
            self.logger.error(f"Unexpected error in main loop: {e}")
        finally:
            await self.cleanup_resources()


async def main() -> None:
    """Parse command-line arguments and run the MainApp."""
    parser = argparse.ArgumentParser(
        description='Hazard detection from DB configs or JSON file',
    )
    parser.add_argument(
        '--poll',
        type=int,
        default=300,
        help='LISTEN health fallback interval in seconds',
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to JSON config file for stream configs',
    )
    args = parser.parse_args()

    if args.config:
        # Load configs from JSON file
        with open(args.config, encoding='utf-8') as f:
            configs = json.load(f)
        app = MainApp(poll_interval=args.poll)
        active_configs = [cfg for cfg in configs if app._can_run_stream(cfg)]
        if active_configs:
            app._ensure_yolo_worker(active_configs)
            # Start a process for every enabled, non-expired config.
            procs = []
            for cfg in active_configs:
                yolo_request_queue, yolo_result_queue = app._yolo_worker_slot(
                    cfg,
                )
                proc = Process(
                    target=process_single_stream,
                    args=(cfg, yolo_request_queue, yolo_result_queue),
                )
                proc.start()
                procs.append(proc)
        else:
            procs = []
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
        await app.run()


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


def _positive_int_env(name: str, default: int = 0) -> int:
    """Read a non-negative integer environment setting."""
    try:
        return max(0, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _model_camera_limits_env() -> dict[str, int]:
    """Read positive per-model camera limits from a comma-separated setting.

    The value format is ``model_key=camera_count``. Invalid entries are ignored
    so a typo falls back to ``YOLO_WORKER_CAMERAS_PER_ENGINE``.
    """
    limits: dict[str, int] = {}
    raw_value = os.getenv('YOLO_WORKER_CAMERAS_PER_ENGINE_BY_MODEL', '')
    for entry in raw_value.split(','):
        model_key, separator, raw_limit = entry.partition('=')
        if not separator:
            continue
        normalized_model_key = model_key.strip().lower()
        try:
            camera_limit = int(raw_limit.strip())
        except ValueError:
            continue
        if normalized_model_key and camera_limit > 0:
            limits[normalized_model_key] = camera_limit
    return limits


def _handle_sigterm(_signum: int, _frame: FrameType | None) -> None:
    """Route SIGTERM through the normal graceful shutdown path."""
    raise KeyboardInterrupt


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, _handle_sigterm)
    multiprocessing.set_start_method('spawn', force=True)
    asyncio.run(main())
\
"""Python main.py --poll 15.

uv run python main.py --poll 15
"""
