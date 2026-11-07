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

from asyncpg import create_pool  # type: ignore[import-untyped]
from asyncpg.pool import Pool  # type: ignore[import-untyped]
from dotenv import load_dotenv
from sqlalchemy.engine.url import make_url

from examples.YOLO_server_api.model_registry import revision_for
from src.auth_tokens import TokenManager
from src.configuration_watcher import ConfigurationWatcher
from src.monitor_logger import LoggerConfig
from src.redis_client import RedisManager
from src.runtime_utils import is_expired
from src.stream_identity import stream_identity
from src.stream_processor import delete_stream_live_metadata
from src.stream_processor import process_single_stream
from src.stream_processor import StreamConfig
from src.violation_delivery import OutboxDispatcher
from src.worker_supervisor import StreamProcessState
from src.worker_supervisor import WorkerSupervisor

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
        self.running_processes: dict[str, StreamProcessState] = {}
        self._reload_lock = asyncio.Lock()
        self.lock = asyncio.Lock()  # Prevent overlapping reloads
        self.db_pool: Pool | None = None  # PostgreSQL async connection pool
        self._last_config_summary: tuple[int, int, int] | None = None
        self.workers = WorkerSupervisor(self.logger)
        self.delivery = OutboxDispatcher()
        self.watcher = ConfigurationWatcher(
            self._listener_pool,
            self._reload_from_notification,
            self.logger,
        )

    async def _reload_from_notification(self) -> None:
        await self.reload_configurations()

    async def _listener_pool(self) -> Pool | None:
        await self._ensure_db_pool()
        return self.db_pool

    def start_process(self, cfg: StreamConfig) -> Process:
        return self.workers.start_process(cfg)

    def stop_process(self, process: Process) -> None:
        self.workers.stop_process(process)

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
            if url.get_backend_name() != 'postgresql':
                raise RuntimeError(
                    'DATABASE_URL must use PostgreSQL',
                )
            port = url.port
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
                '[database] Stream configuration database connection ' 'ready',
            )

    async def fetch_stream_configs(self) -> list[StreamConfig]:
        """Query the database for current stream configurations.

        Returns:
            list[StreamConfig]: All configured stream records.
        """
        await self._ensure_db_pool()
        if self.db_pool is None:
            raise RuntimeError(
                'Database pool is not initialized. Check DATABASE_URL '
                'and DB connectivity.',
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
               sc.detect_machinery_close_to_pole,
               sc.site_id,
               catalog.definition::text AS model_definition
        FROM stream_configs sc
        JOIN sites s ON sc.site_id = s.id
        LEFT JOIN detection_model_catalog catalog
            ON catalog.model_key = sc.model_key
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
                site_id,
            ) = row[:16]

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
                    site_id=int(site_id),
                    video_url=video_url,
                    updated_at=updated_at.isoformat(),
                    model_key=model_key,
                    model_revision=(
                        revision_for(json.loads(row[16]))
                        if len(row) > 16 and row[16]
                        else ''
                    ),
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
                await self.watcher.start()
            except Exception as e:
                self.logger.exception(f"[poll] Reload error: {e}")
            await asyncio.sleep(self._next_configuration_reload_delay())

    def _next_configuration_reload_delay(self) -> float:
        """Return the earlier of the health interval and next stream expiry."""
        delay = float(self.poll_interval)
        now = datetime.now(timezone.utc)
        for process_info in self.running_processes.values():
            raw_expiry = process_info.cfg.get('expire_date')
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
        async with self._reload_lock:
            configs = await self.fetch_stream_configs()
            active = {
                self._stream_process_key(cfg): cfg
                for cfg in configs
                if self._can_run_stream(cfg)
            }
            if active:
                replaced = await self.workers.reconcile(list(active.values()))
            else:
                await self.workers.close_async()
                replaced = False
            stopping = {
                key: state
                for key, state in self.running_processes.items()
                if key not in active
                or self._stream_needs_restart(
                    state,
                    active[key],
                    replaced and key in self.workers.restarted_cameras,
                )
            }
            await self.workers.reaper.reap(
                [state.process for state in stopping.values()],
            )
            await asyncio.gather(
                *(
                    self._delete_stream_redis_keys(state.cfg)
                    for state in stopping.values()
                ),
            )
            async with self.lock:
                for key in stopping:
                    self.running_processes.pop(key, None)
                for key, cfg in active.items():
                    if key not in self.running_processes:
                        self.running_processes[key] = StreamProcessState(
                            process=self.start_process(cfg),
                            updated_at=cfg['updated_at'],
                            cfg=cfg,
                        )
                self._log_config_summary(len(configs), len(active))

    @staticmethod
    def _stream_process_key(cfg: StreamConfig) -> str:
        """Return the durable process key for a configured camera.

        The fallback is only for programmatic callers predating ``stream_id``;
        records loaded from PostgreSQL always use the immutable primary key.
        """
        return stream_identity(cfg)

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

    async def _restart_stream_process(
        self,
        stream_key: str,
        proc_info: StreamProcessState,
        cfg: StreamConfig,
    ) -> None:
        """Restart one stream process and refresh its process metadata."""
        reason = self._restart_reason(proc_info, cfg)
        self.logger.info(
            'Restart stream %s (%s)',
            self._stream_log_label(cfg),
            reason,
        )
        await self.workers.reaper.reap([proc_info.process])
        await self._delete_stream_redis_keys(proc_info.cfg)
        new_proc = self.start_process(cfg)
        self.running_processes[stream_key] = StreamProcessState(
            process=new_proc,
            updated_at=cfg['updated_at'],
            cfg=cfg,
        )

    @staticmethod
    def _stream_needs_restart(
        proc_info: StreamProcessState,
        cfg: StreamConfig,
        worker_broker_replaced: bool,
    ) -> bool:
        """Return True when an existing stream should be relaunched."""
        proc = proc_info.process
        return (
            worker_broker_replaced
            or cfg['updated_at'] != proc_info.updated_at
            or not proc.is_alive()
        )

    @staticmethod
    def _restart_reason(
        proc_info: StreamProcessState,
        cfg: StreamConfig,
    ) -> str:
        """Build a compact restart reason for logs."""
        if cfg['updated_at'] != proc_info.updated_at:
            return 'updated_at changed'
        if not proc_info.process.is_alive():
            return 'process exited'
        return 'YOLO worker restarted'

    async def _delete_stream_redis_keys(self, cfg: StreamConfig) -> None:
        """Delete compact live metadata for one configured camera."""
        await delete_stream_live_metadata(cfg)

    async def cleanup_resources(self) -> None:
        """Clean up all resources."""
        await self.watcher.cancel_reload()
        async with self._reload_lock:
            await self.workers.reaper.reap(
                [info.process for info in self.running_processes.values()],
            )
            self.running_processes.clear()
            await self.workers.close_async()
        await self.delivery.close()

        # Close database connection pool
        if self.db_pool:
            await self.watcher.close()
            await self.db_pool.close()
            self.db_pool = None

    async def _reset_db_pool(self) -> None:
        """Close the current database pool so the next poll reconnects."""
        pool = self.db_pool
        self.db_pool = None
        await self.watcher.cancel_reload()
        if pool is not None:
            await self.watcher.close()
            with suppress(Exception):
                await pool.close()

    async def run(self) -> None:
        """Start the application loop that continuously checks the stream
        configs."""
        try:
            self.logger.info(
                '[startup] Stream supervisor started; PostgreSQL LISTEN '
                'enabled with %s-second fallback',
                self.poll_interval,
            )
            self.delivery.start()
            await self.watcher.start()
            await self.poll_and_reload()
        except KeyboardInterrupt:
            self.logger.info('Received keyboard interrupt, shutting down...')
        except Exception as e:
            self.logger.error(f"Unexpected error in main loop: {e}")
        finally:
            await self.cleanup_resources()


async def validate_runtime_dependencies() -> None:
    """Fail before starting workers when shared dependencies are
    unavailable.
    """
    # Check shared dependencies once before spawning camera/GPU processes.
    redis_manager = RedisManager()
    try:
        await redis_manager.redis.ping()
    finally:
        await redis_manager.redis.aclose()
    token_manager = TokenManager()
    try:
        await token_manager.get_valid_token()
    finally:
        await token_manager.close()


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

    await validate_runtime_dependencies()

    if args.config:
        # Load configs from JSON file
        with open(args.config) as f:
            configs = json.load(f)
        app = MainApp(poll_interval=args.poll)
        active_configs = [cfg for cfg in configs if app._can_run_stream(cfg)]
        procs = []
        try:
            app.delivery.start()
            if active_configs:
                app.workers.ensure(active_configs)
            for cfg in active_configs:
                procs.append(app.start_process(cfg))
            while any(p.is_alive() for p in procs):
                for p in procs:
                    await asyncio.to_thread(p.join, timeout=1)
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


def _handle_sigterm(_signum: int, _frame: FrameType | None) -> None:
    """Route SIGTERM through the normal graceful shutdown path."""
    raise KeyboardInterrupt


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, _handle_sigterm)
    multiprocessing.set_start_method('spawn', force=True)
    asyncio.run(main())

"""Python main.py --poll 15.

uv run python main.py --poll 15
"""

__all__ = [
    'MainApp',
    'StreamConfig',
    'process_single_stream',
]
