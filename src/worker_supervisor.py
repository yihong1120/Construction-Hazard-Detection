from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing
from contextlib import suppress
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from multiprocessing import Process
from typing import Any

from src.process_reaper import ProcessReaper
from src.process_startup_lock import ProcessStartupLock
from src.settings.inference import WorkerPoolSettings
from src.stream_identity import stream_identity
from src.stream_processor import process_single_stream
from src.stream_processor import StreamConfig
from src.yolo_worker import YOLO_WORKER_STOP_MESSAGE
from src.yolo_worker import YoloWorker


@dataclass
class StreamProcessState:
    process: Process
    updated_at: str
    cfg: StreamConfig = field(default_factory=StreamConfig)


@dataclass(frozen=True)
class ShardSpec:
    slot_id: int
    model_key: str
    revision: str
    cameras: tuple[str, ...]
    capacity: int
    device: str

    @property
    def runtime_identity(self) -> tuple:
        return self.model_key, self.revision, self.cameras, self.device


@dataclass
class WorkerShard:
    spec: ShardSpec
    request_queue: Any
    result_queues: dict[str, Any]
    process: Process | None = None
    generation: int = 1


class WorkerSupervisor:
    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger
        self.reaper = ProcessReaper()
        self._transition_lock = asyncio.Lock()
        self.shards: dict[int, WorkerShard] = {}
        self._retired: list[WorkerShard] = []
        self._next_slot_id = 0
        # Published IPC views are derived from shard-owned resources.
        self.request_queues: list[Any] = []
        self.result_queues: dict[str, Any] = {}
        self.processes: list[Process] = []
        self.slots: dict[str, int] = {}
        self.camera_slots: dict[str, int] = {}
        self.topology_signature: str | None = None
        self.startup_lock: Any | None = None
        self.restarted_cameras: set[str] = set()
        self._generations: dict[int, int] = {}

    def _plan(
        self,
        configs: list[StreamConfig] | None,
    ) -> dict[int, ShardSpec]:
        pending = {stream_identity(cfg): cfg for cfg in configs or []}
        if len(pending) != len(configs or []):
            raise ValueError(
                'Duplicate stream identity in worker configuration',
            )
        self.settings = WorkerPoolSettings()
        default = self.settings.cameras_per_engine
        limits = self.settings.cameras_per_engine_by_model
        selected: dict[int, ShardSpec] = {}
        specification = _ShardLayout(self.settings).specification

        for slot, shard in sorted(self.shards.items()):
            old = shard.spec
            members = [
                key
                for key in old.cameras
                if key in pending
                and pending[key]['model_key'] == old.model_key
            ]
            if not members:
                continue
            revision = (
                old.revision
                if any(
                    pending[key].get('model_revision', '') == old.revision
                    for key in members
                )
                else pending[members[0]].get('model_revision', '')
            )
            members = [
                key
                for key in members
                if pending[key].get('model_revision', '') == revision
            ]
            members = members[: limits.get(old.model_key.lower(), default)]
            selected[slot] = specification(
                slot,
                old.model_key,
                revision,
                members,
            )
            for key in members:
                del pending[key]

        next_slot = self._next_slot_id
        for key, cfg in sorted(
            pending.items(),
            key=_stream_plan_order,
        ):
            model, revision = cfg['model_key'], cfg.get('model_revision', '')
            target = next(
                (
                    spec
                    for spec in selected.values()
                    if spec.model_key == model
                    and spec.revision == revision
                    and len(spec.cameras) < spec.capacity
                ),
                None,
            )
            if target is not None:
                selected[target.slot_id] = specification(
                    target.slot_id,
                    model,
                    revision,
                    [*target.cameras, key],
                )
            else:
                selected[next_slot] = specification(
                    next_slot,
                    model,
                    revision,
                    [key],
                )
                next_slot += 1
        return selected

    @staticmethod
    def _signature(plan: dict[int, ShardSpec]) -> str:
        return json.dumps(
            [asdict(spec) for _, spec in sorted(plan.items())],
            sort_keys=True,
        )

    def _publish_views(self) -> None:
        ordered = [shard for _, shard in sorted(self.shards.items())]
        self.request_queues = [shard.request_queue for shard in ordered]
        self.processes = [
            shard.process for shard in ordered if shard.process is not None
        ]
        self.result_queues = {
            key: queue
            for shard in ordered
            for key, queue in shard.result_queues.items()
        }
        self.slots = {
            key: index
            for index, shard in enumerate(ordered)
            for key in shard.spec.cameras
        }
        self.camera_slots = self.slots.copy()

    @staticmethod
    def _close_queues(shard: WorkerShard) -> None:
        for queue in [shard.request_queue, *shard.result_queues.values()]:
            with suppress(Exception):
                queue.cancel_join_thread()
                queue.close()

    def _retire(self, plan: dict[int, ShardSpec]) -> list[WorkerShard]:
        retired = []
        self.restarted_cameras = set()
        if not self.shards and self.processes:
            self.close()
        for slot, shard in list(self.shards.items()):
            selected = plan.get(slot)
            if (
                selected is not None
                and selected.runtime_identity == shard.spec.runtime_identity
            ):
                shard.spec = selected
                continue
            self.restarted_cameras.update(shard.spec.cameras)
            if selected is not None:
                self.restarted_cameras.update(selected.cameras)
            retired.append(shard)
            self._retired.append(shard)
            del self.shards[slot]
        self._publish_views()
        return retired

    def ensure(self, configs: list[StreamConfig] | None = None) -> bool:
        plan = self._plan(configs)
        for shard in self._retire(plan):
            if shard.process is not None:
                self.stop_process(shard.process)
            self._close_queues(shard)
            self._retired.remove(shard)
        return self._install(plan)

    async def reconcile(
        self,
        configs: list[StreamConfig] | None = None,
    ) -> bool:
        async with self._transition_lock:
            plan = self._plan(configs)
            self._retire(plan)
            try:
                await self.reaper.reap(
                    [
                        shard.process
                        for shard in self._retired
                        if shard.process is not None
                    ],
                )
            finally:
                for shard in list(self._retired):
                    if shard.process is None or not shard.process.is_alive():
                        self._close_queues(shard)
                        self._retired.remove(shard)
            return self._install(plan)

    async def close_async(self) -> None:
        async with self._transition_lock:
            retired = list(self._retired)
            processes = [
                *self.processes,
                *[
                    shard.process
                    for shard in retired
                    if shard.process is not None
                ],
            ]
            queues = [
                *self.request_queues,
                *self.result_queues.values(),
                *[
                    queue
                    for shard in retired
                    for queue in (
                        shard.request_queue,
                        *shard.result_queues.values(),
                    )
                ],
            ]
            for queue in self.request_queues:
                with suppress(Exception):
                    queue.put(YOLO_WORKER_STOP_MESSAGE, block=False)
            try:
                await self.reaper.reap(processes, graceful=True)
            finally:
                if not any(process.is_alive() for process in processes):
                    for queue in queues:
                        with suppress(Exception):
                            queue.cancel_join_thread()
                            queue.close()
                    self.shards.clear()
                    self._retired.clear()
                    self._publish_views()
                    self.topology_signature = None
                    if self.startup_lock is not None:
                        self.startup_lock.close()
                    self.startup_lock = None
                    self.restarted_cameras.clear()

    def _install(self, plan: dict[int, ShardSpec]) -> bool:
        if plan and self.startup_lock is None:
            self.startup_lock = ProcessStartupLock()
        for slot, spec in sorted(plan.items()):
            if slot in self.shards:
                continue
            self.restarted_cameras.update(spec.cameras)
            self.shards[slot] = self._new_shard(spec)
            self._next_slot_id = max(self._next_slot_id, slot + 1)
        self._publish_views()
        for index, (_, shard) in enumerate(sorted(self.shards.items())):
            if shard.process is None:
                shard.process = self.start_worker(
                    index,
                    shard.request_queue,
                    shard.spec.device,
                )
        self._publish_views()
        self.restart_dead()
        self.topology_signature = self._signature(plan) if plan else None
        if not plan:
            if self.startup_lock is not None:
                self.startup_lock.close()
            self.startup_lock = None
        return bool(self.restarted_cameras)

    def topology(
        self,
        configs: list[StreamConfig] | None,
    ) -> tuple[int, dict[str, int], str | None]:
        plan = self._plan(configs)
        slots = {
            camera: index
            for index, (_, spec) in enumerate(sorted(plan.items()))
            for camera in spec.cameras
        }
        return len(plan), slots, self._signature(plan) if plan else None

    def has_broker(
        self,
        worker_count: int,
        topology_signature: str | None = None,
    ) -> bool:
        """Return whether existing worker IPC can be reused safely."""
        return (
            len(self.request_queues) == worker_count
            and len(self.processes) == worker_count
            and (
                topology_signature is None
                or self.topology_signature == topology_signature
            )
        )

    def restart_dead(self) -> None:
        replacements = []
        for index, (_, shard) in enumerate(sorted(self.shards.items())):
            process = shard.process
            if process is None or process.is_alive():
                continue
            with suppress(Exception):
                process.join(timeout=0)
            replacement = self._new_shard(shard.spec)
            self.shards[shard.spec.slot_id] = replacement
            self.restarted_cameras.update(shard.spec.cameras)
            self._close_queues(shard)
            replacements.append((index, replacement))
            self.logger.warning(
                'YOLO shard %s replacing IPC generation %s with %s',
                shard.spec.slot_id,
                shard.generation,
                replacement.generation,
            )
        self._publish_views()
        for index, shard in replacements:
            shard.process = self.start_worker(
                index,
                shard.request_queue,
                shard.spec.device,
            )
        self._publish_views()

    def _new_shard(self, spec: ShardSpec) -> WorkerShard:
        request: Any = multiprocessing.Queue(maxsize=self.settings.queue_size)
        results: dict[str, Any] = {}
        try:
            for key in spec.cameras:
                results[key] = multiprocessing.Queue(
                    maxsize=self.settings.result_queue_size,
                )
        except BaseException:
            self._close_queues(WorkerShard(spec, request, results))
            raise
        previous = self.shards.get(spec.slot_id)
        generation = (
            max(
                self._generations.get(spec.slot_id, 0),
                previous.generation if previous is not None else 0,
            )
            + 1
        )
        self._generations[spec.slot_id] = generation
        return WorkerShard(spec, request, results, generation=generation)

    def start_worker(
        self,
        worker_index: int,
        request_queue: Any,
        device: str,
    ) -> Process:
        """Start one worker against fixed native request/result queues."""
        result_queues = {
            camera_id: result_queue
            for camera_id, result_queue in self.result_queues.items()
            if self.camera_slots.get(camera_id) == worker_index
        }
        worker = YoloWorker(
            request_queue,
            device,
            result_queues=result_queues,
            startup_lock=self.startup_lock,
        )
        process = Process(target=worker.run, daemon=True)
        process.start()
        self.logger.info(
            '[YOLO-Worker] process %s started on %s',
            worker_index,
            device,
        )
        return process

    def slot(
        self,
        cfg: StreamConfig,
    ) -> tuple[object | None, object | None]:
        """Return the worker and dedicated result queue assigned to a camera.

        Sharded mode pins up to the configured number of same-model cameras to
        one engine worker.
        """
        if not self.request_queues:
            return None, None
        result_queue = self.result_queue(cfg)
        if result_queue is None:
            return None, None
        slot_index = self.slots.get(stream_identity(cfg))
        if slot_index is not None:
            return (
                self.request_queues[slot_index],
                result_queue,
            )
        return None, None

    def result_queue(self, cfg: StreamConfig) -> Any | None:
        return self.result_queues.get(stream_identity(cfg))

    def close(self) -> None:
        """Stop the shared YOLO worker pool and close native IPC queues."""
        for request_queue in self.request_queues:
            try:
                request_queue.put(YOLO_WORKER_STOP_MESSAGE, block=False)
            except Exception as e:
                self.logger.error(f"Error signalling YOLO worker: {e}")
        for process in self.processes:
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join()
        for worker_queue in [
            *self.request_queues,
            *self.result_queues.values(),
        ]:
            with suppress(Exception):
                worker_queue.cancel_join_thread()
                worker_queue.close()
        self.request_queues.clear()
        self.result_queues.clear()
        self.processes.clear()
        self.slots.clear()
        self.camera_slots.clear()
        self.topology_signature = None
        if self.startup_lock is not None:
            self.startup_lock.close()
        self.startup_lock = None
        self.shards.clear()
        self.restarted_cameras.clear()

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

    def start_process(self, cfg: StreamConfig) -> Process:
        """Launch a new child process to handle stream detection.

        Args:
            cfg (StreamConfig): Configuration for the stream.

        Returns:
            Process: The new multiprocessing.Process object.
        """
        if not self.processes:
            self.ensure([cfg])
        yolo_request_queue, yolo_result_queue = self.slot(cfg)
        p = Process(
            target=process_single_stream,
            args=(cfg, yolo_request_queue, yolo_result_queue),
        )
        p.start()
        return p


def _stream_plan_order(pair: tuple[str, StreamConfig]) -> tuple[str, str]:
    return pair[1]['model_key'], pair[0]


class _ShardLayout:
    def __init__(self, settings: WorkerPoolSettings):
        self.default = settings.cameras_per_engine
        self.limits = settings.cameras_per_engine_by_model
        self.devices = settings.devices

    def specification(self, slot, model, revision, members):
        return ShardSpec(
            slot,
            model,
            revision,
            tuple(sorted(members)),
            self.limits.get(model.lower(), self.default),
            self.devices[slot % len(self.devices)],
        )
