from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Self

import numpy as np

from src.yolo_worker_protocol import FrameArray


class _SharedFrameArray(np.ndarray):
    """NumPy view that keeps its shared-memory mapping alive for inference."""

    shared_memory_handle: shared_memory.SharedMemory | None


@dataclass
class _SharedFrameRing:
    """Fixed shared-memory slots owned by one stream-process camera client."""

    shape: tuple[int, ...]
    dtype: np.dtype[np.uint8]
    slots: int
    shared_memory_handle: shared_memory.SharedMemory
    _available_slots: asyncio.Queue[int]

    @classmethod
    def create(
        cls: type[Self],
        shape: tuple[int, ...],
        dtype: np.dtype[np.uint8],
        slots: int,
    ) -> _SharedFrameRing:
        """Allocate a bounded shared-memory ring for one frame format."""
        slot_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        shared_memory_handle = shared_memory.SharedMemory(
            create=True,
            size=slot_bytes * slots,
        )
        available_slots: asyncio.Queue[int] = asyncio.Queue(maxsize=slots)
        for slot in range(slots):
            available_slots.put_nowait(slot)
        return cls(
            shape=shape,
            dtype=dtype,
            slots=slots,
            shared_memory_handle=shared_memory_handle,
            _available_slots=available_slots,
        )

    @property
    def name(self) -> str:
        """Return the POSIX shared-memory name used by worker requests."""
        return self.shared_memory_handle.name

    @property
    def slot_bytes(self) -> int:
        """Return the byte capacity of one fixed frame slot."""
        return int(np.prod(self.shape, dtype=np.int64)) * self.dtype.itemsize

    async def acquire_slot(self, timeout_seconds: float) -> int:
        """Reserve one slot until the corresponding worker result arrives."""
        try:
            return await asyncio.wait_for(
                self._available_slots.get(),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                'YOLO worker shared frame ring is full',
            ) from exc

    def release_slot(self, slot: int) -> None:
        """Make a finished slot available for the next camera frame."""
        with suppress(asyncio.QueueFull):
            self._available_slots.put_nowait(slot)

    def write(self, slot: int, frame: FrameArray) -> None:
        """Copy one contiguous frame into its reserved shared-memory slot."""
        target: FrameArray = np.ndarray(
            self.shape,
            dtype=self.dtype,
            buffer=self.shared_memory_handle.buf,
            offset=slot * self.slot_bytes,
        )
        target[:] = frame

    def close(self) -> None:
        """Close and unlink this ring after all users have stopped."""
        self.shared_memory_handle.close()
        with suppress(FileNotFoundError):
            self.shared_memory_handle.unlink()
