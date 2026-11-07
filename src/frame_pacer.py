from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass


@dataclass
class FrameTiming:
    frames_sent: int = 0
    backpressure_skips: int = 0
    missed_deadlines: int = 0
    max_lateness_seconds: float = 0.0
    write_seconds: float = 0.0
    first_write_at: float | None = None
    last_write_at: float | None = None

    @property
    def actual_fps(self) -> float:
        if self.first_write_at is None or self.last_write_at is None:
            return 0.0
        elapsed = self.last_write_at - self.first_write_at
        return (self.frames_sent - 1) / elapsed if elapsed > 0 else 0.0


class FramePacer:
    """Own the raw-video clock and measure actual stdin delivery cadence."""

    def __init__(self, fps: float) -> None:
        self.interval = 1.0 / fps
        self.timing = FrameTiming()
        self.deadline: float | None = None
        self.started_at = 0.0
        self.reported_at: float | None = None

    def reset_deadline(self) -> None:
        self.deadline = None

    def begin(self) -> None:
        self.started_at = asyncio.get_running_loop().time()
        if self.deadline is None:
            self.deadline = self.started_at
        if self.reported_at is None:
            self.reported_at = self.started_at
        self.timing.max_lateness_seconds = max(
            self.timing.max_lateness_seconds,
            self.started_at - self.deadline,
        )

    async def wait(self) -> None:
        if self.deadline is not None:
            delay = self.deadline - asyncio.get_running_loop().time()
            if delay > 0:
                await asyncio.sleep(delay)
        self.begin()

    def complete(self, sent: bool) -> None:
        now = asyncio.get_running_loop().time()
        if sent:
            timing = self.timing
            timing.frames_sent += 1
            timing.write_seconds += now - self.started_at
            if timing.first_write_at is None:
                timing.first_write_at = self.started_at
            timing.last_write_at = self.started_at
        else:
            self.timing.backpressure_skips += 1
        assert self.deadline is not None
        self.deadline += self.interval
        if self.deadline < now - 1e-9:
            missed = math.ceil((now - self.deadline) / self.interval)
            self.deadline += missed * self.interval
            self.timing.missed_deadlines += missed

    def report_due(self) -> bool:
        now = asyncio.get_running_loop().time()
        if self.reported_at is not None and now - self.reported_at >= 30:
            self.reported_at = now
            return True
        return False
