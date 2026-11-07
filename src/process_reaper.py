from __future__ import annotations

import asyncio
import time
from multiprocessing import Process


class ProcessReaper:
    def __init__(self, timeout: float = 10.0) -> None:
        self.timeout = timeout

    def stop(
        self,
        process: Process,
        deadline: float,
        graceful: bool = False,
    ) -> None:
        if not graceful and process.is_alive():
            process.terminate()
        process.join(timeout=max(0.0, deadline - time.monotonic()))
        if process.is_alive():
            process.kill()
            process.join(timeout=2)
        if process.is_alive():
            raise RuntimeError('Child process did not stop after kill')

    async def reap(
        self,
        processes: list[Process],
        graceful: bool = False,
    ) -> None:
        await _ReapBatch(self, graceful).run(processes)


class _ReapBatch:
    def __init__(self, reaper: ProcessReaper, graceful: bool):
        self.reaper = reaper
        self.graceful = graceful
        self.deadline = time.monotonic() + reaper.timeout
        self.capacity = asyncio.Semaphore(4)

    async def run(self, processes: list[Process]) -> None:
        work = asyncio.gather(
            *(self.stop(process) for process in processes),
            return_exceptions=True,
        )
        try:
            results = await asyncio.shield(work)
        except asyncio.CancelledError:
            await work
            raise
        for result in results:
            if isinstance(result, BaseException):
                raise result

    async def stop(self, process: Process) -> None:
        async with self.capacity:
            await asyncio.to_thread(
                self.reaper.stop,
                process,
                self.deadline,
                self.graceful,
            )
