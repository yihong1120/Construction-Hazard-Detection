from __future__ import annotations

import asyncio
from collections.abc import Awaitable


async def run_coupled_tasks(tasks: set[asyncio.Task[object]]) -> None:
    """Stop and drain a coupled pipeline when any worker exits."""
    try:
        done, _ = await asyncio.wait(
            tasks,
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in done:
            task.result()
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def finish_on_cancel[T](operation: Awaitable[T]) -> T:
    """Keep ownership until an irreversible operation reports its outcome."""
    task = asyncio.ensure_future(operation)
    cancellation = None
    while True:
        try:
            await asyncio.shield(task)
            break
        except asyncio.CancelledError as error:
            cancellation = error
            if task.done():
                break
        except BaseException:
            if cancellation is not None:
                raise cancellation
            raise
    if cancellation is not None:
        if not task.cancelled():
            task.exception()
        raise cancellation
    return task.result()


def observe_task_exception(done: asyncio.Future) -> None:
    if not done.cancelled():
        done.exception()
