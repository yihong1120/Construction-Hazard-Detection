from __future__ import annotations

import asyncio


async def cancel_on_first_failure(tasks: set[asyncio.Task[object]]) -> None:
    """Wait for task failure, then cancel and drain all remaining work."""
    done, pending = await asyncio.wait(
        tasks,
        return_when=asyncio.FIRST_EXCEPTION,
    )
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    for task in done:
        exception = task.exception()
        if exception is not None:
            raise exception
