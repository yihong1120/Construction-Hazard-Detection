from __future__ import annotations

from collections.abc import Callable

from src.native_work import NativeWorkPool

_pool: NativeWorkPool | None = None


async def run_image_work[T](operation: Callable[[], T]) -> T:
    global _pool
    if _pool is None or _pool._closed:
        _pool = NativeWorkPool(code='IMAGE_WORK_BUSY', name='image')
    return await _pool.run(operation)


async def close_image_work():
    global _pool
    pool, _pool = _pool, None
    if pool is not None:
        await pool.close()
