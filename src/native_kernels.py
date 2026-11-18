"""Optional, versioned kernels sharing the camera postprocessing backend."""
from __future__ import annotations

import os
from functools import lru_cache

from src.settings.postprocess import PostprocessSettings


def optional_kernel(name: str, capability: str):
    return _load_kernel(
        name, capability, os.environ.get('POSTPROCESS_BACKEND', 'auto'),
    )


@lru_cache(maxsize=16)
def _load_kernel(name: str, capability: str, backend: str):
    from src.native_postprocess import load_postprocess_backend

    settings = PostprocessSettings(backend=backend)
    extension = load_postprocess_backend(settings.backend)
    if extension is None or getattr(extension, capability, None) != 1:
        return None
    operation = getattr(extension, name, None)
    if not callable(operation):
        raise RuntimeError(f'Incomplete native kernel: {name}')
    return operation
