from __future__ import annotations

from typing import TypeAlias

WorkerPrecisionMode: TypeAlias = str

_WORKER_PRECISION_ALIASES = {
    '32': 'f32',
    'float32': 'f32',
    'fp32': 'f32',
    'f32': 'f32',
    '16': 'f16',
    'float16': 'f16',
    'fp16': 'f16',
    'f16': 'f16',
    '8': 'int8',
    'engine': 'int8',
    'int8': 'int8',
}


def _parse_worker_precision(
    raw_value: str | None,
) -> WorkerPrecisionMode | None:
    """Return a simplified worker precision mode from environment text."""
    if raw_value is None:
        return None
    value = raw_value.strip().lower()
    if value in {'', 'none', 'null', 'default', 'auto', 'legacy'}:
        return None
    try:
        return _WORKER_PRECISION_ALIASES[value]
    except KeyError as exc:
        supported = ', '.join(sorted(set(_WORKER_PRECISION_ALIASES)))
        raise ValueError(
            f'Unsupported YOLO_WORKER_PRECISION: {raw_value!r}. '
            f'Use f32, f16, or int8. Supported aliases: {supported}.',
        ) from exc
