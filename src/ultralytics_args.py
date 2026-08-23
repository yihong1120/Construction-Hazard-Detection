from __future__ import annotations

from typing import TypeAlias

QuantizeValue: TypeAlias = int | str
PrecisionValue: TypeAlias = bool | int | str

_FP16_VALUES = {'16', 'fp16', 'w16a16'}
_FP32_VALUES = {'32', 'fp32', 'w32a32'}
_INT8_VALUES = {'8', 'int8', 'w8a8', 'w8a16'}
_SUPPORTED_QUANTIZE_VALUES = _FP16_VALUES | _FP32_VALUES | _INT8_VALUES


def parse_quantize_value(raw_value: str | None) -> QuantizeValue | None:
    """Parse an optional Ultralytics quantize setting from an env value."""
    if raw_value is None:
        return None

    value = raw_value.strip().lower()
    if value in {'', 'none', 'null', 'default', 'auto'}:
        return None
    if value not in _SUPPORTED_QUANTIZE_VALUES:
        supported = ', '.join(sorted(_SUPPORTED_QUANTIZE_VALUES))
        raise ValueError(
            f"Unsupported quantize value: {raw_value!r}. "
            f"Use one of: {supported}.",
        )
    if value.isdigit():
        return int(value)
    return value


def precision_kwargs(
    enabled: bool,
    quantize: QuantizeValue | None = None,
) -> dict[str, PrecisionValue]:
    """Return precision options for the pinned Ultralytics release."""
    return {
        'quantize': (
            quantize if quantize is not None else (16 if enabled else 32)
        ),
    }
