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
            f'Unsupported quantize value: {raw_value!r}. '
            f'Use one of: {supported}.',
        )
    if value.isdigit():
        return int(value)
    return value


def precision_kwargs(
    enabled: bool,
    quantize: QuantizeValue | None = None,
) -> dict[str, PrecisionValue]:
    """Return the supported Ultralytics precision flag for this install.

    Newer Ultralytics versions renamed the inference-time ``half`` flag to
    ``quantize``. Older versions still only accept ``half``. Checking the
    runtime config keeps both environments working without deprecation noise.
    """
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT
    except Exception:
        return {'half': _legacy_half_value(enabled, quantize)}

    if 'quantize' in DEFAULT_CFG_DICT:
        if quantize is not None:
            return {'quantize': quantize}
        return {'quantize': 16 if enabled else 32}
    return {'half': _legacy_half_value(enabled, quantize)}


def _legacy_half_value(
    enabled: bool,
    quantize: QuantizeValue | None,
) -> bool:
    """Map an explicit quantize request to the closest legacy half setting."""
    if quantize is None:
        return enabled

    value = str(quantize).strip().lower()
    if value in _FP32_VALUES:
        return False
    return True
