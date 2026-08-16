from __future__ import annotations

import pytest

from src.ultralytics_args import parse_quantize_value
from src.ultralytics_args import precision_kwargs


def test_parse_quantize_value_accepts_supported_values() -> None:
    """Quantize env values can request lower precision explicitly."""
    assert parse_quantize_value(None) is None
    assert parse_quantize_value('') is None
    assert parse_quantize_value('8') == 8
    assert parse_quantize_value('16') == 16
    assert parse_quantize_value('32') == 32
    assert parse_quantize_value('int8') == 'int8'
    assert parse_quantize_value('fp16') == 'fp16'


def test_parse_quantize_value_rejects_unknown_values() -> None:
    """Invalid quantize settings should fail early during startup."""
    with pytest.raises(ValueError):
        parse_quantize_value('lowest')


def test_precision_kwargs_uses_pinned_quantize_api() -> None:
    """The pinned Ultralytics release accepts the quantize option."""
    assert precision_kwargs(True) == {'quantize': 16}
    assert precision_kwargs(False) == {'quantize': 32}
    assert precision_kwargs(True, 8) == {'quantize': 8}
