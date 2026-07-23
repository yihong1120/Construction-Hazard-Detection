from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

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


def test_precision_kwargs_uses_quantize_when_supported(
    monkeypatch: Any,
) -> None:
    """Newer Ultralytics installs use quantize instead of half."""
    cfg_module = ModuleType('ultralytics.cfg')
    cfg_module.DEFAULT_CFG_DICT = {'quantize': False}
    monkeypatch.setitem(sys.modules, 'ultralytics.cfg', cfg_module)

    assert precision_kwargs(True) == {'quantize': 16}
    assert precision_kwargs(False) == {'quantize': 32}
    assert precision_kwargs(True, 8) == {'quantize': 8}


def test_precision_kwargs_falls_back_to_half_when_needed(
    monkeypatch: Any,
) -> None:
    """Older Ultralytics installs still accept half."""
    cfg_module = ModuleType('ultralytics.cfg')
    cfg_module.DEFAULT_CFG_DICT = {'half': False}
    monkeypatch.setitem(sys.modules, 'ultralytics.cfg', cfg_module)

    assert precision_kwargs(False) == {'half': False}
    assert precision_kwargs(True, 8) == {'half': True}
    assert precision_kwargs(True, 32) == {'half': False}
