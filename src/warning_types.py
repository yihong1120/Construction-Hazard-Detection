from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias

WarningParams: TypeAlias = Mapping[str, object]
Warnings: TypeAlias = Mapping[str, WarningParams]
MutableWarningParams: TypeAlias = dict[str, object]
MutableWarnings: TypeAlias = dict[str, MutableWarningParams]
