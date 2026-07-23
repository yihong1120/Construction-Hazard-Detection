from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ViolationTypeDefinition:
    """Canonical type metadata derived from structured warning keys."""

    code: str
    label: str
    warning_keys: tuple[str, ...]


VIOLATION_TYPE_DEFINITIONS: tuple[ViolationTypeDefinition, ...] = (
    ViolationTypeDefinition(
        code='no_safety_helmet',
        label='未戴安全帽',
        warning_keys=('warning_no_hardhat',),
    ),
    ViolationTypeDefinition(
        code='no_safety_vest',
        label='未穿安全背心',
        warning_keys=('warning_no_safety_vest',),
    ),
    ViolationTypeDefinition(
        code='near_vehicle',
        label='人員靠近車輛',
        warning_keys=('warning_close_to_vehicle',),
    ),
    ViolationTypeDefinition(
        code='near_machinery',
        label='人員靠近機具',
        warning_keys=('warning_close_to_machinery',),
    ),
    ViolationTypeDefinition(
        code='restricted_area',
        label='進入管制區',
        warning_keys=('warning_people_in_controlled_area',),
    ),
    ViolationTypeDefinition(
        code='utility_pole_restricted_area',
        label='進入電桿管制區',
        warning_keys=('warning_people_in_utility_pole_controlled_area',),
    ),
    ViolationTypeDefinition(
        code='machinery_close_to_pole',
        label='機具靠近電桿',
        warning_keys=('detect_machinery_close_to_pole',),
    ),
)

VIOLATION_TYPE_BY_CODE: dict[str, ViolationTypeDefinition] = {
    definition.code: definition
    for definition in VIOLATION_TYPE_DEFINITIONS
}

# Keep existing analytics clients working while publishing only canonical codes.
VIOLATION_TYPE_ALIASES: dict[str, str] = {
    'no_helmet': 'no_safety_helmet',
    'no_hardhat': 'no_safety_helmet',
    'no_vest': 'no_safety_vest',
    'controlled_area': 'restricted_area',
}


def normalise_violation_type(code: str) -> str | None:
    """Return a canonical type code, accepting documented legacy aliases."""
    normalized = code.strip()
    if not normalized:
        return None
    normalized = VIOLATION_TYPE_ALIASES.get(normalized, normalized)
    return normalized if normalized in VIOLATION_TYPE_BY_CODE else None


def violation_type_codes_from_warnings(
    warnings: str | Mapping[str, Any] | None,
) -> list[str]:
    """Derive canonical codes from the warning payload's structured keys."""
    payload: object = warnings
    if isinstance(warnings, str):
        try:
            payload = json.loads(warnings)
        except json.JSONDecodeError:
            return []

    if not isinstance(payload, Mapping):
        return []

    return [
        definition.code
        for definition in VIOLATION_TYPE_DEFINITIONS
        if any(
            _is_active_warning(payload.get(key))
            for key in definition.warning_keys
        )
    ]


def _is_active_warning(value: object) -> bool:
    """Treat zero-count warning entries as inactive."""
    if isinstance(value, Mapping) and 'count' in value:
        count = value['count']
        if isinstance(count, bool):
            return count
        if isinstance(count, (int, float)):
            return count > 0
    return bool(value)
