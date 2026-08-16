from __future__ import annotations

from dataclasses import dataclass

from examples.violation_records.schemas import ViolationWarning
from examples.violation_records.schemas import ViolationWarningPayload


@dataclass(frozen=True)
class ViolationTypeDefinition:
    """Define canonical metadata for one violation type.

    Attributes:
        code: Stable API and database identifier.
        label: Localised display label for clients.
        warning_keys: Detector warning keys that activate the type.
    """

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

# Index canonical definitions once for request-time filter validation.
VIOLATION_TYPE_BY_CODE: dict[str, ViolationTypeDefinition] = {
    definition.code: definition for definition in VIOLATION_TYPE_DEFINITIONS
}


def normalise_violation_type(code: str) -> str | None:
    """Return a supported canonical violation-type code.

    Args:
        code: Client-supplied type code.

    Returns:
        Canonical code, or ``None`` when unsupported or blank.
    """
    normalized = code.strip()
    if not normalized:
        return None
    return normalized if normalized in VIOLATION_TYPE_BY_CODE else None


def parse_warning_payload(
    warnings_json: str | None,
) -> ViolationWarningPayload | None:
    """Decode the canonical warning payload stored with a violation.

    Args:
        warnings_json: Optional persisted detector-warning JSON.

    Returns:
        Validated warning payload, or ``None`` when no warning data exists.
    """
    if warnings_json is None:
        return None
    return ViolationWarningPayload.model_validate_json(warnings_json)


def violation_type_codes_from_warnings(
    warnings_json: str | None,
) -> list[str]:
    """Derive canonical codes from structured detector-warning keys.

    Args:
        warnings_json: Optional persisted detector-warning JSON.

    Returns:
        Active canonical violation-type codes.
    """
    payload = parse_warning_payload(warnings_json)
    if payload is None:
        return []

    return [
        definition.code
        for definition in VIOLATION_TYPE_DEFINITIONS
        if any(
            _is_active_warning(payload.root.get(key))
            for key in definition.warning_keys
        )
    ]


def _is_active_warning(value: ViolationWarning | None) -> bool:
    """Return whether a detector warning has a positive count.

    Args:
        value: Optional structured detector warning.

    Returns:
        ``True`` when the warning exists and records one or more instances.
    """
    return value is not None and value.count > 0
